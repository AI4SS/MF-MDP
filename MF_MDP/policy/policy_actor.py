import os
import torch
import torch.nn as nn
from typing import Optional, Union
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM


class PolicyActor(nn.Module):
    """
    改进的 Policy Actor 模型，在 IB-Tune Actor 基础上集成:
    1. LoRA 微调的 LLM
    2. 串行 MLP 预测头 (参考 Mean-Field-LLM 的设计)
    3. 支持 K 步状态预测
    4. 支持 Soft Best-of-4 采样策略
    5. 支持 Beta 动态调度

    Args:
        pretrain_or_model: 预训练模型名称或模型实例
        k_steps: 预测未来 K 步的状态
        state_dim: 状态维度 (默认3: pos, neg, neu)
        lora_rank: LoRA 秩
        lora_alpha: LoRA alpha 参数
        gamma: 时间衰减系数
        label_smoothing: 标签平滑系数
        loss_threshold: KL loss 阈值，用于过滤高误差样本
        use_soft_best_of: 是否使用 soft best-of-4 采样策略
        soft_best_beta: soft best-of-4 的选择强度参数（固定值）
        beta_schedule: Beta 调度策略 ('constant', 'linear', 'cosine')
        beta_warmup_ratio: Beta 预热比例（前多少比例的步骤用于预热）
        beta_min: Beta 最小值（训练初期）
        beta_max: Beta 最大值（训练后期）
    """

    def __init__(
        self,
        pretrain_or_model,
        k_steps=10,
        state_dim=3,
        lora_rank=8,
        lora_alpha=16,
        gamma=0.9,
        label_smoothing=0.1,
        loss_threshold=20.0,
        use_flash_attention_2=False,
        bf16=True,
        use_soft_best_of=False,
        soft_best_beta=1.0,
        beta_schedule='constant',
        beta_warmup_ratio=0.1,
        beta_min=0.5,
        beta_max=5.0,
        **kwargs
    ):
        super().__init__()
        self.k_steps = k_steps
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.loss_threshold = loss_threshold
        self.use_soft_best_of = use_soft_best_of
        self.soft_best_beta = soft_best_beta

        # Beta 调度相关
        self.beta_schedule = beta_schedule
        self.beta_warmup_ratio = beta_warmup_ratio
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.current_step = 0  # 当前训练步数（用于计算动态 beta）

        # 1. 加载基座 LLM
        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"
            base_model = AutoModelForCausalLM.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
            )
        else:
            base_model = pretrain_or_model

        # 记录 hidden_size
        hidden_size = base_model.config.hidden_size

        # 2. 配置并应用 LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        self.llm = get_peft_model(base_model, peft_config)

        # 3. 定义 K 个预测头（串行结构）
        # 第 i 个头能看到前面 i 个预测结果
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size + state_dim + i * state_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, state_dim)
            ) for i in range(k_steps)
        ])

        # 确保 MLP 权重是 float32 且开启梯度
        for head in self.prediction_heads:
            head.to(torch.float32)
            for param in head.parameters():
                param.requires_grad = True

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        current_state=None,
        future_states=None,
        return_output=False,
        **kwargs
    ):
        """
        Forward pass:
        1. 计算 LLM 的 text loss (带 label smoothing)
        2. 提取 [EOS] token 的 hidden state
        3. 通过串行 MLP 预测未来 K 步状态
        4. 计算 KL divergence loss (带阈值过滤)

        如果启用 use_soft_best_of，则调用 forward_with_soft_best_of
        """
        if self.use_soft_best_of:
            return self.forward_with_soft_best_of(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                current_state=current_state,
                future_states=future_states,
                return_output=return_output
            )

        # 原始 forward 逻辑（单次生成）

        # 1. LLM forward (不使用自带的 loss，手动计算以支持 label smoothing)
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # 手动计算 loss
            output_hidden_states=True
        )

        # 2. 手动计算 text loss with label smoothing
        text_loss = None
        if labels is not None:
            logits = outputs.logits
            vocab_size = logits.shape[-1]

            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten
            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)

            # 只计算非 -100 的位置
            mask = shift_labels != -100
            shift_logits = shift_logits[mask]
            shift_labels = shift_labels[mask]

            # 计算 cross-entropy with label smoothing
            if self.label_smoothing > 0 and len(shift_labels) > 0:
                n_classes = vocab_size
                smoothing = self.label_smoothing

                log_prob = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                nll_loss = -log_prob.gather(dim=-1, index=shift_labels.unsqueeze(1)).squeeze(1)
                smooth_loss = -log_prob.mean(dim=-1)

                text_loss = (1 - smoothing) * nll_loss + smoothing * smooth_loss
                text_loss = text_loss.mean()
            elif len(shift_labels) > 0:
                text_loss = torch.nn.functional.cross_entropy(
                    shift_logits,
                    shift_labels,
                    reduction='mean'
                )
            else:
                text_loss = torch.tensor(0.0, device=logits.device)

        # 3. 提取特征向量 (使用 [EOS] 位置的 hidden state)
        last_hidden_state = outputs.hidden_states[-1]
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        feature_vector = last_hidden_state[torch.arange(batch_size, device=input_ids.device), sequence_lengths]
        feature_vector = feature_vector.to(torch.float32)

        # 4. 串行状态预测
        per_sample_pred_loss = torch.zeros(batch_size, device=input_ids.device)
        predicted_trajectories = []

        # 用于计算准确率指标
        total_mse = 0.0
        total_mae = 0.0
        direction_correct = 0
        direction_total = 0

        if current_state is not None:
            # 串行预测：每一步都能看到前面所有步骤的预测结果
            for i, head in enumerate(self.prediction_heads):
                # 构造输入：[llm_hidden_state, current_state, pred_dist_0, ..., pred_dist_{i-1}]
                if i == 0:
                    combined_input = torch.cat([feature_vector, current_state], dim=-1)
                else:
                    # 拼接前面所有的预测结果
                    previous_preds = torch.cat(predicted_trajectories, dim=-1)
                    combined_input = torch.cat([feature_vector, current_state, previous_preds], dim=-1)

                delta = head(combined_input)
                pred_dist = torch.softmax(current_state + delta, dim=-1)
                predicted_trajectories.append(pred_dist)

                if future_states is not None:
                    target_dist = future_states[:, i, :]

                    # 使用 reduction='none' 计算每个样本的 KL
                    step_kl_per_sample = nn.functional.kl_div(
                        pred_dist.log(),
                        target_dist,
                        reduction='none'
                    ).sum(dim=-1)

                    # 累加带有时间衰减的 loss
                    per_sample_pred_loss += (self.gamma ** i) * step_kl_per_sample

                    # 计算准确率指标
                    mse = nn.functional.mse_loss(pred_dist, target_dist, reduction='mean')
                    total_mse += mse

                    mae = nn.functional.l1_loss(pred_dist, target_dist, reduction='mean')
                    total_mae += mae

                    # 方向准确率
                    pred_delta = pred_dist - current_state
                    target_delta = target_dist - current_state
                    direction_match = ((pred_delta * target_delta) >= 0).all(dim=-1)
                    direction_correct += direction_match.sum().item()
                    direction_total += direction_match.shape[0]

        # 5. 应用阈值过滤策略
        final_pred_loss = torch.tensor(0.0, device=input_ids.device)
        drop_rate = 0.0

        if future_states is not None:
            # 创建 Mask：误差 < 阈值的保留 (1.0)，否则丢弃 (0.0)
            valid_mask = (per_sample_pred_loss < self.loss_threshold).float()

            num_valid = valid_mask.sum()
            num_total = batch_size
            drop_rate = 1.0 - (num_valid / num_total).item()

            # 只计算有效样本的平均 Loss
            if num_valid > 0:
                final_pred_loss = (per_sample_pred_loss * valid_mask).sum() / num_valid
            else:
                # 极端情况：整个 Batch 都超阈值
                final_pred_loss = per_sample_pred_loss.mean() * 0.0

        # 6. 计算平均指标
        metrics = {}
        if future_states is not None and current_state is not None:
            n_steps = len(self.prediction_heads)
            metrics['pred/mse'] = total_mse / n_steps
            metrics['pred/mae'] = total_mae / n_steps
            metrics['pred/direction_accuracy'] = direction_correct / direction_total if direction_total > 0 else 0.0

        output_dict = {
            "loss": text_loss,
            "pred_loss": final_pred_loss,
            "logits": outputs.logits,
            "predicted_trajectories": torch.stack(predicted_trajectories, dim=1) if predicted_trajectories else None,
            "drop_rate": torch.tensor(drop_rate),
            "raw_mean_loss": per_sample_pred_loss.mean(),
            "metrics": metrics
        }

        if return_output:
            return output_dict
        else:
            return output_dict

    # def forward_with_soft_best_of(
    #     self,
    #     input_ids,
    #     attention_mask,
    #     labels=None,
    #     current_state=None,
    #     future_states=None,
    #     return_output=False,
    #     temperature=0.8,
    #     top_p=0.9
    # ):
    #     """
    #     Soft Best-of-4 采样策略的 Forward Pass

    #     核心思路：
    #     1. 对每个样本，用 LLM 采样生成 4 条不同的候选评论
    #     2. 对每条候选，提取 hidden state，通过 MLP 预测状态分布
    #     3. 计算四条候选的 pred loss（KL divergence）
    #     4. 用 softmax 将 loss 转换为权重（loss 小的权重大）
    #     5. 权重计算使用 detach()，防止梯度耦合
    #     6. 加权组合得到最终的 pred loss

    #     Args:
    #         input_ids: [batch, seq_len]
    #         attention_mask: [batch, seq_len]
    #         labels: [batch, seq_len] - 用于计算 text loss
    #         current_state: [batch, 3] - 当前状态分布
    #         future_states: [batch, k_steps, 3] - 未来状态真实分布
    #         return_output: 是否返回详细信息
    #         temperature: 采样温度
    #         top_p: nucleus sampling 参数

    #     Returns:
    #         output_dict: 包含 loss, pred_loss, metrics 等
    #     """
    #     batch_size = input_ids.shape[0]
    #     device = input_ids.device



    #     candidates_outputs = []
    #     candidates_features = []

    #     # 设置模型为 training 模式以启用 dropout，但使用 no_grad 不保存梯度
    #     was_training = self.training
    #     self.train()  # 启用 dropout

    #     with torch.no_grad():
    #         # 候选 1: 第一次 forward（dropout 随机性 A）
    #         outputs_1 = self.llm(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             output_hidden_states=True
    #         )
    #         feature_1 = self._extract_last_hidden_state(
    #             outputs_1.hidden_states[-1],
    #             attention_mask
    #         )
    #         candidates_outputs.append(outputs_1)
    #         candidates_features.append(feature_1)

    #         # 候选 2: 第二次 forward（dropout 随机性 B，可能不同于 A）
    #         outputs_2 = self.llm(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             output_hidden_states=True
    #         )
    #         feature_2 = self._extract_last_hidden_state(
    #             outputs_2.hidden_states[-1],
    #             attention_mask
    #         )
    #         candidates_outputs.append(outputs_2)
    #         candidates_features.append(feature_2)

    #     # 恢复原始模式
    #     if not was_training:
    #         self.eval()

    #     # =====================================================================
    #     # Step 3: 计算每个候选的 text loss 和 pred loss
    #     # =====================================================================
    #     text_losses = []
    #     pred_losses = []
    #     predicted_trajectories_list = []

    #     for idx, (outputs, feature) in enumerate(zip(candidates_outputs, candidates_features)):
    #         # 3.1 计算 text loss (带 label smoothing)
    #         text_loss = self._compute_text_loss(
    #             outputs.logits,
    #             labels,
    #             self.label_smoothing
    #         )
    #         text_losses.append(text_loss)

    #         # 3.2 计算 pred loss 和预测轨迹
    #         if current_state is not None:
    #             pred_loss, pred_traj = self._compute_prediction_loss(
    #                 feature,
    #                 current_state,
    #                 future_states
    #             )
    #             pred_losses.append(pred_loss)
    #             predicted_trajectories_list.append(pred_traj)
    #         else:
    #             pred_losses.append(torch.tensor(0.0, device=device))
    #             predicted_trajectories_list.append(None)

    #     # =====================================================================
    #     # Step 4: Soft best-of-2 权重计算
    #     # =====================================================================
    #     # 核心：使用 detach() 防止权重影响 loss 的梯度
    #     if len(pred_losses) > 0 and future_states is not None:
    #         # 将 pred losses 堆叠 [2, batch]
    #         pred_losses_tensor = torch.stack(pred_losses)  # [2]

    #         # 转换为权重（loss 小的权重大）
    #         # alpha = softmax(-beta * loss)
    #         weights = torch.softmax(-self.soft_best_beta * pred_losses_tensor.detach(), dim=0)  # [2]

    #         # 加权组合 pred loss
    #         # L = alpha_1 * loss_1 + alpha_2 * loss_2
    #         final_pred_loss = (weights[0] * pred_losses[0] + weights[1] * pred_losses[1])

    #         # 同样加权组合 text loss（可选，也可以简单平均）
    #         final_text_loss = (weights[0] * text_losses[0] + weights[1] * text_losses[1])

    #         # 记录权重供监控
    #         weights_for_log = weights.detach().cpu().numpy()

    #     else:
    #         # 没有 future_states 时，简单平均
    #         final_pred_loss = torch.tensor(0.0, device=device)
    #         final_text_loss = (text_losses[0] + text_losses[1]) / 2.0
    #         weights_for_log = [0.5, 0.5]

    #     # =====================================================================
    #     # Step 5: 计算详细指标
    #     # =====================================================================
    #     metrics = {}
    #     if current_state is not None and future_states is not None:
    #         # 使用加权后的预测轨迹计算指标
    #         weighted_traj = None
    #         if predicted_trajectories_list[0] is not None:
    #             weighted_traj = (
    #                 weights[0].view(-1, 1, 1) * predicted_trajectories_list[0] +
    #                 weights[1].view(-1, 1, 1) * predicted_trajectories_list[1]
    #             )
    #             metrics = self._compute_prediction_metrics(
    #                 weighted_traj,
    #                 current_state,
    #                 future_states
    #             )

    #     # =====================================================================
    #     # Step 6: 构建输出
    #     # =====================================================================
    #     output_dict = {
    #         "loss": final_text_loss,
    #         "pred_loss": final_pred_loss,
    #         "logits": candidates_outputs[0].logits,  # 使用第一个候选的 logits
    #         "predicted_trajectories": weighted_traj,
    #         "drop_rate": torch.tensor(0.0),  # soft best-of 不使用丢弃策略
    #         "raw_mean_loss": torch.stack(pred_losses).mean() if pred_losses else torch.tensor(0.0),
    #         "metrics": metrics,
    #         # 额外信息用于监控
    #         "soft_best_weights": weights_for_log.tolist(),  # [weight_1, weight_2]
    #         "candidate_losses": [l.detach().item() for l in pred_losses] if pred_losses else [],
    #     }

    #     return output_dict
    def forward_with_soft_best_of(
        self,
        input_ids,
        attention_mask,
        labels=None,
        current_state=None,
        future_states=None,
        return_output=False,
        noise_std=0.01,  # 控制随机扰动的强度
        seed_offset=0    # 基础种子偏移量
    ):
        """
        基于随机种子采样（非 Dropout）的 Soft Best-of-4 策略
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 1. 基础 Forward 获取 Logits 和 Hidden States
        # 注意：这里关闭了所有内部随机性（如 dropout），只通过外部注入噪声
        was_training = self.training
        self.eval() 

        with torch.no_grad():
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # 提取原始特征向量
            base_feature = self._extract_last_hidden_state(
                outputs.hidden_states[-1],
                attention_mask
            )

        # 2. 生成四个带有不同随机种子的候选特征
        # 我们对特征向量注入噪声，模拟"模型在不同采样下的不同表现"
        candidates_features = []

        # 为了保证训练的确定性，可以结合当前步数生成种子
        seeds = [
            123 + seed_offset + self.current_step,
            456 + seed_offset + self.current_step,
            789 + seed_offset + self.current_step,
            012 + seed_offset + self.current_step
        ]

        for s in seeds:
            # 创建独立的生成器以保证四次噪声不相关
            generator = torch.Generator(device=device).manual_seed(s)
            # 生成扰动噪声: noise ~ N(0, noise_std)
            noise = torch.randn(
                base_feature.shape, 
                generator=generator, 
                device=device, 
                dtype=base_feature.dtype
            ) * noise_std
            # 注入噪声形成不同的“候选理解”
            candidates_features.append(base_feature + noise)

        # 3. 计算每个候选的损失
        text_losses = []
        pred_losses = []
        predicted_trajectories_list = []

        # 计算 Text Loss (基于原始输出，因为 labels 是固定的)
        # 如果你想让 Text Loss 也受影响，则需要两次带梯度的完整 Forward
        text_loss_main = self._compute_text_loss(outputs.logits, labels, self.label_smoothing)

        for feature in candidates_features:
            # 计算预测头 Loss (带梯度)
            if current_state is not None:
                # 注意：此时 feature 已经带有了随机种子的扰动
                p_loss, p_traj = self._compute_prediction_loss(
                    feature,
                    current_state,
                    future_states
                )
                pred_losses.append(p_loss)
                predicted_trajectories_list.append(p_traj)

        # 恢复原始训练模式
        if was_training:
            self.train()

        # 4. Soft best-of-4 权重计算 (与之前逻辑一致)
        if len(pred_losses) == 4 and future_states is not None:
            pred_losses_tensor = torch.stack(pred_losses)
            # alpha = softmax(-beta * loss)
            weights = torch.softmax(-self.soft_best_beta * pred_losses_tensor.detach(), dim=0)

            # 加权组合最终的 pred loss
            final_pred_loss = (
                weights[0] * pred_losses[0] +
                weights[1] * pred_losses[1] +
                weights[2] * pred_losses[2] +
                weights[3] * pred_losses[3]
            )

            # 由于 text_loss 共享同一个 LLM base，我们这里直接使用 text_loss_main
            # 或者也可以根据 MLP 的表现来加权更新 LLM (类似于 RL 中的优势加权)
            final_text_loss = text_loss_main

            weighted_traj = (
                weights[0].view(-1, 1, 1) * predicted_trajectories_list[0] +
                weights[1].view(-1, 1, 1) * predicted_trajectories_list[1] +
                weights[2].view(-1, 1, 1) * predicted_trajectories_list[2] +
                weights[3].view(-1, 1, 1) * predicted_trajectories_list[3]
            )
            
            metrics = self._compute_prediction_metrics(weighted_traj, current_state, future_states)
        else:
            final_pred_loss = torch.tensor(0.0, device=device)
            final_text_loss = text_loss_main
            weighted_traj = None
            metrics = {}

        # 5. 构建输出
        return {
            "loss": final_text_loss,
            "pred_loss": final_pred_loss,
            "logits": outputs.logits,
            "predicted_trajectories": weighted_traj,
            "metrics": metrics,
            "soft_best_weights": weights.detach().cpu().numpy().tolist() if 'weights' in locals() else [0.25, 0.25, 0.25, 0.25]
        }
    def _extract_last_hidden_state(self, last_hidden_state, attention_mask):
        """
        提取最后一个有效 token 的 hidden state

        Args:
            last_hidden_state: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len]

        Returns:
            feature: [batch, hidden_dim]
        """
        batch_size = last_hidden_state.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        feature_vector = last_hidden_state[
            torch.arange(batch_size, device=last_hidden_state.device),
            sequence_lengths
        ]
        return feature_vector.to(torch.float32)

    def _compute_text_loss(self, logits, labels, label_smoothing=0.0):
        """
        计算 text generation loss (带 label smoothing)

        Args:
            logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len]
            label_smoothing: 标签平滑系数

        Returns:
            loss: scalar
        """
        if labels is None:
            return torch.tensor(0.0, device=logits.device)

        vocab_size = logits.shape[-1]

        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # 只计算非 -100 的位置
        mask = shift_labels != -100
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        # 计算 cross-entropy with label smoothing
        if label_smoothing > 0:
            log_prob = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            nll_loss = -log_prob.gather(dim=-1, index=shift_labels.unsqueeze(1)).squeeze(1)
            smooth_loss = -log_prob.mean(dim=-1)
            loss = (1 - label_smoothing) * nll_loss + label_smoothing * smooth_loss
            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(
                shift_logits,
                shift_labels,
                reduction='mean'
            )

    def _compute_prediction_loss(self, feature_vector, current_state, future_states):
        """
        计算 MLP 预测 loss

        Args:
            feature_vector: [batch, hidden_dim]
            current_state: [batch, 3]
            future_states: [batch, k_steps, 3]

        Returns:
            pred_loss: scalar
            predicted_trajectories: [batch, k_steps, 3]
        """
        batch_size = feature_vector.shape[0]
        device = feature_vector.device

        per_sample_pred_loss = torch.zeros(batch_size, device=device)
        predicted_trajectories = []

        # 串行状态预测
        for i, head in enumerate(self.prediction_heads):
            if i == 0:
                combined_input = torch.cat([feature_vector, current_state], dim=-1)
            else:
                previous_preds = torch.cat(predicted_trajectories, dim=-1)
                combined_input = torch.cat([feature_vector, current_state, previous_preds], dim=-1)

            delta = head(combined_input)
            pred_dist = torch.softmax(current_state + delta, dim=-1)
            predicted_trajectories.append(pred_dist)

            if future_states is not None:
                target_dist = future_states[:, i, :]
                step_kl_per_sample = nn.functional.kl_div(
                    pred_dist.log(),
                    target_dist,
                    reduction='none'
                ).sum(dim=-1)
                per_sample_pred_loss += (self.gamma ** i) * step_kl_per_sample

        # 应用阈值过滤
        if future_states is not None:
            valid_mask = (per_sample_pred_loss < self.loss_threshold).float()
            num_valid = valid_mask.sum()

            if num_valid > 0:
                final_pred_loss = (per_sample_pred_loss * valid_mask).sum() / num_valid
            else:
                final_pred_loss = per_sample_pred_loss.mean() * 0.0
        else:
            final_pred_loss = torch.tensor(0.0, device=device)

        return final_pred_loss, torch.stack(predicted_trajectories, dim=1) if predicted_trajectories else None

    def _compute_prediction_metrics(self, predicted_trajectories, current_state, future_states):
        """
        计算预测准确率指标

        Args:
            predicted_trajectories: [batch, k_steps, 3]
            current_state: [batch, 3]
            future_states: [batch, k_steps, 3]

        Returns:
            metrics: dict
        """
        metrics = {}
        total_mse = 0.0
        total_mae = 0.0
        direction_correct = 0
        direction_total = 0

        for i in range(self.k_steps):
            pred_dist = predicted_trajectories[:, i, :]
            target_dist = future_states[:, i, :]

            # MSE
            mse = nn.functional.mse_loss(pred_dist, target_dist, reduction='mean')
            total_mse += mse.item()

            # MAE
            mae = nn.functional.l1_loss(pred_dist, target_dist, reduction='mean')
            total_mae += mae.item()

            # 方向准确率
            pred_delta = pred_dist - current_state
            target_delta = target_dist - current_state
            direction_match = ((pred_delta * target_delta) >= 0).all(dim=-1)
            direction_correct += direction_match.sum().item()
            direction_total += direction_match.shape[0]

        n_steps = self.k_steps
        metrics['pred/mse'] = total_mse / n_steps
        metrics['pred/mae'] = total_mae / n_steps
        metrics['pred/direction_accuracy'] = direction_correct / direction_total if direction_total > 0 else 0.0

        return metrics

    def save_pretrained(self, path):
        """自定义保存逻辑：保存 LoRA 权重和 MLP 权重"""
        if not os.path.exists(path):
            os.makedirs(path)
        # 1. 保存 LoRA (adapter_model.bin)
        self.llm.save_pretrained(path)
        # 2. 保存所有 MLP 头 (prediction_heads.pt)
        torch.save(self.prediction_heads.state_dict(), os.path.join(path, "prediction_heads.pt"))
        print(f"[SUCCESS] Model and MLP heads saved to {path}")

    def set_training_step(self, current_step, total_steps):
        """
        设置当前训练步数，用于动态计算 beta

        Args:
            current_step: 当前训练步数
            total_steps: 总训练步数
        """
        self.current_step = current_step
        self.total_steps = total_steps

        # 根据调度策略更新 beta
        if self.beta_schedule != 'constant':
            self.soft_best_beta = self._get_scheduled_beta(current_step, total_steps)

    def _get_scheduled_beta(self, current_step, total_steps):
        """
        根据训练进度计算动态 beta

        调度策略：
        - constant: 固定值 beta
        - linear: 线性从 beta_min 增加到 beta_max
        - cosine: 余弦退火从 beta_min 增加到 beta_max

        Args:
            current_step: 当前步数
            total_steps: 总步数

        Returns:
            beta: 当前步数的 beta 值
        """
        if total_steps == 0:
            return self.soft_best_beta

        progress = current_step / total_steps

        if self.beta_schedule == 'linear':
            # 线性调度
            beta = self.beta_min + (self.beta_max - self.beta_min) * progress

        elif self.beta_schedule == 'cosine':
            # 余弦调度
            import math
            beta = self.beta_min + (self.beta_max - self.beta_min) * (1 - math.cos(progress * math.pi)) / 2

        else:
            # constant 或未知策略
            beta = self.soft_best_beta

        return beta

    def get_current_beta(self):
        """获取当前 beta 值（用于日志记录）"""
        return self.soft_best_beta

    @torch.no_grad()
    def generate(self, input_ids, **kwargs):
        """生成接口，兼容 IB-Tune 的调用方式"""
        return self.llm.generate(**kwargs)

    def print_trainable_parameters(self):
        """打印可训练参数"""
        self.llm.print_trainable_parameters()
