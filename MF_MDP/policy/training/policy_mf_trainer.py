import os
from abc import ABC
import math
from tqdm import tqdm
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import Optimizer
# from openrlhf.models import GPTLMLoss
# from openrlhf.utils.distributed_sampler import DistributedSampler
# from openrlhf.utils import DatasetLoader


class PolicyMFTrainer(ABC):
    """
    改进的 Mean Field Policy Trainer

    主要改进:
    1. 结合 LLM text loss 和 状态预测 KL loss
    2. 支持串行状态预测头
    3. 支持阈值过滤高误差样本
    4. 兼容 IB-Tune 的分布式训练框架

    Args:
        model: PolicyActor 模型
        strategy: 训练策略 (IB-Tune)
        optim: 优化器
        train_dataloader: 训练数据加载器
        eval_dataloader: 验证数据加载器
        scheduler: 学习率调度器
        max_norm: 梯度裁剪阈值
        pretrain_mode: 是否预训练模式
        batch_size: batch size
        max_epochs: 最大训练轮数
        tokenizer: tokenizer
        lambda_coeff: text loss 和 pred loss 的权重系数 (默认 0.5)
        k_steps: 预测未来 K 步状态
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        pretrain_mode: bool = False,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
        lambda_coeff: float = 0.5,
        k_steps: int = 10,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args
        self.lambda_coeff = lambda_coeff
        self.k_steps = k_steps

        # 用于 beta 调度
        self.max_steps = None  # 将在 fit 中设置

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb
            import stat
            self._wandb = wandb

            wandb_dir = os.path.abspath("./wandb_logs")
            if not os.path.exists(wandb_dir):
                try:
                    os.makedirs(wandb_dir)
                    print(f"目录 {wandb_dir} 已创建。")
                except Exception as e:
                    print(f"创建目录 {wandb_dir} 时发生错误: {e}")
                    exit(1)

            try:
                os.chmod(wandb_dir, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
                print(f"目录 {wandb_dir} 权限已设置为可写。")
            except Exception as e:
                print(f"修改目录 {wandb_dir} 权限时发生错误: {e}")
                exit(1)

            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                dir=wandb_dir,
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        """
        训练主循环
        """
        # Set evaluation and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")

        # 计算总步数（用于 beta 调度）
        total_steps = self.epochs * num_update_steps_per_epoch if num_update_steps_per_epoch else None
        self.max_steps = total_steps

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )

        for epoch in range(start_epoch, self.epochs):
            # 设置 epoch（如果使用了分布式采样器）
            if hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(len(self.train_dataloader)),
                desc=f"Train step of epoch {epoch}",
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            for step_idx, batch in enumerate(self.train_dataloader):
                # 更新 beta 调度器（在 forward 之前）
                global_step = step // self.strategy.accumulated_gradient
                if self.model.use_soft_best_of and total_steps:
                    self.model.set_training_step(global_step, total_steps)

                # 新的数据格式：batch 是字典列表
                if len(batch) == 0:
                    print(f"[step {step_idx}] Empty batch encountered, skip.")
                    continue

                try:
                    # 从 batch 中提取数据
                    input_ids = torch.stack([item['input_ids'] for item in batch]).squeeze(1).to(self.strategy.device)
                    attention_mask = torch.stack([item['attention_mask'] for item in batch]).squeeze(1).to(self.strategy.device)
                    labels = torch.stack([item['labels'] for item in batch]).squeeze(1).to(self.strategy.device)
                    current_state = torch.stack([item['current_state'] for item in batch]).to(self.strategy.device)
                    future_states = torch.stack([item['future_states'] for item in batch]).to(self.strategy.device)

                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        current_state=current_state,
                        future_states=future_states,
                        return_output=True
                    )

                    # 计算组合 loss
                    text_loss = outputs["loss"]
                    pred_loss = outputs["pred_loss"]
                    drop_rate = outputs.get("drop_rate", 0.0)
                    metrics = outputs.get("metrics", {})

                    # 组合 loss
                    total_loss = (1 - self.lambda_coeff) * text_loss + (self.lambda_coeff * pred_loss)

                    # Backward
                    self.strategy.backward(total_loss, self.model, self.optimizer)
                    self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                    # 记录日志
                    logs_dict = {
                        "loss/text_ce": text_loss.detach().item(),
                        "loss/pred_kl": pred_loss.detach().item(),
                        "loss/total": total_loss.detach().item(),
                        "metrics/drop_rate": drop_rate.item() if isinstance(drop_rate, torch.Tensor) else drop_rate,
                        "lr": self.scheduler.get_last_lr()[0],
                    }

                    # 添加额外指标
                    for key, value in metrics.items():
                        if isinstance(value, torch.Tensor):
                            logs_dict[f"metrics/{key}"] = value.item()
                        else:
                            logs_dict[f"metrics/{key}"] = value

                    # 添加 Soft Best-of-2 特定指标
                    if "soft_best_weights" in outputs:
                        weights = outputs["soft_best_weights"]
                        logs_dict["soft_best/weight_0"] = weights[0] if len(weights) > 0 else 0.0
                        logs_dict["soft_best/weight_1"] = weights[1] if len(weights) > 1 else 0.0

                    if "candidate_losses" in outputs:
                        cand_losses = outputs["candidate_losses"]
                        if len(cand_losses) >= 2:
                            logs_dict["soft_best/candidate_loss_0"] = cand_losses[0]
                            logs_dict["soft_best/candidate_loss_1"] = cand_losses[1]
                            logs_dict["soft_best/loss_gap"] = abs(cand_losses[0] - cand_losses[1])

                    # 记录当前 beta 值（如果使用了动态调度）
                    if self.model.use_soft_best_of and hasattr(self.model, 'get_current_beta'):
                        logs_dict["soft_best/beta"] = self.model.get_current_beta()

                    step_bar.set_postfix(logs_dict)
                    step_bar.update()

                    # Save logs/checkpoints/evaluation
                    if step % self.strategy.accumulated_gradient == 0:
                        global_step = step // self.strategy.accumulated_gradient
                        client_states = {"consumed_samples": global_step * args.train_batch_size}
                        self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                    step += 1

                except Exception as e:
                    print(f"[Error] Step {step_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        """保存日志和检查点"""
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if global_step % args.eval_steps == 0:
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)

        # save ckpt
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.model.llm, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            )
            # 保存 MLP 头
            mlp_path = os.path.join(args.ckpt_path, tag, "prediction_heads.pt")
            torch.save(self.model.prediction_heads.state_dict(), mlp_path)
            print(f"[INFO] Saved MLP heads to {mlp_path}")

    def evaluate(self, eval_dataloader, steps=0):
        """
        评估模型
        """
        self.model.eval()
        total_loss = 0
        total_text_loss = 0
        total_pred_loss = 0
        num_batches = 0

        with torch.no_grad():
            step_bar = tqdm(
                range(len(eval_dataloader)),
                desc=f"Eval stage at step {steps}",
                disable=not self.strategy.is_rank_0(),
            )

            for batch in eval_dataloader:
                if len(batch) == 0:
                    continue

                try:
                    # 从 batch 中提取数据
                    input_ids = torch.stack([item['input_ids'] for item in batch]).squeeze(1).to(self.strategy.device)
                    attention_mask = torch.stack([item['attention_mask'] for item in batch]).squeeze(1).to(self.strategy.device)
                    labels = torch.stack([item['labels'] for item in batch]).squeeze(1).to(self.strategy.device)
                    current_state = torch.stack([item['current_state'] for item in batch]).to(self.strategy.device)
                    future_states = torch.stack([item['future_states'] for item in batch]).to(self.strategy.device)

                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        current_state=current_state,
                        future_states=future_states,
                        return_output=True
                    )

                    # 计算组合 loss
                    text_loss = outputs["loss"]
                    pred_loss = outputs["pred_loss"]
                    total_loss_batch = (1 - self.lambda_coeff) * text_loss + (self.lambda_coeff * pred_loss)

                    total_loss += total_loss_batch.item()
                    total_text_loss += text_loss.item()
                    total_pred_loss += pred_loss.item()
                    num_batches += 1

                    step_bar.update()

                except Exception as e:
                    print(f"[Eval Error] {e}")
                    continue

        # 计算平均 loss
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_text_loss = total_text_loss / num_batches
            avg_pred_loss = total_pred_loss / num_batches

            logs_dict = {
                "loss/text_ce": avg_text_loss,
                "loss/pred_kl": avg_pred_loss,
                "loss/total": avg_loss,
            }

            # Log to wandb/tensorboard
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {f"eval/{k}": v for k, v in {**logs_dict, "global_step": steps}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, steps)

            step_bar.set_postfix(logs_dict)

        self.model.train()
