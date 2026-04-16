"""
单卡训练脚本 - Policy Model with Soft Best-of-4
不依赖 OpenRLHF，直接使用 transformers 和 PyTorch
"""
import os
import sys
import torch
import argparse
from datetime import datetime
from torch.utils.data import random_split, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

# 设置 HF 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加项目路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from LCT.policy.policy_actor import PolicyActor
from LCT.policy.datasets.policy_mf_dataset import PolicyMFDataset, load_and_process_csv_data


def custom_collate_fn(batch):
    """
    自定义 collate 函数，处理 PolicyMFDataset 返回的字典列表。

    Args:
        batch: List[Dict], 每个 dict 包含:
            - input_ids: Tensor [seq_len]
            - attention_mask: Tensor [seq_len]
            - labels: Tensor [seq_len]
            - current_state: Tensor [3]
            - future_states: Tensor [K, 3]
            - question: str
            - response: str
            - file_name: str
            - idx: int

    Returns:
        Dict[str, Tensor/str]: batched data
    """
    # 堆叠张量字段
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    current_state = torch.stack([item['current_state'] for item in batch])
    future_states = torch.stack([item['future_states'] for item in batch])

    # 保持字符串和整数字段为列表
    questions = [item['question'] for item in batch]
    responses = [item['response'] for item in batch]
    file_names = [item['file_name'] for item in batch]
    idxs = [item['idx'] for item in batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'current_state': current_state,
        'future_states': future_states,
        'question': questions,
        'response': responses,
        'file_name': file_names,
        'idx': idxs
    }


class SimplePolicyMFTrainer:
    """简化的 Policy MF 训练器 - 单卡训练专用"""

    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        scheduler,
        tokenizer,
        device,
        lambda_coeff=0.5,
        max_epochs=3,
        save_path="./checkpoints",
        use_wandb=False,
        wandb_project=None,
        wandb_run_name=None,
        save_step=500,
        eval_step=500,
        gradient_accumulation_steps=1,
        use_fp16=False,
        k_steps=10
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.device = device
        self.lambda_coeff = lambda_coeff
        self.max_epochs = max_epochs
        self.save_path = save_path
        self.global_step = 0
        self.use_wandb = use_wandb
        self.save_step = save_step
        self.eval_step = eval_step
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_fp16 = use_fp16
        self.k_steps = k_steps

        # 计算总步数（用于 beta 调度）
        self.total_steps = max_epochs * len(train_dataloader) // gradient_accumulation_steps

        # FP16 GradScaler
        self.scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)

        # 追踪最佳模型
        self.best_eval_loss = float('inf')
        self.best_eval_step = 0

        # 训练状态
        self.current_epoch = 0

        # 初始化 WandB
        if self.use_wandb:
            import wandb
            self.wandb = wandb

            if wandb_project is None:
                wandb_project = "policy-mf-training"

            if wandb_run_name is None:
                wandb_run_name = f"policy-mf-{datetime.now().strftime('%m%d-%H%M')}"

            # 初始化 wandb
            self.wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "lambda_coeff": lambda_coeff,
                    "max_epochs": max_epochs,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "k_steps": k_steps,
                }
            )

            print(f"[INFO] WandB initialized: {wandb_project}/{wandb_run_name}")

        # 打印训练配置
        print(f"\n{'='*60}")
        print("Training Configuration:")
        print(f"{'='*60}")
        print(f"  Device: {device}")
        print(f"  Gradient Accumulation Steps: {gradient_accumulation_steps}")
        print(f"  Mixed Precision (FP16): {'Enabled' if use_fp16 else 'Disabled'}")
        print(f"  Lambda Coefficient: {lambda_coeff}")
        print(f"  K Steps: {k_steps}")
        print(f"  Max Epochs: {max_epochs}")
        print(f"  Total Steps: {self.total_steps}")
        print(f"  Save Path: {save_path}")
        print(f"{'='*60}\n")

    def fit(self, resume_from=None):
        """训练主循环"""
        # 尝试恢复训练
        start_epoch = 0
        if resume_from:
            start_epoch = self._load_checkpoint(resume_from)
            print(f"[INFO] Resumed from epoch {start_epoch}")

        # 根据 global_step 计算应该从哪个 batch 开始
        # global_step 是 optimizer step 的数量，需要转换为 DataLoader batch 索引
        start_batch_idx = self.global_step * self.gradient_accumulation_steps

        for epoch in range(start_epoch, self.max_epochs):
            self.current_epoch = epoch
            self.model.train()

            # 计算当前 epoch 的起始 batch 索引
            current_start_batch = start_batch_idx if epoch == start_epoch else 0

            epoch_bar = tqdm(
                range(len(self.train_dataloader)),
                desc=f"Train epoch {epoch}/{self.max_epochs-1}",
                disable=not self.device.type == 'cuda',
                initial=current_start_batch,  # 设置初始位置
                total=len(self.train_dataloader)
            )

            for step_idx, batch in enumerate(self.train_dataloader):
                # 跳过已完成的 batch（用于断点续训）
                if epoch == start_epoch and step_idx < current_start_batch:
                    continue
                if len(batch) == 0:
                    continue

                # 更新 beta 调度器（在 forward 之前）
                if self.model.use_soft_best_of:
                    self.model.set_training_step(self.global_step, self.total_steps)

                try:
                    # 从 batch 中提取数据 (collate_fn 已经处理过)
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    current_state = batch['current_state'].to(self.device)
                    future_states = batch['future_states'].to(self.device)

                    # Forward pass
                    if self.use_fp16 and self.scaler is not None:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                current_state=current_state,
                                future_states=future_states,
                                return_output=True
                            )

                            text_loss = outputs["loss"]
                            pred_loss = outputs["pred_loss"]
                            total_loss = (1 - self.lambda_coeff) * text_loss + (self.lambda_coeff) * pred_loss
                            scaled_loss = total_loss / self.gradient_accumulation_steps

                        self.scaler.scale(scaled_loss).backward()
                    else:
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            current_state=current_state,
                            future_states=future_states,
                            return_output=True
                        )

                        text_loss = outputs["loss"]
                        pred_loss = outputs["pred_loss"]
                        total_loss = (1 - self.lambda_coeff) * text_loss + (self.lambda_coeff) * pred_loss
                        scaled_loss = total_loss / self.gradient_accumulation_steps

                        scaled_loss.backward()

                    # 梯度累积和优化器步进
                    if (step_idx + 1) % self.gradient_accumulation_steps == 0:
                        if self.use_fp16 and self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()

                        self.scheduler.step()
                        self.optimizer.zero_grad()

                        # 更新全局步数
                        self.global_step += 1

                        # 记录日志
                        logs = self._compute_logs(outputs, total_loss)

                        # 更新进度条 (只显示关键指标)
                        epoch_bar.set_postfix({
                            "loss": f"{logs['train/total_loss']:.4f}",
                            "text": f"{logs['train/text_loss']:.4f}",
                            "pred": f"{logs['train/pred_loss']:.4f}"
                        })
                        epoch_bar.update(self.gradient_accumulation_steps)

                        # 记录到 WandB (每 10 个 optimizer step 记录一次)
                        if self.use_wandb and self.global_step % 10 == 0:
                            self.wandb.log(logs, step=self.global_step)

                        # 保存和评估
                        if self.global_step % self.save_step == 0:
                            self._save_checkpoint(self.global_step)

                        if self.global_step % self.eval_step == 0:
                            eval_metrics = self._evaluate()

                            # WandB 记录评估指标
                            if self.use_wandb:
                                self.wandb.log(eval_metrics, step=self.global_step)

                            # 更新最佳模型
                            eval_loss = eval_metrics.get("eval/total_loss", 0)
                            if eval_loss < self.best_eval_loss:
                                self.best_eval_loss = eval_loss
                                self.best_eval_step = self.global_step
                                self._save_checkpoint(self.global_step, is_best=True)

                                # WandB 记录最佳模型指标
                                if self.use_wandb:
                                    self.wandb.log({
                                        "best/eval_loss": self.best_eval_loss,
                                        "best/eval_step": self.best_eval_step
                                    }, step=self.global_step)

                except Exception as e:
                    print(f"[Error] Step {step_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # 训练结束
        print(f"\n{'='*60}")
        print("Training Completed!")
        print(f"  Best Eval Loss: {self.best_eval_loss:.4f} at step {self.best_eval_step}")
        print(f"{'='*60}\n")

        if self.use_wandb:
            self.wandb.finish()

    def _compute_logs(self, outputs, total_loss):
        """计算训练日志"""
        logs = {
            "train/total_loss": total_loss.item(),
            "train/text_loss": outputs["loss"].item(),
            "train/pred_loss": outputs["pred_loss"].item(),
            "train/learning_rate": self.scheduler.get_last_lr()[0]
        }

        # 添加额外指标 (MSE, MAE, 方向准确率等)
        if "metrics" in outputs:
            metrics = outputs.get("metrics", {})
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    logs[f"train/{key}"] = value.item()
                else:
                    logs[f"train/{key}"] = value

        # Soft Best-of-4 特定指标
        if "soft_best_weights" in outputs:
            weights = outputs["soft_best_weights"]
            if len(weights) > 0:
                logs["train/soft_best_w0"] = float(weights[0])
            if len(weights) > 1:
                logs["train/soft_best_w1"] = float(weights[1])
            if len(weights) > 2:
                logs["train/soft_best_w2"] = float(weights[2])
            if len(weights) > 3:
                logs["train/soft_best_w3"] = float(weights[3])

        if "candidate_losses" in outputs:
            cand_losses = outputs["candidate_losses"]
            if len(cand_losses) >= 1:
                logs["train/candidate_loss_0"] = float(cand_losses[0])
            if len(cand_losses) >= 2:
                logs["train/candidate_loss_1"] = float(cand_losses[1])
            if len(cand_losses) >= 3:
                logs["train/candidate_loss_2"] = float(cand_losses[2])
            if len(cand_losses) >= 4:
                logs["train/candidate_loss_3"] = float(cand_losses[3])

        # 记录 beta
        if self.model.use_soft_best_of:
            logs["train/beta"] = self.model.get_current_beta()

        # 记录 drop_rate (如果有)
        if "drop_rate" in outputs:
            drop_rate = outputs["drop_rate"]
            logs["train/drop_rate"] = drop_rate.item() if isinstance(drop_rate, torch.Tensor) else drop_rate

        return logs

    @torch.no_grad()
    def _evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_text_loss = 0
        total_pred_loss = 0
        num_batches = 0

        for batch in self.eval_dataloader:
            if len(batch) == 0:
                continue

            try:
                # 从 batch 中提取数据 (collate_fn 已经处理过)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                current_state = batch['current_state'].to(self.device)
                future_states = batch['future_states'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    current_state=current_state,
                    future_states=future_states,
                    return_output=True
                )

                text_loss = outputs["loss"]
                pred_loss = outputs["pred_loss"]
                batch_loss = (1 - self.lambda_coeff) * text_loss + (self.lambda_coeff) * pred_loss

                total_loss += batch_loss.item()
                total_text_loss += text_loss.item()
                total_pred_loss += pred_loss.item()
                num_batches += 1

            except Exception as e:
                print(f"[Eval Error] {e}")
                continue

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_text_loss = total_text_loss / num_batches if num_batches > 0 else 0
        avg_pred_loss = total_pred_loss / num_batches if num_batches > 0 else 0

        print(f"  Eval Loss: {avg_loss:.4f} (text: {avg_text_loss:.4f}, pred: {avg_pred_loss:.4f})")

        self.model.train()

        # 返回评估指标字典（供 WandB 记录）
        return {
            "eval/total_loss": avg_loss,
            "eval/text_loss": avg_text_loss,
            "eval/pred_loss": avg_pred_loss,
        }

    def _save_checkpoint(self, step, is_best=False):
        """保存 checkpoint"""
        checkpoint_dir = os.path.join(self.save_path, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 保存模型
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # 保存训练状态
        train_state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "best_eval_step": self.best_eval_step,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

        if self.use_fp16 and self.scaler is not None:
            train_state["scaler"] = self.scaler.state_dict()

        torch.save(train_state, os.path.join(checkpoint_dir, "trainer_state.pt"))

        # 保存为 last_checkpoint
        last_dir = os.path.join(self.save_path, "last_checkpoint")
        if os.path.exists(last_dir):
            import shutil
            shutil.rmtree(last_dir)
        import shutil
        shutil.copytree(checkpoint_dir, last_dir)

        if is_best:
            best_dir = os.path.join(self.save_path, "best_checkpoint")
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            shutil.copytree(checkpoint_dir, best_dir)
            print(f"  [Best] Saved checkpoint {step} (eval_loss={self.best_eval_loss:.4f})")
        else:
            print(f"  [Save] Saved checkpoint {step}")

    def _load_checkpoint(self, checkpoint_path):
        """加载 checkpoint"""
        # 加载 LoRA 权重
        try:
            self.model.llm.load_adapter(checkpoint_path, adapter_name="default")
            print(f"✓ Loaded LoRA weights from {checkpoint_path}")
        except Exception as e:
            print(f"✗ Failed to load LoRA weights: {e}")
            return 0

        # 加载 MLP 权重
        mlp_path = os.path.join(checkpoint_path, "prediction_heads.pt")
        if os.path.exists(mlp_path):
            try:
                self.model.prediction_heads.load_state_dict(torch.load(mlp_path, map_location=self.device))
                print(f"✓ Loaded MLP weights from {mlp_path}")
            except Exception as e:
                print(f"✗ Failed to load MLP weights: {e}")
                return 0

        # 加载训练状态
        state_path = os.path.join(checkpoint_path, "trainer_state.pt")
        if os.path.exists(state_path):
            try:
                train_state = torch.load(state_path, map_location=self.device)
                self.global_step = train_state["global_step"]
                self.best_eval_loss = train_state.get("best_eval_loss", float('inf'))
                self.best_eval_step = train_state.get("best_eval_step", 0)
                self.optimizer.load_state_dict(train_state["optimizer"])
                self.scheduler.load_state_dict(train_state["scheduler"])

                if self.use_fp16 and "scaler" in train_state and self.scaler is not None:
                    self.scaler.load_state_dict(train_state["scaler"])

                # 根据 global_step 计算对应的 batch_idx（用于信息显示）
                calculated_batch_idx = self.global_step * self.gradient_accumulation_steps

                print(f"✓ Loaded trainer state from {state_path}")
                print(f"  Global step: {self.global_step}")
                print(f"  Calculated batch index: {calculated_batch_idx}")
                print(f"  Best eval loss: {self.best_eval_loss:.4f} at step {self.best_eval_step}")

                return train_state["epoch"]
            except Exception as e:
                print(f"✗ Failed to load trainer state: {e}")
                return 0
        else:
            print(f"✗ Trainer state file not found: {state_path}")
            return 0


def main():
    parser = argparse.ArgumentParser()

    # 模型参数
    parser.add_argument("--pretrain", type=str, required=True, help="Pretrained model path")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")

    # LoRA 参数
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--target_modules", type=str, default="all-linear")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_betas", type=str, default="0.9,0.95")
    parser.add_argument("--l2", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=1024)

    # Policy 模型参数
    parser.add_argument("--k_steps", type=int, default=10)
    parser.add_argument("--lambda_coeff", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--loss_threshold", type=float, default=20.0)

    # Soft Best-of-4 参数
    parser.add_argument("--use_soft_best_of", action="store_true", default=False)
    parser.add_argument("--soft_best_beta", type=float, default=1.0)
    parser.add_argument("--beta_schedule", type=str, default="constant",
                        choices=["constant", "linear", "cosine"])
    parser.add_argument("--beta_min", type=float, default=0.5)
    parser.add_argument("--beta_max", type=float, default=5.0)

    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--train_data_n", type=int, default=-1)
    parser.add_argument("--eval_data_n", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)

    # 断点续训
    parser.add_argument("--resume_from", type=str, default=None)

    # WandB
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)

    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 配置 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 配置模型
    model = PolicyActor(
        args.pretrain,
        k_steps=args.k_steps,
        state_dim=3,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        gamma=args.gamma,
        label_smoothing=args.label_smoothing,
        loss_threshold=args.loss_threshold,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        use_soft_best_of=args.use_soft_best_of,
        soft_best_beta=args.soft_best_beta,
        beta_schedule=args.beta_schedule,
        beta_min=args.beta_min,
        beta_max=args.beta_max
    )
    model = model.to(device)
    model.print_trainable_parameters()

    # 梯度检查点
    if args.gradient_checkpointing:
        model.llm.gradient_checkpointing_enable()

    # 准备数据
    print(f"[INFO] Loading data from {args.dataset}")
    raw_data = load_and_process_csv_data(args.dataset)
    print(f"[INFO] Loaded {len(raw_data)} samples")

    full_dataset = PolicyMFDataset(raw_data, tokenizer, max_length=args.max_len, k_steps=args.k_steps)

    # 划分数据集
    if args.eval_data_n > 0:
        val_size = min(args.eval_data_n, len(full_dataset) - args.train_data_n if args.train_data_n > 0 else args.eval_data_n)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        print(f"[INFO] Dataset split: {train_size} train, {val_size} validation")
    else:
        train_dataset = full_dataset
        val_dataset = None

    # 创建 dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    ) if val_dataset else None

    # 配置优化器和 scheduler
    adam_betas = tuple(float(x) for x in args.adam_betas.split(","))
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=adam_betas, weight_decay=args.l2)

    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(args.warmup_ratio * max_train_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps
    )

    print(f"[INFO] Training steps: {max_train_steps}, Warmup steps: {num_warmup_steps}")

    # 创建训练器
    trainer = SimplePolicyMFTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        device=device,
        lambda_coeff=args.lambda_coeff,
        max_epochs=args.epochs,
        save_path=args.output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        save_step=args.save_steps,
        eval_step=args.eval_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        k_steps=args.k_steps
    )

    # 开始训练
    trainer.fit(resume_from=args.resume_from)


if __name__ == "__main__":
    main()
