# MFSim/training/train_event_state_transition.py
import datetime
import os
import sys
import argparse
import logging
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from MF_MDP.state_transition.encoders import build_text_encoder
from MF_MDP.state_transition.event_transformer_net import CausalEventTransformerNet
from datasets.event_state_datasets import (
    EventStateTransitionDataset,
    FullEventDataset,
    collate_event_batch,
    preencode_all_mf_files,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import glob

# 尝试导入配置模块
try:
    from config.settings import get_config
    config = get_config()
except ImportError:
    config = None


@dataclass
class TrainConfig:
    # --- 目录配置（从配置文件读取，优先使用环境变量） ---
    event_data_dir: str = (
        config.get('paths.event_data_dir') if config
        else "/root/Mean-Field-LLM/mf_llm/data/rumdect/Weibo/test"
    )
    mf_dir: str = (
        config.get('paths.mf_data_dir') if config
        else "/root/ICML/data/test_mf/gt"
    )
    state_trajectory_dir: str = (
        config.get('paths.state_trajectory_dir') if config
        else "/root/ICML/data/test_state_distribution/gt"
    )

    # --- 模型与训练参数 ---
    encoder_type: str = "bert"
    model_name: str = "bert-base-chinese"
    text_emb_dim: int = 768
    agent_feat_dim: int = 768

    # Transformer config
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 3
    dropout: float = 0.1
    max_len: int = 4096  # if your events have >256 steps, increase

    train_batch_size: int = 4   # event-level batch, usually smaller
    max_event: int = 100
    num_agents: int = 16
    num_epochs: int = 20
    lr: float = 2e-5
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 10

    save_dir: str = os.path.join(ROOT_DIR, "checkpoints/event")
    save_name: str = "event_transformer_best.pt"


def build_full_event_dataset(cfg: TrainConfig) -> FullEventDataset:
    traj_pattern = os.path.join(cfg.state_trajectory_dir, "*_trajectory.csv")
    all_traj_files = sorted(glob.glob(traj_pattern))
    if not all_traj_files:
        raise ValueError(f"未在 {cfg.state_trajectory_dir} 下找到任何 *_trajectory.csv 文件")

    # Separate English-starting and other events
    en_events = []  # English-starting events (PRIORITY, always included)
    other_events = []  # Other events (subject to max_event limit)

    for traj_path in all_traj_files:
        filename = os.path.basename(traj_path)
        event_id = filename.replace("_trajectory.csv", "")

        # Skip cluster/profile files
        if "cluster" in event_id or "profile" in event_id:
            continue

        # Check if event_id starts with English letter (a-z, A-Z)
        if event_id and event_id[0].isalpha() and ord(event_id[0]) < 128:
            en_events.append((event_id, traj_path))
        else:
            other_events.append((event_id, traj_path))

    # Apply max_event limit only to non-English events
    if cfg.max_event is not None and cfg.max_event > 0:
        other_events = other_events[: cfg.max_event]
        logger.info(f"非英文事件: 选取 {len(other_events)} 个（max_event={cfg.max_event}）")

    logger.info(f"英文事件（优先）: {len(en_events)} 个（必选，无限制）")

    # Combine: English events FIRST, then other events
    selected_events = en_events + other_events
    logger.info(f"总计 {len(selected_events)} 个事件（英文优先在前）")

    encoder_config = {"type": cfg.encoder_type, "model_name": cfg.model_name}

    # 打印英文事件列表
    if en_events:
        logger.info(f"英文事件列表: {[e[0] for e in en_events]}")

    # Collect all MF paths for pre-encoding
    mf_paths_to_encode = []
    event_info_list = []  # (event_id, traj_path, json_path, mf_path)

    for event_id, traj_path in selected_events:
        json_path = os.path.join(cfg.event_data_dir, f"{event_id}.json")
        mf_path = os.path.join(cfg.mf_dir, f"{event_id}_mf.csv")

        if not os.path.exists(json_path):
            logger.warning(f"跳过 {event_id}: 缺少 JSON -> {json_path}")
            continue
        if not os.path.exists(mf_path):
            logger.warning(f"跳过 {event_id}: 缺少 MF -> {mf_path}")
            continue

        mf_paths_to_encode.append(mf_path)
        event_info_list.append((event_id, traj_path, json_path, mf_path))

    if not event_info_list:
        raise RuntimeError("没有找到任何有效事件！")

    # Pre-encode all MF files
    cache_dir = os.path.join(ROOT_DIR, "cache", "text_embeddings")
    logger.info("=" * 60)
    logger.info("预编码所有MF文件...")
    logger.info(f"缓存目录: {cache_dir}")

    embeddings = preencode_all_mf_files(
        mf_paths_to_encode,
        encoder_config,
        cache_dir,
        device=cfg.device,
    )
    logger.info("✅ 预编码完成!")
    logger.info("=" * 60)

    # Build datasets with pre-encoded embeddings
    event_datasets: List[EventStateTransitionDataset] = []

    for event_id, traj_path, json_path, mf_path in tqdm(event_info_list, desc="创建数据集"):
        if mf_path not in embeddings:
            logger.warning(f"跳过 {event_id}: 没有找到预编码的嵌入")
            continue

        mf_text_emb, state_emb = embeddings[mf_path]

        try:
            ds = EventStateTransitionDataset(
                trajectory_path=traj_path,
                mf_path=mf_path,
                mf_text_emb=mf_text_emb,
                state_emb=state_emb,
                test_data_path=json_path,
                batch_size=cfg.num_agents,
                max_steps=None,
            )
            event_datasets.append(ds)
        except Exception as e:
            logger.exception(f"加载事件 {event_id} 失败: {e}")

    if not event_datasets:
        raise RuntimeError("没有成功加载任何有效事件数据集！")

    logger.info(f"✅ 成功加载 {len(event_datasets)} 个事件")
    return FullEventDataset(event_datasets)


def build_dataloader(cfg: TrainConfig) -> DataLoader:
    full_dataset = build_full_event_dataset(cfg)
    loader = DataLoader(
        full_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_event_batch,
    )
    return loader


def build_models(cfg: TrainConfig):
    # Only the causal event transformer (text_encoder removed, using pre-encoded embeddings)
    model = CausalEventTransformerNet(
        text_emb_dim=cfg.text_emb_dim,
        agent_feat_dim=cfg.agent_feat_dim,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
    )
    return model


def train_one_epoch(
    epoch: int,
    cfg: TrainConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    writer: SummaryWriter,
):
    model.train()

    total_loss = 0.0
    total_mae = 0.0
    total_acc = 0.0
    total_steps = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.num_epochs}", ncols=120)

    for step, batch in enumerate(pbar):
        mu_prev_seq = batch["mu_prev_seq"].to(cfg.device)            # (B,T,3)
        target_dist_seq = batch["target_dist_seq"].to(cfg.device)    # (B,T,3)
        profile_vecs_seq = batch["profile_vecs_seq"].to(cfg.device)  # (B,T,N,768)
        text_emb_seq = batch["mf_text_emb_seq"].to(cfg.device)       # (B,T,768) - pre-encoded!
        attn_mask = batch["attn_mask"].to(cfg.device)                # (B,T)

        B, T, _, = mu_prev_seq.shape

        # ---- pool agent feats per step: (B,T,N,768) -> (B,T,768)
        agent_feat_seq = profile_vecs_seq.mean(dim=2)

        # ---- forward (causal AR)
        mu_pred_seq = model(
            mu_prev_seq=mu_prev_seq,
            text_emb_seq=text_emb_seq,
            agent_feat_seq=agent_feat_seq,
            attn_mask=attn_mask,
        )  # (B,T,3)

        # ---- KL loss per position (mask padded)
        log_pred = torch.log(mu_pred_seq + 1e-8)  # (B,T,3)
        kl = F.kl_div(log_pred, target_dist_seq, reduction="none").sum(dim=-1)  # (B,T)

        # apply mask
        kl = kl * attn_mask.float()
        loss = kl.sum() / (attn_mask.float().sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.grad_clip,
            )

        optimizer.step()

        # metrics
        with torch.no_grad():
            mae = (torch.abs(mu_pred_seq - target_dist_seq) * attn_mask.unsqueeze(-1)).sum() / (
                attn_mask.sum() * 3 + 1e-8
            )
            pred_label = torch.argmax(mu_pred_seq, dim=-1)         # (B,T)
            target_label = torch.argmax(target_dist_seq, dim=-1)   # (B,T)
            acc = ((pred_label == target_label).float() * attn_mask.float()).sum() / (attn_mask.sum() + 1e-8)

        global_step = (epoch - 1) * len(loader) + step
        writer.add_scalar("Loss/train_kl", loss.item(), global_step)
        writer.add_scalar("Metric/MAE", mae.item(), global_step)
        writer.add_scalar("Metric/Accuracy", acc.item(), global_step)
        writer.add_scalar("Debug/LR", optimizer.param_groups[0]["lr"], global_step)

        # wandb logging every step
        wandb.log({
            "Loss/train_kl": loss.item(),
            "Metric/MAE": mae.item(),
            "Metric/Accuracy": acc.item(),
            "Debug/LR": optimizer.param_groups[0]["lr"],
            "epoch": epoch,
            "step": step,
        }, step=global_step)

        total_loss += loss.item()
        total_mae += mae.item()
        total_acc += acc.item()
        total_steps += 1

        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'MAE': f'{mae.item():.3f}',
            'Acc': f'{acc.item():.1%}',
            'AvgLoss': f'{total_loss/total_steps:.4f}'
        })

    avg_loss = total_loss / max(1, total_steps)
    avg_mae = total_mae / max(1, total_steps)
    avg_acc = total_acc / max(1, total_steps)
    logger.info(f"[Epoch {epoch}] Finished - Avg Loss: {avg_loss:.6f} | MAE: {avg_mae:.4f} | Acc: {avg_acc:.2%}")
    return avg_loss


def save_checkpoint(cfg: TrainConfig, model, optimizer, epoch, loss, is_best=False):
    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": str(cfg),
    }

    last_path = os.path.join(cfg.save_dir, "checkpoint_last.pt")
    torch.save(ckpt, last_path)

    if is_best:
        best_path = os.path.join(cfg.save_dir, cfg.save_name)
        torch.save(ckpt, best_path)
        logger.info(f"🌟 Best model saved: {best_path} (Loss: {loss:.6f})")

    logger.info(f"💾 Checkpoint saved: {last_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4, help="Event-level batch size")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.num_epochs = args.epochs
    cfg.train_batch_size = args.batch_size

    beijing_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    log_dir = f"{cfg.save_dir}/runs/run_{beijing_now.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)

    # Initialize wandb
    wandb.init(
        project="event-transformer",
        name=f"run_{beijing_now.strftime('%Y%m%d_%H%M%S')}",
        config={
            "epochs": cfg.num_epochs,
            "batch_size": cfg.train_batch_size,
            "lr": cfg.lr,
            "max_event": cfg.max_event,
            "num_agents": cfg.num_agents,
            "d_model": cfg.d_model,
            "num_layers": cfg.num_layers,
            "nhead": cfg.nhead,
            "model_name": cfg.model_name,
        }
    )

    logger.info("=" * 60)
    logger.info("训练配置:")
    logger.info(f"  - Device: {cfg.device}")
    logger.info(f"  - Epochs: {cfg.num_epochs}")
    logger.info(f"  - Batch Size: {cfg.train_batch_size}")
    logger.info(f"  - Learning Rate: {cfg.lr}")
    logger.info(f"  - Max Event: {cfg.max_event}")
    logger.info(f"  - Num Agents: {cfg.num_agents}")
    logger.info(f"  - Save Dir: {cfg.save_dir}")
    logger.info("=" * 60)

    model = build_models(cfg)
    model.to(cfg.device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量:")
    logger.info(f"  - Event Transformer: {total_params:,}")
    logger.info(f"  - Text embeddings: PRE-ENCODED (cached)")

    loader = build_dataloader(cfg)
    logger.info(f"数据加载完成: {len(loader)} batches/epoch")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    start_epoch = 1
    best_loss = float("inf")

    if args.resume:
        ckpt_path = args.checkpoint if args.checkpoint else os.path.join(cfg.save_dir, "checkpoint_last.pt")
        if os.path.exists(ckpt_path):
            logger.info(f"🔄 resume from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=cfg.device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_loss = ckpt.get("loss", float("inf"))
        else:
            logger.warning(f"⚠️ checkpoint not found: {ckpt_path}")

    logger.info("开始训练...")
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        loss = train_one_epoch(epoch, cfg, model, optimizer, loader, writer)
        is_best = loss < best_loss
        if is_best:
            best_loss = loss
        save_checkpoint(cfg, model, optimizer, epoch, loss, is_best=is_best)

    writer.close()
    wandb.finish()
    logger.info("训练完成!")


if __name__ == "__main__":
    main()
