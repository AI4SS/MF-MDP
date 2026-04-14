#!/usr/bin/env python3
"""
使用 Transformer 事件模型生成预测的状态分布轨迹文件
这些文件将作为 eval/run_simulation.sh 的缓存输入
"""
import os
import sys
import argparse
import logging
import hashlib
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from tqdm import tqdm

# 添加父目录到路径以导入模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from MF_MDP.state_transition.encoders import build_text_encoder
from MF_MDP.state_transition.event_transformer_net import CausalEventTransformerNet
from datasets.event_state_datasets import preencode_all_mf_files, get_cache_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 尝试导入配置模块
try:
    from config.settings import get_config
    config = get_config()
except ImportError:
    config = None


# ============================= Dataset for Testing =============================

class EventTestDataset(Dataset):
    """事件级测试数据集，使用预编码的嵌入"""

    def __init__(
        self,
        trajectory_path: str,
        mf_text_emb: torch.Tensor,
        state_emb: torch.Tensor,
        batch_size: int = 16,
        max_steps: Optional[int] = None,
    ):
        self.trajectory_path = trajectory_path
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.mf_text_emb = mf_text_emb
        self.state_emb = state_emb
        self.traj_df = pd.read_csv(trajectory_path)
        if self.max_steps is not None:
            self.traj_df = self.traj_df.iloc[: self.max_steps].reset_index(drop=True)

    def __len__(self) -> int:
        return 1

    def _build_step_profiles(self, step_idx: int) -> torch.Tensor:
        """Build (N, 768) profile vecs for a given step index"""
        start_idx = step_idx * self.batch_size
        num_states = len(self.state_emb)
        indices = [(start_idx + i) % num_states for i in range(self.batch_size)]
        return self.state_emb[indices]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a full event sequence"""
        T = len(self.traj_df)

        mu_prev_seq = torch.zeros(T, 3, dtype=torch.float32)
        target_dist_seq = torch.zeros(T, 3, dtype=torch.float32)
        profile_vecs_seq = torch.zeros(T, self.batch_size, 768, dtype=torch.float32)
        mf_text_emb_seq = torch.zeros(T, 768, dtype=torch.float32)

        for t in range(T):
            row = self.traj_df.iloc[t]
            target_dist_seq[t] = torch.tensor(
                [row["batch_ratio_pos"], row["batch_ratio_neu"], row["batch_ratio_neg"]],
                dtype=torch.float32,
            )
            if t == 0:
                mu_prev_seq[t] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
            else:
                prev_row = self.traj_df.iloc[t - 1]
                mu_prev_seq[t] = torch.tensor(
                    [prev_row["cum_ratio_pos"], prev_row["cum_ratio_neu"], prev_row["cum_ratio_neg"]],
                    dtype=torch.float32,
                )
            mf_idx = (t * self.batch_size + 1) % len(self.mf_text_emb)
            mf_text_emb_seq[t] = self.mf_text_emb[mf_idx]
            profile_vecs_seq[t] = self._build_step_profiles(t)

        attn_mask = torch.ones(T, dtype=torch.long)

        return {
            "mu_prev_seq": mu_prev_seq,
            "target_dist_seq": target_dist_seq,
            "profile_vecs_seq": profile_vecs_seq,
            "mf_text_emb_seq": mf_text_emb_seq,
            "attn_mask": attn_mask,
            "seq_len": T,
        }


# ============================= Test Functions =============================

def test_single_event(
    model: torch.nn.Module,
    event_data: Dict[str, Any],
    device: str,
    warmup_steps: int = 5,
) -> Dict[str, Any]:
    """测试单个事件，使用自回归预测"""
    model.eval()

    mu_prev_seq = event_data["mu_prev_seq"].unsqueeze(0).to(device)
    target_dist_seq = event_data["target_dist_seq"].unsqueeze(0).to(device)
    profile_vecs_seq = event_data["profile_vecs_seq"].unsqueeze(0).to(device)
    text_emb_seq = event_data["mf_text_emb_seq"].unsqueeze(0).to(device)
    attn_mask = event_data["attn_mask"].unsqueeze(0).to(device)

    B, T, _ = mu_prev_seq.shape

    results = {
        "pred_batches": [],
        "pred_globals": [],
        "target_batches": [],
        "target_globals": [],
        "mu_prevs": [],
        "losses": [],
        "maes": [],
        "accs": []
    }

    with torch.no_grad():
        agent_feat_seq = profile_vecs_seq.mean(dim=2)
        mu_prev = mu_prev_seq[:, 0:1, :].clone()

        for t in range(T):
            curr_mu_prev = mu_prev_seq[:, :t+1, :]
            curr_text_emb = text_emb_seq[:, :t+1, :]
            curr_agent_feat = agent_feat_seq[:, :t+1, :]
            curr_attn_mask = attn_mask[:, :t+1]

            target_batch = target_dist_seq[:, t, :]

            if t < warmup_steps:
                mu_pred = target_batch.unsqueeze(1)
            else:
                mu_pred_seq = model(
                    mu_prev_seq=curr_mu_prev,
                    text_emb_seq=curr_text_emb,
                    agent_feat_seq=curr_agent_feat,
                    attn_mask=curr_attn_mask,
                )
                mu_pred = mu_pred_seq[:, -1:, :]

            # Compute cumulative distribution
            if t == 0:
                cum_pred = mu_pred.squeeze(1)
            else:
                prev_cum = mu_prev.squeeze(1)
                curr_batch = mu_pred.squeeze(1)
                cum_pred = (prev_cum * t + curr_batch) / (t + 1)

            mu_prev = cum_pred.unsqueeze(1)

            pred_batch = mu_pred.squeeze(1)
            target_cum = mu_prev_seq[:, t, :]

            loss = F.kl_div(
                torch.log(pred_batch + 1e-8),
                target_batch,
                reduction='batchmean'
            ).item()

            mae = F.l1_loss(pred_batch, target_batch).item()

            pred_label = torch.argmax(pred_batch, dim=1)
            target_label = torch.argmax(target_batch, dim=1)
            acc = (pred_label == target_label).float().mean().item()

            results["pred_batches"].append(pred_batch.cpu())
            results["pred_globals"].append(cum_pred.cpu())
            results["target_batches"].append(target_batch.cpu())
            results["target_globals"].append(target_cum.cpu())
            results["mu_prevs"].append(mu_prev.squeeze(1).cpu())
            results["losses"].append(loss)
            results["maes"].append(mae)
            results["accs"].append(acc)

    return results


def save_predictions(results: Dict[str, Any], output_path: str, num_agents: int = 16):
    """保存预测结果到CSV文件"""
    data = []

    for i in range(len(results["pred_batches"])):
        pred_batch = results["pred_batches"][i]
        pred_global = results["pred_globals"][i]

        if pred_batch.dim() > 1:
            pred_batch = pred_batch.squeeze(0)
        if pred_global.dim() > 1:
            pred_global = pred_global.squeeze(0)

        pred_batch = pred_batch.numpy()
        pred_global = pred_global.numpy()

        batch_id = i
        processed_count = (i + 1) * num_agents

        pred_pos = pred_batch[0]
        pred_neu = pred_batch[1]
        pred_neg = pred_batch[2]

        mean_field_state = pred_pos * 1.0 + pred_neu * 0.0 + pred_neg * (-1.0)

        data.append({
            "batch_id": batch_id,
            "processed_count": processed_count,
            "mean_field_state": mean_field_state,
            "batch_avg": mean_field_state,
            "batch_ratio_pos": pred_pos,
            "batch_ratio_neu": pred_neu,
            "batch_ratio_neg": pred_neg,
            "cum_ratio_pos": pred_global[0],
            "cum_ratio_neu": pred_global[1],
            "cum_ratio_neg": pred_global[2],
            "injected_shares_batch": 0.0
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, float_format='%.15f')
    logger.info(f"预测结果已保存到: {output_path}")


# ============================= Config =============================

@dataclass
class TestConfig:
    # 数据路径
    mf_dir: str = "./data/test_mf"
    state_trajectory_dir: str = "./data/test_state_distribution"

    # 模型配置
    text_emb_dim: int = 768
    agent_feat_dim: int = 768
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 3
    dropout: float = 0.1
    max_len: int = 4096

    # 测试配置
    num_agents: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Warmup配置
    warmup_steps: int = 5

    # 模型路径
    checkpoint_path: str = "./checkpoints/event_transformer_best.pt"

    # 输出路径
    output_dir: str = "./data/pred_state_distribution"

    # Encoder config
    encoder_type: str = "bert"
    model_name: str = "bert-base-chinese"


def process_single_event(
    event_name: str,
    cfg: TestConfig,
    model: torch.nn.Module,
    embeddings_cache: Dict[str, Any],
    mf_dir: str,
    traj_dir: str,
):
    """处理单个事件"""
    mf_path = os.path.join(mf_dir, f"{event_name}_mf.csv")
    traj_path = os.path.join(traj_dir, f"{event_name}_trajectory.csv")

    if not os.path.exists(mf_path):
        logger.warning(f"跳过 {event_name}: MF文件不存在 -> {mf_path}")
        return None
    if not os.path.exists(traj_path):
        logger.warning(f"跳过 {event_name}: Trajectory文件不存在 -> {traj_path}")
        return None

    logger.info(f"处理事件: {event_name}")

    if mf_path not in embeddings_cache:
        logger.warning(f"跳过 {event_name}: 没有找到预编码的嵌入")
        return None

    mf_text_emb, state_emb = embeddings_cache[mf_path]

    output_path = os.path.join(cfg.output_dir, f"{event_name}_trajectory.csv")
    os.makedirs(cfg.output_dir, exist_ok=True)

    test_dataset = EventTestDataset(
        trajectory_path=traj_path,
        mf_text_emb=mf_text_emb,
        state_emb=state_emb,
        batch_size=cfg.num_agents
    )

    event_data = test_dataset[0]
    logger.info(f"序列长度: {event_data['seq_len']}")

    results = test_single_event(model, event_data, cfg.device, cfg.warmup_steps)
    save_predictions(results, output_path, cfg.num_agents)

    avg_loss = sum(results["losses"]) / len(results["losses"])
    avg_mae = sum(results["maes"]) / len(results["maes"])
    avg_acc = sum(results["accs"]) / len(results["accs"])

    logger.info(f"Avg Loss: {avg_loss:.4f} | MAE: {avg_mae:.3f} | Acc: {avg_acc:.1%}")

    return {
        "event_name": event_name,
        "avg_loss": avg_loss,
        "avg_mae": avg_mae,
        "avg_acc": avg_acc,
        "num_steps": len(results["pred_batches"])
    }


def main():
    parser = argparse.ArgumentParser(
        description="生成状态分布缓存文件用于仿真"
    )
    parser.add_argument("--event_name", type=str, default=None,
                       help="单个事件名 (如: jiangping, wuhan). 不指定则处理所有事件")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="模型checkpoint文件路径")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="批次大小 (默认: 16)")
    parser.add_argument("--warmup_steps", type=int, default=5,
                       help="使用真实数据的预热步数 (默认: 5)")
    parser.add_argument("--mf_dir", type=str, required=True,
                       help="MF数据目录 (非递归)")
    parser.add_argument("--traj_dir", type=str, required=True,
                       help="轨迹数据目录 (非递归)")
    parser.add_argument("--output_dir", type=str, default="./data/pred_state_distribution",
                       help="输出目录 (默认: ./data/pred_state_distribution)")
    parser.add_argument("--cache_dir", type=str, default="./cache/text_embeddings",
                       help="文本嵌入缓存目录 (默认: ./cache/text_embeddings)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="设备 (默认: cuda if available else cpu)")

    args = parser.parse_args()

    # 从配置文件获取默认值（如果未通过命令行指定）
    if args.output_dir == "./data/pred_state_distribution" and config:
        args.output_dir = config.get('paths.pred_state_dir', './data/pred_state_distribution')
    if args.cache_dir == "./cache/text_embeddings" and config:
        args.cache_dir = config.get('paths.cache_dir', './cache/text_embeddings')

    # 配置
    cfg = TestConfig()
    cfg.num_agents = args.batch_size
    cfg.warmup_steps = args.warmup_steps
    cfg.checkpoint_path = args.checkpoint
    cfg.mf_dir = args.mf_dir
    cfg.state_trajectory_dir = args.traj_dir
    cfg.output_dir = args.output_dir
    cfg.device = args.device

    # 检查模型文件
    if not os.path.exists(cfg.checkpoint_path):
        logger.error(f"模型文件不存在: {cfg.checkpoint_path}")
        logger.error("请使用 --checkpoint 指定正确的模型路径")
        return

    # 创建输出目录
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger.info(f"设备: {cfg.device}")
    logger.info(f"模型文件: {cfg.checkpoint_path}")
    logger.info(f"MF目录: {cfg.mf_dir}")
    logger.info(f"轨迹目录: {cfg.state_trajectory_dir}")
    logger.info(f"输出目录: {cfg.output_dir}")

    # 1. 构建模型
    logger.info("构建模型...")
    model = CausalEventTransformerNet(
        text_emb_dim=cfg.text_emb_dim,
        agent_feat_dim=cfg.agent_feat_dim,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
    )
    model.to(cfg.device)

    # 2. 加载checkpoint
    logger.info(f"加载checkpoint: {cfg.checkpoint_path}")
    checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("模型加载成功")

    # 3. Pre-encode all MF files
    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    encoder_config = {"type": cfg.encoder_type, "model_name": cfg.model_name}

    # 确保 MF 目录存在
    os.makedirs(cfg.mf_dir, exist_ok=True)

    logger.info("预编码MF文件（使用缓存）...")
    embeddings_cache = preencode_all_mf_files(
        [os.path.join(cfg.mf_dir, f) for f in os.listdir(cfg.mf_dir) if f.endswith("_mf.csv")],
        encoder_config,
        cache_dir,
        device=cfg.device,
    )
    logger.info(f"预编码完成，共 {len(embeddings_cache)} 个文件")

    # 4. 确定要处理的事件列表
    if args.event_name:
        event_names = [args.event_name]
    else:
        import glob
        mf_files = glob.glob(os.path.join(cfg.mf_dir, "*_mf.csv"))
        traj_files = glob.glob(os.path.join(cfg.state_trajectory_dir, "*_trajectory.csv"))

        mf_events = set()
        for f in mf_files:
            event_name = os.path.basename(f).replace("_mf.csv", "")
            mf_events.add(event_name)

        traj_events = set()
        for f in traj_files:
            event_name = os.path.basename(f).replace("_trajectory.csv", "")
            traj_events.add(event_name)

        event_names = sorted(list(mf_events & traj_events))
        event_names = [e for e in event_names if "cluster" not in e and "profile" not in e]

        logger.info(f"发现 {len(event_names)} 个有效事件")

    # 5. 处理所有事件
    all_results = []
    for event_name in event_names:
        result = process_single_event(
            event_name, cfg, model, embeddings_cache,
            cfg.mf_dir, cfg.state_trajectory_dir
        )
        if result:
            all_results.append(result)

    # 6. 打印总结
    logger.info(f"\n所有事件处理完成! 共处理 {len(all_results)} 个事件")

    if len(all_results) > 1:
        logger.info(f"{'事件名':<20} {'步数':<8} {'Loss':<10} {'MAE':<10} {'Acc':<10}")
        logger.info("-" * 60)
        for r in all_results:
            logger.info(f"{r['event_name']:<20} {r['num_steps']:<8} {r['avg_loss']:<10.4f} {r['avg_mae']:<10.3f} {r['avg_acc']:<10.1%}")


if __name__ == "__main__":
    main()
