# MFSim/datasets/event_state_datasets.py

import os
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

from LCT.state_transition.encoders import build_text_encoder


# ============================= Pre-encoding utilities =============================

def get_cache_path(file_path: str, cache_dir: str) -> str:
    """Generate cache file path based on original file path hash."""
    # Use filename + size + mtime + version as cache key
    # Version "v2" includes empty text skipping logic
    stat = os.stat(file_path)
    cache_key = f"{os.path.basename(file_path)}_{stat.st_size}_{stat.st_mtime}_v2"
    hash_key = hashlib.md5(cache_key.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"{hash_key}.pt")


def preencode_mf_file(
    mf_path: str,
    text_encoder,
    cache_dir: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-encode MF texts and states, save to cache.

    Returns:
        mf_text_emb: (num_rows, 768) - encoded mean_field texts
        state_emb: (num_rows, 768) - encoded state texts
    """
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = get_cache_path(mf_path, cache_dir)

    # Check cache
    if os.path.exists(cache_path):
        logger.info(f"Loading cached embeddings from {cache_path}")
        cached = torch.load(cache_path)
        return cached["mf_text_emb"], cached["state_emb"]

    # Encode and cache
    logger.info(f"Pre-encoding {mf_path}...")
    mf_df = pd.read_csv(mf_path)
    mf_texts = mf_df["mean_field"].tolist()
    mf_states = mf_df["state"].tolist()

    device = next(text_encoder.parameters()).device
    text_encoder.eval()

    mf_text_emb_list = []
    state_emb_list = []

    # Encode MF texts (skip empty/invalid texts)
    skipped_count = 0
    for text in tqdm(mf_texts, desc=f"Encoding MF texts ({os.path.basename(mf_path)})"):
        # Skip empty or invalid texts
        text_str = str(text).strip()
        if text_str in ["", "[]", "['']", '""', "''", "nan", "None"]:
            # Use zero embedding for empty texts
            mf_text_emb_list.append(torch.zeros(768, dtype=torch.float32))
            skipped_count += 1
            continue

        inputs = text_encoder.tokenizer(
            text_str,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        ).to(device)
        with torch.no_grad():
            vec = text_encoder(inputs["input_ids"], inputs["attention_mask"])
            if isinstance(vec, tuple):
                vec = vec[0]
            if vec.dim() == 3:
                vec = vec[:, 0, :]
        mf_text_emb_list.append(vec.squeeze(0).cpu())

    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count}/{len(mf_texts)} empty mean_field texts in {os.path.basename(mf_path)}")

    # Encode states
    for text in tqdm(mf_states, desc=f"Encoding states ({os.path.basename(mf_path)})"):
        inputs = text_encoder.tokenizer(
            str(text),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
        ).to(device)
        with torch.no_grad():
            vec = text_encoder(inputs["input_ids"], inputs["attention_mask"])
            if isinstance(vec, tuple):
                vec = vec[0]
            if vec.dim() == 3:
                vec = vec[:, 0, :]
        state_emb_list.append(vec.squeeze(0).cpu())

    mf_text_emb = torch.stack(mf_text_emb_list, dim=0)  # (num_rows, 768)
    state_emb = torch.stack(state_emb_list, dim=0)      # (num_rows, 768)

    # Save to cache
    torch.save({
        "mf_text_emb": mf_text_emb,
        "state_emb": state_emb,
    }, cache_path)
    logger.info(f"Saved embeddings to {cache_path}")

    return mf_text_emb, state_emb


def preencode_all_mf_files(
    mf_paths: List[str],
    encoder_config: dict,
    cache_dir: str,
    device: str = "cuda",
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Pre-encode all MF files and return a dict mapping mf_path -> (mf_text_emb, state_emb).
    """
    text_encoder = build_text_encoder(encoder_config)
    text_encoder = text_encoder.to(device)
    text_encoder.eval()

    embeddings = {}

    for mf_path in tqdm(mf_paths, desc="Pre-encoding MF files"):
        try:
            mf_text_emb, state_emb = preencode_mf_file(mf_path, text_encoder, cache_dir)
            embeddings[mf_path] = (mf_text_emb, state_emb)
        except Exception as e:
            logger.error(f"Failed to encode {mf_path}: {e}")

    return embeddings


class EventStateTransitionDataset(Dataset):
    """
    Event-level autoregressive dataset.
    Uses pre-encoded embeddings for MF texts and state texts.
    """

    def __init__(
        self,
        trajectory_path: str,       # "*_trajectory.csv"
        mf_path: str,               # "*_mf.csv"
        mf_text_emb: torch.Tensor,  # (num_rows, 768) pre-encoded MF texts
        state_emb: torch.Tensor,    # (num_rows, 768) pre-encoded states
        test_data_path: str,        # "*.json" (contains uid stream)
        batch_size: int = 16,
        max_steps: Optional[int] = None,
    ):
        self.trajectory_path = trajectory_path
        self.mf_path = mf_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.max_steps = max_steps

        # Pre-encoded embeddings (already computed and cached)
        self.mf_text_emb = mf_text_emb  # (num_rows, 768)
        self.state_emb = state_emb      # (num_rows, 768)

        # 1) Load trajectory (GT distributions)
        self.traj_df = pd.read_csv(self.trajectory_path)
        if self.max_steps is not None:
            self.traj_df = self.traj_df.iloc[: self.max_steps].reset_index(drop=True)

        # 2) Load test users stream (List[Dict] or JSONL) - kept for potential future use
        with open(self.test_data_path, "r", encoding="utf-8") as f:
            try:
                self.test_users = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                self.test_users = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        # One event = one sample
        return 1

    def _build_step_profiles(self, step_idx: int) -> torch.Tensor:
        """
        Build (N, 768) profile vecs for a given step index using pre-encoded state embeddings.
        start_idx = step_idx * batch_size
        """
        start_idx = step_idx * self.batch_size
        num_states = len(self.state_emb)

        indices = [(start_idx + i) % num_states for i in range(self.batch_size)]
        return self.state_emb[indices]  # (N, 768)

    # ----------------------------
    # Core event sample
    # ----------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a full event sequence (length T).
        """
        T = len(self.traj_df)

        mu_prev_seq = torch.zeros(T, 3, dtype=torch.float32)
        target_dist_seq = torch.zeros(T, 3, dtype=torch.float32)
        profile_vecs_seq = torch.zeros(T, self.batch_size, 768, dtype=torch.float32)
        mf_text_emb_seq = torch.zeros(T, 768, dtype=torch.float32)  # Pre-encoded, not text!

        for t in range(T):
            row = self.traj_df.iloc[t]

            # target distribution at step t
            target_dist_seq[t] = torch.tensor(
                [row["batch_ratio_pos"], row["batch_ratio_neu"], row["batch_ratio_neg"]],
                dtype=torch.float32,
            )

            # mu_prev at step t: same logic as your original dataset:
            # t==0 -> [0,1,0], else previous row's cumulative distribution
            if t == 0:
                mu_prev_seq[t] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
            else:
                prev_row = self.traj_df.iloc[t - 1]
                mu_prev_seq[t] = torch.tensor(
                    [prev_row["cum_ratio_pos"], prev_row["cum_ratio_neu"], prev_row["cum_ratio_neg"]],
                    dtype=torch.float32,
                )

            # mf text embedding aligned by step index
            mf_idx = (t * self.batch_size + 1) % len(self.mf_text_emb)
            mf_text_emb_seq[t] = self.mf_text_emb[mf_idx]

            # profile vectors for this step
            profile_vecs_seq[t] = self._build_step_profiles(t)

        attn_mask = torch.ones(T, dtype=torch.long)  # valid positions = 1

        return {
            "mu_prev_seq": mu_prev_seq,                 # (T, 3)
            "target_dist_seq": target_dist_seq,         # (T, 3)
            "profile_vecs_seq": profile_vecs_seq,       # (T, N, 768)
            "mf_text_emb_seq": mf_text_emb_seq,         # (T, 768) - pre-encoded!
            "attn_mask": attn_mask,                     # (T,)
            "seq_len": T,
        }


class FullEventDataset(Dataset):
    """
    A wrapper dataset that holds multiple events.
    One item = one event sequence.
    """

    def __init__(
        self,
        event_datasets: List[EventStateTransitionDataset],
    ):
        self.event_datasets = event_datasets

    def __len__(self) -> int:
        return len(self.event_datasets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.event_datasets[idx][0]


def collate_event_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pad variable-length event sequences into a batch.
    Now uses pre-encoded embeddings instead of text.
    """
    B = len(batch)
    T_max = max(x["seq_len"] for x in batch)

    mu_prev = torch.zeros(B, T_max, 3, dtype=torch.float32)
    target = torch.zeros(B, T_max, 3, dtype=torch.float32)
    profiles = torch.zeros(B, T_max, batch[0]["profile_vecs_seq"].shape[1], 768, dtype=torch.float32)
    attn_mask = torch.zeros(B, T_max, dtype=torch.long)
    mf_text_emb = torch.zeros(B, T_max, 768, dtype=torch.float32)  # Pre-encoded embeddings

    for i, item in enumerate(batch):
        T = item["seq_len"]
        mu_prev[i, :T] = item["mu_prev_seq"]
        target[i, :T] = item["target_dist_seq"]
        profiles[i, :T] = item["profile_vecs_seq"]
        attn_mask[i, :T] = item["attn_mask"]
        mf_text_emb[i, :T] = item["mf_text_emb_seq"]  # Pre-encoded embeddings

    return {
        "mu_prev_seq": mu_prev,             # (B, T, 3)
        "target_dist_seq": target,          # (B, T, 3)
        "profile_vecs_seq": profiles,       # (B, T, N, 768)
        "mf_text_emb_seq": mf_text_emb,     # (B, T, 768) - pre-encoded!
        "attn_mask": attn_mask,             # (B, T)
        "seq_len": torch.tensor([x["seq_len"] for x in batch], dtype=torch.long),
    }
