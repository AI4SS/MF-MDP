# MFSim/model/state_transition/event_transformer_net.py
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalEventTransformerNet(nn.Module):
    """
    Event-level causal (autoregressive) transformer.
    Predict mu_t using history up to t (via causal mask), given:
      token_t = [mu_prev_t, mf_text_emb_t, pooled_agent_feat_t]
    """

    def __init__(
        self,
        text_emb_dim: int = 768,
        agent_feat_dim: int = 768,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()

        self.text_emb_dim = text_emb_dim
        self.agent_feat_dim = agent_feat_dim
        self.d_model = d_model
        self.max_len = max_len

        in_dim = 3 + text_emb_dim + agent_feat_dim

        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        self.pos_emb = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.out_proj = nn.Linear(d_model, 3)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        """
        Transformer 'mask' where True/float(-inf) indicates blocked attention.
        We'll return float mask compatible with nn.TransformerEncoder:
          shape (T, T), float with -inf on upper triangle.
        """
        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)  # block attending to future
        return mask

    def forward(
        self,
        mu_prev_seq: torch.Tensor,      # (B, T, 3)
        text_emb_seq: torch.Tensor,     # (B, T, 768)
        agent_feat_seq: torch.Tensor,   # (B, T, 768) pooled
        attn_mask: Optional[torch.Tensor] = None,  # (B, T) 1=valid, 0=pad
    ) -> torch.Tensor:
        B, T, _ = mu_prev_seq.shape
        device = mu_prev_seq.device

        if T > self.max_len:
            raise ValueError(f"T={T} exceeds max_len={self.max_len}. Please increase max_len.")

        x = torch.cat([mu_prev_seq, text_emb_seq, agent_feat_seq], dim=-1)  # (B, T, in_dim)
        h = self.in_proj(x)  # (B, T, d_model)

        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        h = h + self.pos_emb(pos_ids)

        # key padding mask: True means "ignore"
        key_padding_mask = None
        if attn_mask is not None:
            key_padding_mask = (attn_mask == 0)

        causal = self._causal_mask(T, device=device)

        z = self.encoder(h, mask=causal, src_key_padding_mask=key_padding_mask)  # (B, T, d_model)
        logits = self.out_proj(z)  # (B, T, 3)
        mu = F.softmax(logits, dim=-1)
        return mu
