# encoders.py
# 路径: MFSim/models/state_transition/encoders.py

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel, AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# ============================================================
# Base Class
# ============================================================

class BaseTextEncoder(nn.Module):
    """
    统一的文本编码器基类。
    子类必须实现 forward(input_ids, attention_mask) -> (batch_size, hidden_dim)
    """
    def __init__(self):
        super().__init__()

    def forward(self, input_ids, attention_mask):
        raise NotImplementedError("Subclasses must implement forward()")


# ============================================================
# 1. Simple GRU Encoder
# ============================================================

class SimpleGRUTextEncoder(BaseTextEncoder):
    """
    轻量级文本编码器：Embedding + GRU → 句向量
    用于训练状态转移模型中的文本平均场 C_{t-1}
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        pad_token_id: int = 0
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=pad_token_id
        )
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.hidden_dim = hidden_dim

    def forward(self, input_ids, attention_mask):
        """
        input_ids:      (batch_size, seq_len)
        attention_mask: (batch_size, seq_len) → 1=有效，0=padding
        return:         (batch_size, hidden_dim)
        """
        emb = self.embedding(input_ids)  # (B, L, emb_dim)

        # mask pad tokens
        emb = emb * attention_mask.unsqueeze(-1)

        _, h_n = self.gru(emb)      # h_n: (1, B, hidden_dim)
        sent_emb = h_n.squeeze(0)   # (B, hidden_dim)

        return sent_emb


# ============================================================
# 2. HuggingFace BERT Encoder
# ============================================================

class BERTTextEncoder(BaseTextEncoder):
    """
    使用 HuggingFace BERT / RoBERTa 作为文本平均场编码器 g_ψ。
    可选择冻结 / 微调。
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_dim: int = None,
        freeze: bool = True
    ):
        """
        model_name: HF 模型名称
        output_dim: 若非 None，则使用线性层降维到该维度
        freeze: 若为 True，则冻结 BERT 参数作为纯特征提取器
        """
        super().__init__()

        if not HF_AVAILABLE:
            raise ImportError("transformers not installed. Run: pip install transformers")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

        hidden_size = self.bert.config.hidden_size

        if output_dim is not None:
            self.proj = nn.Linear(hidden_size, output_dim)
            self.output_dim = output_dim
        else:
            self.proj = None
            self.output_dim = hidden_size

    def forward(self, input_ids, attention_mask):
        """
        BERT 输出 CLS embedding 作为句子向量 C_{t-1}
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token embedding → (B, hidden_dim)
        cls_emb = outputs.last_hidden_state[:, 0, :]

        if self.proj:
            cls_emb = self.proj(cls_emb)

        return cls_emb


# ============================================================
# 3. Factory Function（可选）
# ============================================================

def build_text_encoder(config: dict):
    """
    根据 config 创建文本编码器，方便管理。
    config 例如：
    {
        "type": "gru",
        "vocab_size": 5000,
        "emb_dim": 128,
        "hidden_dim": 256
    }

    或：
    {
        "type": "bert",
        "model_name": "bert-base-uncased",
        "output_dim": 256,
        "freeze": true
    }
    """

    encoder_type = config.get("type", "gru").lower()

    if encoder_type == "gru":
        return SimpleGRUTextEncoder(
            vocab_size=config["vocab_size"],
            emb_dim=config.get("emb_dim", 128),
            hidden_dim=config.get("hidden_dim", 256),
            pad_token_id=config.get("pad_token_id", 0)
        )

    elif encoder_type == "bert":
        return BERTTextEncoder(
            model_name=config.get("model_name", "bert-base-uncased"),
            output_dim=config.get("output_dim", None),
            freeze=config.get("freeze", True)
        )

    else:
        raise ValueError(f"Unknown text encoder type: {encoder_type}")
