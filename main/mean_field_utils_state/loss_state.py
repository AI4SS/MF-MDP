import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """
    def __init__(self, ring_attn_group=None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        self.ring_attn_group = ring_attn_group
        if self.ring_attn_group:
            self.ring_attn_rank = dist.get_rank(self.ring_attn_group)
            self.ring_attn_world_size = dist.get_world_size(self.ring_attn_group)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 对齐 Logits 和 Labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 计算交叉熵
        loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

def calculate_log_probs(model, tokenizer, prompts, true_actions, max_length, device):
    """
    使用原子级拼接 (Token ID 拼接) 计算 NLL Loss，解决对齐导致的 12.38 异常。
    """
    # 记录原始方向并强制切换为 Right Padding 以保证矩阵对齐
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    
    # 直接初始化当前文件中的类
    loss_fn = GPTLMLoss()
    
    # 1. 预处理：确保评论不为空
    processed_true_actions = [
        a if (a and a.strip() != "") else "转发微博" for a in true_actions
    ]
    
    all_input_ids = []
    all_labels = []
    
    # 2. 原子级拼接：绕过字符串拼接带来的空格/特殊符偏移
    for p, a in zip(prompts, processed_true_actions):
        # Prompt 编码 (含 BOS)
        p_ids = tokenizer.encode(p, add_special_tokens=True)
        # Action 编码 (绝对不加 BOS/起始符，防止位置偏移)
        a_ids = tokenizer.encode(a, add_special_tokens=False)
        
        # 拼接 Token IDs
        combined_ids = (p_ids + a_ids)[:max_length]
        # 构造 Labels: Prompt 长度部分全部填充 IGNORE_INDEX
        label_ids = ([-100] * len(p_ids) + a_ids)[:max_length]
        
        all_input_ids.append(torch.tensor(combined_ids))
        all_labels.append(torch.tensor(label_ids))

    # 3. 转化为 Batch 矩阵
    input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    labels = pad_sequence(all_labels, batch_first=True, padding_value=-100).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    # 4. 推理与对齐校验
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        
        # --- 核心调试打印：检查模型到底在算哪个词的 Loss ---
        for i in range(min(1, labels.size(0))): 
            active_indices = (labels[i] != -100)
            # 排除 padding，只看真正被计算 Loss 的 Token
            actual_active = active_indices & (input_ids[i] != tokenizer.pad_token_id)
            active_tokens = input_ids[i][actual_active]
            
            # print(f"\n" + "="*25 + "【NLL 对齐校验】" + "="*25)
            # print(f"[样本索引 {i}]")
            # print(f">> 目标评论: '{processed_true_actions[i]}'")
            # print(f">> 模型实际计算 Loss 的词: '{tokenizer.decode(active_tokens)}'")
            # print(f">> Token ID 序列: {active_tokens.tolist()}")
            # print(f">> Prompt 长度 (已遮蔽): {(labels[i] == -100).sum().item()}")
            # print("="*66 + "\n")

        loss = loss_fn(outputs.logits, labels)

    # 还原 Tokenizer 方向
    tokenizer.padding_side = old_side
    
    # 5. 返回结果，如果是 NaN 则返回默认高值
    return loss.item() if not torch.isnan(loss) else 10.0