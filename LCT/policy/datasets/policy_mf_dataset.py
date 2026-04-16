import torch
from torch.utils.data import Dataset
import pandas as pd

class PolicyMFDataset(Dataset):
    """
    改进的 Policy Dataset，参考 Mean-Field-LLM 的数据构造方式。

    主要改进:
    1. 更详细的 prompt 模板 (包含用户画像、话题、舆论场等)
    2. ChatML 格式 (Qwen 兼容)
    3. 支持用户状态 [pos, neg, neu] 和未来状态预测
    4. 更好的 label masking (只计算回答部分的 loss)
    """
    def __init__(self, data, tokenizer, max_length=1024, k_steps=10):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.k_steps = k_steps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # =================================================================
        # 1. 字段提取
        # =================================================================
        topic = item.get('topic', '')

        # 优先使用 profile_text
        user_profile_str = item.get('profile_text', item.get('user_profile', '未知画像'))

        # 允许为空的字段
        hot_comment = item.get('hot_comment', None)
        related_cases_info = item.get('related_cases_info', None)

        # 舆论场数据 (兼容 字符串 或 列表)
        mean_field = item.get('mf_text', '')
        if isinstance(mean_field, list) and len(mean_field) > 0:
            mf_text = mean_field[-1]
        else:
            mf_text = mean_field

        # =================================================================
        # 2. 构建 Prompt (严格对齐 Mean-Field-LLM 的逻辑)
        # =================================================================
        prompt_content = f"当前讨论的话题是：{topic}\n"

        if hot_comment and hot_comment != "暂无最新热门评论":
            prompt_content += f"热门背景参考：{hot_comment}\n"

        if mf_text and mf_text != '' and mf_text != ['']:
            prompt_content += f"舆论环境总结：{mf_text}\n"

        if related_cases_info:
            prompt_content += f"相关案例：{related_cases_info}\n"

        # 插入用户画像
        prompt_content += f"网友资料：{user_profile_str}\n"

        # 格式约束指令
        format_instruction = (
            "请推测该网友可能的情绪、观点和立场，模拟该网友进行社交媒体互动：\n"
            "1. 如果你决定【转发】，请直接输出：转发微博\n"
            "2. 如果你决定【评论】，请直接输出具体的评论内容文本。\n"
            "注意：必须输出模拟内容，禁止输出任何多余的解释或分析。"
        )
        prompt_content += format_instruction

        # 本地训练/推理时的强引导后缀
        prompt_content += "\n\n模拟内容输出："

        # =================================================================
        # 3. 封装 ChatML 格式
        # =================================================================
        full_prompt_str = (
            f"<|im_start|>user\n{prompt_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # 构造 Target Text
        target_text = str(item['real_comments']) + "<|im_end|>"

        # 拼接 Input + Output
        full_text = full_prompt_str + target_text

        # =================================================================
        # 4. Tokenize
        # =================================================================
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encodings.input_ids.squeeze()
        attention_mask = encodings.attention_mask.squeeze()

        # =================================================================
        # 5. 构造 Labels (Masking)
        # =================================================================
        labels = input_ids.clone()

        # 计算 prompt 部分的长度，用于 mask
        prompt_encodings = self.tokenizer(
            full_prompt_str,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False
        )
        prompt_len = len(prompt_encodings.input_ids)

        # 将 Prompt 部分设为 -100 (不计算 Loss)
        labels[:prompt_len] = -100
        # 将 Padding 部分设为 -100
        labels[attention_mask == 0] = -100

        # =================================================================
        # 6. 状态数据 (MLP 部分 - 用于状态预测)
        # =================================================================
        # current_state: [3] -> [Pos, Neg, Neu]
        current_state = torch.tensor(item['user_state'], dtype=torch.float32)

        # future_states: [K, 3] -> 未来 K 步的真实分布
        future_states = torch.tensor(item['future_states_ground_truth'], dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "current_state": current_state,
            "future_states": future_states,
            # 保存原始文本供后续使用
            "question": full_prompt_str,
            "response": target_text,
            "file_name": item.get("file_name", "unknown"),
            "idx": idx
        }


def load_and_process_csv_data(data_dir_or_file):
    """
    处理CSV数据，转换为PolicyMFDataset期望的格式。

    映射逻辑：
    - user_profile: profile_text
    - user_state: [pre_pos, pre_neg, pre_neu] (Pos, Neg, Neu 顺序)
    - topic: topic
    - mf_text: batch_mf
    - real_comments: real_comments
    - future_states_ground_truth: 提取 dist_t0 到 dist_t10 的状态序列
    """
    import os
    import json

    if data_dir_or_file and os.path.exists(data_dir_or_file):
        if os.path.isdir(data_dir_or_file):
            # 目录模式，合并所有csv
            all_records = []
            for fname in sorted(os.listdir(data_dir_or_file)):
                if fname.endswith(".csv"):
                    fpath = os.path.join(data_dir_or_file, fname)
                    df = pd.read_csv(fpath)
                    processed_records = _process_csv_data(df)
                    all_records.extend(processed_records)
            print(f"Loaded {len(all_records)} samples from {data_dir_or_file}")
            return all_records
        elif data_dir_or_file.endswith(".csv"):
            df = pd.read_csv(data_dir_or_file)
            processed_records = _process_csv_data(df)
            print(f"Loaded {len(processed_records)} samples from {data_dir_or_file}")
            return processed_records
        else:
            with open(data_dir_or_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    else:
        print("No data path provided or file not found.")
        return []


def _process_csv_data(df):
    """
    处理CSV数据，转换为PolicyModelDataset期望的格式。
    """
    processed_records = []

    for idx, row in df.iterrows():
        # 1. 提取当前用户状态 [Pos, Neg, Neu]
        user_state = [
            float(row['pre_pos']),
            float(row['pre_neg']),
            float(row['pre_neu'])
        ]

        # 2. 提取未来 K 步的状态序列 (从 t0 到 t10)
        future_states = []
        for k in range(11):
            pos_col = f'dist_t{k}_pos'
            neg_col = f'dist_t{k}_neg'
            neu_col = f'dist_t{k}_neu'

            # 检查列是否存在
            if pos_col in row:
                future_states.append([
                    float(row[pos_col]),
                    float(row[neg_col]),
                    float(row[neu_col])
                ])

        # 3. 构建 Dataset 所需的记录
        record = {
            "user_profile": str(row['profile_text']),
            "user_state": user_state,
            "topic": str(row['topic']),
            "mf_text": str(row['batch_mf']) if pd.notna(row['batch_mf']) else "",
            "real_comments": str(row['real_comments']),
            "future_states_ground_truth": future_states,
            "file_name": f"{row.get('uid', 'unknown')}.csv" if 'uid' in row else f"data_{idx}.csv",
            "idx": idx
        }

        processed_records.append(record)

    return processed_records
