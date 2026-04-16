import os
import json
import numpy as np
import random
import torch
from datetime import datetime
import sys


# 获取脚本所在目录，然后添加 eval 目录到 sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(SCRIPT_DIR)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from mean_field_utils_state import calculate_log_probs, calculate_mean_field, build_state, build_prompt


# 1. 定义基础目录 - 使用相对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # .../release
BASE_DIR = RELEASE_DIR

# 2. 简化 sys.path，只保留项目根目录和 main
# 尝试从配置文件读取项目根目录，否则使用自动计算的路径
try:
    from config.settings import get_config
    config = get_config()
    PROJECT_ROOT = os.getenv('PROJECT_ROOT') or str(config.project_root)
except ImportError:
    PROJECT_ROOT = BASE_DIR
    config = None

MAIN_DIR = os.path.join(PROJECT_ROOT, "main")

for p in [PROJECT_ROOT, MAIN_DIR]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# 3. 尝试导入推理模块（不提前导入 TrainConfig）
try:
    from LCT.state_transition.encoders import build_text_encoder
    from LCT.state_transition.event_transformer_net import CausalEventTransformerNet
    print("✅ [成功] ST-Net 推理模块已成功识别")
except ImportError as e:
    print(f"❌ [失败] 推理模块导入失败: {e}")
    print(f"当前 Python 搜索路径前5项: {sys.path[:5]}")

seed_num = 46
# 设置随机种子与设备
random.seed(seed_num)
np.random.seed(seed_num)
torch.manual_seed(seed_num)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from transformers import AutoTokenizer, AutoModelForCausalLM

import pandas as pd

import asyncio
import aiohttp
import re

# 状态索引映射
IDX2STATE = {0: "积极 (Positive)", 1: "中立 (Neutral)", 2: "消极 (Negative)"}

async def async_call_single(i, prompt, model_name, api_base, temperature,top_p,max_tokens):
    url = f"{api_base}/chat/completions"
    headers = {"Authorization": f"Bearer EMPTY"}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "要求思考过程简短。" + prompt},
            {"role": "assistant", "content": "<think>\n</think>\n\n"}
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                data = await resp.json()
                content = data['choices'][0]['message']['content']
                cleaned = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip().split('\n')[0]
                return i, cleaned
    except Exception as e:
        return i, "（OpenAI API 调用失败）"

async def async_generate_with_vllm(batch_prompts,temperature,top_p,max_tokens, model_name="deepseek-chat", api_base=None):
    if api_base is None:
        api_base = config.get('api.vllm_base_url', 'http://localhost:8000/v1') if config else 'http://localhost:8000/v1'
    tasks = [async_call_single(i, prompt, model_name, api_base, temperature, top_p, max_tokens) for i, prompt in enumerate(batch_prompts)]
    results = await asyncio.gather(*tasks)
    results.sort()  # 保证按顺序返回
    return [r[1] for r in results]


# 将以下逻辑放入 run_simulation 函数开始处
def prepare_profile_dict():
    """
    加载聚类用户画像数据：直接从 JSONL 文件读取 user_id -> profile 映射
    返回 (profile_dict, profile_list) 用于精确匹配和随机兜底
    """
    # 从配置文件读取路径，否则使用默认路径
    profile_path_default = config.get('paths.profile_path') if config else None
    if profile_path_default:
        jsonl_path = profile_path_default
    else:
        jsonl_path = os.path.join(BASE_DIR, "main/data/profile/cluster_core_user_profile.jsonl")

    profile_dict = {}
    profile_list = []
    try:
        # JSONL 文件直接是 user_id -> 画像的映射
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    user_id = str(item.get('user_id', ''))
                    if user_id:
                        profile_dict[user_id] = item
                        profile_list.append(item)

        print(f"✅ 加载聚类用户画像成功: 共 {len(profile_dict)} 个用户")
        return profile_dict, profile_list
    except Exception as e:
        print(f"⚠️ 预加载画像失败: {e}")
        return {}, []

def run_simulation(tweets, comment_n, tokenizer, a_model, mf_model,
                   st_model_bundle,
                   batch_size=16, alg="mf", model_type='1.5B',
                   generate_true=True, simulation_start=2):
    
    # --- 1. 解包状态转移模型包 ---
    if st_model_bundle is not None:
        st_encoder, st_model, st_cfg = st_model_bundle
    else:
        st_encoder, st_model, st_cfg = None, None, None

    # 状态索引映射
    IDX2STATE = {0: "Positive", 1: "Neutral", 2: "Negative"}
    

    # --- 准备画像数据 ---
    print(f"╔══════════════════════════════════════════════════════════════════════════════╗")
    print(f"║  初始化用户画像系统                                                              ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════════╝")
    uid_to_vec = {}
    user_profile_dict = {}
    user_profile_list = []  # 初始化为空列表

    # 加载聚类用户画像（无论是否有 ST-Net 模型都加载）
    user_profile_dict, user_profile_list = prepare_profile_dict()

    # 打印聚类用户画像加载状态
    if user_profile_list:
        print(f"✅ 聚类用户画像已加载: {len(user_profile_dict)} 个精确映射 + {len(user_profile_list)} 个画像模板")
        print(f"   ├─ 精确匹配: 当用户 UID 在画像字典中时使用")
        print(f"   └─ 随机兜底: 当 UID 不匹配时，使用 UID 哈希随机选择一个画像模板")
        # 显示画像类型分布
        stance_types = {}
        for p in user_profile_list:
            stance = p.get('stance_label', 'Unknown')
            stance_types[stance] = stance_types.get(stance, 0) + 1
        print(f"   画像类型分布:")
        for stance, count in sorted(stance_types.items(), key=lambda x: -x[1]):
            print(f"     • {stance}: {count} 个")
    else:
        print(f"⚠️  警告: 聚类用户画像未加载，将使用基础字段构建用户状态")

    if st_model_bundle:
        _, _, st_cfg = st_model_bundle
        # 注意: 新的 TrainConfig 没有 profile_path 属性
        # 如果需要加载额外的 profile 向量，需要添加到 TrainConfig 中
        # 这里暂时跳过这部分逻辑
        pass

    topic = tweets[0]['text']
    sorted_tweets = sorted(tweets[1:], key=lambda x: x.get('t', 0))

    # 如果 comment_n 为 None，使用全部数据；否则限制数量
    if comment_n is None:
        comment_n = len(sorted_tweets)
        print(f"使用全部数据: {comment_n} 条评论")
    else:
        comment_n = min(len(sorted_tweets), comment_n)

    sampled_indices = list(range(comment_n))
    print(f"╔══════════════════════════════════════════════════════════════════════════════╗")
    print(f"║  仿真配置                                                                        ║")
    print(f"╠══════════════════════════════════════════════════════════════════════════════╣")
    topic_short = topic[:47] + "..." if len(topic) > 50 else topic
    print(f"║  话题: {topic_short:<50}                               ║")
    print(f"║  Agent 数量: {len(sorted_tweets):>6}                                                                   ║")
    print(f"║  仿真数量: {comment_n:>6}                                                                    ║")
    print(f"║  仿真起点: {simulation_start:>6}                                                                    ║")
    print(f"║  算法: {alg:<14}                                                               ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════════╝")

    # 确定模型要求的维度
    expected_feat_dim = 768
    if st_model_bundle:
        expected_feat_dim = st_model_bundle[2].agent_feat_dim

    # # --- 2. 构建 Agents ---
    # agents = []
    # for idx in sampled_indices:
    #     current_uid = str(sorted_tweets[idx]['uid'])
    #     raw_vec = uid_to_vec.get(current_uid, np.zeros(expected_feat_dim))
    #     if len(raw_vec) < expected_feat_dim:
    #         profile_vec = np.pad(raw_vec, (0, expected_feat_dim - len(raw_vec)), 'constant')
    #     else:
    #         profile_vec = raw_vec[:expected_feat_dim]

    #     agents.append({
    #         'uid': current_uid,
    #         'text': sorted_tweets[idx]['text'] if sorted_tweets[idx]['text'] else "转发微博" + sorted_tweets[idx]['original_text'],
    #         'user_description': sorted_tweets[idx].get('user_description', "未知"),
    #         'friends_count': sorted_tweets[idx].get('friends_count', 0),
    #         'followers_count': sorted_tweets[idx].get('followers_count', 0),
    #         'hot_count': (sorted_tweets[idx].get('reposts_count', 0) + 
    #                       sorted_tweets[idx].get('attitudes_count', 0) + 
    #                       sorted_tweets[idx].get('comments_count', 0)),
    #         'fans_count': sorted_tweets[idx].get('followers_count', 0),
    #         'original_index': idx,
    #         'profile_vec': profile_vec
    #     })
    agents = []
    for idx in sampled_indices:
        item = sorted_tweets[idx]  # 获取当前数据项
        current_uid = str(item.get('uid', ''))
        
        # 向量处理逻辑（保持不变）
        raw_vec = uid_to_vec.get(current_uid, np.zeros(expected_feat_dim))
        if len(raw_vec) < expected_feat_dim:
            profile_vec = np.pad(raw_vec, (0, expected_feat_dim - len(raw_vec)), 'constant')
        else:
            profile_vec = raw_vec[:expected_feat_dim]

        # === 核心修改：区分新旧数据源 ===
        
        # 判断依据：新数据一定包含 'profile_text' 字段
        if 'profile_text' in item:
            # === 新数据处理逻辑 ===
            text_content = item.get('text', "")
            
            # 使用 profile_text 作为描述
            desc = item.get('profile_text', "未知")
            
            # 缺失的社交字段补 0
            friends_count = 0
            followers_count = 0
            
            # 【关键】热度映射：使用 like_count，并强转为 int
            # 注意：JSON里可能是字符串 "82"，需要 int()
            try:
                hot_count = int(item.get('like_count') or 0)
            except ValueError:
                hot_count = 0
                
            # 将 profile_text 存入 agent 字典，以便 build_state 调用
            prof_txt_value = item.get('profile_text', "")
            
        else:
            # === 旧数据（微博）处理逻辑（保持原样）===
            text_content = item['text'] if item.get('text') else "转发微博" + item.get('original_text', '')
            desc = item.get('user_description', "未知")
            friends_count = item.get('friends_count', 0)
            followers_count = item.get('followers_count', 0)
            
            # 微博热度计算公式
            hot_count = (item.get('reposts_count', 0) + 
                         item.get('attitudes_count', 0) + 
                         item.get('comments_count', 0))
            
            # 旧数据没有 profile_text 字段，置为空
            prof_txt_value = ""

        # === 统一添加到列表 ===
        agents.append({
            'uid': current_uid,
            'text': text_content,
            'user_description': desc,
            'friends_count': friends_count,
            'followers_count': followers_count,
            'hot_count': hot_count,       # 新数据是 like_count，旧数据是求和
            'fans_count': followers_count, # 新数据默认为 0
            'original_index': idx,
            'profile_vec': profile_vec,
            'profile_text': prof_txt_value # 【关键】存入这个字段，供 build_state 使用
        })
    
    results = []
    total_steps = (len(agents) + batch_size - 1) // batch_size
    mean_field = '' 
    mf_loss = torch.tensor([0.], device=device)

    # 初始群体分布 mu_prev
    if st_model_bundle:
        mu_prev = torch.tensor([[0.33, 0.34, 0.33]], device=device)
    else:
        mu_prev = None

    # 初始化 Client
    if "gpt" in model_type:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    elif "DeepSeek" in model_type:
        from openai import OpenAI
        client = OpenAI(api_key="EMPTY", base_url="http://172.18.129.58:23369/v1")
    else:
        client = None
        
    for step in range(total_steps):
        start_idx = step * batch_size
        end_idx = min(start_idx + batch_size, len(agents))
        batch_agents_list = agents[start_idx:end_idx]

        # 打印批次信息
        print(f"\n{'='*80}")
        print(f"🚀 Batch {step+1}/{total_steps} | 处理 Agent {start_idx+1}-{end_idx}")
        print(f"{'='*80}")
        
        batch_agents = pd.DataFrame(batch_agents_list) 
        current_batch_size = len(batch_agents_list)
        
        # 初始化辅助列表
        latest_hot_comment_texts = ['暂无最新热门评论'] * current_batch_size
        related_cases_infos = [""] * current_batch_size 
        
        # hot/pre 处理
        if 'hot' in alg:
            relevant_results = results[max(len(results) - 30, 0):]
            top_hot_comments = sorted(relevant_results, key=lambda x: x.get('fans_count', 0), reverse=True)[:5]
            if top_hot_comments:
                all_hot_comment = "当前最热门的评论如下："
                label = 'real_comment' if start_idx < simulation_start else 'generated_comment'
                for i, comment in enumerate(top_hot_comments):
                    all_hot_comment += f"第{i+1}位网友评论：{comment[label]}。\n"
                latest_hot_comment_texts = [all_hot_comment] * current_batch_size

        # --- 3. 调用 build_state (传入 profile_list 用于随机兜底) ---
        batch_states = [build_state(agent, user_profile_dict, user_profile_list) for agent in batch_agents_list]

        # --- 4. 状态转移预测 (优先级: ST-Net > 默认值) ---
        batch_predicted_states = [None] * current_batch_size
        mu_curr_dist = [0.33, 0.33, 0.34] # 默认兜底
        used_source = "DEFAULT"  # 追踪数据来源

        # [优先级1] 使用 ST-Net 模型预测
        if st_model_bundle and st_model and st_encoder:
            try:
                # ---- 强健版：确保所有输入都是正确的 (B, T, D) 形状 ----

                # ---- mu_prev_seq: 强制变成 (1,1,3) ----
                if mu_prev is None:
                    mu_prev = torch.tensor([[0.33, 0.34, 0.33]], device=device, dtype=torch.float32)
                if mu_prev.dim() == 1:          # (3,)
                    mu_prev = mu_prev.unsqueeze(0)   # (1,3)
                mu_prev_seq = mu_prev.unsqueeze(1)   # (1,1,3)

                # ---- text_emb: 强制变成 (1,1,768) ----
                mf_input_text = " ".join(mean_field) if isinstance(mean_field, list) else mean_field
                if not mf_input_text:
                    mf_input_text = "目前尚无舆论总结。"

                tokens = st_encoder.tokenizer([mf_input_text], padding=True, truncation=True,
                                              max_length=128, return_tensors="pt").to(device)

                with torch.no_grad():
                    text_emb = st_encoder(tokens["input_ids"], tokens.get("attention_mask"))
                    if isinstance(text_emb, tuple):
                        text_emb = text_emb[0]
                    # 常见两种输出：(1,768) 或 (1,L,768)
                    if text_emb.dim() == 3:
                        text_emb = text_emb[:, 0, :]          # 取 CLS -> (1,768)
                    text_emb_seq = text_emb.unsqueeze(1)      # (1,1,768)

                # ---- agent_feat_seq: pooled 后变成 (1,1,768) ----
                # 将所有 agent 的特征聚合成一个向量（因为是预测整个群体的分布）
                agent_feat = torch.stack([
                    torch.tensor(a['profile_vec'], dtype=torch.float32, device=device)
                    for a in batch_agents_list
                ], dim=0)                         # (N,768)
                agent_feat_seq = agent_feat.mean(dim=0).unsqueeze(0).unsqueeze(1)   # (1,1,768)

                # ---- 打印调试信息 ----
                print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print(f"🔍 ST-Net 模型推理 - 张量形状调试:")
                print(f"   mu_prev shape: {mu_prev.shape} (dim: {mu_prev.dim()})")
                print(f"   mu_prev_seq shape: {mu_prev_seq.shape} (dim: {mu_prev_seq.dim()})")
                print(f"   text_emb shape: {text_emb.shape} (dim: {text_emb.dim()})")
                print(f"   text_emb_seq shape: {text_emb_seq.shape} (dim: {text_emb_seq.dim()})")
                print(f"   agent_feat shape: {agent_feat.shape} (dim: {agent_feat.dim()})")
                print(f"   agent_feat_seq shape: {agent_feat_seq.shape} (dim: {agent_feat_seq.dim()})")
                print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

                # ---- 调用模型 ----
                with torch.no_grad():
                    mu_pred = st_model(mu_prev_seq=mu_prev_seq,
                                       text_emb_seq=text_emb_seq,
                                       agent_feat_seq=agent_feat_seq)

                # mu_pred shape: (B, T, 3) = (1, 1, 3)
                mu_prev = mu_pred[:, 0, :]                 # (1,3)
                mu_curr_dist = mu_pred[0, 0].cpu().tolist()  # 转为 list
                # 对于当前批次，使用相同的分布（因为模型只预测了一个时间步）
                for i in range(current_batch_size):
                    batch_predicted_states[i] = IDX2STATE[int(np.argmax(mu_curr_dist))]

                used_source = "ST-NET-MODEL"
            except Exception as e:
                print(f"⚠️  ST-Net 模型预测失败: {e}, 使用默认值...")
                import traceback
                traceback.print_exc()
        else:
            # ST-Net 不可用的原因诊断
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"🔍 ST-Net 模型状态诊断:")
            if st_model_bundle is None:
                print(f"   ❌ st_model_bundle = None (模型未加载)")
                print(f"      → 原因: 模型加载代码被注释或未实现")
                print(f"      → 位置: 第770-818行")
            else:
                st_enc, st_mdl, _ = st_model_bundle
                if st_model is None:
                    print(f"   ❌ st_model = None (模型对象为空)")
                if st_encoder is None:
                    print(f"   ❌ st_encoder = None (编码器为空)")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # [优先级2] 默认值 (当 ST-Net 不可用或失败时使用)
        if used_source == "DEFAULT":
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"⚠️  使用默认状态分布 [0.33, 0.33, 0.34]")
            print(f"   原因: ST-Net 模型未加载或预测失败")
            print(f"         st_model_bundle = {st_model_bundle is not None}")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        print(f"Batch {step+1} mean field: {mean_field}")

        # 打印当前分布（无论来源是 ST-NET/DEFAULT）
        # 根据来源设置标签和样式
        if used_source == "ST-NET-MODEL":
            # ST-NET 模型预测 - 最醒目的显示
            pos_pct, neu_pct, neg_pct = [f"{x*100:.1f}%" for x in mu_curr_dist]
            print(f"")
            print(f"╔══════════════════════════════════════════════════════════════════════════════╗")
            print(f"║  ✅✅✅  ST-NET 模型预测成功  ✅✅✅                                                  ║")
            print(f"╠══════════════════════════════════════════════════════════════════════════════╣")
            print(f"║  Batch {step+1} 状态分布 (由 ST-Net 模型预测)                                         ║")
            print(f"╠══════════════════════════════════════════════════════════════════════════════╣")
            print(f"║  积极: {pos_pct:>6}  │  中立: {neu_pct:>6}  │  消极: {neg_pct:>6}                     ║")
            print(f"╚══════════════════════════════════════════════════════════════════════════════╝")
            print(f"")
        else:
            # DEFAULT - 警告显示
            pos_pct, neu_pct, neg_pct = [f"{x*100:.1f}%" for x in mu_curr_dist]
            print(f"")
            print(f"╔══════════════════════════════════════════════════════════════════════════════╗")
            print(f"║  ⚠️⚠️⚠️  使用默认状态分布  ⚠️⚠️⚠️                                                  ║")
            print(f"╠══════════════════════════════════════════════════════════════════════════════╣")
            print(f"║  Batch {step+1} 状态分布 [DEFAULT - 固定值]                                            ║")
            print(f"╠══════════════════════════════════════════════════════════════════════════════╣")
            print(f"║  积极: {pos_pct:>6}  │  中立: {neu_pct:>6}  │  消极: {neg_pct:>6}                     ║")
            print(f"╠══════════════════════════════════════════════════════════════════════════════╣")
            print(f"║  💡 提示: 状态分布未动态预测，所有 Agent 使用相同分布                                ║")
            print(f"╚══════════════════════════════════════════════════════════════════════════════╝")
            print(f"")

        # --- 构建提示词 ---
        # 1. 主实验组：传入 mu_curr_dist，使其能在 Prompt 中显示统计数据
        print(f"📝 构建提示词 (实验组 - 包含状态分布)...")
        batch_prompts = [
            build_prompt(
                state,
                topic,
                latest_hot,
                mean_field,
                related_info,
                alg=alg,
                model_type=model_type,
                state_dist=mu_curr_dist  # <--- [关键] 传入当前计算出的分布
            )
            for state, latest_hot, related_info in zip(
                batch_states, latest_hot_comment_texts, related_cases_infos
            )
        ]

        # 打印第一个提示词示例（验证状态分布是否正确注入）
        if batch_prompts:
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"📋 Agent 0 完整提示词:")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(batch_prompts[0])
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # 2. 对比组 (No MF)：显式传入 None 或依赖默认值
        # 这里的 mean_field="" 已经会阻断逻辑进入，但显式传 None 更安全规范
        batch_prompts_wo_mf = [
            build_prompt(
                state,
                topic,
                [''] * current_batch_size,
                "",           # mean_field 置空
                related_info,
                alg=alg,
                model_type=model_type,
                # state_dist=None  # <--- [修改] 对比组不应包含分布信息
            )
            for state, related_info in zip(batch_states, related_cases_infos)
        ]
        
        # --- 批量生成评论 ---
        batch_true_actions = list(batch_agents['text'])
        generated_comments = list(batch_agents['text']) # 初始化

        if generate_true and (start_idx + current_batch_size > simulation_start):
            print("========= simulation ========")
            if "gpt" in model_type:
                try:
                    for i, prompt in enumerate(batch_prompts):
                        response = client.chat.completions.create(
                            model=a_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.8, max_tokens=50, top_p=0.85,
                        )
                        generated_comments[i] = response.choices[0].message.content.split('\n')[0]
                except Exception as e:
                    print(f"调用OpenAI API出错: {e}")
                    generated_comments = ["（OpenAI API 调用失败）"] * len(batch_agents_list)
            
            elif "DeepSeek" in model_type: 
                try:
                    generated_comments = asyncio.run(
                        async_generate_with_vllm(
                            batch_prompts=batch_prompts, model_name=a_model,
                            temperature=0.8, top_p=0.85, max_tokens=500
                        )
                    )
                except Exception as e:
                    print(f"调用 vLLM 异步接口出错: {e}")
                    generated_comments = ["（vLLM 调用失败）"] * len(batch_agents_list)

            elif "ds" in model_type:  
                try:
                    for i, prompt in enumerate(batch_prompts):
                        response = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[{"role": "system", "content": "You are a helpful assistant"},
                                        {"role": "user", "content": prompt}],
                            temperature=0.8, max_tokens=50, top_p=0.85,
                        )
                        generated_comments[i] = response.choices[0].message.content.split('\n')[0]
                except Exception as e:
                    print(f"调用DeepSeek API出错: {e}")
                    generated_comments = ["（DeepSeek API 调用失败）"] * len(batch_agents_list)
            else:
                generated_comments = generate_batch_comments(batch_prompts, tokenizer, a_model, device)
        else:
            print(f"Batch {step + 1} -- True comment = {batch_true_actions}")

        # --- 干预实验 ---
        if alg == "mf_w_inter":
            if step == 4:
                if len(generated_comments) > 7:
                    generated_comments[1] = "真的假的？"
                    generated_comments[3] = "求真相"
                    generated_comments[5] = "我不相信，奥运会又不是闹着玩的？"
                    generated_comments[7] = "等待真相，不要传谣！请大家理智看待！！"
        elif alg == "mf_w_key":
            if step == 4: generated_comments[-1] = "真的假的？"
            if step == 5 and len(generated_comments) > 3:
                generated_comments[1] = batch_true_actions[1]
                generated_comments[3] = batch_true_actions[3]

        print(f"Batch {step + 1} ---------------------------------------------")
        if len(generated_comments) != len(batch_agents_list):
            print("Generation failed, fallback to real comments")
            generated_comments = batch_true_actions[:len(batch_agents_list)]
        print('\n'.join(generated_comments[:3]) + "\n...") 

        # 计算 Log Probability
        if "gpt" not in model_type and "DeepSeek" not in model_type:
            batch_log_probs = calculate_log_probs(
                a_model, tokenizer, batch_prompts, batch_true_actions, 600, device
            )
        else:
            batch_log_probs = 10.0

        # 保存结果
        for i, agent in enumerate(batch_agents_list):
            results.append({
                'step': step + 1,
                'agent_index': start_idx + i + 1,
                'topic': topic,
                'state': batch_states[i],
                'generated_comment': generated_comments[i],
                'real_comment': agent['text'],
                'log_prob': batch_log_probs if 'batch_log_probs' in locals() else 0.0,
                'mf_loss': mf_loss.item() if (isinstance(mf_loss, torch.Tensor)) else 0.0,
                'mean_field': mean_field,
                'mu_dist': mu_curr_dist if 'mu_curr_dist' in locals() else [0.33, 0.33, 0.34],
                'hot_count': agent.get('hot_count', 0)
            })

        if "mf" in alg:
            mean_field, mf_loss = calculate_mean_field(
                topic,
                batch_states,
                generated_comments,
                mean_field,
                mf_model,
                device=device,
                alg=alg,
                model_type=model_type,
                client=client,
                tokenizer=tokenizer,
                state_distribution=mu_curr_dist
            )
            
    return pd.DataFrame(results)


import torch


def generate_batch_comments(batch_prompts, tokenizer, a_model, device, max_new_tokens=50):
    """
    使用本地 Hugging Face 模型批量生成文本。

    参数:
        batch_prompts (List[str]): 输入提示列表
        tokenizer: Hugging Face tokenizer 对象
        a_model: Hugging Face 模型对象
        device: 模型使用的设备 (如 'cuda' 或 'cpu')
        max_new_tokens (int): 每个提示最大生成的新 token 数

    返回:
        List[str]: 生成的文本列表
    """
    try:
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)
        
        input_tokens = inputs["input_ids"]
        input_len = inputs['attention_mask'].shape[1]
        
        with torch.no_grad():
            outputs = a_model.generate(
                input_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.85,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        outputs_list = outputs.tolist()
        generated_comments = []
        
        for i, output in enumerate(outputs_list):
            new_tokens = output[input_len:]  # 只取生成的新 token
            generated_text = \
            tokenizer.decode(new_tokens, skip_special_tokens=True).strip().replace("\u10000b", "").split('\n')[0]
            generated_comments.append(generated_text)

        # print(generated_comments)
        # sys.exit(0)  
        
        return generated_comments
    
    except Exception as e:
        print(f"调用本地模型出错: {e}")
        return []

def evaluate(test_dir, test_file_name, comment_n, alg, model_type, tokenizer,
             a_model, mf_model, batch_size, generate_true=True, st_model_bundle=None):

    selected_files = [test_dir + '/' + test_file_name]
    all_results = pd.DataFrame()

    for file_i, file_path in enumerate(selected_files):
        print("-" * 50)
        print(f"TEST FILE {file_i} | Topic: {test_file_name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        file_results = run_simulation(
            data, comment_n, tokenizer, a_model, mf_model,
            st_model_bundle=st_model_bundle,
            batch_size=batch_size, alg=alg,
            model_type=model_type,
            generate_true=generate_true,
            simulation_start=simulation_start
        )
        
        # --- [关键修改]：安全计算指标，防止 KeyError ---
        # 1. 计算 Log Prob
        avg_log_p = file_results['log_prob'].mean() if 'log_prob' in file_results.columns else "N/A"
        
        # 2. 计算 MF Loss
        avg_mf_loss = file_results['mf_loss'].mean() if 'mf_loss' in file_results.columns else "N/A"
        
        # 3. [新增] 打印 ST-Net 演化状态 (取最后一个 Batch 的分布作为代表)
        last_mu = "N/A"
        if 'mu_dist' in file_results.columns:
            last_mu = file_results['mu_dist'].iloc[-1]

        print(f"Results for {test_file_name}:")
        print(f" >> Average Log Prob: {avg_log_p}")
        print(f" >> Average MF Loss: {avg_mf_loss}")
        if last_mu != "N/A":
            print(f" >> Final State Distribution (Pos/Neu/Neg): {last_mu}")
        
        # 标记文件来源并合并
        file_results['file'] = file_path
        all_results = pd.concat([all_results, file_results], ignore_index=True)
        
    return all_results

def load_st_model(device, checkpoint_path=None):
    import importlib

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))  # /root/MF-MDP

    print(f"🔍 [调试] load_st_model 路径信息:")
    print(f"   __file__: {__file__}")
    print(f"   script_dir: {script_dir}")
    print(f"   project_root: {project_root}")

    # 只保留最关键的路径优先级
    if project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)

    # 关键：强制使用本地 datasets，避免命中其他已缓存模块
    if "datasets" in sys.modules:
        old_mod = sys.modules["datasets"]
        old_file = getattr(old_mod, "__file__", "") or ""
        print(f"   [调试] 发现已缓存 datasets: {old_file}")
        if not old_file.startswith(project_root):
            print("   [调试] 删除非本地 datasets 缓存")
            del sys.modules["datasets"]

    try:
        import datasets
        print(f"   [调试] 当前 datasets 来源: {getattr(datasets, '__file__', None)}")

        # 先显式测试本地数据集模块
        import importlib
        event_ds_mod = importlib.import_module("datasets.event_state_datasets")
        print(f"   [调试] datasets.event_state_datasets 导入成功: {event_ds_mod.__file__}")

        print("   尝试导入 TrainConfig...")
        from LCT.state_transition.training.train_event_transformer import TrainConfig
        print("   TrainConfig 导入成功!")

        from LCT.state_transition.encoders import build_text_encoder
        from LCT.state_transition.event_transformer_net import CausalEventTransformerNet

    except ImportError as e:
        print(f"❌ [导入失败] 无法导入 ST-Net 依赖模块: {e}")
        print(f"   sys.path 前5项: {sys.path[:5]}")
        import traceback
        traceback.print_exc()
        return None, None, None

    if checkpoint_path is None:
        checkpoint_path = os.environ.get(
            "ST_MODEL_CHECKPOINT",
            os.path.join(BASE_DIR, "checkpoints/state_transition_best_a=1.pt")
        )

    try:
        cfg = TrainConfig()

        if not os.path.exists(checkpoint_path):
            print(f"错误：找不到模型权重文件 {checkpoint_path}")
            return None, None, None

        checkpoint = torch.load(
            checkpoint_path,
            map_location=lambda storage, loc: storage.cuda() if torch.cuda.is_available() else storage.cpu(),
            weights_only=False
        )

        print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")

        st_model = CausalEventTransformerNet(
            text_emb_dim=cfg.text_emb_dim,
            agent_feat_dim=cfg.agent_feat_dim,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            max_len=cfg.max_len
        )

        st_model.load_state_dict(checkpoint["model_state_dict"])
        st_model.to(device).eval()

        if "encoder_state_dict" in checkpoint:
            print("📂 检测到旧格式 checkpoint，加载 encoder_state_dict")
            encoder = build_text_encoder({
                "type": cfg.encoder_type,
                "model_name": cfg.model_name,
                "output_dim": cfg.text_emb_dim,
                "freeze": True
            })
            encoder.load_state_dict(checkpoint["encoder_state_dict"])
            encoder.to(device).eval()
        else:
            print("📂 检测到新格式 checkpoint (event_transformer)，重新初始化 encoder")
            encoder = build_text_encoder({
                "type": cfg.encoder_type,
                "model_name": cfg.model_name,
                "output_dim": cfg.text_emb_dim,
                "freeze": True
            })
            encoder.to(device).eval()

        print(f"✅ [成功] 成功加载 ST-Net 模型：{checkpoint_path}")
        return encoder, st_model, cfg

    except Exception as e:
        print(f"❌ [错误] ST-Net 加载过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# def runner(alg, model_type, test_name, comment_n, a_MODEL_NAME, mf_MODEL_NAME, generate_true=True):
def runner(alg, model_type, test_name, comment_n, a_MODEL_NAME, mf_MODEL_NAME,
           generate_true=True, st_model_path=None):
    # 直接使用模型名称，不拼接 MODEL_PATH
    if os.path.isabs(a_MODEL_NAME):
        a_model_name = a_MODEL_NAME
    else:
        a_model_name = os.path.join(MODEL_PATH, a_MODEL_NAME) if MODEL_PATH else a_MODEL_NAME

    if os.path.isabs(mf_MODEL_NAME):
        mf_model_name = mf_MODEL_NAME
    else:
        mf_model_name = os.path.join(MODEL_PATH, mf_MODEL_NAME) if MODEL_PATH else mf_MODEL_NAME

    print(f"Actor model: {a_model_name}")
    print(f"MF model: {mf_model_name}")

    # 处理 test_name：如果是完整路径，则解析出目录和文件名
    if '/' in test_name:
        # 完整路径模式: "./main/data/xxx.json"
        test_dir = os.path.dirname(test_name)
        test_file_name = os.path.basename(test_name)
    else:
        # 旧模式: 只有文件名，使用默认目录
        test_dir = './main/data'  # 相对于 release 目录
        test_file_name = test_name

    # --- 新增：加载 ST-Net 模型包 ---
    st_model_bundle = None
    st_model_status = "NOT_CONFIGURED"  # 追踪 ST-Net 状态

    print(f"╔══════════════════════════════════════════════════════════════════════════════╗")
    print(f"║  ST-Net 状态转移模型检查                                                         ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════════╝")

    # 检查是否需要加载 ST-Net 模型
    load_st_net = "mf" in alg or "state" in alg

    if load_st_net:
        print(f"🔍 算法 '{alg}' 需要状态转移模型，尝试加载...")
        print(f"   检查点: 需要 ST-Net 模型文件路径")


        # 从命令行参数获取 ST-Net 模型路径
        st_checkpoint = st_model_path

        if st_checkpoint:
            print(f"   📂 模型路径: {st_checkpoint}")

            if os.path.exists(st_checkpoint):
                print(f"   ✅ 模型文件存在!")
                try:
                    # 调用实际的模型加载函数
                    st_encoder, st_model, st_cfg = load_st_model(device, st_checkpoint)
                    if st_encoder is not None and st_model is not None:
                        st_model_bundle = (st_encoder, st_model, st_cfg)
                        st_model_status = "LOADED_SUCCESS"
                        print(f"   ✅ ST-Net 模型加载成功!")
                        print(f"   📊 模型配置:")
                        print(f"      - encoder_type: {st_cfg.encoder_type}")
                        print(f"      - text_emb_dim: {st_cfg.text_emb_dim}")
                        print(f"      - agent_feat_dim: {st_cfg.agent_feat_dim}")
                    else:
                        st_model_status = "LOAD_FAILED"
                        print(f"   ❌ ST-Net 模型加载失败: 返回值为 None")
                except Exception as e:
                    st_model_status = f"LOAD_FAILED: {e}"
                    print(f"   ❌ ST-Net 模型加载失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                st_model_status = "FILE_NOT_FOUND"
                print(f"   ❌ 模型文件不存在: {st_checkpoint}")
        else:
            st_model_status = "NO_PATH_PROVIDED"
            print(f"   ❌ 未提供 ST-Net 模型路径")
            print(f"   📝 请使用 --st_model_path 参数指定模型文件")
            print(f"   📋 或在脚本中设置 STATE_MODEL_CHECKPOINT 变量")

        print(f"   💡 若 ST-Net 模型不可用，将使用默认值 [0.33, 0.33, 0.34]")
    else:
        st_model_status = "NOT_REQUIRED"
        print(f"ℹ️  算法 '{alg}' 不需要状态转移模型")
        print(f"   将使用默认值 [0.33, 0.33, 0.34]")

    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"")

    if "gpt" in model_type:
        a_model = model_type
        tokenizer = None
        mf_model = model_type
    elif "ds" in model_type:
        a_model = "deepseek-chat"
        tokenizer = None
        mf_model = None
    elif "DeepSeek" in model_type:
        a_model = model_type
        tokenizer = None
        mf_model = model_type

    else:
        if "mf" in alg:
            mf_model = AutoModelForCausalLM.from_pretrained(mf_model_name, torch_dtype=torch.float16).to(device)
            if a_MODEL_NAME != mf_MODEL_NAME:
                a_model = AutoModelForCausalLM.from_pretrained(a_model_name, torch_dtype=torch.float16).to(device)
            else:
                a_model = mf_model
        else:
            a_model = AutoModelForCausalLM.from_pretrained(a_model_name, torch_dtype=torch.float16).to(device)
            mf_model = None
        tokenizer = AutoTokenizer.from_pretrained(
            a_model_name, 
            fix_mistral_regex=True  # 显式启用修复模式以消除警告并确保分词一致性
        )
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

    evaluation_results = evaluate(
        test_dir, test_file_name, comment_n, alg, model_type, tokenizer,
        a_model, mf_model, batch_size,
        generate_true=generate_true,
        st_model_bundle=st_model_bundle  # <--- 使用前面加载的模型包
    )

    return evaluation_results

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment_n", type=int, default=None,
                       help="Number of comments to simulate. If not specified, uses all available data.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--simulation_start", type=int, default=100)
    parser.add_argument("--alg", type=str, default='mf')
    parser.add_argument("--model", type=str, default='1.5B') # "DeepSeek-R1-Distill-Qwen-32B"
    parser.add_argument("--task", type=str, default='')
    parser.add_argument("--file_name", type=str, required=True,
                       help="File path to JSON data file (e.g., ./main/data/xxx.json)")
    parser.add_argument("--policy_model_path", type=str, default=None,
                       help="Path to policy model (Actor model). If not specified, uses default based on --model and --alg")
    parser.add_argument("--mf_model_path", type=str, default=None,
                       help="Path to mean field model. If not specified, uses default based on --model and --alg")
    parser.add_argument("--st_model_path", type=str, default=None,
                       help="Path to State Transition model (ST-Net) checkpoint file for predicting state distributions")
    parser.add_argument("--model_path", type=str,
                       default=(config.get('paths.model_base_dir') if config else "./models/"),
                       help="Base directory for models (default: ./models/)")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    comment_n = args.comment_n  # 可以为 None，表示使用全部数据
    generate_true = True
    ALG = args.alg  #  'hot', 'mf', 'gpt' 等
    simulation_start = args.simulation_start
    use_vllm = False
    MODEL_PATH = args.model_path  # 使用命令行参数或默认值
    file_name = args.file_name
    model_type = args.model
    batch_size = args.batch_size

    # file_name 必须是直接的 JSON 文件路径
    import os
    if not file_name.endswith('.json'):
        print(f"错误: --file_name 必须是 .json 文件路径，当前值: {file_name}")
        exit(1)

    # 从文件路径提取 basename 用于输出文件名
    basename = os.path.basename(file_name).replace('.json', '')
    file_class = basename

    # 如果用户通过命令行参数指定了模型路径，优先使用
    if args.policy_model_path is not None:
        a_MODEL_NAME = args.policy_model_path
    else:
        # 否则使用默认的模型名称选择逻辑
        if ALG == 'state' or ALG == 'hot' or ALG == 'pre':
            a_MODEL_NAME = f"Qwen2-{model_type}-Instruct"
        elif ALG == 'mf' or ALG == 'mf_w_key' or ALG == 'mf_w_inter':
            if model_type == "1.5B":
                a_MODEL_NAME = "qwen2-1.5-mfPolicyIntervene"
            elif model_type == "7B":
                a_MODEL_NAME = f"Qwen2-{model_type}-Instruct"
            elif "gpt" in model_type:
                a_MODEL_NAME = model_type
            elif "DeepSeek" in model_type:
                a_MODEL_NAME = model_type
        elif ALG == 'mf_wo_sfta':
            a_MODEL_NAME = "Qwen2-1.5B-Instruct"
        elif ALG == 'mf_wo_sft_mf':
            a_MODEL_NAME = "qwen2-1.5B-policy-0324-prompt"
        elif ALG == 'mf_wo_sft_mf_a':
            a_MODEL_NAME = "Qwen2-1.5B-Instruct"
        elif ALG == 'state_sft':
            a_MODEL_NAME = "qwen2-1.5B-state-0220"
        else:
            print("ERROR ALG NAME!")
            a_MODEL_NAME = None

    if args.mf_model_path is not None:
        mf_MODEL_NAME = args.mf_model_path
    else:
        # 否则使用默认的模型名称选择逻辑
        if ALG == 'state' or ALG == 'hot' or ALG == 'pre':
            mf_MODEL_NAME = f"Qwen2-{model_type}-Instruct"
        elif ALG == 'mf' or ALG == 'mf_w_key' or ALG == 'mf_w_inter':
            if model_type == "1.5B":
                mf_MODEL_NAME = "qwen2-1.5B-mf-sft"
            elif model_type == "7B":
                mf_MODEL_NAME = f"Qwen2-{model_type}-Instruct"
            elif "gpt" in model_type:
                mf_MODEL_NAME = model_type
            elif "DeepSeek" in model_type:
                mf_MODEL_NAME = model_type
        elif ALG == 'mf_wo_sfta':
            mf_MODEL_NAME = "qwen2-1.5B-mf-0319"
        elif ALG == 'mf_wo_sft_mf':
            mf_MODEL_NAME = "Qwen2-1.5B-Instruct"
        elif ALG == 'mf_wo_sft_mf_a':
            mf_MODEL_NAME = "Qwen2-1.5B-Instruct"
        elif ALG == 'state_sft':
            mf_MODEL_NAME = "Qwen2-1.5B-Instruct"
        else:
            print("ERROR ALG NAME!")
            mf_MODEL_NAME = None

    args.a_MODEL_NAME = a_MODEL_NAME
    args.mf_MODEL_NAME = mf_MODEL_NAME

    print(f"Policy model path: {a_MODEL_NAME}")
    print(f"MF model path: {mf_MODEL_NAME}")

    # 直接使用传入的文件路径
    test_files = file_name

    evaluation_results = runner(ALG,model_type, test_files, comment_n, a_MODEL_NAME, mf_MODEL_NAME,
                               generate_true=True, st_model_path=args.st_model_path)

    # 保存结果
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if mf_MODEL_NAME == a_MODEL_NAME:
        path = mf_MODEL_NAME
    else:
        path = mf_MODEL_NAME + "/" + a_MODEL_NAME

    OUTPUT_PATH = f'main/result/{path}/evaluation/'
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # 处理 comment_n 为 None 的情况
    comment_n_str = "all" if comment_n is None else str(comment_n)
    output_file = os.path.join(OUTPUT_PATH, f'{file_class}_{ALG}_start_{simulation_start}_n_{comment_n_str}_s_{seed_num}_evaluation_results_{current_time}.csv')
    evaluation_results.to_csv(output_file, index=False)
    print(f"评估结果已保存至 {output_file}")


    # 将 args 保存为字典并记录到 JSON 文件
    args_dict = vars(args)  # 将 argparse.Namespace 转换为字典
    args_dict['output_file'] = output_file  # 添加输出文件路径

    task = args.task
    if task != "":
        saved_paths_file = f'main/result/{task}/{model_type}/saved_file_paths.json'
    else:
        saved_paths_file = 'main/result/saved_file_paths.json'


    dir_path = os.path.dirname(saved_paths_file)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    # 如果文件已存在，读取现有内容
    if os.path.exists(saved_paths_file):
        with open(saved_paths_file, 'r') as f:
            saved_data = json.load(f)
    else:
        saved_data = []
    # 添加新的记录
    saved_data.append(args_dict)

    # 写回 JSON 文件
    with open(saved_paths_file, 'w') as f:
        json.dump(saved_data, f, indent=4)

    print(f"文件路径和参数已记录到 {saved_paths_file}")
