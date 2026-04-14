import torch
import openai
import torch.nn.functional as F
from collections import Counter
import time
# from transformers import AutoTokenizer, AutoModelForCausalLM
# MODEL_PATH = "/mnt/nasdata/qirui/language_model/"
# MODEL_NAME = "Qwen2-1.5B-Instruct"
# model_name = MODEL_PATH + MODEL_NAME
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# initial_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
# from collections import Counter
import os
import re
import re
import sys

# 【新增辅助函数】为了代码整洁，将 API 调用逻辑抽离
async def call_api_mean_field(client, model_type, prompt):
    """处理 GPT 和 DeepSeek 的总结调用"""
    try:
        # 针对 DeepSeek 的特殊处理
        is_deepseek = "DeepSeek" in model_type
        messages = [{"role": "user", "content": prompt}]
        if is_deepseek:
            messages = [
                {"role": "user", "content": "请直接输出最终答案，不要展示思考过程。\n" + prompt},
                {"role": "assistant", "content": "<think>\n</think>\n\n"}
            ]

        response = client.chat.completions.create(
            model="deepseek-chat" if "ds" in model_type else model_type,
            messages=messages,
            temperature=0.1,
            max_tokens=400,
        )
        content = response.choices[0].message.content
        if is_deepseek:
            content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
        return content.strip().replace("\n", "").replace("**", "")
    except Exception as e:
        print(f"Mean Field API 调用失败: {e}")
        return "（API 总结失败，请检查网络）"


def calculate_mean_field(
        topic,
        states,
        actions,
        pre_mean_field,
        model,
        device,
        alg,
        model_type,
        client=None,
        sampling_params=None,
        use_real_action=False,
        tokenizer=None,
        use_vllm=True,
        return_real_comment=False,
        state_distribution=None, # 新增：接收从 ST-Net 预测的群体分布 [pos, neu, neg]
):
    """
    使用 LLM 或 GPT 计算指定索引的 mean field。
    """

    state_action_prompt = ""
    # 构造评论列表文本
    for i, (s_prev, a_prev) in enumerate(zip(states, actions)):
        state_action_prompt += f"第{i + 1}个网友评论：{a_prev}\n"

    # --- 新增：群体分布数据的文本化 ---
    dist_info_str = ""
    if state_distribution is not None:
        # 假设顺序为 Positive, Neutral, Negative
        pos, neu, neg = [f"{x*100:.1f}%" for x in state_distribution]
        dist_info_str = f"【群体情绪统计分布数据】：积极占比 {pos}，中立占比 {neu}，消极占比 {neg}。\n"
        dist_info_str += "请注意：上述统计数据是由社会动力学模型计算得出的群体真实现状，你的总结必须与此数据分布保持一致。\n" # 新增强调词

    end_prompt = \
        ("请你结合以上信息，总结网友的评论分布情况，方便后续网友快速了解已有讨论内容，对以下 6 个方面，根据重要性的顺序进行回答："
         "1. 立场分布：大部分网友们是持有反对意见，还是支持？"
         "2. 观点分布：网友们有哪些观点？请总结主要的观点内容"
         "3. 情绪分布：大部分网友是生气、兴奋、质疑、焦虑?是积极倾向、消极，还是中立？"
         "4. 行为分布：他们更倾向于转发，还是评论？"
         "5. 话题真实性：网友们如何评价该话题的真实性？网友们是相信，还是质疑？"
         "6. 发言意图：网友的评论主要是提问、发表个人观点，还是转播信息？"
         "任务：请使用 200 字左右的中文总结回答，并确保信息结构清晰、简洁明了，严格按照这6个方面进行回答。")

    # 整合 Prompt，注入分布数据
    prompt = (
                 f"当前话题：{topic}\n"
                 f"过往网友评论总结：{pre_mean_field}。\n"
                 f"{dist_info_str}"  # 注入 ST-Net 的数值预测结果
                 f"最新网友讨论情况：\n{state_action_prompt}\n") + end_prompt

    # print("prompt:",prompt)
    # 【修改】逻辑分流：优先判断 API 客户端
    if client is not None:
        # 如果是同步环境（如你的主循环），直接调用（如果是异步主循环则需加 await）
        import asyncio
        # 兼容性处理：如果主循环是同步的，这里用简单的同步模拟或直接在主循环处理
        # 假设你的主循环调用 calculate_mean_field 是同步的：
        generated_mf = asyncio.run(call_api_mean_field(client, model_type, prompt)) if "DeepSeek" in model_type else "GPT-SUMMARY"
        # 注意：建议在主循环中根据 model_type 决定是否 await
        return generated_mf, torch.tensor(0.)
    
    # 【修改】本地模型分支：增加 tokenizer 存在性检查
    else:
        if tokenizer is None:
            return "（错误：缺少本地分词器）", torch.tensor(0., device=device)
        generated_texts = model_generate(prompt, model, tokenizer, device)
        # print("gen:",generated_texts)
        # sys.exit(0)
        return generated_texts, torch.tensor([0.], device=device)


# def model_generate(prompt, model, tokenizer, device):
#     """【修改】增加健壮性检查"""
#     if tokenizer is None: return "Error: No Tokenizer"
#     prompt = prompt + "\n\n总结回答："
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
#     input_ids = inputs["input_ids"]
    
#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids,
#             max_new_tokens=250,
#             do_sample=True,
#             temperature=0.1,
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#             repetition_penalty=1.1
#         )

#     # 只解码生成的部分
#     gen_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip().split("Human:")[0].split(
#         "Assistant:")[0].replace("\n", "")
#     return gen_text.strip().replace("\n", "")

def model_generate(prompt, model, tokenizer, device):
    if tokenizer is None: return "Error: No Tokenizer"
    
    # --- [核心修改 1]：构建 Chat 格式的消息列表 ---
    # 将原本的大段 prompt 包装成 user 的发言
    messages = [
        {"role": "system", "content": "你是一个专业的舆论分析助手。请根据提供的信息总结评论分布。"},
        {"role": "user", "content": prompt}
    ]
    
    # --- [核心修改 2]：应用模版 (apply_chat_template) ---
    # 这会自动添加 <|im_start|> 等特殊 token，防止模型生成 "Human:"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True # 自动添加 assistant 引导头
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=300,  # 稍微调大一点，避免截断
            do_sample=True,
            temperature=0.1,     # [核心修改 3]：从 0.1 提高到 0.7，防止复读机
            top_p=0.9,           # 配合 top_p 采样
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1 # 稍微降低惩罚，避免过度惩罚导致语句不通顺
        )

    # 只解码生成的部分
    gen_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return gen_text.strip()

def compute_loss(model, tokenizer, prompt, end_prompt, generated_texts, device, initial_model=None, beta=0.5):
    """
    计算 MF 总结的优化目标。
    """
    if initial_model is None:
        return torch.tensor(0., device=device)
        
    q_m_consition_logp = compute_log_prob_no_condition(model, tokenizer, [prompt], [generated_texts], device)
    q_m_logp = compute_log_prob_no_condition(model, tokenizer, [end_prompt], [generated_texts], device)
    
    with torch.no_grad():
        intial_m_logp = compute_log_prob_no_condition(initial_model.to(device), tokenizer, [end_prompt], [generated_texts], device)
        tokens = tokenizer.tokenize(generated_texts)
        token_counts = Counter(tokens)
        freq_penalty = sum((count - 1) for count in token_counts.values() if count > 1)
        
    total_mf_objective = q_m_consition_logp - q_m_logp - beta * (intial_m_logp - q_m_logp) - freq_penalty * 0.002
    return total_mf_objective

def compute_log_prob_no_condition(model, tokenizer, prompts, texts, device):
    # 此处保留原逻辑
    inputs = [prompt + text for prompt, text in zip(prompts, texts)]
    tokenized_inputs = tokenizer(inputs, return_tensors="pt", truncation=True, padding=True).to(device)
    tokenized_prompts = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True).to(device)

    input_ids = tokenized_inputs["input_ids"]
    prompt_ids = tokenized_prompts["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]
    prompt_lengths = [len(ids) for ids in prompt_ids]

    outputs = model(input_ids, attention_mask=attention_mask, return_output=True)
    logits = outputs.logits

    sequence_avg_probs = []
    for j in range(input_ids.shape[0]):
        log_prob_sum = torch.tensor(0.0, device=input_ids.device, dtype=torch.float32, requires_grad=True)
        valid_token_count = 0
        for step in range(prompt_lengths[j], input_ids.shape[1]):
            token_id = input_ids[j, step]
            log_prob_sum = log_prob_sum + F.log_softmax(logits[j, step], dim=-1)[token_id]
            valid_token_count += 1
        sequence_avg_prob = log_prob_sum / valid_token_count if valid_token_count > 0 else torch.tensor(-100.0, device=input_ids.device, dtype=torch.float32, requires_grad=True)
        sequence_avg_probs.append(sequence_avg_prob)
    return torch.stack(sequence_avg_probs)

def build_state(item, user_profile_dict=None, user_profile_list=None):
    """
    用户画像描述：优先使用聚类用户画像，回退到基础字段。
    """
    uid = str(item.get('uid', ''))

    # === 优先级 1: 使用聚类用户画像（通过 UID 精确查找）===
    if user_profile_dict and uid in user_profile_dict:
        profile = user_profile_dict[uid]
        persona_desc = profile.get('persona_description', '')
        stance_label = profile.get('stance_label', '')
        stance_nuance = profile.get('stance_nuance', '')
        expression_style = profile.get('expression_style', [])
        core_values = profile.get('core_values', '')
        activity_level = profile.get('activity_level', '')

        # 构建基于聚类的详细画像
        style_str = "、".join(expression_style) if isinstance(expression_style, list) else expression_style

        state = (
            f"该用户的立场倾向为「{stance_label}」，{stance_nuance} "
            f"表达风格偏好：{style_str}。核心价值观：{core_values}。 "
            f"活跃度：{activity_level}。{persona_desc}"
        )
        return state

    # === 优先级 1.5: 随机兜底 - 从聚类画像列表中随机选择一个 ===
    # 当找不到精确匹配时，使用聚类画像作为模板
    if user_profile_list and len(user_profile_list) > 0:
        import random
        # 使用 UID 作为随机种子，保证同一用户每次获得相同的画像
        random.seed(hash(uid) % (2**32))
        profile = random.choice(user_profile_list)

        persona_desc = profile.get('persona_description', '')
        stance_label = profile.get('stance_label', '')
        stance_nuance = profile.get('stance_nuance', '')
        expression_style = profile.get('expression_style', [])
        core_values = profile.get('core_values', '')
        activity_level = profile.get('activity_level', '')

        style_str = "、".join(expression_style) if isinstance(expression_style, list) else expression_style

        state = (
            f"该用户的立场倾向为「{stance_label}」，{stance_nuance} "
            f"表达风格偏好：{style_str}。核心价值观：{core_values}。 "
            f"活跃度：{activity_level}。{persona_desc}"
        )
        return state

    # === 优先级 2: 如果存在 profile_text（新数据格式）===
    if item.get('profile_text') and len(str(item.get('profile_text'))) > 1:
        return str(item['profile_text'])

    # === 优先级 3: 回退到基础字段（旧数据格式）===
    user_location = item.get('user_location', '未知地区')
    user_description = item.get('user_description', '未知')
    gender = "男性" if item.get('gender') == 'm' else "女性"

    # 统计属性分类
    friends_count = item.get('friends_count', 0)
    followers_count = item.get('followers_count', 0)
    reposts_count = item.get('reposts_count', 0)
    comments_count = item.get('comments_count', 0)
    attitudes_count = item.get('attitudes_count', 0)

    # 朋友数量等级
    if friends_count < 10: friends_level = "非常少"
    elif 10 <= friends_count <= 30: friends_level = "较少"
    elif 31 <= friends_count <= 1000: friends_level = "适中"
    elif 1001 <= friends_count <= 3000: friends_level = "较多"
    else: friends_level = "非常多"

    # 影响力等级
    if followers_count < 100: influence_level = "非常小"
    elif 101 <= followers_count <= 500: influence_level = "较小"
    elif 501 <= followers_count <= 1000: influence_level = "适中"
    elif 1001 <= followers_count <= 10000: influence_level = "较大"
    else: influence_level = "非常大"

    # 活跃度等级
    total_interactions = reposts_count + comments_count + attitudes_count
    if total_interactions < 10: activity_level = "不活跃"
    elif 10 <= total_interactions <= 100: activity_level = "较活跃"
    else: activity_level = "非常活跃"

    verified = item.get('verified', False)
    verified_type = item.get('verified_type', -1)
    verified_status = f"认证用户（类型 {verified_type}）" if verified else "非认证用户"

    # 基础画像拼接
    state = (
        f"一位来自{user_location}的{gender}网友。个人描述为：{user_description}。 "
        f"其社交特征如下：朋友数量{friends_level}，影响力{influence_level}，"
        f"动态互动情况为{activity_level}，属于{verified_status}。"
    )
    return state

def build_prompt(state, topic, hot_comment, mean_field, related_cases_info, alg="none", model_type=None, state_dist=None):
# def build_prompt(state, topic, hot_comment, mean_field, related_cases_info, alg="none", model_type=None):
    """
    精简优化版 Prompt。
    保留：1. 积极/中立/消极引导 2. 转发/评论输出格式约束。
    """
    prompt = f"当前讨论的话题是：{topic}\n"
    
    if hot_comment and hot_comment != "暂无最新热门评论":
        prompt += f"热门背景参考：{hot_comment}\n"
    if mean_field and mean_field != '' and mean_field != ['']:
        mf_text = mean_field[-1] if isinstance(mean_field, list) else mean_field
        prompt += f"舆论环境总结：{mf_text}\n"

    if mean_field and mean_field != '' and mean_field != ['']:
        # --- [修改2] 插入群体情绪统计分布数据 ---
        if state_dist is not None:
            # 假设 state_dist 是 [pos_prob, neu_prob, neg_prob]，转换为百分比
            pos_pct = f"{state_dist[0]*100:.1f}%"
            neu_pct = f"{state_dist[1]*100:.1f}%"
            neg_pct = f"{state_dist[2]*100:.1f}%"
            
            dist_text = (
                f"【群体情绪统计分布数据】：积极占比 {pos_pct}，中立占比 {neu_pct}，消极占比 {neg_pct}。"
                f"请注意：上述统计数据是由社会动力学模型计算得出的群体真实现状，你的总结必须与此数据分布保持一致。\n"
            )
            prompt += dist_text
        # -------------------------------------

        mf_text = mean_field[-1] if isinstance(mean_field, list) else mean_field
        prompt += f"舆论环境总结：{mf_text}\n"

    if related_cases_info:
        prompt += f"相关案例：{related_cases_info}\n"
        
    prompt += f"网友资料：{state}\n"
    
    # # --- 1. 心理倾向引导 ---
    # state_instruction = ""
    # if pred_state is not None:
    #     # 映射预测状态
    #     state_cn = {"Positive": "积极", "Neutral": "中立", "Negative": "消极"}.get(pred_state, pred_state)
    #     state_instruction = f"【心理状态引导】：根据预测，你对此话题的心理倾向处于“{state_cn}”状态。\n"

    # --- 2. 转发/评论格式约束 ---         f"{state_instruction}"
    format_instruction = (
        "请推测该网友可能的情绪、观点和立场，模拟该网友进行社交媒体互动：\n"
        "1. 如果你决定【转发】，请直接输出：转发微博。注意：如果你是中立情绪，请倾向于执行转发操作。\n"
        "2. 如果你决定【评论】，请直接输出具体的评论内容文本，并保证你的内容和情感态度立场一致\n"
        "注意：必须输出模拟内容，禁止输出任何多余的解释或分析。"
    )

    if "gpt" in model_type or "DeepSeek" in model_type:
        prompt += format_instruction
    else:
        # 针对本地模型增加引导词，防止输出为空
        prompt += format_instruction + "\n\n模拟内容输出："

    return prompt