#!/bin/bash

# =============================================================================
# ICML 2026 - 仿真运行脚本
#
# 此脚本用于运行 mean field 仿真实验
# 支持生成状态分布缓存文件（使用 Transformer 模型）
#
# 使用方法: cd /root/ICML/release && bash main/script/run_simulation.sh
# =============================================================================

# -----------------------------------------------------------------------------
# 获取脚本所在目录的绝对路径
# -----------------------------------------------------------------------------
# 获取脚本实际所在路径（解析符号链接）
SCRIPT_SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SCRIPT_SOURCE" ]; do
    SCRIPT_DIR="$(cd -P "$(dirname "$SCRIPT_SOURCE")" && pwd)"
    SCRIPT_SOURCE="$(readlink "$SCRIPT_SOURCE")"
    [[ $SCRIPT_SOURCE != /* ]] && SCRIPT_SOURCE="$SCRIPT_DIR/$SCRIPT_SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SCRIPT_SOURCE")" && pwd)"

# release 目录是 script 目录的上两级
# SCRIPT_DIR = .../release/main/script
# 我们需要到 .../release
RELEASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 切换到 release 目录，确保所有相对路径都正确
cd "$RELEASE_DIR" || exit 1

echo "脚本目录: $SCRIPT_DIR"
echo "工作目录: $(pwd)"
echo ""

# -----------------------------------------------------------------------------
# 配置部分
# -----------------------------------------------------------------------------

# 1. 生成状态分布缓存的配置
# 是否在运行仿真前先生成状态分布缓存文件
GENERATE_CACHE=false

# State Transformer 模型路径 (用于生成状态分布缓存)
# 注意: event_transformer_best.pt 是正确的模型 (训练时使用预编码)
STATE_MODEL_CHECKPOINT="./checkpoints/event_transformer/event_transformer_best.pt"

# 缓存相关路径 (相对于 release 目录)
MF_DATA_DIR="./data/test_mf"
TRAJ_DATA_DIR="./data/test_state_distribution"
STATE_CACHE_OUTPUT_DIR="./data/pred_state_distribution"
TEXT_EMBED_CACHE_DIR="./cache/text_embeddings"

# 2. 仿真配置
# 默认使用 run_mf_mdp.py 脚本
PYTHON_SCRIPT="main/script/run_mf_mdp.py"

# 模型路径配置
POLICY_MODEL_PATH="/root/qwen2-1.5B-policy"  # 策略模型路径
MF_MODEL_PATH="/root/qwen2-1.5B-mf"      # 均值场模型路径
MODEL_BASE_DIR="/root/models/"  # 模型基础目录（备用）

# 数据目录 (自动搜索 batch > 1000 的文件)
DATA_DIR="./data"

# 仿真参数
alg="mf"  # 算法类型: mf, hot, state 等 (使用 mf 或 state 才会加载 ST-Net 模型)
simulation_start=50
model_type="1.5B"  # 对应微调的 qwen2-1.5B-mf-sft
batch_size=16
task="full_simulation_run"

# -----------------------------------------------------------------------------
# 函数定义
# -----------------------------------------------------------------------------

# 查找所有 batch > 1000 的数据文件
function find_large_batch_files() {
    local data_dir="$1"
    local min_batch=10000
    local found_files=()

    # 将所有调试信息输出到 stderr，避免被 mapfile 捕获
    echo "" >&2
    echo "============================================================================" >&2
    echo "搜索数据文件 (batch > $min_batch)" >&2
    echo "============================================================================" >&2
    echo "目录: $data_dir" >&2
    echo "============================================================================" >&2
    echo "" >&2

    if [ ! -d "$data_dir" ]; then
        echo "错误: 数据目录不存在: $data_dir" >&2
        return 1
    fi

    # 遍历所有 JSON 文件
    for json_file in "$data_dir"/*.json; do
        if [ -f "$json_file" ]; then
            # 使用 Python 快速计算 JSON 数组长度
            array_length=$(python3 -c "
import json
import sys
try:
    with open('$json_file', 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 数组长度 - 1 = 评论数量 (第一条是原始微博)
        print(len(data) - 1)
except Exception as e:
    print(0)
" 2>/dev/null)

            # 检查是否大于阈值
            if [ "$array_length" -gt "$min_batch" ]; then
                basename=$(basename "$json_file" .json)
                found_files+=("$basename")
                echo "✓ 找到: $basename (batch=$array_length)" >&2
            fi
        fi
    done

    echo "" >&2
    echo "============================================================================" >&2
    echo "搜索完成: 找到 ${#found_files[@]} 个符合条件的文件" >&2
    echo "============================================================================" >&2
    echo "" >&2

    # 返回文件列表 (只输出到 stdout)
    printf '%s\n' "${found_files[@]}"
    return 0
}

# 生成状态分布缓存文件
function generate_state_cache() {
    echo ""
    echo "============================================================================"
    echo "生成状态分布缓存文件"
    echo "============================================================================"
    echo "模型: $STATE_MODEL_CHECKPOINT"
    echo "MF数据目录: $MF_DATA_DIR"
    echo "轨迹数据目录: $TRAJ_DATA_DIR"
    echo "输出目录: $STATE_CACHE_OUTPUT_DIR"
    echo "============================================================================"
    echo ""

    # 检查模型文件是否存在
    if [ ! -f "$STATE_MODEL_CHECKPOINT" ]; then
        echo "错误: 模型文件不存在: $STATE_MODEL_CHECKPOINT"
        echo "跳过缓存生成，使用现有数据文件..."
        return 1
    fi

    # 生成状态分布缓存
    poetry run python main/script/generate_state_trajectory.py \
        --checkpoint "$STATE_MODEL_CHECKPOINT" \
        --mf_dir "$MF_DATA_DIR" \
        --traj_dir "$TRAJ_DATA_DIR" \
        --output_dir "$STATE_CACHE_OUTPUT_DIR" \
        --cache_dir "$TEXT_EMBED_CACHE_DIR" \
        --batch_size $batch_size \
        --warmup_steps 5 \
        --device "cuda"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ 状态分布缓存生成成功!"
        echo "缓存文件保存在: $STATE_CACHE_OUTPUT_DIR"
        return 0
    else
        echo ""
        echo "✗ 状态分布缓存生成失败"
        return 1
    fi
}

# 运行单个仿真
function run_simulation() {
    local file_path=$1

    echo "----------------------------------------------------------------"
    echo "Processing: $file_path"
    echo "Time: $(date)"
    echo "当前工作目录: $(pwd)"
    echo "算法类型: $alg"
    echo "Policy Model Path: $POLICY_MODEL_PATH"
    echo "State Model Path: $STATE_MODEL_CHECKPOINT"
    echo "----------------------------------------------------------------"

    # 构建命令行参数 (去掉 comment_n，使用全部数据)
    local cmd_args=(
        --file_name "$file_path"
        --alg "$alg"
        --simulation_start "$simulation_start"
        --model "$model_type"
        --task "$task"
        --batch_size $batch_size
        --model_path "$MODEL_BASE_DIR"
        --st_model_path "$STATE_MODEL_CHECKPOINT"
    )
    echo "添加参数: --st_model_path $STATE_MODEL_CHECKPOINT"

    # 如果指定了模型路径，添加到参数中
    if [ -n "$POLICY_MODEL_PATH" ]; then
        cmd_args+=(--policy_model_path "$POLICY_MODEL_PATH")
        echo "添加参数: --policy_model_path $POLICY_MODEL_PATH"
    fi

    if [ -n "$MF_MODEL_PATH" ]; then
        cmd_args+=(--mf_model_path "$MF_MODEL_PATH")
        echo "添加参数: --mf_model_path $MF_MODEL_PATH"
    fi

    echo "执行命令: poetry run python $PYTHON_SCRIPT ${cmd_args[@]}"
    echo "----------------------------------------------------------------"

    poetry run python "$PYTHON_SCRIPT" "${cmd_args[@]}"

    if [ $? -ne 0 ]; then
        echo "Error occurred on $file_path, skipping to next..."
        return 1
    fi
    return 0
}

# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------

echo ""
echo "============================================================================"
echo "ICML 2026 - Mean Field 仿真"
echo "============================================================================"
echo "开始时间: $(date)"
echo ""

# Step 1: 生成状态分布缓存 (如果启用)
if [ "$GENERATE_CACHE" = true ]; then
    generate_state_cache
    cache_result=$?

    if [ $cache_result -ne 0 ]; then
        echo "警告: 缓存生成失败或跳过，将使用现有数据文件"
    fi
else
    echo "缓存生成已禁用 (GENERATE_CACHE=false)"
    echo "将使用现有的状态分布数据文件"
fi

echo ""
echo "============================================================================"
echo "开始仿真"
echo "============================================================================"
echo "数据目录: $DATA_DIR"
echo "算法类型: $alg"
echo "仿真起点: $simulation_start"
echo "模型: $model_type"
echo "============================================================================"
echo ""

# Step 2: 自动搜索并运行仿真
total=0
success=0

# 自动搜索 batch > 1000 的文件
mapfile -t found_files < <(find_large_batch_files "$DATA_DIR")

if [ ${#found_files[@]} -eq 0 ]; then
    echo "警告: 没有找到符合条件的文件 (batch > 1000)"
    echo "退出程序..."
    exit 1
fi

# 对每个文件运行仿真
for file_basename in "${found_files[@]}"; do
    # 构建完整文件路径 (相对于 release 目录)
    file_path="$DATA_DIR/${file_basename}.json"

    total=$((total + 1))
    run_simulation "$file_path"
    if [ $? -eq 0 ]; then
        success=$((success + 1))
    fi
done

echo ""
echo "============================================================================"
echo "仿真完成!"
echo "============================================================================"
echo "完成时间: $(date)"
echo "总计: $total 个任务"
echo "成功: $success 个任务"
echo "失败: $((total - success)) 个任务"
echo "============================================================================"
