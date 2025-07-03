#!/bin/bash

# 配置参数（修改这里）
GPUS=(4 7)          # 要监控的GPU编号（可修改）
MEMORY_THRESHOLD=1000 # 显存使用阈值MB（建议50-100）
UTIL_THRESHOLD=5    # GPU利用率阈值%（建议1-5）
CHECK_INTERVAL=60   # 检查间隔秒数（建议60）
TARGET_SCRIPT="./scripts/stage2.sh" # 要执行的脚本路径

# 函数：检查GPU是否空闲
check_gpu_free() {
    local gpu_id=$1
    # 获取GPU状态（需要nvidia-smi）
    local gpu_info=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i $gpu_id)
    
    # 提取显存使用和GPU利用率
    local mem_used=$(echo $gpu_info | awk -F', ' '{print $1}')
    local gpu_util=$(echo $gpu_info | awk -F', ' '{print $2}')
    
    # 检查是否低于阈值
    if [ "$mem_used" -lt "$MEMORY_THRESHOLD" ] && [ "$gpu_util" -lt "$UTIL_THRESHOLD" ]; then
        return 0 # 空闲
    else
        return 1 # 忙碌
    fi
}

# 主监控循环
while true; do
    all_free=true
    
    # 检查所有指定GPU
    for gpu in "${GPUS[@]}"; do
        if ! check_gpu_free $gpu; then
            all_free=false
            break
        fi
    done
    
    # 全部空闲时执行脚本
    if [ "$all_free" = true ]; then
        echo "[$(date)] GPU ${GPUS[@]} 空闲，执行脚本: $TARGET_SCRIPT"
        bash "$TARGET_SCRIPT"
        exit 0 # 执行后退出（如需持续监控请删除此行）
    else
        echo "[$(date)] GPU ${GPUS[@]} 忙碌，等待..."
    fi
    
    sleep $CHECK_INTERVAL
done
