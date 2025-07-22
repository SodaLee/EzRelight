#!/bin/bash

# 配置参数（修改这里）
GPUS=(6 7)  # 监控所有8个GPU
MEMORY_THRESHOLD=1000   # 显存使用阈值MB（建议50-100）
UTIL_THRESHOLD=5        # GPU利用率阈值%（建议1-5）
CHECK_INTERVAL=60       # 检查间隔秒数（建议60）
TARGET_SCRIPT="./scripts/video.sh" # 要执行的脚本路径
REQUIRED_FREE_GPUS=2    # 需要的空闲GPU数量

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

# 函数：修改脚本中的CUDA_VISIBLE_DEVICES
modify_cuda_devices() {
    local free_gpus=$1
    local temp_script=$(mktemp)
    local cuda_devices=$(echo "$free_gpus" | tr ' ' ',')
    if [ -f "$TARGET_SCRIPT" ]; then
        cp "$TARGET_SCRIPT" "$TARGET_SCRIPT.bak.$(date +%Y%m%d%H%M%S)"
        # 用 awk 替换首个 CUDA_VISIBLE_DEVICES 行，否则插入
        awk -v cuda_devices="$cuda_devices" '
        BEGIN {replaced=0}
        /^[[:space:]]*CUDA_VISIBLE_DEVICES="[0-9,]+"/ && replaced==0 {
            print "CUDA_VISIBLE_DEVICES=\"" cuda_devices "\" \\";
            replaced=1
            next
        }
        {print}
        END {
            if(replaced==0) {
                print "(WARN: 未检测到原有 CUDA_VISIBLE_DEVICES 行，已插入新行)" > "/dev/stderr"
            }
        }
        ' "$TARGET_SCRIPT" > "$temp_script"
        if [ -s "$temp_script" ]; then
            mv "$temp_script" "$TARGET_SCRIPT"
            chmod +x "$TARGET_SCRIPT"
            echo "[$(date)] 已修改 $TARGET_SCRIPT 中的 CUDA_VISIBLE_DEVICES=\"$cuda_devices\" \\" 
        else
            echo "[$(date)] 错误：生成的临时脚本为空，未覆盖原文件。内容如下："
            cat "$TARGET_SCRIPT"
            rm -f "$temp_script"
            return 1
        fi
    else
        echo "[$(date)] 错误：目标脚本 $TARGET_SCRIPT 不存在"
        rm -f "$temp_script"
        return 1
    fi
}

# 主监控循环
while true; do
    free_gpus=()
    
    # 检查所有指定GPU
    for gpu in "${GPUS[@]}"; do
        if check_gpu_free $gpu; then
            free_gpus+=($gpu)
        fi
    done
    
    # 检查是否有足够的空闲GPU
    if [ ${#free_gpus[@]} -ge $REQUIRED_FREE_GPUS ]; then
        echo "[$(date)] 找到 ${#free_gpus[@]} 个空闲GPU: ${free_gpus[@]}"
        
        # 选择前两个空闲的GPU
        selected_gpus="${free_gpus[0]} ${free_gpus[1]}"
        echo "[$(date)] 选择GPU: $selected_gpus"
        
        # 修改目标脚本中的CUDA_VISIBLE_DEVICES
        if modify_cuda_devices "$selected_gpus"; then
            echo "[$(date)] 执行脚本: $TARGET_SCRIPT"
        bash "$TARGET_SCRIPT"
        exit 0 # 执行后退出（如需持续监控请删除此行）
        else
            echo "[$(date)] 修改脚本失败，继续监控..."
        fi
    else
        echo "[$(date)] 当前空闲GPU数量: ${#free_gpus[@]} (需要 $REQUIRED_FREE_GPUS 个)，等待..."
        if [ ${#free_gpus[@]} -gt 0 ]; then
            echo "[$(date)] 空闲GPU: ${free_gpus[@]}"
        fi
    fi
    
    sleep $CHECK_INTERVAL
done
