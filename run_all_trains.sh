#!/bin/bash

# 批量运行训练脚本（除了rent-train.sh）
# 使用nohup在后台运行，输出重定向到日志文件

# 加载环境变量
source ./set_path.sh

# 创建日志目录
DATE=$(date +%Y%m%d)
TIME_TAG=$(date +%H%M%S)
LOG_DIR="$USERPath/checkpoints/logs/${DATE}-${TIME_TAG}"
mkdir -p "$LOG_DIR"

echo "开始运行所有训练脚本..."
echo "当前时间: $(date)"
echo "日志目录: $LOG_DIR"

# 运行EMPO训练
echo "启动EMPO训练..."
bash empo-train.sh > "$LOG_DIR/empo_train.log" 2>&1
echo "EMPO训练已完成，日志文件: $LOG_DIR/empo_train.log"

# 运行Intuitor训练
echo "启动Intuitor训练..."
bash intuitor-train.sh > "$LOG_DIR/intuitor_train.log" 2>&1
echo "Intuitor训练已完成，日志文件: $LOG_DIR/intuitor_train.log"

# 运行TTRL训练
echo "启动TTRL训练..."
bash ttrl-train.sh > "$LOG_DIR/ttrl_train.log" 2>&1
echo "TTRL训练已完成，日志文件: $LOG_DIR/ttrl_train.log"

echo "所有训练脚本已串行执行完成！"

echo "查看日志文件:"
echo "  cat $LOG_DIR/empo_train.log"
echo "  cat $LOG_DIR/intuitor_train.log"
echo "  cat $LOG_DIR/ttrl_train.log"

# 同时在当前目录保存一个指向最新日志目录的链接文件
echo "$LOG_DIR" > latest_log_dir.txt

echo "最新日志目录路径已保存到 latest_log_dir.txt"
echo "脚本执行完成！"