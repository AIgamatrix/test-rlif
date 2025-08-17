#!/bin/bash

# 训练进程管理脚本

# 加载环境变量
source ./set_path.sh

# 获取最新日志目录
function get_latest_log_dir() {
    if [ -f "latest_log_dir.txt" ]; then
        cat latest_log_dir.txt
    else
        echo ""
    fi
}

function show_help() {
    echo "训练管理脚本"
    echo "用法: bash manage_trains.sh [选项]"
    echo ""
    echo "选项:"
    echo "  status    - 查看训练完成状态"
    echo "  progress  - 显示当前训练进度"
    echo "  logs      - 查看所有日志文件"
    echo "  clean     - 清理日志文件"
    echo "  list      - 列出所有日志目录"
    echo "  help      - 显示此帮助信息"
}

function check_status() {
    echo "检查训练完成状态..."
    echo "=========================================="
    
    LOG_DIR=$(get_latest_log_dir)
    if [ -z "$LOG_DIR" ]; then
        echo "未找到最新的日志目录，请先运行训练脚本"
        return
    fi
    
    echo "当前日志目录: $LOG_DIR"
    echo ""
    
    # 检查EMPO
    if [ -f "$LOG_DIR/empo_train.log" ]; then
        if grep -q "EMPO训练已完成" "$LOG_DIR/empo_train.log" 2>/dev/null; then
            echo "✓ EMPO训练已完成"
        else
            echo "○ EMPO训练日志存在但完成状态不明"
        fi
    else
        echo "✗ EMPO训练日志文件不存在"
    fi
    
    # 检查Intuitor
    if [ -f "$LOG_DIR/intuitor_train.log" ]; then
        if grep -q "Intuitor训练已完成" "$LOG_DIR/intuitor_train.log" 2>/dev/null; then
            echo "✓ Intuitor训练已完成"
        else
            echo "○ Intuitor训练日志存在但完成状态不明"
        fi
    else
        echo "✗ Intuitor训练日志文件不存在"
    fi
    
    # 检查TTRL
    if [ -f "$LOG_DIR/ttrl_train.log" ]; then
        if grep -q "TTRL训练已完成" "$LOG_DIR/ttrl_train.log" 2>/dev/null; then
            echo "✓ TTRL训练已完成"
        else
            echo "○ TTRL训练日志存在但完成状态不明"
        fi
    else
        echo "✗ TTRL训练日志文件不存在"
    fi
}

function show_progress() {
    echo "显示训练进度..."
    echo "=========================================="
    
    LOG_DIR=$(get_latest_log_dir)
    if [ -z "$LOG_DIR" ]; then
        echo "未找到最新的日志目录，请先运行训练脚本"
        return
    fi
    
    echo "当前日志目录: $LOG_DIR"
    echo ""
    
    # 显示EMPO进度
    if [ -f "$LOG_DIR/empo_train.log" ]; then
        echo "=== EMPO训练进度 ==="
        tail -5 "$LOG_DIR/empo_train.log"
        echo ""
    fi
    
    # 显示Intuitor进度
    if [ -f "$LOG_DIR/intuitor_train.log" ]; then
        echo "=== Intuitor训练进度 ==="
        tail -5 "$LOG_DIR/intuitor_train.log"
        echo ""
    fi
    
    # 显示TTRL进度
    if [ -f "$LOG_DIR/ttrl_train.log" ]; then
        echo "=== TTRL训练进度 ==="
        tail -5 "$LOG_DIR/ttrl_train.log"
        echo ""
    fi
}

function show_logs() {
    echo "查看训练日志..."
    echo "=========================================="
    
    LOG_DIR=$(get_latest_log_dir)
    if [ -z "$LOG_DIR" ]; then
        echo "未找到最新的日志目录，请先运行训练脚本"
        return
    fi
    
    echo "当前日志目录: $LOG_DIR"
    echo ""
    
    echo "EMPO训练日志 (最后10行):"
    if [ -f "$LOG_DIR/empo_train.log" ]; then
        tail -10 "$LOG_DIR/empo_train.log"
    else
        echo "日志文件不存在"
    fi
    
    echo ""
    echo "Intuitor训练日志 (最后10行):"
    if [ -f "$LOG_DIR/intuitor_train.log" ]; then
        tail -10 "$LOG_DIR/intuitor_train.log"
    else
        echo "日志文件不存在"
    fi
    
    echo ""
    echo "TTRL训练日志 (最后10行):"
    if [ -f "$LOG_DIR/ttrl_train.log" ]; then
        tail -10 "$LOG_DIR/ttrl_train.log"
    else
        echo "日志文件不存在"
    fi
}

function list_logs() {
    echo "列出所有日志目录..."
    echo "=========================================="
    
    LOGS_BASE_DIR="$USERPath/checkpoints/logs"
    if [ -d "$LOGS_BASE_DIR" ]; then
        echo "日志目录: $LOGS_BASE_DIR"
        echo ""
        ls -la "$LOGS_BASE_DIR" | grep "^d" | awk '{print $9}' | grep -v "^\.$" | grep -v "^\.\.$" | sort -r
        echo ""
        echo "当前使用的日志目录:"
        get_latest_log_dir
    else
        echo "日志目录不存在: $LOGS_BASE_DIR"
    fi
}

function clean_files() {
    echo "清理日志文件..."
    echo "=========================================="
    
    LOG_DIR=$(get_latest_log_dir)
    if [ -z "$LOG_DIR" ]; then
        echo "未找到最新的日志目录"
        return
    fi
    
    # 清理当前日志目录中的日志文件
    rm -f "$LOG_DIR/empo_train.log" "$LOG_DIR/intuitor_train.log" "$LOG_DIR/ttrl_train.log"
    echo "已删除日志文件: $LOG_DIR"
    
    # 清理latest_log_dir.txt
    rm -f latest_log_dir.txt
    echo "已删除latest_log_dir.txt"
    
    # 如果日志目录为空，则删除该目录
    if [ -d "$LOG_DIR" ] && [ -z "$(ls -A "$LOG_DIR")" ]; then
        rmdir "$LOG_DIR"
        echo "已删除空的日志目录: $LOG_DIR"
    fi
}

# 主逻辑
case "$1" in
    status)
        check_status
        ;;
    progress)
        show_progress
        ;;
    logs)
        show_logs
        ;;
    list)
        list_logs
        ;;
    clean)
        clean_files
        ;;
    help|--help|-h)
        show_help
        ;;
    "")
        show_help
        ;;
    *)
        echo "未知选项: $1"
        show_help
        exit 1
        ;;
esac