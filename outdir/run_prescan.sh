#!/usr/bin/env bash
# 用 tmux 并行跑所有 prescan 批次，每个批次一个窗口
# 用法：
#   bash outdir/run_prescan.sh              # 默认 max_parallel=4
#   bash outdir/run_prescan.sh 8            # 同时跑 8 个批次
#   bash outdir/run_prescan.sh 4 500        # 每批 500 个场景

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MAX_PARALLEL="${1:-4}"
BATCH_SIZE="${2:-1000}"
TOTAL_SCENES=6813
SESSION="prescan"

# 计算批次总数
NUM_BATCHES=$(( (TOTAL_SCENES + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "Total scenes:  $TOTAL_SCENES"
echo "Batch size:    $BATCH_SIZE"
echo "Num batches:   $NUM_BATCHES (0..$((NUM_BATCHES-1)))"
echo "Max parallel:  $MAX_PARALLEL"
echo "tmux session:  $SESSION"
echo ""

# 如果 session 已存在，提示用户
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists."
    echo "  Attach:  tmux attach -t $SESSION"
    echo "  Kill:    tmux kill-session -t $SESSION"
    exit 1
fi

# 创建 session（第一个窗口用作 monitor）
tmux new-session -d -s "$SESSION" -n "monitor"
tmux send-keys -t "$SESSION:monitor" \
    "echo '=== Prescan Monitor ==='; echo 'Batches: 0..$((NUM_BATCHES-1))'; echo 'Use Ctrl-b w to list windows'; watch -n 5 'wc -l $PROJECT_DIR/prescan_output/prescan_batch_*.csv 2>/dev/null || echo waiting...'" Enter

running=0
batch=0

while [ $batch -lt $NUM_BATCHES ]; do
    # 等待有空位
    while [ $running -ge $MAX_PARALLEL ]; do
        sleep 5
        # 统计还活着的 batch 窗口数
        running=$(tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null \
            | grep -c '^batch_' || true)
    done

    win_name="batch_${batch}"
    tmux new-window -t "$SESSION" -n "$win_name"
    # 窗口里跑 blenderproc，完成后暂停 30 秒让用户看到结果，然后自动关闭
    tmux send-keys -t "$SESSION:$win_name" \
        "cd $PROJECT_DIR && echo '>>> Batch $batch starting...' && blenderproc run outdir/prescan_placements.py --batch $batch --batch-size $BATCH_SIZE; echo '>>> Batch $batch finished (exit \$?)'; sleep 30" Enter

    echo "Launched batch $batch in window '$win_name'"
    running=$((running + 1))
    batch=$((batch + 1))

    # 稍微错开启动，避免同时初始化 Blender
    sleep 3
done

echo ""
echo "All $NUM_BATCHES batches launched."
echo "Attach to tmux:  tmux attach -t $SESSION"
echo "List windows:    Ctrl-b w"
echo "Switch window:   Ctrl-b <number>"
