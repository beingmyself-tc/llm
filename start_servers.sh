#!/bin/bash
# Use absolute path so this script works from anywhere
cd /Users/seb/code/llm
source .venv/bin/activate

echo "Starting Qwen 14B (Smart) on port 8080..."
mlx_lm.server --model mlx-community/Qwen2.5-Coder-14B-Instruct-4bit --draft-model mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit --port 8080 &
PID1=$!

echo "Starting Qwen 7B (Fast) on port 8081..."
mlx_lm.server --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --draft-model mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit --port 8081 &
PID2=$!

# Handle shutdown
trap "kill $PID1 $PID2; exit" SIGINT SIGTERM

echo "MLX Servers are running."
echo " - 14B: http://localhost:8080"
echo " - 7B:  http://localhost:8081"
echo "Press Ctrl+C to stop both servers."

wait
