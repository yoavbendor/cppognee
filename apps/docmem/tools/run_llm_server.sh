#!/usr/bin/env bash
# Serve a GGUF model as an OpenAI-compatible API (llama-cpp-python server).
#
# Usage: run_llm_server.sh MODEL.gguf [PORT=8080] [N_CTX=8192]
# Endpoints: http://127.0.0.1:PORT/v1
#
# The chat template is read from the GGUF metadata (no --chat_format override).
# The interpreter is $LLM_PYTHON, default: the llmvenv built for llama-cpp-python[server].
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 MODEL.gguf [PORT=8080] [N_CTX=8192]" >&2
  exit 2
fi

MODEL="$1"
PORT="${2:-8080}"
N_CTX="${3:-8192}"
PYTHON="${LLM_PYTHON:-/tmp/claude-0/-home-user/8df192dc-e0a5-50ec-ba29-4d1c2c18b24f/scratchpad/llmvenv/bin/python}"

if [ ! -f "$MODEL" ]; then
  echo "model not found: $MODEL" >&2
  exit 1
fi

exec "$PYTHON" -m llama_cpp.server \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --n_ctx "$N_CTX" \
  --n_threads 4
