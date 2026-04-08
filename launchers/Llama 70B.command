#!/bin/bash
# Llama 70B — Claude Code on Llama 3.3 70B Abliterated (8-bit MLX)
# Double-click to launch
#
# THE WISE ONE — ~7 tok/s, ~70 GB RAM, full 8-bit precision, abliterated.
# Slower but the most capable reasoning in the lineup. Needs 96+ GB unified memory.

CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"
MLX_SERVER="$HOME/.local/mlx-native-server/server.py"
MLX_PYTHON="$HOME/.local/mlx-server/bin/python3"

# Default points at our own abliterated MLX upload:
#   https://huggingface.co/divinetribe/Llama-3.3-70B-Instruct-abliterated-8bit-mlx
# Override with MLX_MODEL=<your-path-or-hf-id>
MLX_MODEL_DEFAULT="divinetribe/Llama-3.3-70B-Instruct-abliterated-8bit-mlx"

# Start MLX server if not running
if ! lsof -i :4000 >/dev/null 2>&1; then
  MLX_MODEL="${MLX_MODEL:-$MLX_MODEL_DEFAULT}" \
  "$MLX_PYTHON" "$MLX_SERVER" >/tmp/mlx-server.log 2>&1 &
  echo "  Loading Llama 3.3 70B Abliterated on MLX (~7 tok/s, 8-bit full precision)..."
  while ! curl -s http://localhost:4000/health 2>/dev/null | grep -q "ok"; do
    sleep 2
  done
fi

clear
echo ""
echo "  → Claude Code with LOCAL AI (Llama 3.3 70B Abliterated)"
echo "  → MLX Native: ~7 tok/s, 8-bit full precision, abliterated"
echo "  → Running on Apple Silicon — no cloud, no API fees"
echo ""

ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
CLAUDE_SESSION_LABEL="Llama 70B · Local" \
exec "$CLAUDE_BIN" --model claude-sonnet-4-6 \
  --permission-mode auto \
  --bare \
  --append-system-prompt-file "$HOME/.claude/CLAUDE.md" \
  --mcp-config "$HOME/.claude.json"
