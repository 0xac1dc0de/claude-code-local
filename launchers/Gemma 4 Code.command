#!/bin/bash
# Gemma 4 Code — Claude Code on Gemma 4 31B Abliterated (4-bit MLX)
# Double-click to launch
#
# THE QUICK ONE — ~15 tok/s, ~18 GB RAM, abliterated, instruction-tuned.
# Best balance of speed and quality for daily coding.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/claude-local-common.sh"

CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"

# Override with MLX_MODEL=<your-path-or-hf-id>
MLX_MODEL_DEFAULT="divinetribe/gemma-4-31b-it-abliterated-4bit-mlx"

ensure_mlx_server "${MLX_MODEL:-$MLX_MODEL_DEFAULT}" \
  "  Loading Gemma 4 31B Abliterated on MLX (~15 tok/s, 4-bit)..."

clear
echo ""
echo "  → Claude Code with LOCAL AI (Gemma 4 31B Abliterated)"
echo "  → MLX Native: 4-bit IT, abliterated, instruction tuned for coding"
echo "  → Running on Apple Silicon — no cloud, no API fees"
echo ""

ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
CLAUDE_SESSION_LABEL="Gemma 4 · Local" \
exec "$CLAUDE_BIN" --model claude-sonnet-4-6 \
  --permission-mode auto \
  --bare \
  --append-system-prompt-file "$HOME/.claude/CLAUDE.md" \
  --mcp-config "$HOME/.claude.json"
