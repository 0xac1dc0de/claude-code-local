#!/bin/bash
# Narrative Gemma — Local AI Claude Code with auto-narration
# Double-click to launch
#
# Boots Gemma 4 31B Abliterated on MLX, then opens Claude Code inside the
# NarrativeGemma project folder so the CLAUDE.md narration rules are loaded
# automatically — every reply gets spoken aloud through your TTS of choice.
#
# OPTIONAL DEPENDENCY:
#   ~/.local/bin/speak — a CLI that takes a string and speaks it through
#   your speakers. Stub it with `say "$@"` (macOS built-in) if you don't
#   have a fancier voice setup. The CLAUDE.md persona expects this binary.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/claude-local-common.sh"

CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/NarrativeGemma"
COMBINED_PROMPT="/tmp/narrative_gemma_combined_prompt.md"

# Override the model with: MLX_MODEL=<your-path-or-hf-id>
MLX_MODEL_DEFAULT="divinetribe/gemma-4-31b-it-abliterated-4bit-mlx"

# ── Build combined system prompt ──────────────────────────────────────
# --bare disables auto-memory, so we hand-stitch the narration rules into
# a file that the patched MLX server appends to its mode-specific prompt.
{
  cat "$PROJECT_DIR/CLAUDE.md"
} > "$COMBINED_PROMPT"

# Always restart the MLX server so it picks up the MLX_APPEND_SYSTEM_PROMPT_FILE
# env var. Env vars can only be applied at process start, so even if Gemma is
# already running we need a fresh process with this var in its environment.
export MLX_APPEND_SYSTEM_PROMPT_FILE="$COMBINED_PROMPT"
force_restart_mlx_server "${MLX_MODEL:-$MLX_MODEL_DEFAULT}" \
  "  Loading Gemma 4 31B Abliterated with narration rules..."

clear
echo ""
echo "  → NARRATIVE GEMMA — Local AI with auto-narration"
echo "  → Gemma 4 31B Abliterated · 4-bit · ~15 tok/s"
echo "  → Every response spoken aloud via ~/.local/bin/speak"
echo "  → Running on Apple Silicon — no cloud, no API fees"
echo ""
cd "$PROJECT_DIR" || exit 1

# NOTE: The MLX server's "code mode" silently REPLACES Claude Code's
# system prompt with a generic coding-assistant prompt that says NOT
# to use tools for greetings. That kills narration. The exported
# MLX_APPEND_SYSTEM_PROMPT_FILE above tells the server to append the
# narration prompt instead.
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
CLAUDE_SESSION_LABEL="Narrative Gemma · Local" \
exec "$CLAUDE_BIN" --model claude-sonnet-4-6 \
  --permission-mode auto \
  --bare \
  --append-system-prompt-file "$COMBINED_PROMPT" \
  --mcp-config "$HOME/.claude.json"
