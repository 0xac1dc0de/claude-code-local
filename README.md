<p align="center">
  <h1 align="center">Claude Code Local</h1>
  <p align="center">
    <strong>Run a 122 billion parameter AI on your MacBook. No cloud. No fees. No data leaves your machine.</strong>
  </p>
  <p align="center">
    <a href="#benchmarks"><img src="https://img.shields.io/badge/Speed-48_tok%2Fs-brightgreen?style=for-the-badge" alt="Speed"></a>
    <a href="#benchmarks"><img src="https://img.shields.io/badge/Claude_Code-17.6s_per_task-blue?style=for-the-badge" alt="Claude Code"></a>
    <a href="#benchmarks"><img src="https://img.shields.io/badge/vs_llama.cpp-7.5x_faster-orange?style=for-the-badge" alt="7.5x faster"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="MIT"></a>
  </p>
</p>

---

## What Is This?

Your MacBook has a powerful GPU built right into the chip. This project uses that GPU to run a massive AI model — the same kind that powers ChatGPT and Claude — **entirely on your computer**.

No internet needed. No monthly subscription. No one sees your code or data.

You get the full **Claude Code** experience — it can write code, edit files, manage projects, even control your web browser — all powered by AI running on your own machine.

```
         You
          |
     Claude Code         (the AI coding tool)
          |
   MLX Native Server     (our server — 200 lines of Python)
          |
   Qwen 3.5 122B Model   (122 billion parameter brain)
          |
    Apple Silicon GPU     (your M-series chip does the thinking)
```

---

## Benchmarks

We built and tested three different approaches. Each one got faster.

### Speed Comparison

```
                        Tokens per Second
  Ollama (Gen 1)        ██████████████████████████████ 30 tok/s
  llama.cpp (Gen 2)     █████████████████████████████████████████ 41 tok/s
  MLX Native (Gen 3)    ████████████████████████████████████████████████ 48 tok/s
```

### Real-World Claude Code Task

How long it takes to ask Claude Code to write a function:

```
  Ollama + Proxy          ████████████████████████████████████████████ 133 seconds
  llama.cpp + Proxy       ████████████████████████████████████████████ 133 seconds
  MLX Native (no proxy)   ██████ 17.6 seconds

                          7.5x faster
```

### Side-by-Side

| | Ollama | llama.cpp + TurboQuant | **MLX Native (ours)** |
|---|:---:|:---:|:---:|
| **Speed** | 30 tok/s | 41 tok/s | **48 tok/s** |
| **Claude Code task** | 133s | 133s | **17.6s** |
| **Needs a proxy?** | Yes | Yes | **No** |
| **Lines of code** | N/A (external) | N/A (C++ fork) | **~200 Python** |
| **Apple Silicon native?** | No (generic) | No (ported) | **Yes (MLX)** |

### vs Cloud APIs

| | **Our Local Setup** | Claude Sonnet (Cloud) | Claude Opus (Cloud) |
|---|:---:|:---:|:---:|
| Speed | 48 tok/s | ~80 tok/s | ~40 tok/s |
| Monthly cost | **$0** | $20-100+ | $20-100+ |
| Privacy | **100% local** | Cloud | Cloud |
| Works offline | **Yes** | No | No |
| Your data leaves your Mac | **Never** | Always | Always |

---

## How We Got Here

Most people trying to run Claude Code locally hit the same wall: **Claude Code speaks one language (Anthropic API), local models speak another (OpenAI API)**. So everyone builds a proxy to translate between them. That proxy adds latency, complexity, and breaks things.

We took a different approach: **we built a server that speaks Claude Code's language natively**. No translation. No middleware. Direct.

| What everyone else does | What we did |
|---|---|
| Claude Code → **Proxy** → Ollama → Model | Claude Code → **Our Server** → Model |
| 3 processes, 2 API translations | **1 process, 0 translations** |
| 133 seconds per task | **17.6 seconds per task** |

That one change — eliminating the proxy — made it **7.5x faster**.

---

## What You Need

| Your Mac | RAM | Model You Can Run |
|----------|-----|-------------------|
| M1/M2/M3/M4 (base) | 8-16 GB | Small models (4B) |
| M1/M2/M3/M4 Pro | 18-36 GB | Medium models (32B) |
| M2/M3/M4/M5 Max | 64-128 GB | **Large models (122B)** |
| M2/M3/M4 Ultra | 128-192 GB | Multiple large models |

You also need:
- **Python 3.12+** (for MLX)
- **Claude Code** (`npm install -g @anthropic-ai/claude-code`)

---

## Quick Start (4 Steps)

### Step 1: Set up Python environment

```bash
python3.12 -m venv ~/.local/mlx-server
~/.local/mlx-server/bin/pip install mlx-lm
```

### Step 2: Download the AI model

This downloads ~50 GB on first run (one time only):

```bash
~/.local/mlx-server/bin/python3 -c "
from mlx_lm.utils import load
load('mlx-community/Qwen3.5-122B-A10B-4bit')
print('Done!')
"
```

### Step 3: Start the server

```bash
~/.local/mlx-server/bin/python3 proxy/server.py
```

### Step 4: Launch Claude Code

```bash
ANTHROPIC_BASE_URL=http://localhost:4000 \
ANTHROPIC_API_KEY=sk-local \
claude --model claude-sonnet-4-6
```

**Or just double-click** `Claude Local.command` on your Desktop. It does all of this automatically.

---

## How It Works (Simple Version)

```
                    ┌─────────────────────────────────────────┐
                    │         Your MacBook (M5 Max)           │
                    │                                         │
   You type    ───> │  Claude Code                            │
   a question       │      │                                  │
                    │      ▼                                  │
                    │  MLX Server (our code, port 4000)       │
                    │      │                                  │
                    │      ▼                                  │
                    │  Qwen 3.5 122B ── runs on GPU ──────>   │
                    │      │            (Apple Silicon)       │
                    │      ▼                                  │
   You get     <─── │  Answer                                 │
   an answer        │                                         │
                    └─────────────────────────────────────────┘

                    Nothing leaves this box. Ever.
```

## How It Works (Technical)

The server is a single Python file (~200 lines). It does three things:

1. **Loads the model** using Apple's MLX framework — this runs natively on your M-series GPU with unified memory (no copying data between CPU and GPU)

2. **Serves the Anthropic Messages API** — the same API that Claude Code uses to talk to Anthropic's cloud. Our server pretends to be Anthropic, but it's running on your laptop.

3. **Cleans up the output** — the Qwen model "thinks out loud" in `<think>` tags. Our server strips those so Claude Code gets clean responses.

---

## Browser Control

Claude Code can control your web browser too. Not a fake sandboxed browser — **your real one**, with all your logins and sessions.

| | Chrome DevTools (CDP) | Playwright |
|---|---|---|
| **What it controls** | Your real Brave/Chrome | A separate sandboxed browser |
| **Logged in?** | Yes — all your sessions | No — starts fresh |
| **Speed** | Fast | Slower |
| **Best for** | Daily tasks, logged-in work | Automated jobs, scraping |

**Example:** You can say "go to my GitHub and check which PRs need review" and it opens your actual browser, already logged in, and does it.

---

## When To Use This

| Situation | Use This? |
|-----------|-----------|
| On a plane with no wifi | **Yes** — full AI coding, no internet needed |
| Working with sensitive client code | **Yes** — nothing leaves your machine |
| Don't want to pay API fees | **Yes** — $0/month forever |
| Want the absolute fastest responses | Use cloud API — still faster for now |
| Need Claude's full reasoning ability | Use cloud API — local model is good but not Claude-level |

---

## What's In This Repo

```
proxy/
  server.py              ← The entire server. 200 lines. This is the project.

launchers/
  Claude Local.command    ← Double-click to start everything
  Browser Agent.command   ← Double-click for browser automation

scripts/
  download-and-import.sh  ← Download models from HuggingFace
  persistent-download.sh  ← Auto-retry downloader
  start-mlx-server.sh    ← Alternative MLX server config

docs/
  BENCHMARKS.md           ← Detailed speed comparisons
  TWITTER-THREAD.md       ← Social media content

setup.sh                  ← One-command installer
```

---

## Security

We audited every component before running it:

| Component | Source | Network Calls | Dependencies |
|-----------|--------|:---:|:---:|
| **server.py** | We wrote it | 0 | mlx-lm only |
| **MLX framework** | Apple | 0 | Apple's own |
| **Qwen 3.5 model** | HuggingFace verified | 0 | None |

No telemetry. No analytics. No phone-home. No pip packages from strangers. We even [removed LiteLLM](https://x.com/Tahseen_Rahman/status/2035501506242240520) after supply chain concerns were raised in the community.

---

## The Journey

We didn't start here. We went through three generations in one night:

| Gen | What We Tried | Speed | What We Learned |
|:---:|---|:---:|---|
| 1 | Ollama + custom proxy | 30 tok/s | Ollama works but Claude Code can't talk to it directly |
| 2 | llama.cpp TurboQuant + proxy | 41 tok/s | TurboQuant compresses KV cache 4.9x, but the proxy is the bottleneck |
| **3** | **MLX native server** | **48 tok/s** | **Kill the proxy. Speak Anthropic API directly. 7.5x faster.** |

Each generation taught us something. The final insight — that the proxy was the real bottleneck, not the model — changed everything.

---

## Credits

Built on the shoulders of:

| Project | What It Does | Who Made It |
|---------|-------------|-------------|
| [Claude Code](https://claude.ai/claude-code) | AI coding agent | Anthropic |
| [MLX](https://github.com/ml-explore/mlx) | Apple Silicon ML framework | Apple |
| [mlx-lm](https://github.com/ml-explore/mlx-examples) | Model loading + inference | Apple |
| [Qwen 3.5](https://qwenlm.github.io/) | The 122B model | Alibaba |
| [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) | KV cache compression research | Google Research |

Tested on **Apple M5 Max** with **128 GB unified memory**.

---

<p align="center">
  <strong>MIT License</strong> — Use it however you want.
</p>
