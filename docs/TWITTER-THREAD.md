# Twitter/X Thread — Claude Code Local

---

**Tweet 1 (Hook)**

I'm running Claude Code with THREE local AI models on my MacBook.

122B params at 65 tok/s. 70B for hardest reasoning. 31B for daily speed.

No internet. No API fees. No data leaves my machine.

Here's the lineup and how I built it:

---

**Tweet 2 (The Lineup)**

The fighters:

🔵 Qwen 3.5 122B — THE BEAST (65 tok/s, MoE A10B, 4-bit)
🟠 Llama 3.3 70B Abliterated — THE WISE ONE (~7 tok/s, 8-bit affine, 128K ctx) ⭐ I packed and uploaded this one myself: huggingface.co/divinetribe/Llama-3.3-70B-Instruct-abliterated-8bit-mlx
🟢 Gemma 4 31B Abliterated — THE QUICK ONE (~15 tok/s, 4-bit IT)

Same MLX server runs all three. Swap one env var and you swap the brain.

---

**Tweet 3 (The Problem We Solved)**

Claude Code only speaks Anthropic's API. Local models speak OpenAI's API.

Most projects bridge this with a proxy. Proxies add latency, complexity, and break things.

We deleted the proxy.

We wrote a server that speaks Anthropic API natively, with MLX inference under the hood.

Result: 7.5x faster than the proxy approach.

---

**Tweet 4 (Benchmarks)**

The numbers (Qwen 3.5 122B, measured on M5 Max 128 GB):

- 65 tokens/sec sustained generation
- 17.6 seconds for a real Claude Code coding task (down from 133s with a proxy)
- Tool-call reliability: 98/98 tests passing across 7 runs
- Beats cloud Opus on raw tok/s

---

**Tweet 5 (Privacy + Safety)**

Why this matters:

🔒 100% on-device. Nothing ever leaves your Mac.
✈️ Works on a plane. No internet, no problem.
🚫 No telemetry. No analytics. No phone-home.
📜 Every dependency audited. We yanked LiteLLM after the supply-chain scare.
💰 $0/month after hardware.

Same Claude Code experience. Zero data exposure.

---

**Tweet 6 (Modes)**

Four ways to use it:

1. **Code mode** — Claude Code with the local model (Gemma / Llama / Qwen)
2. **Browser mode** — Autonomous Brave browser agent via Chrome DevTools Protocol
3. **Narrative mode** — Every reply spoken aloud through your TTS or cloned voice
4. **Phone mode** — iMessage in, video out, full pipeline from your couch

---

**Tweet 7 (How to Set Up)**

```
git clone https://github.com/nicedreamzapp/claude-code-local
cd claude-code-local
bash setup.sh
```

Auto-detects your RAM, picks a model from the lineup, downloads it, builds the launcher.

Double-click the launcher on your Desktop. You're coding locally.

---

**Tweet 8 (Closer)**

2026 is the year AI goes fully local.

The hardware is here. The models are here. The tools are catching up.

You don't need a cloud subscription to have an AI coding partner anymore.
You don't need to ship your code to a server farm.
You don't need to pay $20-100/mo for what your laptop already runs.

Open source. Link in bio. Go build something.

---

*Suggested images/video per tweet:*
1. Hero shot — three model "fighter cards" side by side
2. Architecture diagram from README
3. Terminal showing the MLX server log on startup
4. Benchmark bar chart from BENCHMARKS.md
5. The data-flow safety diagram from README
6. Four-up screenshot of the four modes running
7. Terminal showing the 3-command setup
8. MacBook running on a plane, no wifi, Claude Code working
