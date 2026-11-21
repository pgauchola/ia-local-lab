# 01 · Hello API

Talking to a local LLM from Python. [LM Studio](https://lmstudio.ai/) exposes an OpenAI-compatible API, so the official `openai` library works by just pointing `base_url` at `http://localhost:1234/v1`.

**Concepts:** messages and roles, parameters, tokens, context window, streaming.

## Scripts

| Script | What it does |
|---|---|
| `01_basico.py` | One request, one answer. Prints the reasoning, `finish_reason` and token usage. |
| `02_streaming.py` | Streams the answer token by token and measures time to first token (TTFT), thinking time and tok/s. Takes the model id as an argument. |
| `03_parametros.py` | Three experiments: `max_tokens`, `temperature` and the system prompt. |
| `04_historial.py` | Terminal chat that keeps the conversation history and shows how the context fills up. `--no-history` mode to see the model forget. |

## How to run

1. LM Studio open, local server **Running**.
2. Load the model **fully on the GPU** (see "Lessons" below for why):
   ```powershell
   lms load openai/gpt-oss-20b --gpu max -y
   ```
3. In this folder:
   ```powershell
   uv sync
   uv run 01_basico.py
   uv run 02_streaming.py                    # or: uv run 02_streaming.py qwen/qwen3.5-9b
   uv run 03_parametros.py
   uv run 04_historial.py                    # interactive: run it in a real terminal
   ```

## Results

RX 9070 XT (16 GB), LM Studio 0.4.28, llama.cpp Vulkan runtime 2.80.1. Prompt: *"Explain how a four-stroke engine works, in about 200 words."*

| Model | Load | VRAM used | Speed | Notes |
|---|---|---|---|---|
| gpt-oss-20b (MXFP4, MoE) | Auto (JIT, via API) | 8.8 GB | ~37 tok/s | Part of the model left on the CPU |
| gpt-oss-20b (MXFP4, MoE) | `--gpu max` | 12.7 GB | **~167 tok/s** | ~7 reasoning tokens with `reasoning_effort="low"` |
| Qwen3.5 9B (Q4_K_M) | Auto (JIT, via API) | n/a | ~90 tok/s | Thought for 8,000+ tokens and filled the 8K context without answering |

## Lessons learned

**Performance**
- **Auto-loading can silently halve your speed (or worse).** When the API asks for a model that isn't loaded, LM Studio loads it on the fly with automatic settings. Here that left part of gpt-oss on the CPU: ~37 tok/s instead of ~167. Load models explicitly with full GPU offload. `lms ps` shows a `TTL` for auto-loaded models.
- **Measure from the client and from the engine.** LM Studio's native endpoint (`/api/v0/chat/completions`) returns engine-side stats. Comparing them with the script's numbers showed the measurement was right and the problem was elsewhere.
- **Open question:** TTFT is ~2.2 s from Python, while the engine reports ~0.04 s. Not investigated yet.

**Reasoning models**
- Reasoning tokens are real tokens: they take time, count towards `max_tokens` and fill the context. With no limit, Qwen3.5 9B reasoned until the context was full and never answered.
- "OpenAI-compatible" doesn't mean identical: LM Studio returns gpt-oss reasoning in `reasoning` and Qwen's in `reasoning_content`. Neither is part of the official schema.
- Once, gpt-oss reasoning leaked into the visible answer (`"Will say yes but depends on game, settings.Sí, ..."`). Code that parses answers should not assume the split is always clean.

**Parameters**
- `max_tokens` does **not** make answers shorter. It only cuts them off (`finish_reason="length"`). gpt-oss went over 400 tokens explaining a CPU cache. To get short answers, ask for them.
- `temperature=0` gave the same answer 3 times out of 3. `temperature=1.5` gave 3 different ones, though still similar ("Bean & ...").
- The system prompt changed the answer completely (one sentence, pirate, a child's explanation in Spanish) and costs tokens on **every** request: 71 → 84-95 prompt tokens.

**Context and memory**
- The model has no memory. The chat's "memory" is a Python list that is resent in full on every request, so prompt tokens grow every turn (99 → 173 → 193 → 231).
- Without history, the model forgot the user's name and even misread *"¿qué gráfica tienes?"* as "what chart do you have?" instead of "what graphics card". Context also gives words their meaning.

**Reliability**
- Models answer confidently even when they are wrong: each stroke of a four-stroke engine is half a crankshaft turn, not a full one. When simplified for a child, RAM was described as keeping data when the PC is off, which is the opposite of what it said a moment earlier.
- Knowledge cutoff: gpt-oss doesn't know the RX 9070 XT (released after its training data) and made up specs instead of saying so. That's what RAG (project 05) is for.

**Windows**
- Printing model output can crash with `UnicodeEncodeError` (cp1252) on characters like a non-breaking hyphen. Fixed with `sys.stdout.reconfigure(encoding="utf-8")`.
