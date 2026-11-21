# ia-local-lab

Small hands-on projects to learn how to **use local LLMs** from code: everything runs on my own GPU, with no cloud APIs and no data leaving the PC.

Each project introduces one new concept, from a first API call up to a small RAG app.

## Setup

| | |
|---|---|
| GPU | AMD Radeon RX 9070 XT, 16 GB VRAM (RDNA 4) |
| OS | Windows 11 |
| LLM server | [LM Studio](https://lmstudio.ai/) (llama.cpp, Vulkan runtime), OpenAI-compatible API at `http://localhost:1234/v1` |
| Python | 3.12, managed with [uv](https://docs.astral.sh/uv/), one environment per project |

Models used:

| Model | Role | Speed on this GPU |
|---|---|---|
| `openai/gpt-oss-20b` (MXFP4, MoE) | Main model | ~160 tok/s |
| `qwen/qwen3.5-9b` (Q4_K_M) | Lighter model, vision | ~90 tok/s |
| `text-embedding-nomic-embed-text-v1.5` | Embeddings | n/a |

## Projects

| # | Project | What it teaches | Status |
|---|---|---|---|
| 01 | **Hello API**: talk to LM Studio with the `openai` library, normal and streaming responses, tok/s, comparing models | Messages and roles, parameters, tokens, context, streaming | In progress |
| 02 | **Batch processing**: a folder of texts → structured JSON per text | Structured output (JSON schema), validation | Planned |
| 03 | **Tool calling by hand**: the model calls Python functions, with a hand-written agent loop | How an agent works inside | Planned |
| 04 | **MCP in LM Studio**: add an MCP server and use it from the chat | The MCP standard, permissions, prompt injection risks | Planned |
| 05 | **Mini RAG**: chat with your own documents | Embeddings, semantic search, chunking | Planned |

## Running a project

1. Open LM Studio and turn the local server on (**Running**).
2. In the project folder:
   ```powershell
   uv sync          # creates .venv and installs the exact versions from uv.lock
   uv run <script>.py
   ```
