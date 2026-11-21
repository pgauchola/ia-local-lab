"""03 - Parameters: see what max_tokens, temperature and the system prompt change.

Usage:
    uv run 03_parametros.py                    # gpt-oss-20b (default)
    uv run 03_parametros.py qwen/qwen3.5-9b
"""

import sys

from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

MODEL = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-20b"
extra = {"reasoning_effort": "low"} if "gpt-oss" in MODEL else {}


def ask(messages, **params):
    """Send one request and return the first choice and the token usage."""
    response = client.chat.completions.create(
        model=MODEL, messages=messages, **extra, **params
    )
    return response.choices[0], response.usage


def title(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


print(f"Model: {MODEL}")

# --- 1. max_tokens -------------------------------------------------------
# Hard limit on how many tokens the model may generate. It does NOT make the
# model write shorter answers: it just cuts them off when the limit is hit.
title("1. max_tokens: same question, different limits")
question = [{"role": "user", "content": "Explain what a CPU cache is."}]

for limit in (40, 400):
    choice, usage = ask(question, max_tokens=limit)
    print(f"\n--- max_tokens={limit} -> finish_reason={choice.finish_reason}, "
          f"completion_tokens={usage.completion_tokens}")
    print(choice.message.content or "(empty answer: the limit was used up by the reasoning)")

# --- 2. temperature ------------------------------------------------------
# The model computes a probability for every possible next token.
# temperature=0 -> always pick the most likely one (deterministic).
# Higher temperature -> flatter probabilities, less likely tokens get picked more.
title("2. temperature: same question 3 times at each temperature")
question = [{"role": "user", "content": "Invent a name for a coffee shop. Reply with the name only."}]

for temp in (0, 1.5):
    print(f"\n--- temperature={temp}")
    for _ in range(3):
        choice, _ = ask(question, temperature=temp, max_tokens=200)
        print("  ", (choice.message.content or "").strip())

# --- 3. System prompt ----------------------------------------------------
# A message with role "system" goes first and sets HOW the model behaves.
# The user question is the same; only the instructions change.
title("3. System prompt: same question, different instructions")
user_msg = {"role": "user", "content": "What is RAM?"}
systems = [
    None,
    "Answer in exactly one sentence.",
    "You are a pirate. Answer like a pirate, in two sentences max.",
    "Answer in Spanish, explained for a 10-year-old, in two sentences max.",
]

for system in systems:
    messages = [{"role": "system", "content": system}, user_msg] if system else [user_msg]
    choice, usage = ask(messages, max_tokens=400)
    print(f"\n--- system: {system or '(none)'}  [prompt_tokens={usage.prompt_tokens}]")
    print(choice.message.content)
