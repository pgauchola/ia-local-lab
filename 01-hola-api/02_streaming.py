"""02 - Streaming: print the answer while it is generated and measure speed.

Usage:
    uv run 02_streaming.py                    # gpt-oss-20b (default)
    uv run 02_streaming.py qwen/qwen3.5-9b    # any model id from /v1/models
"""

import sys
import time

from openai import OpenAI

# Windows may print with the old cp1252 encoding, which crashes on characters
# like "‑" or emojis that models often produce. Force UTF-8.
sys.stdout.reconfigure(encoding="utf-8")

client =OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

MODEL = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-20b"
PROMPT = "Explain how a four-stroke engine works, in about 200 words."

# reasoning_effort only makes sense for gpt-oss, so we send it only to that model.
extra = {"reasoning_effort": "low"} if "gpt-oss" in MODEL else {}

print(f"Model: {MODEL}\n")

start = time.perf_counter()
first_token_at = None   # first chunk of anything (reasoning or answer)
first_answer_at = None  # first chunk of the visible answer
reasoning_field = None  # which field name LM Studio uses for the reasoning
chunks = 0
usage = None
finish_reason = None    # "stop" = ended on its own, "length" = cut off (max_tokens or context full)

stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": PROMPT}],
    stream=True,
    stream_options={"include_usage": True},  # ask for token counts in the last chunk
    **extra,
)

# The response arrives as many small "chunks". Each one carries a "delta":
# the new piece of text since the previous chunk.
for chunk in stream:
    if chunk.usage:  # the final chunk carries the token counts, with no text
        usage = chunk.usage
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if chunk.choices[0].finish_reason:  # only set on the last chunk with a choice
        finish_reason = chunk.choices[0].finish_reason

    # Reasoning comes in a non-standard field; try the two usual names.
    reasoning = None
    for name in ("reasoning_content", "reasoning"):
        value = getattr(delta, name, None)
        if value:
            reasoning, reasoning_field = value, name
            break

    if not reasoning and not delta.content:
        continue

    now = time.perf_counter()
    chunks += 1
    if first_token_at is None:
        first_token_at = now
        if reasoning:
            print("[thinking] ", end="")

    if reasoning:
        print(reasoning, end="", flush=True)
    if delta.content:
        if first_answer_at is None:
            first_answer_at = now
            print("\n\n[answer] " if reasoning_field else "[answer] ", end="")
        print(delta.content, end="", flush=True)  # flush: show it now, don't buffer

end = time.perf_counter()

# --- Metrics -------------------------------------------------------------
tokens = usage.completion_tokens if usage else chunks  # fallback: ~1 token per chunk
gen_time = end - first_token_at

print("\n\n=== Metrics ===")
print(f"Finish reason:       {finish_reason}")
print(f"Reasoning field:     {reasoning_field or '(none)'}")
print(f"Time to first token: {first_token_at - start:.2f} s")
if reasoning_field and first_answer_at:
    print(f"Thinking time:       {first_answer_at - first_token_at:.2f} s")
print(f"Total time:          {end - start:.2f} s")
print(f"Completion tokens:   {tokens} ({'from usage' if usage else 'estimated from chunks'})")
print(f"Speed:               {tokens / gen_time:.1f} tok/s")
