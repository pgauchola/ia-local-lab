"""04 - Chat history: the model has no memory, so we resend the whole conversation.

Usage (run it in a real terminal, it reads your keyboard):
    uv run 04_historial.py                 # chat with memory
    uv run 04_historial.py --no-history    # each message is sent alone: the model "forgets"

Commands inside the chat: /reset (start over), /exit
"""

import sys

from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

MODEL = "openai/gpt-oss-20b"
CONTEXT_LENGTH = 8192  # context the model was loaded with (see `lms ps`)
KEEP_HISTORY = "--no-history" not in sys.argv

SYSTEM = {
    "role": "system",
    "content": "You are a helpful assistant. Answer briefly, in the user's language.",
}

# The whole "memory" of the chat is this Python list. The model itself keeps
# nothing between requests: every request must carry the full conversation.
history = [SYSTEM]

print(f"Model: {MODEL} | history: {'ON' if KEEP_HISTORY else 'OFF'}")
print("Type /reset to start over, /exit to quit.")

while True:
    try:
        user_text = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):  # Ctrl+C / Ctrl+Z also quit
        break
    if not user_text:
        continue
    if user_text == "/exit":
        break
    if user_text == "/reset":
        history = [SYSTEM]
        print("(history cleared)")
        continue

    history.append({"role": "user", "content": user_text})

    # With history: send everything. Without: only the system prompt + last message.
    messages = history if KEEP_HISTORY else [SYSTEM, history[-1]]

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        reasoning_effort="low",
    )

    print("AI:  ", end="", flush=True)
    answer = ""
    usage = None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            piece = chunk.choices[0].delta.content
            answer += piece
            print(piece, end="", flush=True)
    print()

    # Save the answer with role "assistant" so the next request includes it.
    # (Only the visible answer: the reasoning is not sent back.)
    history.append({"role": "assistant", "content": answer})

    if usage:
        used = usage.prompt_tokens + usage.completion_tokens
        print(
            f"     [messages sent: {len(messages)} | prompt_tokens: {usage.prompt_tokens}"
            f" | context used: {used}/{CONTEXT_LENGTH} ({used / CONTEXT_LENGTH:.0%})]"
        )
