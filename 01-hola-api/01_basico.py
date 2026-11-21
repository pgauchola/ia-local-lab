"""01 - Basic call: send one message to a local model and inspect the response."""

import sys

from openai import OpenAI

# Windows may print with the old cp1252 encoding, which crashes on characters
# like "‑" or emojis that models often produce. Force UTF-8.
sys.stdout.reconfigure(encoding="utf-8")

# LM Studio exposes an OpenAI-compatible API, so we only change the address.
# The API key is required by the library but LM Studio ignores it.
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

MODEL = "openai/gpt-oss-20b"

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": "Explain in two sentences what a GPU is."},
    ],
    reasoning_effort="low",  # gpt-oss only: how much it "thinks" before answering
)

choice = response.choices[0]

print("=== Answer ===")
print(choice.message.content)

# gpt-oss reasons before answering. LM Studio returns that text in an extra
# field that is not part of the official OpenAI schema, so we read it carefully.
reasoning = getattr(choice.message, "reasoning_content", None) or getattr(
    choice.message, "reasoning", None
)
if reasoning:
    print("\n=== Reasoning (hidden thinking) ===")
    print(reasoning)

print("\n=== Metadata ===")
print("Model:        ", response.model)
print("Finish reason:", choice.finish_reason)  # "stop" = ended on its own, "length" = hit max_tokens
print("Prompt tokens:    ", response.usage.prompt_tokens)      # what we sent (incl. chat template)
print("Completion tokens:", response.usage.completion_tokens)  # what it generated (incl. reasoning)
print("Total tokens:     ", response.usage.total_tokens)
