from text_normaliser import normalize_for_tts


TEXT = """
## **Interview Question**

How did you use `FastAPI` and an **LLM**
to connect STT, TTS, and WebRTC?

1. Explain the API flow.
2. Explain how JSON is handled.

- Mention AI and ML.
"""


normalized = normalize_for_tts(
    TEXT
)

print("\n--- FINAL RESULT ---\n")

print(normalized)