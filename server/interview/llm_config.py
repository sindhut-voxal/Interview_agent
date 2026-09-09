import os

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"


def gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
