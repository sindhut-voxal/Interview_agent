import re

from loguru import logger

from pipecat.frames.frames import LLMTextFrame

from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
)


ACRONYM_MAP = {
    "HTTPS": "H T T P S",
    "HTTP": "H T T P",
    "JSON": "J S O N",
    "LLM": "L L M",
    "STT": "S T T",
    "TTS": "T T S",
    "API": "A P I",
    "GPU": "G P U",
    "CPU": "C P U",
    "SQL": "S Q L",
    "URL": "U R L",
    "NLP": "N L P",
    "ASR": "A S R",
    "WebRTC": "Web R T C",
    "AI": "A I",
    "ML": "M L",
    "UI": "U I",
    "UX": "U X",
}


def normalize_markdown(text: str) -> str:

    # Remove code fence markers but keep the content.
    text = re.sub(
        r"```(?:\w+)?",
        "",
        text,
    )

    # Remove inline code markers but keep the content.
    text = re.sub(
        r"`([^`]*)`",
        r"\1",
        text,
    )

    # Convert Markdown links:
    # [OpenAI](https://example.com)
    # -> OpenAI
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text,
    )

    # Remove heading markers.
    text = re.sub(
        r"^\s*#{1,6}\s*",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Remove bold and italic markers.
    text = re.sub(
        r"(\*\*|__)",
        "",
        text,
    )

    text = re.sub(
        r"(?<!\*)\*(?!\*)",
        "",
        text,
    )

    text = re.sub(
        r"(?<!_)_(?!_)",
        "",
        text,
    )

    # Remove bullet markers.
    text = re.sub(
        r"^\s*[-*+•]\s+",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Remove numbered-list markers.
    text = re.sub(
        r"^\s*\d+[.)]\s+",
        "",
        text,
        flags=re.MULTILINE,
    )

    return text


def expand_acronyms(text: str) -> str:

    acronyms = sorted(
        ACRONYM_MAP.keys(),
        key=len,
        reverse=True,
    )

    for acronym in acronyms:

        replacement = ACRONYM_MAP[acronym]

        pattern = (
            r"\b"
            + re.escape(acronym)
            + r"\b"
        )

        text = re.sub(
            pattern,
            replacement,
            text,
        )

    return text


def clean_whitespace(text: str) -> str:

    # Preserve sentence/newline boundaries,
    # but remove unnecessary spaces and blank lines.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{2,}",
        "\n",
        text,
    )

    return text.strip()


def normalize_for_tts(
    text: str,
) -> str:

    original_text = text

    text = normalize_markdown(text)

    text = expand_acronyms(text)

    text = clean_whitespace(text)

    if text != original_text:

        logger.debug(
            f"Normalized TTS text:\n"
            f"Before: {original_text}\n"
            f"After: {text}"
        )

    return text


class TextNormalizerProcessor(
    FrameProcessor,
):

    async def process_frame(
        self,
        frame,
        direction: FrameDirection,
    ):

        await super().process_frame(
            frame,
            direction,
        )

        if isinstance(
            frame,
            LLMTextFrame,
        ):

            normalized_text = (
                normalize_for_tts(
                    frame.text
                )
            )

            frame.text = normalized_text

        await self.push_frame(
            frame,
            direction,
        )