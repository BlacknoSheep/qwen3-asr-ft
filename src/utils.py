import numpy as np
import regex
from transformers.audio_utils import resolve_language
from transformers.models.qwen3_asr.processing_qwen3_asr import LANGUAGE_CODE_TO_NAME


def normalize_text(text: str) -> str:
    # qwen支持全角和大小写
    # text = unicodedata.normalize("NFKC", text) # 全角半角统一
    # text = text.lower()
    # text = regex.sub(r"[<\[][^>\]]*[>\]]", "", text)  # remove words between brackets
    # text = regex.sub(r"\(([^)]+?)\)", "", text)  # remove words between parenthesis
    text = regex.sub(r"\s+", " ", text)  # 合并空白
    text = text.strip()  # 去首尾空白
    return text


def normalize_language(language: str | None) -> str | None:
    return resolve_language(language, LANGUAGE_CODE_TO_NAME, False)


def build_messages(
    audio: str | np.ndarray, language: str | None, transcription: str = "", system_prompt: str | None = None
) -> list[dict]:
    """
    system_prompt 可以添加热词，例如："Vocabulary: Quilter, apostle, gospel."
    """
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    audio_type = "path" if isinstance(audio, str) else "audio"
    messages += [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    audio_type: audio,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": f"language {language}<asr_text>{transcription}"}],
        },
    ]
    return messages
