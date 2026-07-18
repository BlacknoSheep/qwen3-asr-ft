from dataclasses import dataclass

from typing import Any, Dict, List

import regex
import torch
from datasets import Audio, Dataset
from transformers.models.qwen3_asr import Qwen3ASRProcessor

from .utils import (
    normalize_text,
    normalize_language,
    build_messages,
    generate_assistant_masks,
)

NON_TEXT_RE = regex.compile(r"[^\p{L}\p{N}]+")
WHITESPACE_RE = regex.compile(r"\s+")

with open("./src/qwen3_asr_chat_template_fixed.jinja", "r") as f:
    CHAT_TEMPLATE = f.read()


@dataclass
class SimpleCollator:
    processor: Qwen3ASRProcessor
    dtype: torch.dtype

    def __init__(self, processor: Qwen3ASRProcessor, dtype: torch.dtype):
        self.processor = processor
        self.dtype = dtype

        assistant_start = "<|im_start|>assistant\n"
        assistant_end = "<|im_end|>"
        self.assistant_start_ids = self.processor.tokenizer.encode(
            assistant_start,
            return_tensors="pt",
            add_special_tokens=False,
        )
        self.assistant_end_ids = self.processor.tokenizer.encode(
            assistant_end,
            return_tensors="pt",
            add_special_tokens=False,
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        print(len(features[0]["messages"][1]["content"][0]["audio"])) # list
        inputs: Any = self.processor.apply_chat_template(
            [feature["messages"] for feature in features],
            chat_template=CHAT_TEMPLATE,
            tokenize=True,
            return_dict=True,
        )
        labels = inputs.input_ids.clone()
        assistant_masks = generate_assistant_masks(
            labels, self.assistant_start_ids, self.assistant_end_ids
        )
        labels[assistant_masks == 0] = -100
        return {
            "input_ids": inputs.input_ids.to(self.dtype),
            "attention_mask": inputs.attention_mask.to(self.dtype),
            "input_features": inputs.input_features.to(self.dtype),
            "input_features_mask": inputs.input_features_mask.to(self.dtype),
            "labels": labels,
        }


class SimpleDataManager:
    def __init__(
        self,
        dataset: Dataset,
        processor: Qwen3ASRProcessor,
        samplerate=16000,
    ) -> None:
        """
        dataset 必须包含这些列：['audio', 'language', 'transcription']
        """
        assert "audio" in dataset.column_names, "Dataset must contain 'audio' column."
        assert "language" in dataset.column_names, (
            "Dataset must contain 'language' column."
        )
        assert "transcription" in dataset.column_names, (
            "Dataset must contain 'transcription' column."
        )

        self.dataset = dataset
        self.processor = processor
        self.samplerate = samplerate

    def _normalize(self, example: Dict[str, Any]) -> Dict[str, Any]:
        audio = example["audio"]["array"]
        language = normalize_language(example["language"])
        text = normalize_text(example["transcription"])
        messages = build_messages(audio, language, text)
        return {"messages": messages}

    def get_dataset(self) -> Dataset:
        """
        返回的数据集包含以下列：["messages"]，具体处理延迟到 collator 便于 batch 和 pad
        """
        self.dataset = self.dataset.cast_column(
            "audio", Audio(sampling_rate=self.samplerate)
        )
        self.dataset = self.dataset.map(
            self._normalize,
            remove_columns=self.dataset.column_names,
            load_from_cache_file=False,
        )
        return self.dataset

    def get_collator(self, dtype: torch.dtype):
        return SimpleCollator(
            processor=self.processor,
            dtype=dtype,
        )
