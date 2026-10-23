from dataclasses import dataclass
from typing import Any

import torch
from datasets import Audio, Dataset
from transformers.models.qwen3_asr import Qwen3ASRProcessor

from .utils import (
    build_messages,
    normalize_language,
    normalize_text,
)


@dataclass
class SimpleCollator:
    processor: Qwen3ASRProcessor
    dtype: torch.dtype

    def __init__(self, processor: Qwen3ASRProcessor, dtype: torch.dtype):
        self.processor = processor
        self.dtype = dtype

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        messages_batch = []
        for feature in features:
            audio = feature["audio"].get_all_samples().data[0].numpy()
            messages = build_messages(audio, feature["language"], feature["transcription"])
            messages_batch.append(messages)

        inputs: Any = self.processor.apply_chat_template(
            messages_batch, tokenize=True, return_dict=True, return_assistant_tokens_mask=True, return_tensors="pt"
        )
        labels = inputs.input_ids.clone()
        labels[inputs.assistant_masks == 0] = -100
        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "input_features": inputs.input_features.to(self.dtype),
            "input_features_mask": inputs.input_features_mask,
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
        assert "language" in dataset.column_names, "Dataset must contain 'language' column."
        assert "transcription" in dataset.column_names, "Dataset must contain 'transcription' column."

        self.dataset = dataset
        self.processor = processor
        self.samplerate = samplerate

    @staticmethod
    def _normalize(example: dict[str, Any]) -> dict[str, Any]:
        return {
            "language": normalize_language(example["language"]),
            "transcription": normalize_text(example["transcription"]),
        }

    def get_dataset(self) -> Dataset:
        """
        返回的数据集至少包含以下列：["audio", "language", "transcription"]，具体处理延迟到 collator 便于 batch 和 pad
        """
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=self.samplerate, num_channels=1))
        self.dataset = self.dataset.map(
            self._normalize,
            load_from_cache_file=False,
        )
        return self.dataset

    def get_collator(self, dtype: torch.dtype):
        return SimpleCollator(
            processor=self.processor,
            dtype=dtype,
        )
