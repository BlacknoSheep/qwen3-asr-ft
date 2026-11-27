import os
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Audio, Dataset, load_dataset
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
            messages = build_messages(
                audio,
                feature["language"],
                feature["transcription"],
                feature.get("system_prompt", None),
            )
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
        system_prompt = example.get("system_prompt", None)
        if system_prompt is not None:
            system_prompt = normalize_text(system_prompt)
            return {
                "language": normalize_language(example["language"]),
                "transcription": normalize_text(example["transcription"]),
                "system_prompt": normalize_text(system_prompt),
            }
        return {
            "language": normalize_language(example["language"]),
            "transcription": normalize_text(example["transcription"]),
        }

    def get_dataset(self, load_from_cache_file=True, num_proc: int | None = None) -> Dataset:
        """
        返回的数据集至少包含以下列：["audio", "language", "transcription"]，具体处理延迟到 collator 便于 batch 和 pad
        """
        self.dataset = self.dataset.map(
            self._normalize,
            load_from_cache_file=load_from_cache_file,
            num_proc=num_proc,
        )
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=self.samplerate, num_channels=1))
        return self.dataset

    def get_collator(self, dtype: torch.dtype):
        return SimpleCollator(
            processor=self.processor,
            dtype=dtype,
        )

    @classmethod
    def from_path(
        cls,
        data_name_or_path: str,
        processor: Qwen3ASRProcessor,
        samplerate=16000,
        *,
        split: str = "train",
        load_from_cache_file: bool = True,
        num_proc: int | None = None,
    ) -> "SimpleDataManager":
        """
        加载 Hugging Face 数据集的指定 split，或将本地 JSON/JSONL 文件作为该 split 加载。
        数据集需包含 audio、language、transcription 列。
        audio 是音频绝对路径或相对路径（相对于该 jsonl 文件所在目录）
        """
        if data_name_or_path.endswith((".json", ".jsonl")):
            data_file = os.path.abspath(data_name_or_path)
            dataset_dir = os.path.dirname(data_file)
            dataset = load_dataset("json", data_files={split: data_file}, split=split)
            dataset = dataset.map(
                lambda example: {"audio": os.path.join(dataset_dir, example["audio"])},
                load_from_cache_file=load_from_cache_file,
                num_proc=num_proc,
            )
        else:
            dataset = load_dataset(data_name_or_path, split=split)
        return cls(dataset=dataset, processor=processor, samplerate=samplerate)
