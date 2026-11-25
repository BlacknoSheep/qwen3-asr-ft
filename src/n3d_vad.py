from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor
from transformers.audio_utils import load_audio

MODEL_NAME = "nvidia/Nemotron-3-Diarization"


class N3DVAD:
    """离线检测人声；同一个实例可重复处理多个文件。"""

    def __init__(
        self,
        model_name: str | Path = MODEL_NAME,
        *,
        device: str | torch.device | None = None,
        local_files_only: bool = True,
    ) -> None:
        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        self.sampling_rate = self.processor.feature_extractor.sampling_rate
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            AutoModelForAudioFrameClassification.from_pretrained(model_name, local_files_only=local_files_only)
            .to(device)
            .eval()
        )

    @torch.inference_mode()
    def get_speech_timestamps(
        self,
        audio: str | Path | np.ndarray,
        *,
        sampling_rate: int | None = None,
        threshold: float = 0.5,
        min_speech_duration_ms: float = 0,
        min_silence_duration_ms: float = 0,
        speakers: bool = False,
    ) -> list[dict[str, float | int]]:
        """返回按起点排序的秒级区间（start、end）。

        文件会解码为模型采样率；数组须为一维单声道，且已采用模型采样率。
        默认合并重叠、相接的区间，并可用 min_silence_duration_ms 桥接短静音。
        speakers=True 时保留 speaker_id 和重叠片段，不桥接静音。
        两种模式均在最后按 min_speech_duration_ms 过滤短片段。
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold 必须在 0 到 1 之间")
        if min_speech_duration_ms < 0 or min_silence_duration_ms < 0:
            raise ValueError("语音和静音时长不能为负")

        if isinstance(audio, (str, Path)):
            audio = load_audio(str(audio), sampling_rate=self.sampling_rate)
            sampling_rate = self.sampling_rate
        elif sampling_rate is None:
            sampling_rate = self.sampling_rate
        if audio.size == 0:
            return []

        duration = len(audio) / sampling_rate
        inputs = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        logits = self.model(**inputs).logits
        (detected,) = self.processor.extract_speaker_dict(
            logits.cpu(), inputs.attention_mask.cpu(), threshold=threshold
        )

        segments = []
        for segment in detected:
            start, end = segment["Start"], min(segment["End"], duration)
            if end <= start:
                continue
            if speakers:
                segments.append({"start": start, "end": end, "speaker_id": segment["Speaker"]})
            elif segments and round((start - segments[-1]["end"]) * 1000, 6) <= min_silence_duration_ms:
                segments[-1]["end"] = max(segments[-1]["end"], end)
            else:
                segments.append({"start": start, "end": end})

        return [
            segment
            for segment in segments
            if round((segment["end"] - segment["start"]) * 1000, 6) >= min_speech_duration_ms
        ]
