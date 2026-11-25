"""用 Nemotron 3 VAD 和前景音量过滤分段，再用 Qwen3-ASR 生成字幕。"""

import argparse
import math
from pathlib import Path

import numpy as np
import regex
from transformers.audio_utils import load_audio

from scripts.generate_srt_qwen3 import (
    SAMPLE_RATE,
    detect_foreground,
    intersect_intervals,
    load_model,
    prepare_segments,
    transcribe,
    write_srt,
)
from src.n3d_vad import MODEL_NAME as N3D_MODEL_NAME
from src.n3d_vad import N3DVAD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen3-ASR-1.7B-hf")
    parser.add_argument("--audio_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--language", default="none", help="none 表示自动识别语言")
    parser.add_argument("--system_prompt", help="可选的系统提示，例如热词")
    parser.add_argument("--ignore_regex", help="将 regex.fullmatch() 匹配的文本置空")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--vad_model_name", default=N3D_MODEL_NAME, help="N3D 模型名称或本地路径")
    parser.add_argument("--vad_device", help="N3D 推理设备，默认自动选择 CUDA 或 CPU")
    parser.add_argument("--vad_threshold", type=float, default=0.5, help="N3D 说话人活动概率阈值")
    parser.add_argument("--min_speech_duration_ms", type=float, default=500, help="过滤后保留的最短语音时长")
    parser.add_argument("--min_silence_duration_ms", type=float, default=500, help="N3D 合并语音时允许的静音间隔")
    parser.add_argument("--max_segment_s", type=float, default=30.0, help="单段转录的时长上限")
    parser.add_argument("--foreground_peak_dbfs", type=float, default=-30.0, help="前景峰值触发阈值（dBFS）")
    parser.add_argument("--foreground_padding_s", type=float, default=0.15, help="前景区域前后保护时长")
    parser.add_argument("--foreground_merge_gap_s", type=float, default=0.15, help="保护后前景区域的最大合并间隔")
    parser.add_argument("--no_foreground_filter", action="store_true", help="关闭低音量背景声过滤")
    args = parser.parse_args()
    if not 0 <= args.vad_threshold <= 1:
        parser.error("vad_threshold 必须在 0 到 1 之间")
    for name in ("min_speech_duration_ms", "min_silence_duration_ms", "foreground_padding_s", "foreground_merge_gap_s"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"{name} 必须为有限非负数")
    if not math.isfinite(args.foreground_peak_dbfs) or args.foreground_peak_dbfs > 0:
        parser.error("foreground_peak_dbfs 必须为不大于 0 的有限值")
    if not math.isfinite(args.max_segment_s) or round(args.max_segment_s * SAMPLE_RATE) < 1:
        parser.error("max_segment_s 必须为有限值且至少为一个采样点")
    if args.batch_size < 1:
        parser.error("batch_size 必须为正整数")
    args.language = None if args.language.lower() == "none" else args.language
    return args


def detect_speech(audio: np.ndarray, args: argparse.Namespace) -> list[tuple[int, int]]:
    vad = N3DVAD(args.vad_model_name, device=args.vad_device)
    timestamps = vad.get_speech_timestamps(
        audio,
        sampling_rate=SAMPLE_RATE,
        threshold=args.vad_threshold,
        min_silence_duration_ms=args.min_silence_duration_ms,
    )
    return [(round(t["start"] * SAMPLE_RATE), round(t["end"] * SAMPLE_RATE)) for t in timestamps]


def main():
    args = parse_args()
    ignore_regex = regex.compile(args.ignore_regex) if args.ignore_regex else None
    audio = load_audio(str(args.audio_path), sampling_rate=SAMPLE_RATE)
    speech = detect_speech(audio, args)
    if not args.no_foreground_filter:
        foreground = detect_foreground(
            audio,
            args.foreground_peak_dbfs,
            round(args.foreground_padding_s * SAMPLE_RATE),
            round(args.foreground_merge_gap_s * SAMPLE_RATE),
        )
        speech = intersect_intervals(speech, foreground)
    min_samples = max(1, math.ceil(args.min_speech_duration_ms * SAMPLE_RATE / 1000))
    speech = [(start, end) for start, end in speech if end - start >= min_samples]
    segments = prepare_segments(audio, speech, padding=0, max_samples=round(args.max_segment_s * SAMPLE_RATE))
    print(f"待转录片段: {len(segments)}")

    texts = []
    if segments:
        processor, model = load_model(args)
        texts = transcribe(
            audio,
            segments,
            processor,
            model,
            batch_size=args.batch_size,
            language=args.language,
            max_new_tokens=256,
            desc="转录",
            system_prompt=args.system_prompt,
            ignore_regex=ignore_regex,
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_srt(args.output_path, segments, texts)


if __name__ == "__main__":
    main()
