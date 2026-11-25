"""
生成 srt 字幕文件
通过滑动窗口重置 vad 状态，漏检率更低
"""

import argparse
import math
from pathlib import Path

import regex
import torch
from silero_vad import get_speech_timestamps, load_silero_vad
from tqdm import tqdm
from transformers.audio_utils import load_audio
from transformers.models.qwen3_asr import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor

from scripts.generate_srt_qwen3 import SAMPLE_RATE, detect_foreground, intersect_intervals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="generate_srt")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-ASR-1.7B-hf")
    parser.add_argument("--audio_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--language", type=str, default="none", help="none for None")
    parser.add_argument("--system_prompt", type=str, default=None, help="可选的系统提示，例如热词")
    parser.add_argument("--ignore_regex", help="跳过 regex.fullmatch() 匹配的字幕")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--vad_window_s", type=float, default=30.0)
    parser.add_argument("--vad_overlap_s", type=float, default=5.0)
    parser.add_argument("--foreground_peak_dbfs", type=float, default=-30.0, help="前景峰值触发阈值（dBFS）")
    parser.add_argument("--foreground_padding_s", type=float, default=0.15, help="前景区域前后保护时长")
    parser.add_argument("--foreground_merge_gap_s", type=float, default=0.15, help="保护后前景区域的最大合并间隔")
    parser.add_argument("--no_foreground_filter", action="store_true", help="关闭低音量背景声过滤")

    args = parser.parse_args()
    if not math.isfinite(args.foreground_peak_dbfs) or args.foreground_peak_dbfs > 0:
        parser.error("foreground_peak_dbfs 必须为不大于 0 的有限值")
    for name in ("foreground_padding_s", "foreground_merge_gap_s"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"{name} 必须为有限非负数")
    return args


samplerate = SAMPLE_RATE

# vad
threshold = 0.2
min_speech_duration_ms = 500
max_speech_duration_s = 30
min_silence_duration_ms = 500

# stt
device = torch.device("cuda")
dtype = torch.bfloat16


def sample2timestamp(sample: int, sr: int = 16000) -> str:
    milliseconds = round(sample / sr * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def get_speech_timestamps_chunked(
    audio,
    vad,
    *,
    sampling_rate: int = 16000,
    window_s: float = 30.0,
    overlap_s: float = 5.0,
) -> list[dict[str, int]]:
    """Use window centers for VAD; return original-audio sample offsets."""
    if window_s <= 0 or overlap_s < 0 or overlap_s >= window_s:
        raise ValueError("Require window_s > overlap_s >= 0")

    window = round(window_s * sampling_rate)
    step = round((window_s - overlap_s) * sampling_rate)
    if window <= 0 or step <= 0:
        raise ValueError("VAD windows are too short")

    def detect(start: int, stop: int) -> list[dict[str, int]]:
        # Every call resets the Silero model's hidden state.
        chunk = torch.as_tensor(audio[start:stop], dtype=torch.float32).clone()
        return get_speech_timestamps(
            chunk,
            vad,
            threshold=threshold,
            sampling_rate=sampling_rate,
            min_speech_duration_ms=min_speech_duration_ms,
            max_speech_duration_s=max_speech_duration_s,
            min_silence_duration_ms=min_silence_duration_ms,
            return_seconds=False,
        )

    candidates: list[dict[str, int]] = []
    for offset in range(0, len(audio), step):
        end = min(offset + window, len(audio))
        # Partition each overlap at its midpoint. Only one window owns each
        # sample, so an edge detection cannot bridge another window's silence.
        owned_start = 0 if offset == 0 else (offset + offset - step + window) // 2
        owned_end = len(audio) if end == len(audio) else (end + offset + step) // 2
        for segment in detect(offset, end):
            start = max(owned_start, offset + int(segment["start"]))
            stop = min(owned_end, offset + int(segment["end"]))
            if stop > start:
                candidates.append({"start": start, "end": stop})
        if end == len(audio):
            break

    # Rejoin speech crossing an ownership boundary, preserving every silence.
    candidates.sort(key=lambda s: (s["start"], s["end"]))
    merged: list[dict[str, int]] = []
    for segment in candidates:
        if merged and segment["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], segment["end"])
        else:
            merged.append(segment.copy())

    # Re-run VAD on long joined speech with uninterrupted context. Silero
    # prefers short silences when enforcing max_speech_duration_s.
    max_samples = round(max_speech_duration_s * sampling_rate)
    if max_samples <= 0:
        raise ValueError("max_speech_duration_s must be positive")
    result: list[dict[str, int]] = []
    for segment in merged:
        start, stop = segment["start"], segment["end"]
        if stop - start <= max_samples:
            result.append(segment)
            continue
        for local in detect(start, stop):
            a = start + int(local["start"])
            b = min(stop, start + int(local["end"]))
            # Enforce the exact cap even if VAD padding exceeds it. Do not
            # redistribute all boundaries by dividing the interval equally.
            for part_start in range(a, b, max_samples):
                result.append({"start": part_start, "end": min(part_start + max_samples, b)})
    return result


def main():
    args = parse_args()
    ignore_regex = regex.compile(args.ignore_regex) if args.ignore_regex else None
    if args.language.lower() == "none":
        args.language = None

    vad = load_silero_vad()
    processor = Qwen3ASRProcessor.from_pretrained(args.model_name, local_files_only=True)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_name,
        local_files_only=True,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        device_map=device,
    )
    model.eval()

    audio = load_audio(args.audio_path, sampling_rate=samplerate)

    timestamps = get_speech_timestamps_chunked(
        audio,
        vad,
        sampling_rate=samplerate,
        window_s=args.vad_window_s,
        overlap_s=args.vad_overlap_s,
    )  # original-audio samples, suitable for ASR and SRT
    if not args.no_foreground_filter:
        foreground = detect_foreground(
            audio,
            args.foreground_peak_dbfs,
            round(args.foreground_padding_s * samplerate),
            round(args.foreground_merge_gap_s * samplerate),
        )
        speech = [(t["start"], t["end"]) for t in timestamps]
        min_samples = round(min_speech_duration_ms * samplerate / 1000)
        timestamps = [
            {"start": start, "end": end}
            for start, end in intersect_intervals(speech, foreground)
            if end - start >= min_samples
        ]
    print(f"待转录片段: {len(timestamps)}")

    # 生成 srt 字幕文件
    # batch stt
    num_segments = len(timestamps)
    num_batches = (num_segments + args.batch_size - 1) // args.batch_size
    counter = 1
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        for batch_idx in tqdm(range(num_batches)):
            batch_timestamps = timestamps[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
            batch_segments = []
            for t in batch_timestamps:
                start, end = t["start"], t["end"]
                segment = audio[start:end]
                batch_segments.append(segment)

            inputs = processor.apply_transcription_request(
                audio=batch_segments,
                prompt=[args.system_prompt] * len(batch_segments),
                language=[args.language] * len(batch_segments),  # type:ignore
            ).to(model.device, model.dtype)
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=256)  # type: ignore
            texts: list[str] = processor.decode(output_ids, return_format="transcription_only")  # type: ignore

            for t, text in zip(batch_timestamps, texts):
                start, end = t["start"], t["end"]
                text = text.strip()
                if ignore_regex is not None and ignore_regex.fullmatch(text):
                    continue
                f.write(f"{counter}\n")
                f.write(f"{sample2timestamp(start)} --> {sample2timestamp(end)}\n")
                f.write(f"{text}\n\n")
                counter += 1


if __name__ == "__main__":
    main()
