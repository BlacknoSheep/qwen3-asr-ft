"""用短窗 ASR 的空白转录探测语音边界，输出诊断 SRT 和可复用的 JSON。"""

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from tqdm import tqdm
from transformers.audio_utils import load_audio
from transformers.models.qwen3_asr import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-ASR-1.7B-hf")
    parser.add_argument("--audio_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--language", type=str, default="none", help="none for None")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--resolution_s", type=float, default=0.5)
    parser.add_argument("--max_new_tokens", type=int, default=64)

    args = parser.parse_args()
    if not math.isfinite(args.resolution_s) or round(samplerate * args.resolution_s) < 1:
        parser.error("resolution_s 必须至少为一个采样点，且为有限值")
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("batch_size 和 max_new_tokens 必须为正整数")
    return args


samplerate = 16000
device = torch.device("cuda")
dtype = torch.bfloat16


def sample2timestamp(sample: int, sr: int = 16000) -> str:
    milliseconds = round(sample / sr * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def make_windows(num_samples: int, resolution: int) -> list[tuple[int, int]]:
    """尾窗独立处理，所有窗口均不超过 resolution，覆盖完整音频。"""
    return [(start, min(start + resolution, num_samples)) for start in range(0, num_samples, resolution)]


def transcribe_windows(
    audio,
    windows: list[tuple[int, int]],
    processor,
    model,
    *,
    batch_size: int,
    language: str | None,
    max_new_tokens: int,
) -> Iterator[tuple[int, int, str]]:
    for offset in tqdm(range(0, len(windows), batch_size), desc="短窗 ASR", unit="batch"):
        batch_windows = windows[offset : offset + batch_size]
        with torch.inference_mode():
            inputs = processor.apply_transcription_request(
                audio=[audio[start:end] for start, end in batch_windows],
                language=language,
            ).to(model.device, model.dtype)
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            # generate 返回输入 + 输出；仅解码新增 token，避免提示词污染空白判定。
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            texts = processor.decode(generated_ids, return_format="transcription_only")

        for (start, end), text in zip(batch_windows, texts, strict=True):
            text = text.strip()
            if text == "system":
                text = ""
            yield start, end, text


def find_segments(records: list[tuple[int, int, str]]) -> list[tuple[int, int]]:
    """每个空白窗口都是边界；保留所有非空短片段，不做平滑或长度过滤。"""
    segments = []
    segment_start = None
    segment_end = 0
    for start, end, text in records:
        if text:
            if segment_start is None:
                segment_start = start
            segment_end = end
        elif segment_start is not None:
            segments.append((segment_start, segment_end))
            segment_start = None
    if segment_start is not None:
        segments.append((segment_start, segment_end))
    return segments


def print_statistics(records: list[tuple[int, int, str]], segments: list[tuple[int, int]], elapsed: float):
    duration = records[-1][1] / samplerate if records else 0
    nonempty = sum(bool(text) for _, _, text in records)
    print(f"音频 {duration:.2f}s；窗口 {len(records)}（非空 {nonempty}）；探测耗时 {elapsed:.2f}s")
    lengths = sorted((end - start) / samplerate for start, end in segments)
    print(f"可切出片段数量: {len(lengths)}；覆盖时长: {sum(lengths):.2f}s")
    if not lengths:
        return
    p95 = lengths[math.ceil(len(lengths) * 0.95) - 1]
    print(f"片段长度: 最短 {lengths[0]:.2f}s / 中位 {median(lengths):.2f}s / P95 {p95:.2f}s / 最长 {lengths[-1]:.2f}s")
    print(f">20s: {sum(t > 20 for t in lengths)}；>30s: {sum(t > 30 for t in lengths)}")
    distribution = Counter(round(length, 2) for length in lengths)
    print("片段长度分布（秒: 数量）:")
    print(" / ".join(f"{length:g}: {count}" for length, count in sorted(distribution.items())))


def main():
    args = parse_args()
    language = None if args.language.lower() == "none" else args.language
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    resolution = round(samplerate * args.resolution_s)
    audio = load_audio(str(args.audio_path), sampling_rate=samplerate)
    windows = make_windows(len(audio), resolution)

    processor = Qwen3ASRProcessor.from_pretrained(args.model_name, local_files_only=True)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_name,
        local_files_only=True,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        device_map=device,
    )
    model.eval()
    started = perf_counter()
    records = []
    with args.output_path.open("w", encoding="utf-8") as f:
        for counter, (start, end, text) in enumerate(
            transcribe_windows(
                audio,
                windows,
                processor,
                model,
                batch_size=args.batch_size,
                language=language,
                max_new_tokens=args.max_new_tokens,
            ),
            start=1,
        ):
            records.append((start, end, text))
            # 诊断 SRT 保留空白窗口；精确边界和空白标记以 JSON 为准。
            f.write(f"{counter}\n{sample2timestamp(start)} --> {sample2timestamp(end)}\n{text}\n\n")
    elapsed = perf_counter() - started
    segments = find_segments(records)
    metadata_path = args.output_path.with_suffix(".segments.json")
    metadata = {
        "audio_path": str(args.audio_path.resolve()),
        "sample_rate": samplerate,
        "num_samples": len(audio),
        "model_name": args.model_name,
        "language": language,
        "resolution_samples": resolution,
        "max_new_tokens": args.max_new_tokens,
        "windows": [{"start": start, "end": end, "text": text} for start, end, text in records],
        "segments": [{"start": start, "end": end} for start, end in segments],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_statistics(records, segments, elapsed)
    print(f"切分结果（采样点，左闭右开）: {metadata_path}")


if __name__ == "__main__":
    main()
