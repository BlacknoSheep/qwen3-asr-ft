"""短窗 ASR 探测边界，再对完整片段重新转录，生成 SRT。"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers.audio_utils import load_audio
from transformers.models.qwen3_asr import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor

SAMPLE_RATE = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen3-ASR-1.7B-hf")
    parser.add_argument("--audio_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--language", default="none", help="none 表示自动识别语言")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--resolution_s", type=float, default=0.5)
    parser.add_argument("--probe_batch_size", type=int, default=512)
    parser.add_argument("--probe_max_new_tokens", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8, help="完整片段转录的 batch 大小")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--padding_s", type=float, default=0.1)
    parser.add_argument("--max_segment_s", type=float, default=30.0)
    parser.add_argument("--debug", action="store_true", help="额外输出 .debug.json 中间结果")
    args = parser.parse_args()
    for name in ("resolution_s", "max_segment_s"):
        value = getattr(args, name)
        if not math.isfinite(value) or round(value * SAMPLE_RATE) < 1:
            parser.error(f"{name} 必须为有限值且至少为一个采样点")
    if not math.isfinite(args.padding_s) or args.padding_s < 0:
        parser.error("padding_s 必须为有限非负数")
    for name in ("probe_batch_size", "probe_max_new_tokens", "batch_size", "max_new_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"{name} 必须为正整数")
    args.language = None if args.language.lower() == "none" else args.language
    return args


def load_model(args):
    processor = Qwen3ASRProcessor.from_pretrained(args.model_name, local_files_only=True)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_name,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map=torch.device("cuda"),
    ).eval()
    return processor, model


def transcribe(
    audio: np.ndarray,
    intervals: list[tuple[int, int]],
    processor,
    model,
    *,
    batch_size: int,
    language: str | None,
    max_new_tokens: int,
    desc: str,
    sort_by_length: bool = False,
) -> list[str]:
    """按需按长度分批，返回与输入片段顺序一致的文本。"""
    results = [""] * len(intervals)
    order = list(range(len(intervals)))
    if sort_by_length:
        order.sort(key=lambda i: intervals[i][1] - intervals[i][0])
    for offset in tqdm(range(0, len(order), batch_size), desc=desc, unit="batch"):
        indices = order[offset : offset + batch_size]
        with torch.inference_mode():
            inputs = processor.apply_transcription_request(
                audio=[audio[intervals[i][0] : intervals[i][1]] for i in indices],
                language=language,
            ).to(model.device, model.dtype)
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            texts = processor.decode(generated_ids, return_format="transcription_only")
        # 释放本批 GPU 张量后再准备下一批，避免两批输入同时占用显存。
        del inputs, output_ids, generated_ids
        for index, text in zip(indices, texts, strict=True):
            text = text.strip()
            results[index] = "" if text == "system" else text
    return results


def find_segments(records: list[tuple[int, int, str]]) -> list[tuple[int, int]]:
    segments = []
    start = None
    end = 0
    for window_start, window_end, text in records:
        if text:
            if start is None:
                start = window_start
            end = window_end
        elif start is not None:
            segments.append((start, end))
            start = None
    if start is not None:
        segments.append((start, end))
    return segments


def choose_cut(audio: np.ndarray, start: int, limit: int, max_samples: int) -> int:
    """在上限前最多 5 秒内找明显低能量的 20ms 帧，否则直接使用上限。"""
    frame = round(0.02 * SAMPLE_RATE)
    search = min(5 * SAMPLE_RATE, max_samples // 6)
    left = max(start + 1, limit - search)
    count = (limit - left) // frame
    if count < 2:
        return limit
    frames = audio[left : left + count * frame].reshape(count, frame)
    energies = np.mean(np.square(frames, dtype=np.float64), axis=1)
    minimum = energies.min()
    # 能量必须比搜索区间的中位数至少低约 6dB，避免在平稳语音中任意提前切。
    if minimum >= np.median(energies) * 0.25:
        return limit
    index = int(np.flatnonzero(energies == minimum)[-1])
    return left + index * frame + frame // 2


def prepare_segments(
    audio: np.ndarray, raw_segments: list[tuple[int, int]], padding: int, max_samples: int
) -> list[tuple[int, int]]:
    result = []
    for i, (raw_start, raw_end) in enumerate(raw_segments):
        left = (raw_segments[i - 1][1] + raw_start) // 2 if i else 0
        right = (raw_end + raw_segments[i + 1][0]) // 2 if i + 1 < len(raw_segments) else len(audio)
        start = max(left, raw_start - padding)
        end = min(right, raw_end + padding)
        while end - start > max_samples:
            cut = choose_cut(audio, start, start + max_samples, max_samples)
            result.append((start, cut))
            start = cut
        result.append((start, end))
    return result


def sample2timestamp(sample: int) -> str:
    milliseconds = round(sample / SAMPLE_RATE * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def write_srt(path: Path, segments: list[tuple[int, int]], texts: list[str]) -> int:
    counter = 0
    with path.open("w", encoding="utf-8") as f:
        for (start, end), text in zip(segments, texts, strict=True):
            if not text:
                continue
            counter += 1
            f.write(f"{counter}\n{sample2timestamp(start)} --> {sample2timestamp(end)}\n{text}\n\n")
    return counter


def main():
    args = parse_args()
    if args.audio_path.resolve() == args.output_path.resolve():
        raise ValueError("音频和 SRT 必须使用不同路径")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = load_audio(str(args.audio_path), sampling_rate=SAMPLE_RATE)
    processor, model = load_model(args)
    resolution = round(args.resolution_s * SAMPLE_RATE)
    windows = [(start, min(start + resolution, len(audio))) for start in range(0, len(audio), resolution)]
    probe_texts = transcribe(
        audio,
        windows,
        processor,
        model,
        batch_size=args.probe_batch_size,
        language=args.language,
        max_new_tokens=args.probe_max_new_tokens,
        desc="探测边界",
    )
    records = [(start, end, text) for (start, end), text in zip(windows, probe_texts, strict=True)]
    raw_segments = find_segments(records)
    padding = round(args.padding_s * SAMPLE_RATE)
    max_samples = round(args.max_segment_s * SAMPLE_RATE)
    segments = prepare_segments(audio, raw_segments, padding, max_samples)
    print(f"探测片段: {len(raw_segments)}；待转录片段: {len(segments)}")
    texts = transcribe(
        audio,
        segments,
        processor,
        model,
        batch_size=args.batch_size,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        desc="完整片段转录",
        sort_by_length=True,
    )
    counter = write_srt(args.output_path, segments, texts)
    if args.debug:
        debug = {
            "sample_rate": SAMPLE_RATE,
            "windows": [{"start": s, "end": e, "text": t} for s, e, t in records],
            "raw_segments": raw_segments,
            "segments": [{"start": s, "end": e, "text": t} for (s, e), t in zip(segments, texts, strict=True)],
        }
        args.output_path.with_suffix(".debug.json").write_text(
            json.dumps(debug, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"已生成 {counter} 条字幕（空白 {len(segments) - counter}）：{args.output_path}")


if __name__ == "__main__":
    main()
