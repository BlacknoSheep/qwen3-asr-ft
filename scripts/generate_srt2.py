"""
生成 srt 字幕文件
"""

import argparse
from pathlib import Path
from transformers.audio_utils import load_audio
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from tqdm import tqdm
from transformers.models.qwen3_asr import Qwen3ASRProcessor
from transformers.models.qwen3_asr import Qwen3ASRForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="generate_srt")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--audio_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--language", type=str, default="none", help="none for None")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--vad_window_s", type=float, default=30.0)
    parser.add_argument("--vad_overlap_s", type=float, default=5.0)

    return parser.parse_args()


samplerate = 16000

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
    if args.language.lower() == "none":
        args.language = None

    vad = load_silero_vad(onnx=True)
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
    print(len(timestamps))

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
                language=[args.language] * len(batch_segments),  # type:ignore
            ).to(model.device, model.dtype)
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=256)  # type: ignore
            texts: list[str] = processor.decode(output_ids, return_format="transcription_only")  # type: ignore

            for t, text in zip(batch_timestamps, texts):
                start, end = t["start"], t["end"]
                text = text.strip()
                f.write(f"{counter}\n")
                f.write(f"{sample2timestamp(start)} --> {sample2timestamp(end)}\n")
                f.write(f"{text}\n\n")
                counter += 1


if __name__ == "__main__":
    main()
