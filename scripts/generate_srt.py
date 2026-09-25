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

    return parser.parse_args()


samplerate = 16000

# vad
threshold = 0.2
min_speech_duration_ms = 500
max_speech_duration_s = 30

# stt
device = torch.device("cuda")
dtype = torch.bfloat16


def sample2timestamp(sample: int, sr: int = 16000) -> str:
    milliseconds = round(sample / sr * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


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

    timestamps = get_speech_timestamps(
        torch.from_numpy(audio),
        vad,
        threshold=threshold,
        sampling_rate=samplerate,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
    )  # samples
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
