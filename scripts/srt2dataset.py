"""
根据 srt 字幕文件和音频文件，生成音频-文本数据集
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据 SRT 字幕和完整音频切分生成 Whisper 训练用 JSONL 数据集。")
    parser.add_argument("--srt_path", type=str)
    parser.add_argument("--audio_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--language", type=str)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--metadata_only", action="store_true", help="仅生成 metadata.jsonl，不切分音频")
    return parser.parse_args()


@dataclass
class Subtitle:
    start: float
    end: float
    text: str


TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})"
)


def parse_timestamp(value: str) -> float:
    hours, minutes, rest = value.replace(",", ".").split(":")
    seconds, milliseconds = rest.split(".")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000


def parse_srt(path: Path) -> list[Subtitle]:
    content = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", content.strip())
    subtitles: list[Subtitle] = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        timestamp_index = next((idx for idx, line in enumerate(lines) if TIMESTAMP_RE.search(line)), None)
        if timestamp_index is None:
            continue

        match = TIMESTAMP_RE.search(lines[timestamp_index])
        if match is None:
            continue

        text = " ".join(lines[timestamp_index + 1 :]).strip()
        if not text:
            continue

        start = parse_timestamp(match.group("start"))
        end = parse_timestamp(match.group("end"))
        if end <= start:
            continue

        subtitles.append(Subtitle(start=start, end=end, text=text))

    return subtitles


def build_dataset(
    srt_path: Path,
    audio_path: Path,
    output_dir: Path,
    language: str,
    sample_rate: int,
    metadata_only: bool = False,
) -> None:
    subtitles = parse_srt(srt_path)
    if not subtitles:
        raise ValueError(f"No valid subtitles found in {srt_path}.")

    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio_dir = output_dir / "audio"
    json_path = output_dir / "metadata.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as f:
        for idx, subtitle in enumerate(tqdm(subtitles), start=1):
            start = round(subtitle.start * sample_rate)
            end = round(subtitle.end * sample_rate)
            if start >= len(audio):
                continue
            end = min(end, len(audio))
            if end <= start:
                continue

            segment_path = audio_dir / f"{idx:06d}.wav"
            if not metadata_only:
                sf.write(segment_path, audio[start:end], sample_rate)

            record = {
                "audio": str(segment_path.relative_to(json_path.parent)),
                "language": language,
                "transcription": subtitle.text,
                "start_time": subtitle.start,
                "end_time": subtitle.end,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    build_dataset(
        srt_path=Path(args.srt_path),
        audio_path=Path(args.audio_path),
        output_dir=Path(args.output_dir),
        language=args.language,
        sample_rate=args.sample_rate,
        metadata_only=args.metadata_only,
    )


if __name__ == "__main__":
    main()
