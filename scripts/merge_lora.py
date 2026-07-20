import argparse
import torch
from transformers.models.qwen3_asr import (
    Qwen3ASRProcessor,
    Qwen3ASRForConditionalGeneration,
)
from peft import PeftModel
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-ASR-1.7B-hf")
    parser.add_argument("--lora_model", type=str)
    parser.add_argument("--output_path", type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    base_model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    peft_model = PeftModel.from_pretrained(
        base_model,
        args.lora_model,
    )
    merged_model = peft_model.merge_and_unload()  # type:ignore

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    merged_model.save_pretrained(
        args.output_path,
        safe_serialization=True,
    )

    processor = Qwen3ASRProcessor.from_pretrained(args.base_model, local_files_only=True)
    processor.save_pretrained(args.output_path)


if __name__ == "__main__":
    main()
