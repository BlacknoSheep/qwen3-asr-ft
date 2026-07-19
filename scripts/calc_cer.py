import argparse
import os
import evaluate
import torch
from datasets import load_dataset
from transformers.models.qwen3_asr import (
    Qwen3ASRProcessor,
    Qwen3ASRForConditionalGeneration,
)
import regex as re

from scripts.utils import SAMPLE_RATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--model_name", default="Qwen/Qwen3-ASR-1.7B-hf")
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--attn_implementation", default="sdpa")

    # dataset
    parser.add_argument("--data_file", default="KYOU-0/Ace-Taffy-voice")
    parser.add_argument("--num_proc", type=int, default=8)

    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_accumulation_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def normalize_text_for_wer(text: str) -> str:
    NON_TEXT_RE = re.compile(r"[^\p{L}\p{N}]+")
    text = text.lower()
    # 删除所有非语言符号（标点符号、空白等）
    text = NON_TEXT_RE.sub("", text)
    return text


def main() -> None:
    from src.data_manager import SimpleDataManager

    args = parse_args()
    dtype = torch.bfloat16

    # ---------------- model ----------------
    processor = Qwen3ASRProcessor.from_pretrained(
        args.model_name, local_files_only=True
    )
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype=dtype,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
    )
    if args.lora_model is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            args.lora_model,
        )

    model.model.requires_grad_(False)

    # ---------------- dataset ----------------
    if args.data_file.endswith(".json") or args.data_file.endswith(".jsonl"):
        dataset = load_dataset(
            "json",
            data_files=args.data_file,
            split="train",
        )
        dataset_dir = os.path.dirname(args.data_file)
        dataset = dataset.map(
            lambda x: {"audio": os.path.join(dataset_dir, x["audio"])},
            num_proc=args.num_proc,
        )
    else:
        dataset = load_dataset(args.data_file, split="train")

    dm = SimpleDataManager(
        dataset=dataset,
        processor=processor,
        samplerate=SAMPLE_RATE,
    )

    dataset = dm.get_dataset()

    # ---------------- train ----------------
    metric_cer = evaluate.load("cer")

    def compute_metrics(pred_ids, label_ids):
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        pred_str = [normalize_text_for_wer(s) for s in pred_str]
        label_str = [normalize_text_for_wer(s) for s in label_str]
        cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)  # type: ignore

        return {"cer": cer}


if __name__ == "__main__":
    main()
