import argparse
from typing import Any

import regex as re
import torch
from tqdm.auto import tqdm
from transformers.models.qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)

from scripts.utils import SAMPLE_RATE
from src.data_manager import SimpleDataManager

NON_LANGUAGE_RE = re.compile(r"[^\p{L}\p{N}]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--model_name", default="Qwen/Qwen3-ASR-1.7B-hf", help="base model or merged model")
    parser.add_argument("--lora_model", type=str, default=None, help="LoRA for the base model")
    parser.add_argument("--attn_implementation", default="sdpa")

    # dataset
    parser.add_argument("--data_file", default="KYOU-0/Ace-Taffy-voice")
    parser.add_argument("--num_proc", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def normalize_text_for_cer(text: str) -> str:
    """保留 Unicode 字母和数字，删除空白、标点及其他非语言字符。"""
    return NON_LANGUAGE_RE.sub("", text).lower()


def edit_distance(prediction: str, reference: str) -> int:
    """计算字符级 Levenshtein 距离，仅保留一行动态规划状态。"""
    if len(prediction) > len(reference):
        prediction, reference = reference, prediction

    previous = list(range(len(prediction) + 1))
    for ref_index, ref_char in enumerate(reference, start=1):
        current = [ref_index]
        for pred_index, pred_char in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[pred_index] + 1,
                    previous[pred_index - 1] + (pred_char != ref_char),
                )
            )
        previous = current
    return previous[-1]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # ---------------- model ----------------
    processor = Qwen3ASRProcessor.from_pretrained(args.model_name, local_files_only=True)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype=dtype,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
    )
    if args.lora_model is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.lora_model)

    model.requires_grad_(False)
    model.to(device)  # type: ignore
    model.eval()

    # ---------------- dataset ----------------
    dm = SimpleDataManager.from_path(
        data_name_or_path=args.data_file,
        processor=processor,
        samplerate=SAMPLE_RATE,
        num_proc=args.num_proc,
    )
    dataset = dm.get_dataset(num_proc=args.num_proc)

    # 不能使用训练 collator：它的 input_ids 包含 assistant 标签，会导致答案泄漏。
    batch_size = args.batch_size
    total_edits = 0
    total_reference_chars = 0

    progress = tqdm(
        range(0, len(dataset), batch_size),
        desc="Calculating CER",
        unit="batch",
    )
    for start in progress:
        rows = dataset[start : start + batch_size]
        conversations = [
            [
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio": audio.get_all_samples().data[0].numpy()}],
                }
            ]
            for audio in rows["audio"]
        ]

        inputs: Any = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        inputs = inputs.to(device, dtype)
        prompt_length = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            generated_ids = model.generate(  # type: ignore
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        # decoder-only generate 返回“输入 + 新 token”，这里只解码 assistant 新输出。
        assistant_ids = generated_ids[:, prompt_length:].cpu()
        predictions = processor.decode(assistant_ids, skip_special_tokens=True)
        for prediction, reference in zip(predictions, rows["transcription"], strict=True):
            # 仅截取转录，不使用会消除重复的解析器；缺少分隔符视为空转录。
            prediction = normalize_text_for_cer(prediction.partition("<asr_text>")[2])
            reference = normalize_text_for_cer(reference)
            total_edits += edit_distance(prediction, reference)
            total_reference_chars += len(reference)

        completed = min(start + batch_size, len(dataset))
        progress.set_postfix(samples=f"{completed}/{len(dataset)}")

    if total_reference_chars == 0:
        raise ValueError("所有标签在删除非语言字符后均为空，无法计算 CER")
    cer = 100 * total_edits / total_reference_chars
    print(f"CER: {cer:.4f}%")


if __name__ == "__main__":
    main()
