import argparse
import os
from typing import Any

import numpy as np
import regex as re
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers.models.qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)

from scripts.utils import SAMPLE_RATE


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


def _copy_message(message: dict[str, Any]) -> dict[str, Any]:
    """复制 datasets 反序列化的消息，并将音频 list 还原为 ndarray。"""
    message = dict(message)
    content = message.get("content")
    if not isinstance(content, list):
        return message

    copied_content = []
    for block in content:
        if not isinstance(block, dict):
            copied_content.append(block)
            continue
        block = dict(block)
        if block.get("type") == "audio" and isinstance(block.get("audio"), list):
            block["audio"] = np.asarray(block["audio"], dtype=np.float32)
        copied_content.append(block)
    message["content"] = copied_content
    return message


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text", "")) for block in content if isinstance(block, dict) and block.get("type") == "text"
    )


def prepare_eval_conversation(
    messages: list[dict[str, Any]],
    processor: Qwen3ASRProcessor,
) -> tuple[list[dict[str, Any]], str]:
    """将 assistant 标签从生成输入中移除，并返回纯转写标签。"""
    prompt_messages = []
    assistant_parts = []
    for raw_message in messages:
        message = _copy_message(raw_message)
        if message.get("role") == "assistant":
            assistant_parts.append(_message_text(message))
        else:
            prompt_messages.append(message)

    if not assistant_parts:
        raise ValueError("评估样本中没有 assistant 标签")

    assistant_text = "".join(assistant_parts)
    reference = processor.extract_transcription(assistant_text)
    if not isinstance(reference, str):
        raise TypeError("assistant 标签解析结果必须是字符串")
    return prompt_messages, reference


def main() -> None:
    from src.data_manager import CHAT_TEMPLATE, SimpleDataManager

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
    if args.data_file.endswith((".json", ".jsonl")):
        dataset = load_dataset("json", data_files=args.data_file, split="train")
        dataset_dir = os.path.dirname(os.path.abspath(args.data_file))
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
        conversations = []
        references = []
        for messages in rows["messages"]:
            conversation, reference = prepare_eval_conversation(messages, processor)
            conversations.append(conversation)
            references.append(reference)

        inputs: Any = processor.apply_chat_template(
            conversations,
            chat_template=CHAT_TEMPLATE,
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
        predictions = processor.decode(
            assistant_ids,
            return_format="transcription_only",
        )
        if isinstance(predictions, str):
            predictions = [predictions]
        normalized_predictions = [normalize_text_for_cer(s) for s in predictions]  # type: ignore
        normalized_references = [normalize_text_for_cer(s) for s in references]
        total_edits += sum(
            edit_distance(prediction, reference)
            for prediction, reference in zip(
                normalized_predictions,
                normalized_references,
                strict=True,
            )
        )
        total_reference_chars += sum(map(len, normalized_references))

        completed = min(start + batch_size, len(dataset))
        progress.set_postfix(samples=f"{completed}/{len(dataset)}")

    if total_reference_chars == 0:
        raise ValueError("所有标签在删除非语言字符后均为空，无法计算 CER")
    cer = 100 * total_edits / total_reference_chars
    print(f"CER: {cer:.4f}%")


if __name__ == "__main__":
    main()
