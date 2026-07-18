import argparse
import os
import evaluate
import torch
from datasets import load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
import regex as re


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="calc_cer")
    parser.add_argument("--output_dir", default="./outputs/finetune")

    # model
    parser.add_argument("--model_name", default="openai/whisper-large-v3-turbo")
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
    output_dir = os.path.join(args.output_dir, args.name)
    dtype = torch.bfloat16

    # ---------------- model ----------------
    processor = WhisperProcessor.from_pretrained(args.model_name, local_files_only=True)
    model_config = WhisperConfig.from_pretrained(args.model_name, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name,
        config=model_config,
        dtype=dtype,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
    )

    model.model.requires_grad_(False)
    model.eval()

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
        dataset = load_dataset("KYOU-0/Ace-Taffy-voice", split="train")

    dm = SimpleDataManager(
        dataset=dataset,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
    )

    dataset = dm.get_dataset()

    # ---------------- train ----------------
    metric_cer = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [normalize_text_for_wer(s) for s in pred_str]
        label_str = [normalize_text_for_wer(s) for s in label_str]
        cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)  # type: ignore

        return {"cer": cer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        bf16=True,
        tf32=True,
        # torch_compile=True, # compile 太慢了
        predict_with_generate=True,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_accumulation_steps=args.eval_accumulation_steps,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=dm.get_collator(dtype=dtype),
        args=training_args,
        compute_metrics=compute_metrics,
        eval_dataset=dataset,  # type: ignore
        processing_class=processor,
    )

    eval_results = trainer.evaluate(language="zh", task="transcribe", max_new_tokens=224)
    print(eval_results)


if __name__ == "__main__":
    main()
