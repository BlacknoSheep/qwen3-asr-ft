import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.models.qwen3_asr import Qwen3ASRProcessor, Qwen3ASRForConditionalGeneration
from peft import get_peft_model, LoraConfig

from scripts.utils import get_utc_time_str, SAMPLE_RATE
from src.data_manager import SimpleDataManager

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="simple_ft")
    parser.add_argument("--output_dir", default="./outputs/finetune")

    # model
    parser.add_argument("--model_name", default="Qwen/Qwen3-ASR-1.7B-hf")
    parser.add_argument("--attn_implementation", default="sdpa")

    # dataset
    parser.add_argument("--data_file", default="KYOU-0/Ace-Taffy-voice")
    parser.add_argument("--valid_size", type=float, default=0)
    parser.add_argument("--num_proc", type=int, default=8)

    # train
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=float, default=0.1)
    parser.add_argument("--save_steps", type=float, default=0.2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to_wandb", action="store_true")

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    experiment_time_str = get_utc_time_str()
    output_dir = os.path.join(args.output_dir, args.name)
    os.environ["WANDB_DIR"] = os.path.abspath(output_dir)
    dtype = torch.bfloat16

    # ---------------- model ----------------
    processor = Qwen3ASRProcessor.from_pretrained(args.model_name, local_files_only=True)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype=dtype,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
    )

    # Lora
    lora_config = LoraConfig(
        r=64,
        target_modules=["model.model.language_model.*.q_proj", "model.model.language_model.*.v_proj", "lm_head"],
        lora_alpha=64,
        ensure_weight_tying=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # trainable params: 19,709,952 || all params: 2,057,762,432 || trainable%: 0.9578

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

    train_dataset = dm.get_dataset()

    # ---------------- train ----------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        bf16=True,
        tf32=True,
        gradient_checkpointing=args.gradient_checkpointing,
        torch_compile=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=args.save_total_limit,
        prediction_loss_only=True,
        remove_unused_columns=False,
        seed=args.seed,
        report_to="wandb" if args.report_to_wandb else "none",
        run_name=f"{args.name}_{experiment_time_str}",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=dm.get_collator(dtype=dtype),
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
    )

    trainer.train()


if __name__ == "__main__":
    main()
