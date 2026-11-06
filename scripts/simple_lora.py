import argparse
import os

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.models.qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)

from scripts.utils import SAMPLE_RATE
from src.data_manager import SimpleDataManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, default="config/examples/languageQV.yaml")
    parser.add_argument("--name", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # update config
    if args.name is not None:
        config["name"] = args.name
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.model_name_or_path is not None:
        config["model_config"]["model_name_or_path"] = args.model_name_or_path
    if args.data_file is not None:
        config["data_config"]["data_file"] = args.data_file
    if args.per_device_train_batch_size is not None:
        config["training_args"]["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None:
        config["training_args"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.seed is not None:
        config["training_args"]["seed"] = args.seed

    output_dir = os.path.join(config["output_dir"], config["name"])
    os.makedirs(output_dir, exist_ok=True)
    # config screenshot
    with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True)

    os.environ["WANDB_DIR"] = os.path.abspath(output_dir)
    dtype = torch.bfloat16
    model_config = config["model_config"]
    data_config = config["data_config"]
    lora_config = config["lora_config"]
    training_args = config["training_args"]

    # ---------------- model ----------------
    processor = Qwen3ASRProcessor.from_pretrained(
        model_config["model_name_or_path"],
        local_files_only=model_config["local_files_only"],
    )
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_config["model_name_or_path"],
        dtype=dtype,
        local_files_only=model_config["local_files_only"],
        attn_implementation=model_config["attn_implementation"],
    )

    # Lora
    lora_config = LoraConfig(**lora_config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------------- dataset ----------------
    if data_config["data_file"].endswith(".json") or data_config["data_file"].endswith(".jsonl"):
        dataset = load_dataset(
            "json",
            data_files=data_config["data_file"],
            split="train",
        )
        dataset_dir = os.path.dirname(data_config["data_file"])
        dataset = dataset.map(
            lambda x: {"audio": os.path.join(dataset_dir, x["audio"])},
            load_from_cache_file=data_config["load_from_cache_file"],
            num_proc=data_config["num_proc"],
        )
    else:
        dataset = load_dataset(data_config["data_file"], split="train")

    dm = SimpleDataManager(
        dataset=dataset,
        processor=processor,
        samplerate=SAMPLE_RATE,
    )

    train_dataset = dm.get_dataset(
        load_from_cache_file=data_config["load_from_cache_file"], num_proc=data_config["num_proc"]
    )

    # ---------------- train ----------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        bf16=True,
        tf32=True,
        prediction_loss_only=True,
        remove_unused_columns=False,
        **training_args,
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
