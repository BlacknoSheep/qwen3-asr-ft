```bash
uv sync

. .venv/activate

hf download Qwen/Qwen3-ASR-1.7B-hf

python -m scripts.simple_lora

python -m scripts.merge_lora --lora_model=<lora_ckpt_path> --output_path=<output_path>
```