```bash
uv sync

. .venv/activate

hf download Qwen/Qwen3-ASR-1.7B-hf

python -m scripts.simple_lora

python -m scripts.merge_lora --lora_model=<lora_ckpt_path> --output_path=<output_path>
```

# Result

- Dataset: KYOU-0/Ace-Taffy-voice

| model                  | cer   |
| :--------------------- | :---- |
| Qwen/Qwen3-ASR-0.6B-hf | 32.75 |
| Qwen/Qwen3-ASR-1.7B-hf | 30.03 |
| simple_lora_50-0.6B    |       |
| simple_lora_50-1.7B    | 18.60 |
