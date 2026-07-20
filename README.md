# 准备

```bash
uv sync

. .venv/bin/activate

hf download Qwen/Qwen3-ASR-1.7B-hf
```

## 数据集

1. jsonl 数据

类似以下格式，其中`audio`字段是音频路径（相对于该 jsonl 文件）

```jsonl
{"audio": "audio/000001.wav", "language": "zh", "transcription": "塔菲去哪儿了"}
{"audio": "audio/000002.wav", "language": "zh", "transcription": "砰，塔菲出现了"}
```

2. huggingface dataset

至少包含`audio`，`language`，`transcription`这三列，例如：https://huggingface.co/datasets/KYOU-0/Ace-Taffy-voice

# Finetune

1. 微调 lora

```bash
python -m scripts.simple_lora
```

# Result

数据集：KYOU-0/Ace-Taffy-voice

| model                  | cer\* |
| :--------------------- | :---- |
| Qwen/Qwen3-ASR-0.6B-hf | 32.75 |
| Qwen/Qwen3-ASR-1.7B-hf | 30.03 |
| simple_lora-0.6B       |       |
| simple_lora-1.7B       | 14.10 |

\*去除空白、标点、特殊符号
