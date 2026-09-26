# 准备

```bash
uv sync

. .venv/bin/activate

hf download Qwen/Qwen3-ASR-1.7B-hf
hf download Qwen/Qwen3-ASR-0.6B-hf
```

## 数据集

1. jsonl 数据

类似以下格式，其中`audio`字段是音频路径（相对于该 jsonl 文件）

```jsonl
{"audio": "audio/000001.wav", "language": "zh", "transcription": "塔菲去哪儿了"}
{"audio": "audio/000002.wav", "language": "zh", "transcription": "砰，塔菲出现了"}
```

可以使用现有模型生成数据集

```bash
# --model_name 为 huggingface 模型名称，或者本地模型路径

# 1. 生成 srt 字幕文件
python -m scripts.generate_srt \
  --model_name="Qwen/Qwen3-ASR-1.7B-hf" \
  --audio_path="./tmp/audio.wav" \
  --output_path="./outputs/data/audio.srt" \
  --language="zh"

# 2. 使用 Aegsub 等软件对字幕进行校对

# 3. 生成数据集
python -m scripts.srt2dataset \
  --srt_path="./outputs/data/audio.srt" \
  --audio_path="./tmp/audio.wav" \
  --output_dir="./outputs/data" \
  --language="zh"
```

2. huggingface dataset

至少包含`audio`，`language`，`transcription`这三列，例如：https://huggingface.co/datasets/KYOU-0/Ace-Taffy-voice

# Finetune

1. 微调 lora

```bash
python -m scripts.simple_lora \
  --name="1.7b" \
  --output_dir="./outputs/lora" \
  --model_name="Qwen/Qwen3-ASR-1.7B-hf" \
  --data_file="./outputs/data/metadata.jsonl"

# hf 数据集：--data_file="KYOU-0/Ace-Taffy-voice"
```

2. 合并 lora 到主模型

```bash
python -m scripts.merge_lora \
  --base_model="Qwen/Qwen3-ASR-1.7B-hf" \
  --lora_model="./outputs/lora/1.7b/checkpoint-50" \
  --output_path="./outputs/merged/1.7b"
```

# Result

- CER，去除空白、标点、特殊符号

| model                  | KYOU-0/Ace-Taffy-voice | 2026-09-25-11_30min |
| :--------------------- | ---------------------: | ------------------: |
| Qwen/Qwen3-ASR-0.6B-hf |                  32.75 |               43.19 |
| Qwen/Qwen3-ASR-1.7B-hf |                  30.03 |               39.12 |
| simple_lora-0.6B       |                  14.95 |               34.86 |
| simple_lora-1.7B       |                  14.10 |               28.69 |

# 推理

## vllm

```bash
uv sync --extra vllm

# WSL2 中，必须设置 VLLM_WSL2_ENABLE_PIN_MEMORY=1 才能启动 vllm
# 启动耗时较长，最后一行输出 `Application startup complete.` 表明启动成功
export VLLM_WSL2_ENABLE_PIN_MEMORY=1 && vllm serve "./outputs/merged/1.7b" \
  --served-model-name qwen3-asr \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --kv-cache-memory-bytes 128M \
  --performance-mode interactivity \
  --host 127.0.0.1 \
  --port 8000

# 测试端口
curl -sS http://localhost:8000/v1/audio/transcriptions \
  -F "model=qwen3-asr" \
  -F "file=@./tests/test.wav" \
  -F "language=zh"
```
