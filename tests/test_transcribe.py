import argparse

import torch
from transformers.audio_utils import load_audio
from transformers.models.qwen3_asr import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-ASR 音频转写")
    parser.add_argument("--model_name", required=True, help="Hugging Face 模型名称或本地模型路径")
    parser.add_argument("--audio_path", required=True, help="音频文件路径")
    parser.add_argument("--language", default="none", help="语言，例如 zh、en；默认 none 自动识别")
    args = parser.parse_args()

    language = None if args.language.lower() == "none" else args.language
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    processor = Qwen3ASRProcessor.from_pretrained(args.model_name)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype=dtype,
        device_map=device,
        attn_implementation="sdpa",
    ).eval()

    audio = load_audio(args.audio_path, sampling_rate=16000)
    inputs = processor.apply_transcription_request(audio=audio, language=language).to(model.device, model.dtype)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :].cpu()
    text = processor.decode(generated_ids, return_format="transcription_only")
    print(text.strip())


if __name__ == "__main__":
    main()
