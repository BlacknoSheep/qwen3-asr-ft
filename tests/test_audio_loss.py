"""诊断真实音频的训练 loss（需要提供准确转写）。

python tests/test_audio_loss.py --audio_path ./tests/test.wav --transcription "音频的真实转写"
python tests/test_audio_loss.py --transcription "音频的真实转写" --inspect_only

reference labels 根据 assistant 前缀边界构造，不依赖模板的 generation 标记；
包含 assistant 正文以及模板输出的结束符、尾部换行。
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers.audio_utils import load_audio
from transformers.models.qwen3_asr import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor

# 同时支持 python tests/test_audio_loss.py 和 python -m tests.test_audio_loss。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_manager import SimpleCollator
from src.utils import build_messages, normalize_language, normalize_text


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 Qwen3-ASR chat_template、labels 和音频 loss")
    parser.add_argument("--model_name", default="Qwen/Qwen3-ASR-0.6B-hf")
    parser.add_argument("--audio_path", default="./tests/test.wav")
    parser.add_argument("--transcription", required=True, help="音频的真实转写，不使用模型生成结果作为标签")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--inspect_only", action="store_true", help="只检查模板及 labels，不加载模型")
    parser.add_argument("--show_all_tokens", action="store_true", help="同时打印未参与监督的 token")
    args = parser.parse_args()
    transcription = normalize_text(args.transcription)
    if not transcription:
        parser.error("--transcription 不能为空")
    language = normalize_language(args.language)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = Qwen3ASRProcessor.from_pretrained(args.model_name, local_files_only=True)
    audio = load_audio(args.audio_path, sampling_rate=16000)
    messages = build_messages(audio, language, transcription)

    print("=== chat_template ===")
    print(processor.chat_template)
    print("\n=== 渲染文本（音频展开前） ===")
    print(repr(processor.apply_chat_template(messages, tokenize=False)))
    print(f"音频: {args.audio_path}, 时长: {len(audio) / 16000:.3f}s")
    print(f"标准化转写: {transcription!r}, language: {language}")

    # 适配 collator 所需的 AudioDecoder 接口，实际 labels 完全由训练 collator 生成。
    feature = {
        "audio": SimpleNamespace(get_all_samples=lambda: SimpleNamespace(data=torch.from_numpy(audio).unsqueeze(0))),
        "language": language,
        "transcription": transcription,
    }
    batch = SimpleCollator(processor, dtype)([feature])
    prompt = processor.apply_chat_template([messages[:-1]], tokenize=True, return_dict=True, add_generation_prompt=True)
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    prefix_length = prompt["input_ids"].shape[1]
    if prefix_length >= input_ids.shape[1] or not torch.equal(input_ids[:, :prefix_length], prompt["input_ids"]):
        raise RuntimeError("完整输入与 assistant prompt 的 token 前缀不一致，无法可靠构造 reference labels")
    reference = input_ids.clone()
    reference[:, :prefix_length] = -100
    reference[batch["attention_mask"] == 0] = -100

    print("\n=== 输入与监督范围 ===")
    for key, value in batch.items():
        print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
    print(f"assistant 正文起点: {prefix_length}")
    for name, target in (("collator", labels), ("reference", reference)):
        selected = target[0] != -100
        print(f"{name}: 有效 next-token 标签数={int((target[:, 1:] != -100).sum())}")
        print(f"{name} 监督文本: {processor.tokenizer.decode(input_ids[0, selected], skip_special_tokens=False)!r}")

    problems = []
    if not (labels[:, 1:] != -100).any():
        problems.append("所有 next-token labels 都是 -100，loss 无有效监督；检查 generation 标记和 assistant_masks")
    if (labels[batch["attention_mask"] == 0] != -100).any():
        problems.append("padding token 参与了监督")
    differences = (labels != reference).nonzero(as_tuple=False)
    print(f"collator/reference labels 差异数: {len(differences)}")
    if len(differences):
        problems.append(f"assistant mask 与独立前缀边界不一致，差异位置: {differences[:, 1].tolist()}")

    token_losses = None
    if not args.inspect_only:
        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            args.model_name, dtype=dtype, device_map=device, attn_implementation="sdpa", local_files_only=True
        ).eval()
        with torch.inference_mode():
            output = model(**{key: value.to(device) for key, value in batch.items()}, use_cache=False)
            # 位置 i 的 logits 预测位置 i+1 的 token，与 causal LM loss 保持一致。
            token_losses = (
                F.cross_entropy(
                    output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1]),
                    input_ids[:, 1:].to(device).reshape(-1),
                    reduction="none",
                )
                .reshape_as(input_ids[:, 1:])
                .cpu()
            )
        actual_loss = output.loss.detach().float().cpu()
        print("\n=== Loss（自然对数，token 均值） ===")
        print(f"model loss: {actual_loss.item():.8f}")
        for name, target in (("collator", labels), ("reference", reference)):
            active = target[:, 1:] != -100
            manual_loss = token_losses[active].mean()
            print(f"manual {name} loss: {manual_loss.item():.8f}")
            if name == "collator" and active.any():
                if not torch.isfinite(actual_loss):
                    problems.append("模型 loss 不是有限值")
                elif not torch.isclose(actual_loss, manual_loss, rtol=1e-4, atol=1e-5):
                    problems.append("模型 loss 与手动 shifted cross entropy 不一致")
            if name == "reference" and not torch.isfinite(manual_loss):
                problems.append("reference loss 不是有限值")

    print("\n=== Token 明细（位置、ID、collator/reference 是否监督、NLL、token） ===")
    for position, token_id in enumerate(input_ids[0].tolist()):
        supervised = labels[0, position].item() != -100
        expected = reference[0, position].item() != -100
        if not (args.show_all_tokens or supervised or expected):
            continue
        nll = f"{token_losses[0, position - 1].item():.6f}" if token_losses is not None and position else "-"
        token = processor.tokenizer.convert_ids_to_tokens(token_id)
        print(f"{position:5d} {token_id:7d} {int(supervised)}/{int(expected)} {nll:>12} {token!r}")

    if problems:
        print("\n检查失败：")
        for problem in problems:
            print(f"- {problem}")
        raise SystemExit(1)
    print("\n检查通过：监督范围一致。" if args.inspect_only else "\n检查通过：监督范围及 loss 计算一致。")


if __name__ == "__main__":
    main()
