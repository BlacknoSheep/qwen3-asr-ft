import regex
import torch
from transformers.models.qwen3_asr.processing_qwen3_asr import resolve_language


def normalize_text(text: str) -> str:
    # qwen支持全角和大小写
    # text = unicodedata.normalize("NFKC", text) # 全角半角统一
    # text = text.lower()
    text = regex.sub(r"[<\[][^>\]]*[>\]]", "", text)  # remove words between brackets
    text = regex.sub(r"\(([^)]+?)\)", "", text)  # remove words between parenthesis
    text = regex.sub(r"\s+", " ", text)  # 合并空白
    text = text.strip()  # 去首尾空白
    return text


def normalize_language(language: str | None) -> str | None:
    return resolve_language(language)


def build_messages(audio, language: str | None, transcription: str | None):
    language = resolve_language(language)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": language}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio": audio,
                },
            ],
        },
    ]
    if transcription is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"language {language}<asr_text>{transcription}"}],
            }
        )
    return messages


def generate_assistant_masks(
    input_ids: torch.Tensor,
    assistant_start_ids: torch.Tensor,
    assistant_end_ids: torch.Tensor,
) -> torch.Tensor:
    """
    为 assistant 回复部分生成 mask。

    规则：
    - 不包括 assistant_start
    - 包括 assistant_end
    - 支持一个序列中出现多段 assistant 回复
    - 未找到对应 assistant_end 时，从 assistant_start 之后一直标记到序列末尾

    Args:
        input_ids: [batch_size, seq_len]
        assistant_start_ids: [start_len] 或 [1, start_len]
        assistant_end_ids: [end_len] 或 [1, end_len]

    Returns:
        assistant_masks: [batch_size, seq_len]
            assistant 内容及 assistant_end 对应位置为 1，其余为 0。
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids 必须是二维张量 [batch_size, seq_len]，实际 shape={tuple(input_ids.shape)}")

    # 兼容 tokenizer.encode(..., return_tensors="pt") 返回的 [1, len]
    assistant_start_ids = assistant_start_ids.reshape(-1).to(
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    assistant_end_ids = assistant_end_ids.reshape(-1).to(
        device=input_ids.device,
        dtype=input_ids.dtype,
    )

    start_len = assistant_start_ids.numel()
    end_len = assistant_end_ids.numel()

    if start_len == 0:
        raise ValueError("assistant_start_ids 不能为空")
    if end_len == 0:
        raise ValueError("assistant_end_ids 不能为空")

    batch_size, seq_len = input_ids.shape
    assistant_masks = torch.zeros_like(input_ids)

    for batch_idx in range(batch_size):
        sequence = input_ids[batch_idx]
        search_pos = 0

        while search_pos <= seq_len - start_len:
            # 查找下一个 assistant_start
            start_candidates = (
                sequence[search_pos:]
                .unfold(0, start_len, 1)
                .eq(assistant_start_ids)
                .all(dim=-1)
                .nonzero(as_tuple=False)
            )

            if start_candidates.numel() == 0:
                break

            start_pos = search_pos + start_candidates[0, 0].item()

            # mask 从 assistant_start 之后开始
            content_start = start_pos + start_len

            # 从内容起始位置查找 assistant_end
            if content_start <= seq_len - end_len:
                end_candidates = (
                    sequence[content_start:]
                    .unfold(0, end_len, 1)
                    .eq(assistant_end_ids)
                    .all(dim=-1)
                    .nonzero(as_tuple=False)
                )
            else:
                end_candidates = torch.empty(
                    (0, 1),
                    dtype=torch.long,
                    device=input_ids.device,
                )

            if end_candidates.numel() == 0:
                # 没有闭合的 assistant_end，标记到序列末尾
                assistant_masks[batch_idx, content_start:] = 1
                break

            end_pos = content_start + end_candidates[0, 0].item()

            # 包括完整的 assistant_end
            mask_end = end_pos + end_len
            assistant_masks[batch_idx, content_start:mask_end] = 1

            # 继续查找下一段 assistant 回复
            search_pos = mask_end

    return assistant_masks
