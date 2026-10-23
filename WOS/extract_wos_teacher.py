#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WOS Teacher 离线缓存提取脚本

目标
====
只让训练好的 8B Teacher 完整前向一次，把 Student 后续蒸馏需要的 Teacher 信息统一缓存：

1) Traditional logits KD 所需信息
   - 不直接保存 128K 维 logits。
   - 保存所有 Completion causal prediction positions 的 Teacher final hidden states：
         H_T_KD: [N_completion_positions, teacher_hidden_size]
   - 另外保存训练后 Teacher 的 common-vocab lm_head 行：
         W_T_common: [V_common, teacher_hidden_size]
   - Student 训练时恢复：
         z_T_common = H_T_KD @ W_T_common^T (+ bias)
     然后继续做原来的 temperature-scaled KL。
   - 因此 Traditional KD 仍然是 logits KD，不是 hidden-state KD。

2) Hidden-state Align / Hier 所需信息
   - Completion 中第二次 L1 special token 的 Teacher final hidden
   - Completion 中第二次 L2 special token 的 Teacher final hidden
   - 不重复额外保存两份向量，而是保存它们在该样本 H_T_KD 中的相对 offset：
         l1_rel_offset
         l2_rel_offset
     训练时：
         h_T_L1 = H_T_KD[sample_start + l1_rel_offset]
         h_T_L2 = H_T_KD[sample_start + l2_rel_offset]

关键一致性
========
- 用 Student tokenizer 构造完整训练序列；
- 再按“相同 token 字符串”把 Student token ID 1→1 翻译到 Teacher ID；
- Teacher 不独立重新 tokenize 原文本，因此 Student/Teacher 序列位置严格一致；
- Prompt、Completion、截断规则与原训练代码保持一致；
- L1/L2 都提取 Completion 中第二次 occurrence；
- Teacher 处于 eval + inference_mode，不更新参数。

缓存格式
========
save_dir/
  manifest.json
  offsets.npy                    int64 [num_samples + 1]
  valid_counts.npy               int32 [num_samples]
  l1_rel_offsets.npy             int32 [num_samples]
  l2_rel_offsets.npy             int32 [num_samples]
  student_input_sha256.npy       S64   [num_samples]
  teacher_completion_hidden.npy  float16/float32 [total_valid_positions, hidden_size]

  student_common_ids.npy         int64 [V_common]
  teacher_common_ids.npy         int64 [V_common]
  teacher_common_lm_head.npy     float16/float32 [V_common, hidden_size]
  teacher_common_lm_head_bias.npy (仅当 Teacher lm_head 有 bias)

后续 Student Dataset 只要保留 sample_index，即使 DataLoader shuffle，
也能通过 offsets[sample_index] 精确读取对应 Teacher cache。
"""

import os

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import random
import shutil
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed as hf_set_seed


# ======================================================================
# 0. 固定随机种子
# ======================================================================
def set_all_seeds(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    hf_set_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# ======================================================================
# 1. 与现有 WOS 训练完全一致的特殊 token / Prompt
# ======================================================================
CTX_START_TOKEN = "<|reserved_special_token_110|>"
CTX_END_TOKEN = "<|reserved_special_token_111|>"
PROBE_TOKEN_L1 = "<|reserved_special_token_112|>"
PROBE_TOKEN_L2 = "<|reserved_special_token_113|>"

REQUIRED_CLEAN_FIELDS = (
    "text",
    "l1_label",
    "l2_label",
    "l1_id",
    "l2_id",
)


def install_student_chat_template(tokenizer) -> None:
    """必须与当前 Student/KD 训练脚本保持一致。"""
    tokenizer.chat_template = (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + "
        "message['role'] + '<|end_header_id|>\\n\\n' + "
        "message['content'] | trim + '<|eot_id|>' %}"
        "{% if loop.index0 == 0 %}"
        "{% set content = bos_token + content %}"
        "{% endif %}"
        "{{ content }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant"
        "<|end_header_id|>\\n\\n' }}"
        "{% endif %}"
    )


def build_user_content(text: str) -> str:
    return (
        f"{CTX_START_TOKEN}\n"
        f"{text}\n"
        f"{CTX_END_TOKEN}\n\n"
        "Classify the document according to the WOS two-level taxonomy.\n\n"
        "Hierarchy definitions:\n"
        "- Macro Domain (L1): the top-level research domain.\n"
        "- Micro Sub-field (L2): the fine-grained research sub-field "
        "under the corresponding L1 domain.\n\n"
        "Hierarchy-token assignments:\n"
        "The following Llama reserved special token is dedicated to "
        "representing the document's Macro Domain (L1) information:\n"
        f"{PROBE_TOKEN_L1}\n\n"
        "The following Llama reserved special token is dedicated to "
        "representing the document's Micro Sub-field (L2) information:\n"
        f"{PROBE_TOKEN_L2}\n\n"
        "Output exactly two lines.\n"
        "On the first line, reproduce the L1 reserved special token shown "
        "above, followed by one space and the correct L1 label.\n"
        "On the second line, reproduce the L2 reserved special token shown "
        "above, followed by one space and the correct L2 label.\n"
        "Stop immediately after the second line.\n"
        "Do not output numbering, colons, explanations, or any other text."
    )


def build_completion_text(l1_label: str, l2_label: str) -> str:
    return (
        f"{PROBE_TOKEN_L1} {l1_label}\n"
        f"{PROBE_TOKEN_L2} {l2_label}\n"
    )


def build_prompt_messages(text: str, system_prompt: str) -> List[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_content(text)},
    ]


def truncate_text_for_prompt(
    text: str,
    tokenizer,
    max_seq_length: int,
    reserve_tokens: int,
) -> str:
    max_text_len = max_seq_length - reserve_tokens
    if max_text_len <= 0:
        raise ValueError(
            f"max_seq_length={max_seq_length} 必须大于 reserve_tokens={reserve_tokens}"
        )

    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(text_tokens) > max_text_len:
        text_tokens = text_tokens[:max_text_len]
        text = tokenizer.decode(
            text_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return text


# ======================================================================
# 2. 数据读取
# ======================================================================
def load_json_or_jsonl(path: str) -> List[dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        raise ValueError(f"数据文件为空: {path}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        rows = []
        for line_no, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise TypeError(
                    f"JSONL 第 {line_no} 行必须是 dict，实际={type(item).__name__}"
                )
            rows.append(item)
        return rows

    if not isinstance(data, list):
        raise TypeError(f"JSON 顶层必须是 list，实际={type(data).__name__}")
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"JSON 第 {idx} 条必须是 dict。")
    return data


def get_clean_sample_fields(
    item: dict,
    split_name: str,
    idx: int,
) -> Tuple[str, str, str, int, int]:
    missing = [k for k in REQUIRED_CLEAN_FIELDS if k not in item]
    if missing:
        raise KeyError(
            f"{split_name} 第 {idx} 条缺少字段 {missing}；"
            f"必须包含 {list(REQUIRED_CLEAN_FIELDS)}"
        )

    text = item["text"]
    l1_label = item["l1_label"]
    l2_label = item["l2_label"]
    l1_id = item["l1_id"]
    l2_id = item["l2_id"]

    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{split_name} 第 {idx} 条 text 必须是非空字符串。")
    if not isinstance(l1_label, str) or not l1_label.strip():
        raise ValueError(f"{split_name} 第 {idx} 条 l1_label 必须是非空字符串。")
    if not isinstance(l2_label, str) or not l2_label.strip():
        raise ValueError(f"{split_name} 第 {idx} 条 l2_label 必须是非空字符串。")
    if isinstance(l1_id, bool) or not isinstance(l1_id, int):
        raise TypeError(f"{split_name} 第 {idx} 条 l1_id 必须是 int。")
    if isinstance(l2_id, bool) or not isinstance(l2_id, int):
        raise TypeError(f"{split_name} 第 {idx} 条 l2_id 必须是 int。")

    return (
        text.strip(),
        l1_label.strip(),
        l2_label.strip(),
        int(l1_id),
        int(l2_id),
    )


# ======================================================================
# 3. Tokenizer 对齐
# ======================================================================
def validate_atomic_reserved_tokens(tokenizer, name: str) -> Dict[str, int]:
    token_map = {
        "ctx_start": CTX_START_TOKEN,
        "ctx_end": CTX_END_TOKEN,
        "l1": PROBE_TOKEN_L1,
        "l2": PROBE_TOKEN_L2,
    }

    vocab = tokenizer.get_vocab()
    ids = {}

    print("\n" + "=" * 80)
    print(f"验证 {name} 四个 WOS reserved tokens")

    for role, token in token_map.items():
        if token not in vocab:
            raise RuntimeError(f"{name} 缺少 token: {token}")

        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(
                f"{name} 的 {token} 不是单 token，encoded={encoded}"
            )

        token_id = int(encoded[0])
        if token_id == tokenizer.unk_token_id:
            raise RuntimeError(f"{name} 的 {token} 被编码成 unk。")

        ids[role] = token_id
        print(f"{role:>9s}: {token} -> {token_id}")

    if len(set(ids.values())) != 4:
        raise RuntimeError(f"{name} 四个 reserved token ID 发生重复: {ids}")

    print("=" * 80)
    return ids


def build_teacher_student_vocab_alignment(
    student_tokenizer,
    teacher_tokenizer,
) -> dict:
    """
    按 token 字符串语义对齐，而不是按数字 ID 对齐。
    common token 按 Student ID 排序，保证后续 Student/Teacher logits 列一一对应。
    """
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()

    common_tokens = set(student_vocab) & set(teacher_vocab)
    if not common_tokens:
        raise RuntimeError("Teacher / Student 没有共同 token。")

    protected_tokens = [
        CTX_START_TOKEN,
        CTX_END_TOKEN,
        PROBE_TOKEN_L1,
        PROBE_TOKEN_L2,
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ]

    for token in protected_tokens:
        if token not in student_vocab or token not in teacher_vocab:
            raise RuntimeError(f"Teacher/Student 至少一边缺少必需 token: {token}")

        s_ids = student_tokenizer.encode(token, add_special_tokens=False)
        t_ids = teacher_tokenizer.encode(token, add_special_tokens=False)
        if len(s_ids) != 1 or len(t_ids) != 1:
            raise RuntimeError(
                f"必需 token 必须在两边都是原子 token: {token}; "
                f"student={s_ids}, teacher={t_ids}"
            )

    common_sorted = sorted(common_tokens, key=lambda tok: student_vocab[tok])

    student_common_ids = np.asarray(
        [int(student_vocab[tok]) for tok in common_sorted],
        dtype=np.int64,
    )
    teacher_common_ids = np.asarray(
        [int(teacher_vocab[tok]) for tok in common_sorted],
        dtype=np.int64,
    )

    student_to_teacher = np.full(
        len(student_tokenizer),
        -1,
        dtype=np.int64,
    )
    for tok in common_sorted:
        student_to_teacher[int(student_vocab[tok])] = int(teacher_vocab[tok])

    common_token_sha256 = hashlib.sha256(
        "\n".join(common_sorted).encode("utf-8")
    ).hexdigest()

    student_only = sorted(set(student_vocab) - set(teacher_vocab))
    teacher_only = sorted(set(teacher_vocab) - set(student_vocab))

    print("\n" + "=" * 88)
    print("Teacher / Student token-string alignment")
    print(f"Student vocab : {len(student_vocab):,}")
    print(f"Teacher vocab : {len(teacher_vocab):,}")
    print(f"Common vocab  : {len(common_sorted):,}")
    print(f"Student-only  : {len(student_only):,}")
    print(f"Teacher-only  : {len(teacher_only):,}")
    print(f"common token list SHA256: {common_token_sha256}")
    print("=" * 88 + "\n")

    return {
        "student_common_ids": student_common_ids,
        "teacher_common_ids": teacher_common_ids,
        "student_to_teacher": student_to_teacher,
        "common_token_count": len(common_sorted),
        "common_token_sha256": common_token_sha256,
        "student_only_tokens": student_only,
        "teacher_only_tokens": teacher_only,
    }


# ======================================================================
# 4. 构造单样本：完全复制训练序列逻辑
# ======================================================================
def sha256_token_ids(token_ids: List[int]) -> str:
    payload = ",".join(str(int(x)) for x in token_ids)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def find_all_positions(ids: List[int], token_id: int) -> List[int]:
    return [i for i, x in enumerate(ids) if int(x) == int(token_id)]


def build_one_sample(
    item: dict,
    idx: int,
    student_tokenizer,
    student_to_teacher: np.ndarray,
    student_token_ids: Dict[str, int],
    system_prompt: str,
    max_seq_length: int,
    reserve_tokens: int,
) -> dict:
    text, l1_label, l2_label, _, _ = get_clean_sample_fields(
        item,
        "TRAIN",
        idx,
    )

    text = truncate_text_for_prompt(
        text=text,
        tokenizer=student_tokenizer,
        max_seq_length=max_seq_length,
        reserve_tokens=reserve_tokens,
    )

    messages = build_prompt_messages(text, system_prompt)

    prompt_ids = student_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_ids = [int(x) for x in prompt_ids]

    completion_ids = student_tokenizer.encode(
        build_completion_text(l1_label, l2_label),
        add_special_tokens=False,
    )
    completion_ids = [int(x) for x in completion_ids]

    eot_id = student_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is None or eot_id == student_tokenizer.unk_token_id:
        eot_id = student_tokenizer.eos_token_id
    if eot_id is None:
        raise RuntimeError("Student tokenizer 缺少 EOT/EOS。")

    completion_ids.append(int(eot_id))

    student_input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids

    if len(student_input_ids) > max_seq_length:
        raise RuntimeError(
            f"TRAIN 第 {idx} 条最终长度 {len(student_input_ids)} "
            f"> max_seq_length={max_seq_length}"
        )

    # 严格布局检查
    l1_positions = find_all_positions(
        student_input_ids,
        student_token_ids["l1"],
    )
    l2_positions = find_all_positions(
        student_input_ids,
        student_token_ids["l2"],
    )

    if len(l1_positions) != 2 or len(l2_positions) != 2:
        raise RuntimeError(
            f"TRAIN 第 {idx} 条 L1/L2 occurrence 异常: "
            f"L1={l1_positions}, L2={l2_positions}"
        )

    l1_pos = int(l1_positions[1])
    l2_pos = int(l2_positions[1])

    if not (l1_positions[0] < l2_positions[0] < l1_pos < l2_pos):
        raise RuntimeError(
            f"TRAIN 第 {idx} 条 special token 顺序异常: "
            f"L1={l1_positions}, L2={l2_positions}"
        )

    # Causal shift:
    # logits[:, :-1, :] <-> labels[:, 1:]
    shift_labels = labels[1:]
    valid_positions = [
        pos for pos, target in enumerate(shift_labels)
        if int(target) != -100
    ]

    if not valid_positions:
        raise RuntimeError(f"TRAIN 第 {idx} 条没有有效 Completion prediction positions。")

    valid_pos_to_rel = {
        int(pos): rel for rel, pos in enumerate(valid_positions)
    }

    if l1_pos not in valid_pos_to_rel:
        raise RuntimeError(
            f"TRAIN 第 {idx} 条第二次 L1 token position={l1_pos} "
            "不在 KD causal mask 中。"
        )
    if l2_pos not in valid_pos_to_rel:
        raise RuntimeError(
            f"TRAIN 第 {idx} 条第二次 L2 token position={l2_pos} "
            "不在 KD causal mask 中。"
        )

    l1_rel_offset = int(valid_pos_to_rel[l1_pos])
    l2_rel_offset = int(valid_pos_to_rel[l2_pos])

    # Student IDs -> Teacher IDs，一一按 token string 翻译
    student_ids_np = np.asarray(student_input_ids, dtype=np.int64)

    if student_ids_np.min() < 0 or student_ids_np.max() >= len(student_to_teacher):
        raise RuntimeError(f"TRAIN 第 {idx} 条 Student ID 超出映射表范围。")

    teacher_ids_np = student_to_teacher[student_ids_np]
    if np.any(teacher_ids_np < 0):
        bad = np.unique(student_ids_np[teacher_ids_np < 0]).tolist()
        raise RuntimeError(
            f"TRAIN 第 {idx} 条使用了 Teacher 不存在的 Student token: {bad[:30]}"
        )

    return {
        "student_input_ids": student_input_ids,
        "teacher_input_ids": teacher_ids_np.tolist(),
        "valid_positions": valid_positions,
        "valid_count": len(valid_positions),
        "l1_rel_offset": l1_rel_offset,
        "l2_rel_offset": l2_rel_offset,
        "student_input_sha256": sha256_token_ids(student_input_ids),
    }


# ======================================================================
# 5. cache 预扫描
# ======================================================================
def prescan_dataset(
    data: List[dict],
    student_tokenizer,
    alignment: dict,
    student_token_ids: Dict[str, int],
    system_prompt: str,
    max_seq_length: int,
    reserve_tokens: int,
    scan_limit: int,
):
    num_samples = len(data) if scan_limit < 0 else min(len(data), scan_limit)
    if num_samples <= 0:
        raise ValueError("实际提取样本数必须 > 0。")

    valid_counts = np.zeros(num_samples, dtype=np.int32)
    l1_rel_offsets = np.zeros(num_samples, dtype=np.int32)
    l2_rel_offsets = np.zeros(num_samples, dtype=np.int32)
    input_hashes = np.empty(num_samples, dtype="S64")

    print("\n" + "=" * 88)
    print(f"第一遍预扫描：{num_samples:,} 条")
    print("目的：确定每条 Completion KD position 数、L1/L2 offset 和总缓存大小")
    print("=" * 88)

    for idx in tqdm(range(num_samples), desc="Prescan"):
        sample = build_one_sample(
            item=data[idx],
            idx=idx,
            student_tokenizer=student_tokenizer,
            student_to_teacher=alignment["student_to_teacher"],
            student_token_ids=student_token_ids,
            system_prompt=system_prompt,
            max_seq_length=max_seq_length,
            reserve_tokens=reserve_tokens,
        )

        valid_counts[idx] = sample["valid_count"]
        l1_rel_offsets[idx] = sample["l1_rel_offset"]
        l2_rel_offsets[idx] = sample["l2_rel_offset"]
        input_hashes[idx] = sample["student_input_sha256"].encode("ascii")

    offsets = np.zeros(num_samples + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(valid_counts, dtype=np.int64)

    total_positions = int(offsets[-1])

    if total_positions <= 0:
        raise RuntimeError("总 Completion KD position 数为 0。")

    if np.any(l1_rel_offsets < 0) or np.any(l1_rel_offsets >= valid_counts):
        raise RuntimeError("存在非法 L1 relative offset。")
    if np.any(l2_rel_offsets < 0) or np.any(l2_rel_offsets >= valid_counts):
        raise RuntimeError("存在非法 L2 relative offset。")

    print("\n预扫描完成")
    print(f"样本数: {num_samples:,}")
    print(f"总有效 Completion prediction positions: {total_positions:,}")
    print(f"平均每样本有效位置数: {total_positions / num_samples:.3f}")
    print(
        f"每样本有效位置范围: "
        f"{int(valid_counts.min())} ~ {int(valid_counts.max())}"
    )

    return {
        "num_samples": num_samples,
        "valid_counts": valid_counts,
        "offsets": offsets,
        "l1_rel_offsets": l1_rel_offsets,
        "l2_rel_offsets": l2_rel_offsets,
        "input_hashes": input_hashes,
        "total_positions": total_positions,
    }


# ======================================================================
# 6. 保存 Teacher common-vocab lm_head
# ======================================================================
def numpy_cache_dtype(name: str):
    name = str(name).lower()
    if name == "float16":
        return np.float16
    if name == "float32":
        return np.float32
    raise ValueError(f"cache_dtype 只支持 float16/float32，实际={name}")


def save_teacher_common_lm_head(
    teacher_model,
    teacher_common_ids: np.ndarray,
    save_dir: str,
    cache_dtype: str,
    row_chunk_size: int,
):
    """
    按块把 Teacher lm_head 的 common-vocab 行写到磁盘，避免一次性在 GPU 上
    额外构造约 1GB 的完整 common weight tensor。
    """
    teacher = teacher_model
    lm_head = teacher.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Teacher 缺少 lm_head/output embeddings。")

    weight = lm_head.weight
    hidden_size = int(weight.shape[1])
    common_size = int(len(teacher_common_ids))

    np_dtype = numpy_cache_dtype(cache_dtype)
    weight_path = os.path.join(save_dir, "teacher_common_lm_head.npy")

    mmap = np.lib.format.open_memmap(
        weight_path,
        mode="w+",
        dtype=np_dtype,
        shape=(common_size, hidden_size),
    )

    device = weight.device
    print("\n" + "=" * 88)
    print("保存 Teacher common-vocab lm_head")
    print(f"common vocab : {common_size:,}")
    print(f"hidden size  : {hidden_size}")
    print(f"dtype        : {cache_dtype}")
    print(f"path         : {weight_path}")
    print("=" * 88)

    with torch.inference_mode():
        for start in tqdm(
            range(0, common_size, row_chunk_size),
            desc="Save lm_head",
        ):
            end = min(start + row_chunk_size, common_size)

            ids = torch.as_tensor(
                teacher_common_ids[start:end],
                dtype=torch.long,
                device=device,
            )

            rows = weight.index_select(0, ids)
            rows_cpu = rows.detach().float().cpu().numpy()

            if np_dtype == np.float16:
                rows_cpu = rows_cpu.astype(np.float16, copy=False)
            else:
                rows_cpu = rows_cpu.astype(np.float32, copy=False)

            mmap[start:end] = rows_cpu

    mmap.flush()
    del mmap

    bias_path = None
    if getattr(lm_head, "bias", None) is not None:
        bias = lm_head.bias.detach()
        bias_mmap_path = os.path.join(
            save_dir,
            "teacher_common_lm_head_bias.npy",
        )
        bias_mmap = np.lib.format.open_memmap(
            bias_mmap_path,
            mode="w+",
            dtype=np_dtype,
            shape=(common_size,),
        )

        with torch.inference_mode():
            for start in range(0, common_size, row_chunk_size):
                end = min(start + row_chunk_size, common_size)
                ids = torch.as_tensor(
                    teacher_common_ids[start:end],
                    dtype=torch.long,
                    device=bias.device,
                )
                chunk = bias.index_select(0, ids).float().cpu().numpy()
                if np_dtype == np.float16:
                    chunk = chunk.astype(np.float16, copy=False)
                else:
                    chunk = chunk.astype(np.float32, copy=False)
                bias_mmap[start:end] = chunk

        bias_mmap.flush()
        del bias_mmap
        bias_path = bias_mmap_path

    return {
        "weight_path": weight_path,
        "bias_path": bias_path,
        "hidden_size": hidden_size,
        "common_size": common_size,
    }


# ======================================================================
# 7. 批处理
# ======================================================================
def pad_teacher_batch(
    samples: List[dict],
    teacher_pad_token_id: int,
    device: torch.device,
):
    max_len = max(len(x["teacher_input_ids"]) for x in samples)
    batch_size = len(samples)

    input_ids = torch.full(
        (batch_size, max_len),
        int(teacher_pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
        device=device,
    )

    for row, sample in enumerate(samples):
        ids = sample["teacher_input_ids"]
        n = len(ids)
        input_ids[row, :n] = torch.tensor(
            ids,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row, :n] = 1

    return input_ids, attention_mask


# ======================================================================
# 8. Teacher final hidden 提取
# ======================================================================
def extract_teacher_hidden_cache(
    data: List[dict],
    prescan: dict,
    student_tokenizer,
    teacher_tokenizer,
    alignment: dict,
    student_token_ids: Dict[str, int],
    teacher_model,
    system_prompt: str,
    max_seq_length: int,
    reserve_tokens: int,
    extraction_batch_size: int,
    cache_dtype: str,
    save_dir: str,
):
    teacher = teacher_model
    decoder = getattr(teacher, "model", None)
    if decoder is None:
        raise RuntimeError(
            "当前 Teacher 不是预期的 LlamaForCausalLM 结构，找不到 teacher.model。"
        )

    hidden_size = int(teacher.config.hidden_size)
    total_positions = int(prescan["total_positions"])
    num_samples = int(prescan["num_samples"])

    np_dtype = numpy_cache_dtype(cache_dtype)
    hidden_path = os.path.join(
        save_dir,
        "teacher_completion_hidden.npy",
    )

    hidden_mmap = np.lib.format.open_memmap(
        hidden_path,
        mode="w+",
        dtype=np_dtype,
        shape=(total_positions, hidden_size),
    )

    teacher_pad_id = teacher_tokenizer.pad_token_id
    if teacher_pad_id is None:
        teacher_pad_id = teacher_tokenizer.eos_token_id
    if teacher_pad_id is None:
        teacher_pad_id = 0

    device = next(teacher.parameters()).device

    print("\n" + "=" * 88)
    print("第二遍：Teacher 8B 真正前向并写入 final hidden cache")
    print(f"样本数              : {num_samples:,}")
    print(f"总 KD positions      : {total_positions:,}")
    print(f"Teacher hidden size  : {hidden_size}")
    print(f"Cache dtype          : {cache_dtype}")
    print(f"Extraction batch size: {extraction_batch_size}")
    print(f"Cache path           : {hidden_path}")
    print("=" * 88)

    teacher.eval()
    teacher.requires_grad_(False)
    teacher.config.use_cache = False

    with torch.inference_mode():
        for batch_start in tqdm(
            range(0, num_samples, extraction_batch_size),
            desc="Teacher extract",
        ):
            batch_end = min(
                batch_start + extraction_batch_size,
                num_samples,
            )

            batch_samples = []
            for idx in range(batch_start, batch_end):
                sample = build_one_sample(
                    item=data[idx],
                    idx=idx,
                    student_tokenizer=student_tokenizer,
                    student_to_teacher=alignment["student_to_teacher"],
                    student_token_ids=student_token_ids,
                    system_prompt=system_prompt,
                    max_seq_length=max_seq_length,
                    reserve_tokens=reserve_tokens,
                )

                # 与第一遍预扫描结果做严格一致性检查
                if sample["valid_count"] != int(
                    prescan["valid_counts"][idx]
                ):
                    raise RuntimeError(
                        f"第 {idx} 条二次构造 valid_count 与 prescan 不一致。"
                    )
                if sample["l1_rel_offset"] != int(
                    prescan["l1_rel_offsets"][idx]
                ):
                    raise RuntimeError(
                        f"第 {idx} 条 L1 offset 与 prescan 不一致。"
                    )
                if sample["l2_rel_offset"] != int(
                    prescan["l2_rel_offsets"][idx]
                ):
                    raise RuntimeError(
                        f"第 {idx} 条 L2 offset 与 prescan 不一致。"
                    )
                expected_hash = prescan["input_hashes"][idx].decode("ascii")
                if sample["student_input_sha256"] != expected_hash:
                    raise RuntimeError(
                        f"第 {idx} 条 Student token hash 与 prescan 不一致。"
                    )

                batch_samples.append(sample)

            teacher_input_ids, attention_mask = pad_teacher_batch(
                batch_samples,
                teacher_pad_token_id=teacher_pad_id,
                device=device,
            )

            outputs = decoder(
                input_ids=teacher_input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

            hidden = outputs.last_hidden_state

            for row, sample in enumerate(batch_samples):
                sample_idx = batch_start + row
                actual_len = len(sample["teacher_input_ids"])

                if actual_len < 2:
                    raise RuntimeError(
                        f"第 {sample_idx} 条序列长度不足 2。"
                    )

                prediction_hidden = hidden[
                    row,
                    : actual_len - 1,
                    :,
                ]

                valid_positions = torch.tensor(
                    sample["valid_positions"],
                    dtype=torch.long,
                    device=prediction_hidden.device,
                )

                selected = prediction_hidden.index_select(
                    dim=0,
                    index=valid_positions,
                )

                expected_count = int(prescan["valid_counts"][sample_idx])
                if selected.shape != (expected_count, hidden_size):
                    raise RuntimeError(
                        f"第 {sample_idx} 条 hidden shape 异常: "
                        f"{tuple(selected.shape)} != "
                        f"({expected_count}, {hidden_size})"
                    )

                abs_start = int(prescan["offsets"][sample_idx])
                abs_end = int(prescan["offsets"][sample_idx + 1])

                selected_cpu = selected.detach().float().cpu().numpy()
                if np_dtype == np.float16:
                    selected_cpu = selected_cpu.astype(
                        np.float16,
                        copy=False,
                    )
                else:
                    selected_cpu = selected_cpu.astype(
                        np.float32,
                        copy=False,
                    )

                hidden_mmap[abs_start:abs_end] = selected_cpu

            # 及时释放本 batch 显存
            del outputs, hidden, teacher_input_ids, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    hidden_mmap.flush()
    del hidden_mmap

    return hidden_path


# ======================================================================
# 9. 校验缓存
# ======================================================================
def validate_written_cache(
    save_dir: str,
    prescan: dict,
    hidden_size: int,
    common_vocab_size: int,
):
    hidden = np.load(
        os.path.join(save_dir, "teacher_completion_hidden.npy"),
        mmap_mode="r",
    )
    lm_head = np.load(
        os.path.join(save_dir, "teacher_common_lm_head.npy"),
        mmap_mode="r",
    )

    expected_hidden_shape = (
        int(prescan["total_positions"]),
        int(hidden_size),
    )
    expected_head_shape = (
        int(common_vocab_size),
        int(hidden_size),
    )

    if hidden.shape != expected_hidden_shape:
        raise RuntimeError(
            f"hidden cache shape 错误: {hidden.shape} != {expected_hidden_shape}"
        )
    if lm_head.shape != expected_head_shape:
        raise RuntimeError(
            f"lm_head cache shape 错误: {lm_head.shape} != {expected_head_shape}"
        )

    # 检查开头/中间/结尾几行都有限
    probe_rows = sorted(set([
        0,
        max(0, hidden.shape[0] // 2),
        hidden.shape[0] - 1,
    ]))
    for row in probe_rows:
        if not np.isfinite(np.asarray(hidden[row], dtype=np.float32)).all():
            raise FloatingPointError(
                f"hidden cache 第 {row} 行出现 NaN/Inf。"
            )

    head_probe_rows = sorted(set([
        0,
        max(0, lm_head.shape[0] // 2),
        lm_head.shape[0] - 1,
    ]))
    for row in head_probe_rows:
        if not np.isfinite(np.asarray(lm_head[row], dtype=np.float32)).all():
            raise FloatingPointError(
                f"lm_head cache 第 {row} 行出现 NaN/Inf。"
            )

    print("\n" + "=" * 88)
    print("离线 Teacher cache 最终校验通过")
    print(f"teacher_completion_hidden: {hidden.shape}, dtype={hidden.dtype}")
    print(f"teacher_common_lm_head    : {lm_head.shape}, dtype={lm_head.dtype}")
    print("=" * 88 + "\n")


# ======================================================================
# 10. 主函数
# ======================================================================
def main(args):
    set_all_seeds(args.seed)

    if args.extraction_batch_size <= 0:
        raise ValueError("--extraction_batch_size 必须 > 0。")
    if args.lm_head_row_chunk_size <= 0:
        raise ValueError("--lm_head_row_chunk_size 必须 > 0。")

    if os.path.exists(args.save_dir):
        if not args.overwrite:
            raise FileExistsError(
                f"save_dir 已存在: {args.save_dir}\n"
                "为防止覆盖旧 cache，默认停止。"
                "若确认重建，请传 --overwrite true"
            )
        shutil.rmtree(args.save_dir)

    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "=" * 100)
    print("WOS Teacher 离线蒸馏缓存提取")
    print(f"Student tokenizer/base: {args.student_model_path}")
    print(f"Teacher best model    : {args.teacher_model_path}")
    print(f"Train data            : {args.train_data_path}")
    print(f"Save dir              : {args.save_dir}")
    print(f"Device                : {args.device}")
    print("=" * 100 + "\n")

    # ------------------------------------------------------------------
    # A. 数据
    # ------------------------------------------------------------------
    data = load_json_or_jsonl(args.train_data_path)
    print(f"读取训练集: {len(data):,} 条")

    # ------------------------------------------------------------------
    # B. Student tokenizer：它定义训练时的权威序列
    # ------------------------------------------------------------------
    student_tokenizer = AutoTokenizer.from_pretrained(
        args.student_model_path
    )
    install_student_chat_template(student_tokenizer)

    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    student_tokenizer.padding_side = "right"

    # ------------------------------------------------------------------
    # C. Teacher tokenizer：只用于 token-string ID 映射
    # ------------------------------------------------------------------
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model_path
    )
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_tokenizer.padding_side = "right"

    student_token_ids = validate_atomic_reserved_tokens(
        student_tokenizer,
        "Student",
    )
    teacher_token_ids = validate_atomic_reserved_tokens(
        teacher_tokenizer,
        "Teacher",
    )

    alignment = build_teacher_student_vocab_alignment(
        student_tokenizer,
        teacher_tokenizer,
    )

    np.save(
        os.path.join(args.save_dir, "student_common_ids.npy"),
        alignment["student_common_ids"],
    )
    np.save(
        os.path.join(args.save_dir, "teacher_common_ids.npy"),
        alignment["teacher_common_ids"],
    )

    # ------------------------------------------------------------------
    # D. 第一遍：只 tokenize / 计算 offset，不跑 8B Teacher
    # ------------------------------------------------------------------
    prescan = prescan_dataset(
        data=data,
        student_tokenizer=student_tokenizer,
        alignment=alignment,
        student_token_ids=student_token_ids,
        system_prompt=args.system_prompt,
        max_seq_length=args.max_seq_length,
        reserve_tokens=args.reserve_tokens,
        scan_limit=args.scan_limit,
    )

    np.save(
        os.path.join(args.save_dir, "offsets.npy"),
        prescan["offsets"],
    )
    np.save(
        os.path.join(args.save_dir, "valid_counts.npy"),
        prescan["valid_counts"],
    )
    np.save(
        os.path.join(args.save_dir, "l1_rel_offsets.npy"),
        prescan["l1_rel_offsets"],
    )
    np.save(
        os.path.join(args.save_dir, "l2_rel_offsets.npy"),
        prescan["l2_rel_offsets"],
    )
    np.save(
        os.path.join(args.save_dir, "student_input_sha256.npy"),
        prescan["input_hashes"],
    )

    # ------------------------------------------------------------------
    # E. 加载冻结 Teacher
    # ------------------------------------------------------------------
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 torch.cuda.is_available()=False。")

    device = torch.device(args.device)

    if args.model_dtype == "bfloat16":
        model_dtype = torch.bfloat16
    elif args.model_dtype == "float16":
        model_dtype = torch.float16
    elif args.model_dtype == "float32":
        model_dtype = torch.float32
    else:
        raise ValueError(
            "--model_dtype 只支持 bfloat16/float16/float32"
        )

    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_path,
        torch_dtype=model_dtype,
        attn_implementation=args.attn_implementation,
        device_map=None,
    )
    teacher_model.to(device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    teacher_model.config.use_cache = False

    teacher_vocab_rows = int(
        teacher_model.get_output_embeddings().weight.shape[0]
    )
    if teacher_vocab_rows != len(teacher_tokenizer):
        raise RuntimeError(
            "Teacher lm_head 行数与 Teacher tokenizer 长度不一致: "
            f"{teacher_vocab_rows} != {len(teacher_tokenizer)}"
        )

    hidden_size = int(teacher_model.config.hidden_size)

    print("\n" + "=" * 88)
    print("Teacher 加载完成")
    print(f"Teacher hidden size : {hidden_size}")
    print(f"Teacher vocab rows  : {teacher_vocab_rows:,}")
    print(f"Teacher dtype       : {next(teacher_model.parameters()).dtype}")
    print(f"Teacher device      : {next(teacher_model.parameters()).device}")
    print("Teacher trainable   : 0")
    print("=" * 88 + "\n")

    # ------------------------------------------------------------------
    # F. 先保存 common-vocab lm_head
    # ------------------------------------------------------------------
    lm_head_info = save_teacher_common_lm_head(
        teacher_model=teacher_model,
        teacher_common_ids=alignment["teacher_common_ids"],
        save_dir=args.save_dir,
        cache_dtype=args.cache_dtype,
        row_chunk_size=args.lm_head_row_chunk_size,
    )

    # ------------------------------------------------------------------
    # G. Teacher 8B 只完整遍历一次数据，保存 final hidden
    # ------------------------------------------------------------------
    hidden_path = extract_teacher_hidden_cache(
        data=data,
        prescan=prescan,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        alignment=alignment,
        student_token_ids=student_token_ids,
        teacher_model=teacher_model,
        system_prompt=args.system_prompt,
        max_seq_length=args.max_seq_length,
        reserve_tokens=args.reserve_tokens,
        extraction_batch_size=args.extraction_batch_size,
        cache_dtype=args.cache_dtype,
        save_dir=args.save_dir,
    )

    # ------------------------------------------------------------------
    # H. manifest
    # ------------------------------------------------------------------
    manifest = {
        "format_version": 1,
        "dataset": "WOS",
        "split": "train",
        "num_samples": int(prescan["num_samples"]),
        "total_completion_prediction_positions": int(
            prescan["total_positions"]
        ),
        "teacher_hidden_size": hidden_size,
        "common_vocab_size": int(alignment["common_token_count"]),
        "cache_dtype": args.cache_dtype,

        "student_model_path": args.student_model_path,
        "teacher_model_path": args.teacher_model_path,
        "train_data_path": args.train_data_path,

        "max_seq_length": int(args.max_seq_length),
        "reserve_tokens": int(args.reserve_tokens),
        "system_prompt": args.system_prompt,

        "student_reserved_token_ids": student_token_ids,
        "teacher_reserved_token_ids": teacher_token_ids,

        "token_alignment_rule": "same token string; token IDs may differ",
        "common_token_list_sha256": alignment["common_token_sha256"],
        "student_only_tokens": alignment["student_only_tokens"],
        "teacher_only_tokens": alignment["teacher_only_tokens"],

        "causal_position_rule": (
            "prediction_hidden = teacher.model(...).last_hidden_state[:, :-1, :]; "
            "select positions where labels[:, 1:] != -100"
        ),
        "l1_hidden_rule": (
            "second occurrence of L1 reserved token; "
            "stored as relative offset inside each sample's selected "
            "Completion prediction hidden sequence"
        ),
        "l2_hidden_rule": (
            "second occurrence of L2 reserved token; "
            "stored as relative offset inside each sample's selected "
            "Completion prediction hidden sequence"
        ),

        "traditional_kd_reconstruction": (
            "teacher_common_logits = cached_completion_hidden "
            "@ teacher_common_lm_head.T (+ cached bias if present); "
            "then temperature-scaled KL"
        ),
        "traditional_kd_is_still_logits_kd": True,
        "hidden_align_source": (
            "cached hidden at L1/L2 second special-token occurrence"
        ),
        "hidden_hier_source": (
            "cached L2 hidden - cached L1 hidden"
        ),

        "files": {
            "offsets": "offsets.npy",
            "valid_counts": "valid_counts.npy",
            "l1_rel_offsets": "l1_rel_offsets.npy",
            "l2_rel_offsets": "l2_rel_offsets.npy",
            "student_input_sha256": "student_input_sha256.npy",
            "teacher_completion_hidden": os.path.basename(hidden_path),
            "student_common_ids": "student_common_ids.npy",
            "teacher_common_ids": "teacher_common_ids.npy",
            "teacher_common_lm_head": os.path.basename(
                lm_head_info["weight_path"]
            ),
            "teacher_common_lm_head_bias": (
                os.path.basename(lm_head_info["bias_path"])
                if lm_head_info["bias_path"] is not None
                else None
            ),
        },
    }

    manifest_path = os.path.join(args.save_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # I. 最终校验
    # ------------------------------------------------------------------
    validate_written_cache(
        save_dir=args.save_dir,
        prescan=prescan,
        hidden_size=hidden_size,
        common_vocab_size=alignment["common_token_count"],
    )

    print("\n" + "=" * 100)
    print("Teacher 离线缓存提取完成")
    print(f"Cache dir: {args.save_dir}")
    print("")
    print("后续 Student 训练阶段：")
    print("1) 不再加载 8B Teacher Transformer；")
    print("2) 用 cached hidden × cached Teacher common lm_head 恢复 Teacher logits；")
    print("3) Traditional KD 仍计算 logits KL；")
    print("4) L1/L2 cached hidden 直接用于你的 hidden Align；")
    print("5) cached L2 hidden - cached L1 hidden 直接用于你的 hidden Hier。")
    print("=" * 100 + "\n")


# ======================================================================
# 11. 参数
# ======================================================================
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract one-time offline Teacher cache for WOS: "
            "Completion final hidden states + common-vocab Teacher lm_head. "
            "The cache supports Traditional logits KD and hidden-state "
            "Align/Hierarchy losses without loading the 8B Transformer "
            "during Student training."
        )
    )

    # 路径
    parser.add_argument(
        "--student_model_path",
        type=str,
        default="/data1/lzp/code/llama-3.2_1b/",
        help="Student base/tokenizer；它定义训练时的权威 token 序列。",
    )
    parser.add_argument(
        "--teacher_model_path",
        type=str,
        default=(
            "/data1/lzp/code/hierarchical_distill/"
            "checkpoints_teacher_wos/"
            "best_sft_model/"
        ),
        help="训练完成的 merged best 8B Teacher。",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="/data1/lzp/code/data/wos/train_formatted.jsonl",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=(
            "/data1/lzp/code/hierarchical_distill/"
            "teacher_wos_hiddenstates/"
        ),
    )

    # 与训练完全一致
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--reserve_tokens", type=int, default=300)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are a specialized academic classifier for the WOS "
            "two-level taxonomy. Follow the required output format exactly. "
            "Do not produce explanations or any extra conversational text."
        ),
    )

    # 提取性能
    parser.add_argument(
        "--extraction_batch_size",
        type=int,
        default=1,
        help="8B Teacher 前向 batch。显存足够可尝试 2/4。",
    )
    parser.add_argument(
        "--lm_head_row_chunk_size",
        type=int,
        default=4096,
        help="保存 common lm_head 时每次从 GPU 复制多少行。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--model_dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Teacher 前向 dtype。",
    )
    parser.add_argument(
        "--cache_dtype",
        type=str,
        choices=["float16", "float32"],
        default="float16",
        help="磁盘缓存 dtype；float16 约为 float32 一半空间。",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
    )

    # 调试：-1 提取全部；先可用 10/100 条做 sanity check
    parser.add_argument(
        "--scan_limit",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--overwrite",
        type=str2bool,
        default=False,
        help="默认禁止覆盖已有 cache。",
    )

    args = parser.parse_args()
    main(args)
