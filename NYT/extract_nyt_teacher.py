#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NYT Teacher 离线缓存提取脚本

与 WOS 单-token 蒸馏缓存保持同一设计，只把两层 L1/L2 扩展到 NYT D1-D8：

1) Traditional logits KD
   - 保存所有 Completion causal prediction positions 的 Teacher final hidden：
       H_T_KD: [total_completion_positions, teacher_hidden_size]
   - 保存 Teacher common-vocab lm_head：
       W_T_common: [V_common, teacher_hidden_size]
   - Student 蒸馏时恢复：
       z_T_common = H_T_KD @ W_T_common.T (+ bias)

2) Depth hidden Align / Hier
   - 每条样本固定有 D1-D8 八个 reserved token 的第二次 occurrence。
   - 不额外复制 8 份 hidden，只保存每个深度 token 在该样本 H_T_KD 中的相对 offset：
       depth_rel_offsets: int32 [num_samples, 8]
   - 蒸馏时：
       h_T_Dk = H_T_KD[sample_start + depth_rel_offsets[i, k-1]]

关键一致性
========
- 以 Student tokenizer 构造权威训练序列；
- 再按“相同 token 字符串”把 Student token ID 翻译到 Teacher ID；
- Teacher 不重新 tokenize 原文本，所以 Teacher/Student 序列位置严格一致；
- Prompt / Completion / 截断规则与当前 NYT Teacher/Student SFT 完全一致；
- D1-D8 均提取 Completion 中第二次 occurrence 对应的 final hidden；
- 空层 [] 也保留对应深度 token，因此 8 个 depth offset 每条样本都存在；
- Teacher 为 eval + inference_mode，不更新参数。

缓存格式
========
save_dir/
  manifest.json
  offsets.npy                       int64 [num_samples + 1]
  valid_counts.npy                  int32 [num_samples]
  depth_rel_offsets.npy             int32 [num_samples, 8]
  student_input_sha256.npy          S64   [num_samples]
  teacher_completion_hidden.npy     float16/float32 [total_valid_positions, hidden]
  student_common_ids.npy            int64 [V_common]
  teacher_common_ids.npy            int64 [V_common]
  teacher_common_lm_head.npy        float16/float32 [V_common, hidden]
  teacher_common_lm_head_bias.npy   可选
"""

import os
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import random
import shutil
from typing import Dict, List, Sequence, Tuple

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
# 1. 与当前 NYT Teacher / Student 训练完全一致的固定定义
# ======================================================================
CTX_START_TOKEN = "<|reserved_special_token_110|>"
CTX_END_TOKEN = "<|reserved_special_token_111|>"
DEPTH_PROBE_TOKENS: Dict[int, str] = {
    depth: f"<|reserved_special_token_{111 + depth}|>"
    for depth in range(1, 9)
}
MAX_DEPTH = 8

REQUIRED_CLEAN_FIELDS = (
    "text",
    "labels",
    "label_ids",
    "top_labels",
    "top_label_ids",
    "leaf_labels",
    "leaf_label_ids",
)

CHAT_TEMPLATE = (
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

DEFAULT_SYSTEM_PROMPT = (
    "You are a specialized classifier for the NYT hierarchical "
    "multi-label taxonomy. Follow the required eight-line "
    "JSON-array output format exactly. Do not produce explanations "
    "or any extra conversational text."
)


def install_student_chat_template(tokenizer) -> None:
    tokenizer.chat_template = CHAT_TEMPLATE


def build_user_content(text: str, max_depth: int = MAX_DEPTH) -> str:
    assignments = "\n".join(
        f"D{depth}={DEPTH_PROBE_TOKENS[depth]}"
        for depth in range(1, max_depth + 1)
    )
    return (
        f"{CTX_START_TOKEN}\n"
        f"{text}\n"
        f"{CTX_END_TOKEN}\n\n"
        "Classify the document according to the NYT hierarchical "
        f"multi-label taxonomy with depths 1-{max_depth}.\n\n"
        "Hierarchy definitions:\n"
        "- Depth Dk contains every label assigned to the document at "
        "exactly taxonomy depth k.\n"
        "- The complete output must be ancestor-closed across all paths.\n"
        "- For each depth, use the document and all previously generated "
        "shallower-depth labels as hierarchical context.\n\n"
        "Hierarchy-token assignments:\n"
        "Each Llama reserved special token below is dedicated to "
        "representing the document's hierarchy information at its assigned "
        "taxonomy depth and serves as the depth-specific prediction-state "
        "marker for all labels at that depth.\n"
        f"{assignments}\n\n"
        f"Output exactly {max_depth} lines in depth order.\n"
        "On each line, reproduce the corresponding reserved special token "
        "shown above, followed by one space and a valid JSON array containing "
        "all labels at that depth.\n"
        "At D1, keep labels in canonical label-ID order. At each deeper "
        "depth Dk, first group labels by the order of their parent labels "
        "in D(k-1), then keep children of the same parent in canonical "
        "label-ID order. Use [] when the depth has no label. Use exact "
        "taxonomy label strings.\n"
        f"Stop immediately after the {max_depth}th line.\n"
        "Do not output numbering, colons, explanations, or any other text."
    )


def sort_depth_labels_by_parent_order(
    labels: Sequence[str],
    current_depth: int,
    previous_depth_labels: Sequence[str],
    label_space: dict,
) -> List[str]:
    label2id = label_space["label2id"]
    parent = label_space["parent"]
    normalized_labels = list(labels)
    if current_depth == 1:
        return sorted(normalized_labels, key=label2id.get)

    parent_rank = {
        parent_label: rank
        for rank, parent_label in enumerate(previous_depth_labels)
    }

    def hierarchy_key(label: str):
        parent_label = parent[label]
        if parent_label in parent_rank:
            return (0, parent_rank[parent_label], label2id[label])
        return (
            1,
            label2id.get(parent_label, len(label2id)),
            label2id[label],
        )

    return sorted(normalized_labels, key=hierarchy_key)


def group_labels_by_depth(
    labels: Sequence[str],
    label_space: dict,
) -> Dict[int, List[str]]:
    max_depth = int(label_space["max_depth"])
    depth_map = label_space["depth"]
    label2id = label_space["label2id"]
    grouped = {d: [] for d in range(1, max_depth + 1)}

    seen = set()
    for raw_label in labels:
        label = str(raw_label).strip()
        if not label:
            raise ValueError("构造 Completion 时遇到空标签字符串。")
        if label in seen:
            raise ValueError(f"构造 Completion 时标签重复: {label!r}")
        if label not in label2id:
            raise ValueError(f"构造 Completion 时遇到未知标签: {label!r}")
        seen.add(label)
        d = int(depth_map[label])
        if d < 1 or d > max_depth:
            raise ValueError(f"标签 {label!r} 深度异常: {d}")
        grouped[d].append(label)

    for d in range(1, max_depth + 1):
        previous = grouped[d - 1] if d > 1 else []
        grouped[d] = sort_depth_labels_by_parent_order(
            grouped[d], d, previous, label_space
        )
    return grouped


def build_completion_text(labels: Sequence[str], label_space: dict) -> str:
    max_depth = int(label_space["max_depth"])
    grouped = group_labels_by_depth(labels, label_space)
    lines = []
    for d in range(1, max_depth + 1):
        labels_json = json.dumps(
            grouped[d], ensure_ascii=False, separators=(",", ":")
        )
        lines.append(f"{DEPTH_PROBE_TOKENS[d]} {labels_json}")
    return "\n".join(lines) + "\n"


def build_prompt_messages(
    text: str,
    system_prompt: str,
    max_depth: int = MAX_DEPTH,
) -> List[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": build_user_content(text, max_depth=max_depth),
        },
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


def _get_eot_id(tokenizer) -> int:
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is None or eot_id == tokenizer.unk_token_id:
        eot_id = tokenizer.eos_token_id
    if eot_id is None:
        raise RuntimeError("Tokenizer 缺少可用 EOT/EOS。")
    return int(eot_id)


# ======================================================================
# 2. 数据与 label_space
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
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise TypeError(f"JSON 顶层必须是 list，实际={type(data).__name__}")
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"JSON 第 {idx} 条必须是 dict。")
    return data


def _normalize_string_list(value, field_name: str, idx: int) -> List[str]:
    if not isinstance(value, list):
        raise TypeError(f"TRAIN 第 {idx} 条 {field_name} 必须是 list。")
    result = []
    for j, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise TypeError(
                f"TRAIN 第 {idx} 条 {field_name}[{j}] 必须是非空 str。"
            )
        result.append(item.strip())
    return result


def _normalize_int_list(value, field_name: str, idx: int) -> List[int]:
    if not isinstance(value, list):
        raise TypeError(f"TRAIN 第 {idx} 条 {field_name} 必须是 list。")
    result = []
    for j, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(
                f"TRAIN 第 {idx} 条 {field_name}[{j}] 必须是 int。"
            )
        result.append(int(item))
    return result


def get_clean_sample_fields(item: dict, idx: int):
    missing = [k for k in REQUIRED_CLEAN_FIELDS if k not in item]
    if missing:
        raise KeyError(
            f"TRAIN 第 {idx} 条缺少字段 {missing}；必须包含 {list(REQUIRED_CLEAN_FIELDS)}"
        )
    text = item["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"TRAIN 第 {idx} 条 text 必须是非空字符串。")
    labels = _normalize_string_list(item["labels"], "labels", idx)
    label_ids = _normalize_int_list(item["label_ids"], "label_ids", idx)
    top_labels = _normalize_string_list(item["top_labels"], "top_labels", idx)
    top_label_ids = _normalize_int_list(item["top_label_ids"], "top_label_ids", idx)
    leaf_labels = _normalize_string_list(item["leaf_labels"], "leaf_labels", idx)
    leaf_label_ids = _normalize_int_list(item["leaf_label_ids"], "leaf_label_ids", idx)
    if not labels:
        raise ValueError(f"TRAIN 第 {idx} 条 labels 为空。")
    return (
        text.strip(), labels, label_ids,
        top_labels, top_label_ids, leaf_labels, leaf_label_ids,
    )


def load_label_space(path: str, expected_num_labels: int, expected_max_depth: int) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"label_space 不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    required = ("num_labels", "max_depth", "label2id", "parent", "depth")
    missing = [k for k in required if k not in raw]
    if missing:
        raise KeyError(f"label_space 缺少字段: {missing}")

    num_labels = int(raw["num_labels"])
    max_depth = int(raw["max_depth"])
    if num_labels != int(expected_num_labels):
        raise ValueError(f"NYT 标签数异常: {num_labels} != {expected_num_labels}")
    if max_depth != int(expected_max_depth):
        raise ValueError(f"NYT 最大深度异常: {max_depth} != {expected_max_depth}")

    label2id = {str(k).strip(): int(v) for k, v in raw["label2id"].items()}
    if sorted(label2id.values()) != list(range(num_labels)):
        raise ValueError("label2id 必须是连续 0-based ID。")
    parent = {str(k).strip(): str(v).strip() for k, v in raw["parent"].items()}
    depth_map = {str(k).strip(): int(v) for k, v in raw["depth"].items()}
    if set(parent) != set(label2id):
        raise ValueError("label_space.parent 必须恰好覆盖全部任务标签。")
    if not set(label2id).issubset(depth_map):
        raise ValueError("label_space.depth 未覆盖全部任务标签。")

    resolved = dict(raw)
    resolved["num_labels"] = num_labels
    resolved["max_depth"] = max_depth
    resolved["label2id"] = label2id
    resolved["parent"] = parent
    resolved["depth"] = depth_map
    return resolved


def validate_train_data(data: List[dict], label_space: dict) -> None:
    label2id = label_space["label2id"]
    depth_map = label_space["depth"]
    parent = label_space["parent"]
    for idx, item in enumerate(tqdm(data, desc="校验 TRAIN", leave=False)):
        _, labels, label_ids, *_ = get_clean_sample_fields(item, idx)
        if len(labels) != len(label_ids):
            raise ValueError(f"TRAIN 第 {idx} 条 labels/label_ids 长度不一致。")
        if len(labels) != len(set(labels)):
            raise ValueError(f"TRAIN 第 {idx} 条 labels 存在重复。")
        unknown = set(labels) - set(label2id)
        if unknown:
            raise ValueError(f"TRAIN 第 {idx} 条未知标签: {sorted(unknown)[:20]}")
        expected_labels = sorted(labels, key=label2id.get)
        expected_ids = [label2id[x] for x in expected_labels]
        if labels != expected_labels or label_ids != expected_ids:
            raise ValueError(f"TRAIN 第 {idx} 条 labels/label_ids 与 label_space 不一致。")
        label_set = set(labels)
        for label in labels:
            current = label
            while parent[current] != "Root":
                current = parent[current]
                if current not in label_set:
                    raise ValueError(f"TRAIN 第 {idx} 条不是祖先闭包: {label!r}")
            d = int(depth_map[label])
            if d < 1 or d > MAX_DEPTH:
                raise ValueError(f"TRAIN 第 {idx} 条标签深度异常: {label!r}->{d}")
    print(f"✅ TRAIN 数据严格校验通过: {len(data):,} 条")


# ======================================================================
# 3. Tokenizer 与 Teacher/Student 词表对齐
# ======================================================================
def validate_atomic_reserved_tokens(tokenizer, name: str) -> Dict[str, object]:
    token_map = {"ctx_start": CTX_START_TOKEN, "ctx_end": CTX_END_TOKEN}
    token_map.update({f"depth_{d}": DEPTH_PROBE_TOKENS[d] for d in range(1, 9)})
    vocab = tokenizer.get_vocab()
    flat = {}
    print("\n" + "=" * 88)
    print(f"验证 {name} NYT reserved tokens")
    for role, token in token_map.items():
        if token not in vocab:
            raise RuntimeError(f"{name} 缺少 token: {token}")
        encoded = tokenizer.encode(token, add_special_tokens=False)
        converted = tokenizer.convert_tokens_to_ids(token)
        if len(encoded) != 1:
            raise RuntimeError(f"{name} 的 {token} 不是单 token: {encoded}")
        token_id = int(encoded[0])
        if int(converted) != token_id:
            raise RuntimeError(f"{name} 的 {token} encode/get_vocab ID 不一致。")
        if token_id == tokenizer.unk_token_id:
            raise RuntimeError(f"{name} 的 {token} 被编码成 unk。")
        flat[role] = token_id
        print(f"{role:>10s}: {token} -> {token_id}")
    if len(set(flat.values())) != len(flat):
        raise RuntimeError(f"{name} reserved token ID 发生重复: {flat}")
    print("=" * 88)
    return {
        "ctx_start": flat["ctx_start"],
        "ctx_end": flat["ctx_end"],
        "depth": {d: flat[f"depth_{d}"] for d in range(1, 9)},
    }


def build_teacher_student_vocab_alignment(student_tokenizer, teacher_tokenizer) -> dict:
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    common_tokens = set(student_vocab) & set(teacher_vocab)
    if not common_tokens:
        raise RuntimeError("Teacher / Student 没有共同 token。")

    protected_tokens = [CTX_START_TOKEN, CTX_END_TOKEN] + [
        DEPTH_PROBE_TOKENS[d] for d in range(1, 9)
    ] + [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ]
    protected_info = {}
    for token in protected_tokens:
        if token not in student_vocab or token not in teacher_vocab:
            raise RuntimeError(f"Teacher/Student 至少一边缺少必需 token: {token}")
        s_ids = student_tokenizer.encode(token, add_special_tokens=False)
        t_ids = teacher_tokenizer.encode(token, add_special_tokens=False)
        if len(s_ids) != 1 or len(t_ids) != 1:
            raise RuntimeError(
                f"必需 token 必须在两边都是原子 token: {token}; student={s_ids}, teacher={t_ids}"
            )
        protected_info[token] = {
            "student_id": int(s_ids[0]),
            "teacher_id": int(t_ids[0]),
        }

    common_sorted = sorted(common_tokens, key=lambda tok: student_vocab[tok])
    student_common_ids = np.asarray(
        [int(student_vocab[tok]) for tok in common_sorted], dtype=np.int64
    )
    teacher_common_ids = np.asarray(
        [int(teacher_vocab[tok]) for tok in common_sorted], dtype=np.int64
    )
    student_to_teacher = np.full(len(student_tokenizer), -1, dtype=np.int64)
    for tok in common_sorted:
        student_to_teacher[int(student_vocab[tok])] = int(teacher_vocab[tok])

    common_sha = hashlib.sha256("\n".join(common_sorted).encode("utf-8")).hexdigest()
    student_only = sorted(set(student_vocab) - set(teacher_vocab))
    teacher_only = sorted(set(teacher_vocab) - set(student_vocab))

    print("\n" + "=" * 88)
    print("Teacher / Student token-string alignment")
    print(f"Student vocab : {len(student_vocab):,}")
    print(f"Teacher vocab : {len(teacher_vocab):,}")
    print(f"Common vocab  : {len(common_sorted):,}")
    print(f"Student-only  : {len(student_only):,}")
    print(f"Teacher-only  : {len(teacher_only):,}")
    print(f"common token SHA256: {common_sha}")
    print("=" * 88 + "\n")

    return {
        "student_common_ids": student_common_ids,
        "teacher_common_ids": teacher_common_ids,
        "student_to_teacher": student_to_teacher,
        "common_token_count": len(common_sorted),
        "common_token_sha256": common_sha,
        "student_only_tokens": student_only,
        "teacher_only_tokens": teacher_only,
        "protected_info": protected_info,
    }


# ======================================================================
# 4. 构造单样本：完全复制 NYT Student 训练序列逻辑
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
    student_token_ids: Dict[str, object],
    label_space: dict,
    system_prompt: str,
    max_seq_length: int,
    reserve_tokens: int,
) -> dict:
    text, labels, *_ = get_clean_sample_fields(item, idx)
    text = truncate_text_for_prompt(
        text, student_tokenizer, max_seq_length, reserve_tokens
    )
    messages = build_prompt_messages(
        text=text,
        system_prompt=system_prompt,
        max_depth=int(label_space["max_depth"]),
    )
    prompt_ids = student_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_ids = [int(x) for x in prompt_ids]

    completion_ids = student_tokenizer.encode(
        build_completion_text(labels, label_space),
        add_special_tokens=False,
    )
    completion_ids = [int(x) for x in completion_ids]
    completion_ids.append(_get_eot_id(student_tokenizer))

    student_input_ids = prompt_ids + completion_ids
    labels_ids = [-100] * len(prompt_ids) + completion_ids

    if len(student_input_ids) > max_seq_length:
        raise RuntimeError(
            f"TRAIN 第 {idx} 条最终长度 {len(student_input_ids)} > max_seq_length={max_seq_length}"
        )

    # Prompt 中每个深度 token 第一次出现；Completion 中第二次出现。
    second_positions = []
    for d in range(1, int(label_space["max_depth"]) + 1):
        token_id = int(student_token_ids["depth"][d])
        positions = find_all_positions(student_input_ids, token_id)
        if len(positions) != 2:
            raise RuntimeError(
                f"TRAIN 第 {idx} 条 D{d} occurrence 异常: {positions}"
            )
        second_positions.append(int(positions[1]))
    if second_positions != sorted(second_positions):
        raise RuntimeError(
            f"TRAIN 第 {idx} 条 D1-D8 第二次 occurrence 顺序异常: {second_positions}"
        )

    # causal shift: logits[:, :-1] <-> labels[:, 1:]
    shift_labels = labels_ids[1:]
    valid_positions = [
        pos for pos, target in enumerate(shift_labels)
        if int(target) != -100
    ]
    if not valid_positions:
        raise RuntimeError(f"TRAIN 第 {idx} 条没有有效 Completion prediction positions。")

    valid_pos_to_rel = {
        int(pos): rel for rel, pos in enumerate(valid_positions)
    }
    depth_rel_offsets = []
    for d, pos in enumerate(second_positions, start=1):
        if pos not in valid_pos_to_rel:
            raise RuntimeError(
                f"TRAIN 第 {idx} 条 D{d} 第二次 token position={pos} 不在 KD causal mask 中。"
            )
        depth_rel_offsets.append(int(valid_pos_to_rel[pos]))

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
        "depth_rel_offsets": depth_rel_offsets,
        "student_input_sha256": sha256_token_ids(student_input_ids),
    }


# ======================================================================
# 5. 预扫描
# ======================================================================
def prescan_dataset(
    data: List[dict],
    student_tokenizer,
    alignment: dict,
    student_token_ids: Dict[str, object],
    label_space: dict,
    system_prompt: str,
    max_seq_length: int,
    reserve_tokens: int,
    scan_limit: int,
):
    num_samples = len(data) if scan_limit < 0 else min(len(data), scan_limit)
    if num_samples <= 0:
        raise ValueError("实际提取样本数必须 > 0。")

    max_depth = int(label_space["max_depth"])
    valid_counts = np.zeros(num_samples, dtype=np.int32)
    depth_rel_offsets = np.zeros((num_samples, max_depth), dtype=np.int32)
    input_hashes = np.empty(num_samples, dtype="S64")

    print("\n" + "=" * 88)
    print(f"第一遍预扫描：{num_samples:,} 条")
    print("目的：确定每条 Completion KD position 数、D1-D8 offset 和总缓存大小")
    print("=" * 88)

    for idx in tqdm(range(num_samples), desc="Prescan"):
        sample = build_one_sample(
            item=data[idx],
            idx=idx,
            student_tokenizer=student_tokenizer,
            student_to_teacher=alignment["student_to_teacher"],
            student_token_ids=student_token_ids,
            label_space=label_space,
            system_prompt=system_prompt,
            max_seq_length=max_seq_length,
            reserve_tokens=reserve_tokens,
        )
        valid_counts[idx] = sample["valid_count"]
        depth_rel_offsets[idx] = np.asarray(
            sample["depth_rel_offsets"], dtype=np.int32
        )
        input_hashes[idx] = sample["student_input_sha256"].encode("ascii")

    offsets = np.zeros(num_samples + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(valid_counts, dtype=np.int64)
    total_positions = int(offsets[-1])
    if total_positions <= 0:
        raise RuntimeError("总 Completion KD position 数为 0。")

    if np.any(depth_rel_offsets < 0):
        raise RuntimeError("存在负数 depth relative offset。")
    for d in range(max_depth):
        if np.any(depth_rel_offsets[:, d] >= valid_counts):
            raise RuntimeError(f"存在非法 D{d+1} relative offset。")

    print("\n预扫描完成")
    print(f"样本数: {num_samples:,}")
    print(f"总有效 Completion prediction positions: {total_positions:,}")
    print(f"平均每样本有效位置数: {total_positions / num_samples:.3f}")
    print(f"每样本有效位置范围: {int(valid_counts.min())} ~ {int(valid_counts.max())}")

    return {
        "num_samples": num_samples,
        "valid_counts": valid_counts,
        "offsets": offsets,
        "depth_rel_offsets": depth_rel_offsets,
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
    lm_head = teacher_model.get_output_embeddings()
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
        for start in tqdm(range(0, common_size, row_chunk_size), desc="Save lm_head"):
            end = min(start + row_chunk_size, common_size)
            ids = torch.as_tensor(
                teacher_common_ids[start:end], dtype=torch.long, device=device
            )
            rows = weight.index_select(0, ids).detach().float().cpu().numpy()
            rows = rows.astype(np_dtype, copy=False)
            mmap[start:end] = rows
    mmap.flush()
    del mmap

    bias_path = None
    if getattr(lm_head, "bias", None) is not None:
        bias = lm_head.bias.detach()
        bias_path = os.path.join(save_dir, "teacher_common_lm_head_bias.npy")
        bias_mmap = np.lib.format.open_memmap(
            bias_path,
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
                bias_mmap[start:end] = chunk.astype(np_dtype, copy=False)
        bias_mmap.flush()
        del bias_mmap

    return {
        "weight_path": weight_path,
        "bias_path": bias_path,
        "hidden_size": hidden_size,
        "common_size": common_size,
    }


# ======================================================================
# 7. 批处理
# ======================================================================
def pad_teacher_batch(samples: List[dict], teacher_pad_token_id: int, device: torch.device):
    max_len = max(len(x["teacher_input_ids"]) for x in samples)
    batch_size = len(samples)
    input_ids = torch.full(
        (batch_size, max_len),
        int(teacher_pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, max_len), dtype=torch.long, device=device
    )
    for row, sample in enumerate(samples):
        ids = sample["teacher_input_ids"]
        n = len(ids)
        input_ids[row, :n] = torch.tensor(ids, dtype=torch.long, device=device)
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
    student_token_ids: Dict[str, object],
    label_space: dict,
    teacher_model,
    system_prompt: str,
    max_seq_length: int,
    reserve_tokens: int,
    extraction_batch_size: int,
    cache_dtype: str,
    save_dir: str,
):
    decoder = getattr(teacher_model, "model", None)
    if decoder is None:
        raise RuntimeError("当前 Teacher 不是预期 LlamaForCausalLM，找不到 teacher.model。")

    hidden_size = int(teacher_model.config.hidden_size)
    total_positions = int(prescan["total_positions"])
    num_samples = int(prescan["num_samples"])
    np_dtype = numpy_cache_dtype(cache_dtype)
    hidden_path = os.path.join(save_dir, "teacher_completion_hidden.npy")
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
    device = next(teacher_model.parameters()).device

    print("\n" + "=" * 88)
    print("第二遍：Teacher 8B 真正前向并写入 final hidden cache")
    print(f"样本数              : {num_samples:,}")
    print(f"总 KD positions      : {total_positions:,}")
    print(f"Teacher hidden size  : {hidden_size}")
    print(f"Cache dtype          : {cache_dtype}")
    print(f"Extraction batch size: {extraction_batch_size}")
    print(f"Cache path           : {hidden_path}")
    print("=" * 88)

    teacher_model.eval()
    teacher_model.requires_grad_(False)
    teacher_model.config.use_cache = False

    with torch.inference_mode():
        for batch_start in tqdm(
            range(0, num_samples, extraction_batch_size),
            desc="Teacher extract",
        ):
            batch_end = min(batch_start + extraction_batch_size, num_samples)
            batch_samples = []
            for idx in range(batch_start, batch_end):
                sample = build_one_sample(
                    item=data[idx],
                    idx=idx,
                    student_tokenizer=student_tokenizer,
                    student_to_teacher=alignment["student_to_teacher"],
                    student_token_ids=student_token_ids,
                    label_space=label_space,
                    system_prompt=system_prompt,
                    max_seq_length=max_seq_length,
                    reserve_tokens=reserve_tokens,
                )
                if sample["valid_count"] != int(prescan["valid_counts"][idx]):
                    raise RuntimeError(f"第 {idx} 条 valid_count 与 prescan 不一致。")
                if not np.array_equal(
                    np.asarray(sample["depth_rel_offsets"], dtype=np.int32),
                    np.asarray(prescan["depth_rel_offsets"][idx], dtype=np.int32),
                ):
                    raise RuntimeError(f"第 {idx} 条 D1-D8 offset 与 prescan 不一致。")
                expected_hash = prescan["input_hashes"][idx].decode("ascii")
                if sample["student_input_sha256"] != expected_hash:
                    raise RuntimeError(f"第 {idx} 条 Student token hash 与 prescan 不一致。")
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
                    raise RuntimeError(f"第 {sample_idx} 条序列长度不足 2。")
                prediction_hidden = hidden[row, : actual_len - 1, :]
                valid_positions = torch.tensor(
                    sample["valid_positions"],
                    dtype=torch.long,
                    device=prediction_hidden.device,
                )
                selected = prediction_hidden.index_select(0, valid_positions)
                expected_count = int(prescan["valid_counts"][sample_idx])
                if selected.shape != (expected_count, hidden_size):
                    raise RuntimeError(
                        f"第 {sample_idx} 条 hidden shape 异常: {tuple(selected.shape)} "
                        f"!= ({expected_count}, {hidden_size})"
                    )
                abs_start = int(prescan["offsets"][sample_idx])
                abs_end = int(prescan["offsets"][sample_idx + 1])
                selected_cpu = selected.detach().float().cpu().numpy()
                hidden_mmap[abs_start:abs_end] = selected_cpu.astype(np_dtype, copy=False)

            del outputs, hidden, teacher_input_ids, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    hidden_mmap.flush()
    del hidden_mmap
    return hidden_path


# ======================================================================
# 9. 最终校验
# ======================================================================
def validate_written_cache(save_dir: str, prescan: dict, hidden_size: int, common_vocab_size: int):
    hidden = np.load(
        os.path.join(save_dir, "teacher_completion_hidden.npy"), mmap_mode="r"
    )
    lm_head = np.load(
        os.path.join(save_dir, "teacher_common_lm_head.npy"), mmap_mode="r"
    )
    expected_hidden_shape = (int(prescan["total_positions"]), int(hidden_size))
    expected_head_shape = (int(common_vocab_size), int(hidden_size))
    if hidden.shape != expected_hidden_shape:
        raise RuntimeError(f"hidden cache shape 错误: {hidden.shape} != {expected_hidden_shape}")
    if lm_head.shape != expected_head_shape:
        raise RuntimeError(f"lm_head cache shape 错误: {lm_head.shape} != {expected_head_shape}")

    for row in sorted(set([0, max(0, hidden.shape[0] // 2), hidden.shape[0] - 1])):
        if not np.isfinite(np.asarray(hidden[row], dtype=np.float32)).all():
            raise FloatingPointError(f"hidden cache 第 {row} 行出现 NaN/Inf。")
    for row in sorted(set([0, max(0, lm_head.shape[0] // 2), lm_head.shape[0] - 1])):
        if not np.isfinite(np.asarray(lm_head[row], dtype=np.float32)).all():
            raise FloatingPointError(f"lm_head cache 第 {row} 行出现 NaN/Inf。")

    depth_offsets = np.load(
        os.path.join(save_dir, "depth_rel_offsets.npy"), mmap_mode="r"
    )
    if depth_offsets.shape != (int(prescan["num_samples"]), MAX_DEPTH):
        raise RuntimeError(
            f"depth_rel_offsets shape 错误: {depth_offsets.shape}"
        )

    print("\n" + "=" * 88)
    print("NYT 离线 Teacher cache 最终校验通过")
    print(f"teacher_completion_hidden: {hidden.shape}, dtype={hidden.dtype}")
    print(f"teacher_common_lm_head    : {lm_head.shape}, dtype={lm_head.dtype}")
    print(f"depth_rel_offsets         : {depth_offsets.shape}, dtype={depth_offsets.dtype}")
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
    if args.expected_max_depth != MAX_DEPTH:
        raise ValueError(f"当前脚本固定 NYT 8 层，expected_max_depth 必须为 {MAX_DEPTH}。")

    if os.path.exists(args.save_dir):
        if not args.overwrite:
            raise FileExistsError(
                f"save_dir 已存在: {args.save_dir}\n"
                "为防止覆盖旧 cache，默认停止；确认重建请传 --overwrite true"
            )
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "=" * 100)
    print("NYT Teacher 离线蒸馏缓存提取")
    print(f"Student tokenizer/base: {args.student_model_path}")
    print(f"Teacher best model    : {args.teacher_model_path}")
    print(f"Train data            : {args.train_data_path}")
    print(f"Label space           : {args.label_space_path}")
    print(f"Save dir              : {args.save_dir}")
    print(f"Device                : {args.device}")
    print("=" * 100 + "\n")

    data = load_json_or_jsonl(args.train_data_path)
    label_space = load_label_space(
        args.label_space_path,
        args.expected_num_labels,
        args.expected_max_depth,
    )
    validate_train_data(data, label_space)
    print(f"读取训练集: {len(data):,} 条")

    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model_path)
    install_student_chat_template(student_tokenizer)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    student_tokenizer.padding_side = "right"

    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_tokenizer.padding_side = "right"

    student_token_ids = validate_atomic_reserved_tokens(student_tokenizer, "Student")
    teacher_token_ids = validate_atomic_reserved_tokens(teacher_tokenizer, "Teacher")
    alignment = build_teacher_student_vocab_alignment(
        student_tokenizer, teacher_tokenizer
    )

    np.save(os.path.join(args.save_dir, "student_common_ids.npy"), alignment["student_common_ids"])
    np.save(os.path.join(args.save_dir, "teacher_common_ids.npy"), alignment["teacher_common_ids"])

    prescan = prescan_dataset(
        data=data,
        student_tokenizer=student_tokenizer,
        alignment=alignment,
        student_token_ids=student_token_ids,
        label_space=label_space,
        system_prompt=args.system_prompt,
        max_seq_length=args.max_seq_length,
        reserve_tokens=args.reserve_tokens,
        scan_limit=args.scan_limit,
    )
    np.save(os.path.join(args.save_dir, "offsets.npy"), prescan["offsets"])
    np.save(os.path.join(args.save_dir, "valid_counts.npy"), prescan["valid_counts"])
    np.save(os.path.join(args.save_dir, "depth_rel_offsets.npy"), prescan["depth_rel_offsets"])
    np.save(os.path.join(args.save_dir, "student_input_sha256.npy"), prescan["input_hashes"])

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 torch.cuda.is_available()=False。")
    device = torch.device(args.device)
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map[args.model_dtype]

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

    teacher_vocab_rows = int(teacher_model.get_output_embeddings().weight.shape[0])
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

    lm_head_info = save_teacher_common_lm_head(
        teacher_model=teacher_model,
        teacher_common_ids=alignment["teacher_common_ids"],
        save_dir=args.save_dir,
        cache_dtype=args.cache_dtype,
        row_chunk_size=args.lm_head_row_chunk_size,
    )

    hidden_path = extract_teacher_hidden_cache(
        data=data,
        prescan=prescan,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        alignment=alignment,
        student_token_ids=student_token_ids,
        label_space=label_space,
        teacher_model=teacher_model,
        system_prompt=args.system_prompt,
        max_seq_length=args.max_seq_length,
        reserve_tokens=args.reserve_tokens,
        extraction_batch_size=args.extraction_batch_size,
        cache_dtype=args.cache_dtype,
        save_dir=args.save_dir,
    )

    manifest = {
        "format_version": 2,
        "dataset": "NYT",
        "split": "train",
        "num_samples": int(prescan["num_samples"]),
        "max_depth": int(label_space["max_depth"]),
        "total_completion_prediction_positions": int(prescan["total_positions"]),
        "teacher_hidden_size": hidden_size,
        "common_vocab_size": int(alignment["common_token_count"]),
        "cache_dtype": args.cache_dtype,
        "student_model_path": args.student_model_path,
        "teacher_model_path": args.teacher_model_path,
        "train_data_path": args.train_data_path,
        "label_space_path": args.label_space_path,
        "max_seq_length": int(args.max_seq_length),
        "reserve_tokens": int(args.reserve_tokens),
        "system_prompt": args.system_prompt,
        "student_reserved_token_ids": {
            "ctx_start": int(student_token_ids["ctx_start"]),
            "ctx_end": int(student_token_ids["ctx_end"]),
            "depth": {str(d): int(student_token_ids["depth"][d]) for d in range(1, 9)},
        },
        "teacher_reserved_token_ids": {
            "ctx_start": int(teacher_token_ids["ctx_start"]),
            "ctx_end": int(teacher_token_ids["ctx_end"]),
            "depth": {str(d): int(teacher_token_ids["depth"][d]) for d in range(1, 9)},
        },
        "token_alignment_rule": "same token string; token IDs may differ",
        "common_token_list_sha256": alignment["common_token_sha256"],
        "student_only_tokens": alignment["student_only_tokens"],
        "teacher_only_tokens": alignment["teacher_only_tokens"],
        "causal_position_rule": (
            "prediction_hidden = teacher.model(...).last_hidden_state[:, :-1, :]; "
            "select positions where labels[:, 1:] != -100"
        ),
        "depth_hidden_rule": (
            "for D1-D8, use the second occurrence of the depth reserved token; "
            "store its relative offset inside each sample's selected Completion "
            "prediction hidden sequence"
        ),
        "traditional_kd_reconstruction": (
            "teacher_common_logits = cached_completion_hidden "
            "@ teacher_common_lm_head.T (+ cached bias if present); "
            "then temperature-scaled KL"
        ),
        "traditional_kd_is_still_logits_kd": True,
        "hidden_align_source": "cached hidden at D1-D8 second special-token occurrence",
        "hidden_hier_source": "adjacent cached depth deltas: h_T_D(k+1) - h_T_Dk, k=1..7",
        "files": {
            "offsets": "offsets.npy",
            "valid_counts": "valid_counts.npy",
            "depth_rel_offsets": "depth_rel_offsets.npy",
            "student_input_sha256": "student_input_sha256.npy",
            "teacher_completion_hidden": os.path.basename(hidden_path),
            "student_common_ids": "student_common_ids.npy",
            "teacher_common_ids": "teacher_common_ids.npy",
            "teacher_common_lm_head": os.path.basename(lm_head_info["weight_path"]),
            "teacher_common_lm_head_bias": (
                os.path.basename(lm_head_info["bias_path"])
                if lm_head_info["bias_path"] is not None else None
            ),
        },
    }
    manifest_path = os.path.join(args.save_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    validate_written_cache(
        args.save_dir,
        prescan,
        hidden_size,
        alignment["common_token_count"],
    )

    print("\n" + "=" * 100)
    print("NYT Teacher 离线缓存提取完成")
    print(f"Cache dir: {args.save_dir}")
    print("后续 Student 蒸馏阶段：")
    print("1) 不再加载 8B Teacher Transformer；")
    print("2) cached hidden × cached Teacher common lm_head 恢复 Teacher logits；")
    print("3) Traditional KD 仍计算 logits KL；")
    print("4) D1-D8 cached hidden 用于 depth hidden Align；")
    print("5) D1->D2 ... D7->D8 相邻差分用于 Hier loss。")
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
            "Extract one-time offline Teacher cache for NYT: Completion final "
            "hidden states + common-vocab Teacher lm_head + D1-D8 relative offsets."
        )
    )

    parser.add_argument(
        "--student_model_path",
        type=str,
        default="/data1/lzp/code/llama-3.2_1b/",
        help="Student base/tokenizer；定义蒸馏训练时的权威 token 序列。",
    )
    parser.add_argument(
        "--teacher_model_path",
        type=str,
        default=(
            "/data1/lzp/code/hierarchical_distill/"
            "checkpoints_teacher_nyt/best_sft_model/"
        ),
        help="当前 NYT 训练完成后导出的 merged best 8B Teacher。",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="/data1/lzp/code/data/nyt/train_formatted.jsonl",
    )
    parser.add_argument(
        "--label_space_path",
        type=str,
        default="/data1/lzp/code/data/nyt/label_space.json",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=(
            "/data1/lzp/code/hierarchical_distill/"
            "teacher_nyt_hiddenstates/"
        ),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_num_labels", type=int, default=166)
    parser.add_argument("--expected_max_depth", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=3072)
    parser.add_argument("--reserve_tokens", type=int, default=1024)
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)

    parser.add_argument("--extraction_batch_size", type=int, default=1)
    parser.add_argument("--lm_head_row_chunk_size", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--model_dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--cache_dtype",
        type=str,
        choices=["float16", "float32"],
        default="float16",
    )
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--scan_limit", type=int, default=-1)
    parser.add_argument("--overwrite", type=str2bool, default=False)

    main(parser.parse_args())
