#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WOS 双 START/END Token Teacher 离线特征提取脚本
================================================

用途
----
使用已经训练完成的 8B Teacher（best_sft_model）只完整前向一次，
为后续“训练好的 Teacher -> 未训练 raw Student”的离线蒸馏建立缓存。

当前双 Token Completion 与 Student/Teacher SFT 保持严格一致：

    L1_START + L1_LABEL + L1_END + L2_START + L2_LABEL + L2_END + EOT

其中：
    L1_START = <|reserved_special_token_112|>
    L1_END   = <|reserved_special_token_113|>
    L2_START = <|reserved_special_token_114|>
    L2_END   = <|reserved_special_token_115|>

Prompt 中上述四个层级 Token 各出现一次用于角色绑定；
Completion 中再次出现。因此层级蒸馏统一使用 Completion 中的第二次 occurrence。

缓存内容
--------
1) Traditional logits KD：
   不直接保存 128K 维 Teacher logits。
   保存所有 Completion causal prediction positions 的 Teacher final hidden：

       H_T_KD: [N_completion_prediction_positions, teacher_hidden_size]

   同时保存 Teacher common-vocab lm_head：

       W_T_common: [V_common, teacher_hidden_size]

   Student 训练时恢复：

       z_T_common = H_T_KD @ W_T_common^T (+ bias)

   然后计算 temperature-scaled KL。
   所以 Traditional KD 仍然是 logits KD，不是 hidden-state KD。

2) 4-token Hidden Align：
   不额外重复保存四份 hidden。
   只保存四个第二次 occurrence 在每个样本 H_T_KD 中的 relative offset：

       l1_start_rel_offset
       l1_end_rel_offset
       l2_start_rel_offset
       l2_end_rel_offset

   训练时直接从同一份 H_T_KD 中读取。

3) 3-edge Hidden Hier：
   后续训练直接由四个 cached hidden 计算：

       L1 internal : h(L1_END)   - h(L1_START)
       L1 -> L2    : h(L2_START) - h(L1_END)
       L2 internal : h(L2_END)   - h(L2_START)

关键一致性
----------
- Student tokenizer 定义权威训练 token 序列；
- Teacher 不重新 tokenize 原文；
- Student token ID 按“相同 token 字符串”翻译到 Teacher ID；
- Prompt / Completion / 截断 / chat template 与当前双 Token SFT 一致；
- 四个层级 Token 全部使用 Completion 中第二次 occurrence；
- L1_END 与 L2_START 必须在 Completion token 序列中严格相邻；
- L2_END 的下一个 target 必须是 EOT；
- Teacher = eval + inference_mode + requires_grad_(False)。

输出格式
--------
save_dir/
  cache_manifest.json
  offsets.npy
  valid_counts.npy

  l1_start_rel_offsets.npy
  l1_end_rel_offsets.npy
  l2_start_rel_offsets.npy
  l2_end_rel_offsets.npy

  student_input_hashes.jsonl

  teacher_completion_hidden.npy
  teacher_common_lm_head.pt

  teacher_student_vocab_alignment.json

该格式与 distill_wos_paired_tokens.py 的 train stage 直接兼容。
"""

import os

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import argparse
import hashlib
import json
import random
import shutil
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed as hf_set_seed,
)


# ======================================================================
# 0. 固定定义
# ======================================================================

CTX_START_TOKEN = "<|reserved_special_token_110|>"
CTX_END_TOKEN = "<|reserved_special_token_111|>"

HIER_START_TOKEN_L1 = "<|reserved_special_token_112|>"
HIER_END_TOKEN_L1 = "<|reserved_special_token_113|>"
HIER_START_TOKEN_L2 = "<|reserved_special_token_114|>"
HIER_END_TOKEN_L2 = "<|reserved_special_token_115|>"

HIER_ROLES = (
    "l1_start",
    "l1_end",
    "l2_start",
    "l2_end",
)

ROLE_TO_TOKEN = {
    "l1_start": HIER_START_TOKEN_L1,
    "l1_end": HIER_END_TOKEN_L1,
    "l2_start": HIER_START_TOKEN_L2,
    "l2_end": HIER_END_TOKEN_L2,
}

REL_OFFSET_FILES = {
    "l1_start": "l1_start_rel_offsets.npy",
    "l1_end": "l1_end_rel_offsets.npy",
    "l2_start": "l2_start_rel_offsets.npy",
    "l2_end": "l2_end_rel_offsets.npy",
}

CACHE_MANIFEST_NAME = "cache_manifest.json"
CACHE_OFFSETS_NAME = "offsets.npy"
CACHE_VALID_COUNTS_NAME = "valid_counts.npy"
CACHE_HASHES_NAME = "student_input_hashes.jsonl"
CACHE_HIDDEN_NAME = "teacher_completion_hidden.npy"
CACHE_COMMON_HEAD_NAME = "teacher_common_lm_head.pt"
CACHE_ALIGNMENT_NAME = "teacher_student_vocab_alignment.json"

REQUIRED_CLEAN_FIELDS = (
    "text",
    "l1_label",
    "l2_label",
    "l1_id",
    "l2_id",
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
    "You are a specialized academic classifier for the WOS two-level taxonomy. "
    "Follow the required output format exactly. "
    "Do not produce explanations or any extra conversational text."
)


# ======================================================================
# 1. 随机种子 / 通用工具
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


def str2bool(v):
    if isinstance(v, bool):
        return v

    v = str(v).lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def numpy_cache_dtype(name: str):
    name = str(name).lower()
    if name == "float16":
        return np.float16
    if name == "float32":
        return np.float32
    raise ValueError(
        f"cache_dtype 只支持 float16/float32，实际={name}"
    )


def torch_cache_dtype(name: str):
    name = str(name).lower()
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(
        f"cache_dtype 只支持 float16/float32，实际={name}"
    )


def sha256_token_ids(token_ids: List[int]) -> str:
    payload = ",".join(str(int(x)) for x in token_ids)
    return hashlib.sha256(
        payload.encode("ascii")
    ).hexdigest()


def find_all_positions(
    ids: List[int],
    token_id: int,
) -> List[int]:
    return [
        i
        for i, x in enumerate(ids)
        if int(x) == int(token_id)
    ]


# ======================================================================
# 2. 与当前双 Token SFT 完全一致的 Prompt / Completion
# ======================================================================

def install_training_chat_template(tokenizer) -> None:
    tokenizer.chat_template = CHAT_TEMPLATE

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token_id is None:
        raise RuntimeError(
            "❌ tokenizer 没有可用 pad_token_id。"
        )

    tokenizer.padding_side = "right"


def build_user_content(text: str) -> str:
    """
    必须与当前双 START/END Token Student/Teacher 训练脚本完全一致。
    """
    return (
        f"{CTX_START_TOKEN}\n"
        f"{text}\n"
        f"{CTX_END_TOKEN}\n\n"

        "Classify the document according to the WOS two-level taxonomy.\n\n"

        "Hierarchy definitions:\n"
        "- Macro Domain (L1): the top-level research domain.\n"
        "- Micro Sub-field (L2): the fine-grained research sub-field "
        "under the corresponding L1 domain.\n\n"

        "Hierarchy-token roles:\n\n"

        f"{HIER_START_TOKEN_L1} is the L1 start token. "
        "It marks the beginning of Macro Domain prediction. "
        "Its representation should capture the document information needed "
        "to predict the L1 label. In the output, the exact L1 label must "
        "begin immediately after this token, with no intervening space "
        "or newline.\n\n"

        f"{HIER_END_TOKEN_L1} is the L1 end token. "
        "It appears immediately after the complete L1 label. "
        "Its representation should summarize the completed L1 prediction "
        "and serve as the hierarchical transition from L1 to L2. "
        "The L2 start token must immediately follow it, with no intervening "
        "space or newline.\n\n"

        f"{HIER_START_TOKEN_L2} is the L2 start token. "
        "It marks the beginning of Micro Sub-field prediction. "
        "Its representation should use both the document information and "
        "the preceding L1 information to predict the L2 label. "
        "The exact L2 label must begin immediately after this token, "
        "with no intervening space or newline.\n\n"

        f"{HIER_END_TOKEN_L2} is the L2 end token. "
        "It appears immediately after the complete L2 label. "
        "Its representation should summarize the L2 information together "
        "with the complete two-level hierarchical prediction.\n\n"

        "Output exactly one continuous hierarchy chain in this semantic order:\n"
        "L1 start token + exact L1 label + L1 end token + "
        "L2 start token + exact L2 label + L2 end token.\n\n"

        "Use the four hierarchy special tokens assigned above themselves; "
        "do not output their role names as ordinary text.\n"
        "Do not insert spaces or line breaks between a hierarchy special "
        "token and its adjacent label or hierarchy special token. "
        "Spaces that are part of a label itself are allowed.\n"
        "Stop immediately after the L2 end token.\n"
        "Do not output numbering, colons, explanations, or any other text."
    )


def build_completion_text(
    l1_label: str,
    l2_label: str,
) -> str:
    """
    Completion：
      L1_START + L1_LABEL + L1_END + L2_START + L2_LABEL + L2_END

    不插入额外普通空格/换行。
    """
    return (
        f"{HIER_START_TOKEN_L1}{l1_label}{HIER_END_TOKEN_L1}"
        f"{HIER_START_TOKEN_L2}{l2_label}{HIER_END_TOKEN_L2}"
    )


def build_prompt_messages(
    text: str,
    system_prompt: str,
) -> List[dict]:
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": build_user_content(text),
        },
    ]


def encode_chat_prompt_consistently(
    tokenizer,
    messages: List[dict],
    sample_idx: int,
) -> List[int]:
    """
    与当前 SFT 一样，检查：
      1) apply_chat_template(tokenize=True)
      2) render 后 add_special_tokens=False 再 tokenize

    两者必须逐 token 相同。
    """
    direct = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    direct = [int(x) for x in direct]

    bos_id = tokenizer.bos_token_id
    if bos_id is not None:
        bos_count = direct.count(int(bos_id))
        if bos_count != 1:
            raise RuntimeError(
                f"❌ TRAIN 第 {sample_idx} 条 Prompt BOS "
                f"应出现 1 次，实际={bos_count}"
            )

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    reencoded = tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    reencoded = [int(x) for x in reencoded]

    if direct != reencoded:
        common = min(
            len(direct),
            len(reencoded),
        )
        mismatch = next(
            (
                i
                for i in range(common)
                if direct[i] != reencoded[i]
            ),
            common,
        )

        raise RuntimeError(
            f"❌ TRAIN 第 {sample_idx} 条 Prompt "
            "两种编码入口不一致。\n"
            f"direct_len={len(direct)}, "
            f"reencoded_len={len(reencoded)}\n"
            f"first_mismatch={mismatch}"
        )

    return direct


def truncate_text_for_prompt(
    text: str,
    tokenizer,
    max_seq_length: int,
    reserve_tokens: int,
) -> str:
    max_text_len = max_seq_length - reserve_tokens

    if max_text_len <= 0:
        raise ValueError(
            f"max_seq_length={max_seq_length} "
            f"必须大于 reserve_tokens={reserve_tokens}"
        )

    text_tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if len(text_tokens) > max_text_len:
        text_tokens = text_tokens[:max_text_len]
        text = tokenizer.decode(
            text_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    return text


# ======================================================================
# 3. 数据读取
# ======================================================================

def load_json_or_jsonl(path: str) -> List[dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"❌ 数据文件不存在: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        raw = f.read().strip()

    if not raw:
        raise ValueError(
            f"❌ 数据文件为空: {path}"
        )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        rows = []

        for line_no, line in enumerate(
            raw.splitlines(),
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            if not isinstance(item, dict):
                raise TypeError(
                    f"❌ JSONL 第 {line_no} 行必须是 dict，"
                    f"实际={type(item).__name__}"
                )

            rows.append(item)

        return rows

    if not isinstance(data, list):
        raise TypeError(
            f"❌ JSON 顶层必须是 list，"
            f"实际={type(data).__name__}"
        )

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(
                f"❌ JSON 第 {idx} 条必须是 dict。"
            )

    return data


def get_clean_sample_fields(
    item: dict,
    split_name: str,
    idx: int,
) -> Tuple[str, str, str, int, int]:
    missing = [
        k
        for k in REQUIRED_CLEAN_FIELDS
        if k not in item
    ]

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
        raise ValueError(
            f"{split_name} 第 {idx} 条 text "
            "必须是非空字符串。"
        )

    if (
        not isinstance(l1_label, str)
        or not l1_label.strip()
    ):
        raise ValueError(
            f"{split_name} 第 {idx} 条 l1_label "
            "必须是非空字符串。"
        )

    if (
        not isinstance(l2_label, str)
        or not l2_label.strip()
    ):
        raise ValueError(
            f"{split_name} 第 {idx} 条 l2_label "
            "必须是非空字符串。"
        )

    if (
        isinstance(l1_id, bool)
        or not isinstance(l1_id, int)
    ):
        raise TypeError(
            f"{split_name} 第 {idx} 条 l1_id "
            "必须是 int。"
        )

    if (
        isinstance(l2_id, bool)
        or not isinstance(l2_id, int)
    ):
        raise TypeError(
            f"{split_name} 第 {idx} 条 l2_id "
            "必须是 int。"
        )

    return (
        text.strip(),
        l1_label.strip(),
        l2_label.strip(),
        int(l1_id),
        int(l2_id),
    )


# ======================================================================
# 4. 六个 Reserved Token + Teacher/Student 词表对齐
# ======================================================================

def validate_atomic_reserved_tokens(
    tokenizer,
    name: str,
) -> Dict[str, int]:
    token_map = {
        "ctx_start": CTX_START_TOKEN,
        "ctx_end": CTX_END_TOKEN,
        "l1_start": HIER_START_TOKEN_L1,
        "l1_end": HIER_END_TOKEN_L1,
        "l2_start": HIER_START_TOKEN_L2,
        "l2_end": HIER_END_TOKEN_L2,
    }

    vocab = tokenizer.get_vocab()
    ids = {}

    print("\n" + "=" * 88)
    print(
        f"🔎 验证 {name} 六个 WOS reserved tokens"
    )

    for role, token in token_map.items():
        if token not in vocab:
            raise RuntimeError(
                f"❌ {name} 缺少 token: {token}"
            )

        encoded = tokenizer.encode(
            token,
            add_special_tokens=False,
        )
        converted = tokenizer.convert_tokens_to_ids(
            token
        )

        if len(encoded) != 1:
            raise RuntimeError(
                f"❌ {name} 的 {token} 不是单 token，"
                f"encoded={encoded}"
            )

        token_id = int(encoded[0])

        if token_id != int(converted):
            raise RuntimeError(
                f"❌ {name} 的 {token} encode/convert ID "
                f"不一致: {token_id} != {converted}"
            )

        if token_id == tokenizer.unk_token_id:
            raise RuntimeError(
                f"❌ {name} 的 {token} 被编码成 unk。"
            )

        ids[role] = token_id
        print(
            f"✅ {role:>9s}: {token} -> {token_id}"
        )

    if len(set(ids.values())) != 6:
        raise RuntimeError(
            f"❌ {name} 六个 reserved token ID "
            f"发生重复: {ids}"
        )

    print("=" * 88)
    return ids


def build_teacher_student_vocab_alignment(
    student_tokenizer,
    teacher_tokenizer,
) -> dict:
    """
    按 token 字符串语义对齐，而不是数字 ID 对齐。

    common token 按 Student ID 排序，保证：
      Student logits[:, student_common_ids]
      Teacher logits[:, teacher_common_ids]

    两边每一列代表完全相同的 token 字符串。
    """
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()

    common_tokens = (
        set(student_vocab)
        & set(teacher_vocab)
    )

    if not common_tokens:
        raise RuntimeError(
            "❌ Teacher / Student 没有共同 token。"
        )

    protected_tokens = [
        CTX_START_TOKEN,
        CTX_END_TOKEN,
        HIER_START_TOKEN_L1,
        HIER_END_TOKEN_L1,
        HIER_START_TOKEN_L2,
        HIER_END_TOKEN_L2,
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ]

    protected_info = {}

    print("\n" + "=" * 92)
    print("🔐 Teacher / Student protected-token 检查")

    for token in protected_tokens:
        if (
            token not in student_vocab
            or token not in teacher_vocab
        ):
            raise RuntimeError(
                "❌ Teacher/Student 至少一边缺少 "
                f"必需 token: {token}"
            )

        s_ids = student_tokenizer.encode(
            token,
            add_special_tokens=False,
        )
        t_ids = teacher_tokenizer.encode(
            token,
            add_special_tokens=False,
        )

        if len(s_ids) != 1 or len(t_ids) != 1:
            raise RuntimeError(
                f"❌ 必需 token 必须在两边都是原子 token: "
                f"{token}; student={s_ids}, teacher={t_ids}"
            )

        if token not in common_tokens:
            raise RuntimeError(
                f"❌ protected token 未进入 common vocab: "
                f"{token}"
            )

        protected_info[token] = {
            "student_id": int(student_vocab[token]),
            "teacher_id": int(teacher_vocab[token]),
        }

        print(
            f"✅ {token} | "
            f"Student={student_vocab[token]} | "
            f"Teacher={teacher_vocab[token]}"
        )

    common_sorted = sorted(
        common_tokens,
        key=lambda tok: int(student_vocab[tok]),
    )

    student_common_ids = np.asarray(
        [
            int(student_vocab[tok])
            for tok in common_sorted
        ],
        dtype=np.int64,
    )

    teacher_common_ids = np.asarray(
        [
            int(teacher_vocab[tok])
            for tok in common_sorted
        ],
        dtype=np.int64,
    )

    student_to_teacher = np.full(
        len(student_tokenizer),
        -1,
        dtype=np.int64,
    )

    for tok in common_sorted:
        sid = int(student_vocab[tok])
        tid = int(teacher_vocab[tok])
        student_to_teacher[sid] = tid

    common_token_sha256 = hashlib.sha256(
        "\n".join(common_sorted).encode("utf-8")
    ).hexdigest()

    student_only = sorted(
        set(student_vocab)
        - set(teacher_vocab)
    )
    teacher_only = sorted(
        set(teacher_vocab)
        - set(student_vocab)
    )

    same_token_wrong_id_count = sum(
        int(student_vocab[tok])
        != int(teacher_vocab[tok])
        for tok in common_sorted
    )

    print("\n📚 Teacher / Student token-string alignment")
    print(
        f"Student vocab        : {len(student_vocab):,}"
    )
    print(
        f"Teacher vocab        : {len(teacher_vocab):,}"
    )
    print(
        f"Common vocab         : {len(common_sorted):,}"
    )
    print(
        f"Student-only         : {len(student_only):,}"
    )
    print(
        f"Teacher-only         : {len(teacher_only):,}"
    )
    print(
        "Common token ID 不同 : "
        f"{same_token_wrong_id_count:,}"
    )
    print(
        f"Common token SHA256  : {common_token_sha256}"
    )
    print(
        "✅ 后续 Teacher 输入由 Student IDs "
        "按 token 字符串一一翻译，不重新 tokenize 原文"
    )
    print("=" * 92 + "\n")

    return {
        "student_common_ids": student_common_ids,
        "teacher_common_ids": teacher_common_ids,
        "student_to_teacher": student_to_teacher,
        "common_token_count": len(common_sorted),
        "common_token_sha256": common_token_sha256,
        "student_only_tokens": student_only,
        "teacher_only_tokens": teacher_only,
        "same_token_wrong_id_count": (
            same_token_wrong_id_count
        ),
        "protected_info": protected_info,
    }


def save_alignment_summary(
    save_dir: str,
    alignment: dict,
) -> None:
    payload = {
        "alignment_rule": (
            "same token string; numeric token IDs may differ"
        ),
        "kd_vocab_rule": (
            "intersection of Student/Teacher token strings"
        ),
        "common_token_count": int(
            alignment["common_token_count"]
        ),
        "common_token_sha256": alignment[
            "common_token_sha256"
        ],
        "same_token_wrong_id_count": int(
            alignment["same_token_wrong_id_count"]
        ),
        "student_only_tokens": alignment[
            "student_only_tokens"
        ],
        "teacher_only_tokens": alignment[
            "teacher_only_tokens"
        ],
        "protected_tokens": alignment[
            "protected_info"
        ],
        "protected_tokens_are_never_dropped": True,
    }

    path = os.path.join(
        save_dir,
        CACHE_ALIGNMENT_NAME,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"✅ 词表对齐摘要已保存: {path}"
    )


# ======================================================================
# 5. 构造单样本：严格复制当前双 Token 训练序列
# ======================================================================

def audit_prompt_body_span(
    prompt_ids: List[int],
    truncated_text: str,
    student_tokenizer,
    student_token_ids: Dict[str, int],
    sample_idx: int,
) -> None:
    """
    严格确认 CTX_START / CTX_END 内部就是 "\n" + text + "\n"。
    """
    start_positions = find_all_positions(
        prompt_ids,
        student_token_ids["ctx_start"],
    )
    end_positions = find_all_positions(
        prompt_ids,
        student_token_ids["ctx_end"],
    )

    if (
        len(start_positions) != 1
        or len(end_positions) != 1
    ):
        raise RuntimeError(
            f"❌ TRAIN 第 {sample_idx} 条正文边界 occurrence 异常: "
            f"START={start_positions}, END={end_positions}"
        )

    start = start_positions[0]
    end = end_positions[0]

    if end <= start + 1:
        raise RuntimeError(
            f"❌ TRAIN 第 {sample_idx} 条正文边界顺序异常。"
        )

    actual = [
        int(x)
        for x in prompt_ids[start + 1:end]
    ]

    expected = student_tokenizer.encode(
        f"\n{truncated_text}\n",
        add_special_tokens=False,
    )
    expected = [int(x) for x in expected]

    if actual != expected:
        raise RuntimeError(
            f"❌ TRAIN 第 {sample_idx} 条正文 token span "
            "与截断正文不一致。"
        )


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
    text, l1_label, l2_label, _, _ = (
        get_clean_sample_fields(
            item,
            "TRAIN",
            idx,
        )
    )

    truncated_text = truncate_text_for_prompt(
        text=text,
        tokenizer=student_tokenizer,
        max_seq_length=max_seq_length,
        reserve_tokens=reserve_tokens,
    )

    messages = build_prompt_messages(
        truncated_text,
        system_prompt,
    )

    prompt_ids = encode_chat_prompt_consistently(
        tokenizer=student_tokenizer,
        messages=messages,
        sample_idx=idx,
    )

    # Prompt 中 6 个 reserved token occurrence 必须符合当前 SFT。
    expected_prompt_counts = {
        student_token_ids["ctx_start"]: 1,
        student_token_ids["ctx_end"]: 1,
        student_token_ids["l1_start"]: 1,
        student_token_ids["l1_end"]: 1,
        student_token_ids["l2_start"]: 1,
        student_token_ids["l2_end"]: 1,
    }

    for token_id, expected_count in (
        expected_prompt_counts.items()
    ):
        actual_count = prompt_ids.count(
            int(token_id)
        )
        if actual_count != expected_count:
            raise RuntimeError(
                f"❌ TRAIN 第 {idx} 条 Prompt 中 token_id={token_id} "
                f"应出现 {expected_count} 次，实际={actual_count}"
            )

    audit_prompt_body_span(
        prompt_ids=prompt_ids,
        truncated_text=truncated_text,
        student_tokenizer=student_tokenizer,
        student_token_ids=student_token_ids,
        sample_idx=idx,
    )

    completion_ids = student_tokenizer.encode(
        build_completion_text(
            l1_label,
            l2_label,
        ),
        add_special_tokens=False,
    )
    completion_ids = [
        int(x)
        for x in completion_ids
    ]

    eot_id = student_tokenizer.convert_tokens_to_ids(
        "<|eot_id|>"
    )

    if (
        eot_id is None
        or eot_id == student_tokenizer.unk_token_id
    ):
        raise RuntimeError(
            "❌ Student tokenizer 缺少原子 <|eot_id|>。"
        )

    eot_id = int(eot_id)

    student_input_ids = (
        prompt_ids
        + completion_ids
        + [eot_id]
    )

    labels = (
        [-100] * len(prompt_ids)
        + completion_ids
        + [eot_id]
    )

    if len(student_input_ids) > max_seq_length:
        raise RuntimeError(
            f"❌ TRAIN 第 {idx} 条最终长度 "
            f"{len(student_input_ids)} "
            f"> max_seq_length={max_seq_length}"
        )

    # --------------------------------------------------------------
    # 四个层级 Token：完整样本中各必须出现 2 次。
    # --------------------------------------------------------------
    position_lists = {}

    for role in HIER_ROLES:
        position_lists[role] = find_all_positions(
            student_input_ids,
            student_token_ids[role],
        )

        if len(position_lists[role]) != 2:
            raise RuntimeError(
                f"❌ TRAIN 第 {idx} 条 {role} "
                f"应出现 2 次，实际={position_lists[role]}"
            )

    second_positions = {
        role: int(position_lists[role][1])
        for role in HIER_ROLES
    }

    # Prompt 四个 role token 的第一次 occurrence 顺序。
    first_positions = {
        role: int(position_lists[role][0])
        for role in HIER_ROLES
    }

    if not (
        first_positions["l1_start"]
        < first_positions["l1_end"]
        < first_positions["l2_start"]
        < first_positions["l2_end"]
        < second_positions["l1_start"]
        < second_positions["l1_end"]
        < second_positions["l2_start"]
        < second_positions["l2_end"]
    ):
        raise RuntimeError(
            f"❌ TRAIN 第 {idx} 条四个 hierarchy token "
            f"occurrence 顺序异常:\n"
            f"first={first_positions}\n"
            f"second={second_positions}"
        )

    completion_start = len(prompt_ids)

    if any(
        pos < completion_start
        for pos in second_positions.values()
    ):
        raise RuntimeError(
            f"❌ TRAIN 第 {idx} 条某个 hierarchy token "
            "第二次 occurrence 未落在 Completion。"
        )

    # 当前设计中 L1_END 直接预测 L2_START。
    if (
        second_positions["l2_start"]
        != second_positions["l1_end"] + 1
    ):
        between = student_input_ids[
            second_positions["l1_end"] + 1
            : second_positions["l2_start"]
        ]

        raise RuntimeError(
            f"❌ TRAIN 第 {idx} 条 L1_END 与 L2_START "
            "之间存在额外 token，无法保持当前连续层级链。 "
            f"between={between}"
        )

    # L1/L2 标签不能为空。
    if (
        second_positions["l1_end"]
        <= second_positions["l1_start"] + 1
    ):
        raise RuntimeError(
            f"❌ TRAIN 第 {idx} 条 L1_START/L1_END "
            "之间没有有效 L1 标签 token。"
        )

    if (
        second_positions["l2_end"]
        <= second_positions["l2_start"] + 1
    ):
        raise RuntimeError(
            f"❌ TRAIN 第 {idx} 条 L2_START/L2_END "
            "之间没有有效 L2 标签 token。"
        )

    # 当前设计：L2_END 下一个 target 必须就是 EOT。
    if (
        second_positions["l2_end"] + 1
        >= len(student_input_ids)
        or student_input_ids[
            second_positions["l2_end"] + 1
        ] != eot_id
    ):
        raise RuntimeError(
            f"❌ TRAIN 第 {idx} 条 L2_END 后不是 EOT。"
        )

    # --------------------------------------------------------------
    # Causal shift:
    # logits/prediction hidden position p
    # 对应 labels[p+1]
    # --------------------------------------------------------------
    shift_labels = labels[1:]

    valid_positions = [
        pos
        for pos, target in enumerate(shift_labels)
        if int(target) != -100
    ]

    if not valid_positions:
        raise RuntimeError(
            f"❌ TRAIN 第 {idx} 条没有有效 "
            "Completion causal prediction positions。"
        )

    valid_pos_to_rel = {
        int(pos): rel
        for rel, pos in enumerate(valid_positions)
    }

    rel_offsets = {}

    for role in HIER_ROLES:
        pos = second_positions[role]

        if pos not in valid_pos_to_rel:
            raise RuntimeError(
                f"❌ TRAIN 第 {idx} 条 {role} "
                f"第二次 occurrence position={pos} "
                "不在 causal KD mask 中。"
            )

        rel_offsets[role] = int(
            valid_pos_to_rel[pos]
        )

    # --------------------------------------------------------------
    # Student IDs -> Teacher IDs
    # 严格按 token string 一一翻译。
    # --------------------------------------------------------------
    student_ids_np = np.asarray(
        student_input_ids,
        dtype=np.int64,
    )

    if (
        student_ids_np.min() < 0
        or student_ids_np.max()
        >= len(student_to_teacher)
    ):
        raise RuntimeError(
            f"❌ TRAIN 第 {idx} 条 Student ID "
            "超出翻译表范围。"
        )

    teacher_ids_np = student_to_teacher[
        student_ids_np
    ]

    if np.any(teacher_ids_np < 0):
        bad = np.unique(
            student_ids_np[
                teacher_ids_np < 0
            ]
        ).tolist()

        bad_tokens = [
            student_tokenizer.convert_ids_to_tokens(
                int(x)
            )
            for x in bad[:30]
        ]

        raise RuntimeError(
            f"❌ TRAIN 第 {idx} 条使用了 Teacher "
            "不存在的 Student token。\n"
            f"student_ids={bad[:30]}\n"
            f"tokens={bad_tokens}"
        )

    return {
        "student_input_ids": student_input_ids,
        "teacher_input_ids": (
            teacher_ids_np.tolist()
        ),
        "valid_positions": valid_positions,
        "valid_count": len(valid_positions),
        "second_positions": second_positions,
        "rel_offsets": rel_offsets,
        "student_input_sha256": (
            sha256_token_ids(
                student_input_ids
            )
        ),
    }


# ======================================================================
# 6. 第一遍预扫描
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
    num_samples = (
        len(data)
        if scan_limit < 0
        else min(
            len(data),
            scan_limit,
        )
    )

    if num_samples <= 0:
        raise ValueError(
            "❌ 实际提取样本数必须 > 0。"
        )

    valid_counts = np.zeros(
        num_samples,
        dtype=np.int32,
    )

    rel_offsets = {
        role: np.zeros(
            num_samples,
            dtype=np.int32,
        )
        for role in HIER_ROLES
    }

    input_hashes = []

    print("\n" + "=" * 92)
    print(
        f"🔎 第一遍预扫描：{num_samples:,} 条"
    )
    print(
        "目的：确定 Completion causal position 数、"
        "四个 hierarchy-token relative offsets 和总 cache 大小"
    )
    print("=" * 92)

    for idx in tqdm(
        range(num_samples),
        desc="Prescan",
    ):
        sample = build_one_sample(
            item=data[idx],
            idx=idx,
            student_tokenizer=student_tokenizer,
            student_to_teacher=alignment[
                "student_to_teacher"
            ],
            student_token_ids=student_token_ids,
            system_prompt=system_prompt,
            max_seq_length=max_seq_length,
            reserve_tokens=reserve_tokens,
        )

        valid_counts[idx] = int(
            sample["valid_count"]
        )

        for role in HIER_ROLES:
            rel_offsets[role][idx] = int(
                sample["rel_offsets"][role]
            )

        input_hashes.append(
            sample["student_input_sha256"]
        )

    offsets = np.zeros(
        num_samples + 1,
        dtype=np.int64,
    )

    offsets[1:] = np.cumsum(
        valid_counts,
        dtype=np.int64,
    )

    total_positions = int(
        offsets[-1]
    )

    if total_positions <= 0:
        raise RuntimeError(
            "❌ 总 Completion causal position 数为 0。"
        )

    for role in HIER_ROLES:
        arr = rel_offsets[role]

        if (
            np.any(arr < 0)
            or np.any(arr >= valid_counts)
        ):
            raise RuntimeError(
                f"❌ 存在非法 {role} relative offset。"
            )

    print("\n✅ 预扫描完成")
    print(
        f"样本数: {num_samples:,}"
    )
    print(
        "总有效 Completion causal prediction positions: "
        f"{total_positions:,}"
    )
    print(
        "平均每样本有效位置数: "
        f"{total_positions / num_samples:.3f}"
    )
    print(
        "每样本有效位置范围: "
        f"{int(valid_counts.min())} ~ "
        f"{int(valid_counts.max())}"
    )

    for role in HIER_ROLES:
        print(
            f"{role:>9s} relative offset range: "
            f"{int(rel_offsets[role].min())} ~ "
            f"{int(rel_offsets[role].max())}"
        )

    return {
        "num_samples": num_samples,
        "valid_counts": valid_counts,
        "offsets": offsets,
        "rel_offsets": rel_offsets,
        "input_hashes": input_hashes,
        "total_positions": total_positions,
    }


def save_prescan_metadata(
    save_dir: str,
    prescan: dict,
) -> None:
    np.save(
        os.path.join(
            save_dir,
            CACHE_OFFSETS_NAME,
        ),
        prescan["offsets"],
    )

    np.save(
        os.path.join(
            save_dir,
            CACHE_VALID_COUNTS_NAME,
        ),
        prescan["valid_counts"],
    )

    for role in HIER_ROLES:
        np.save(
            os.path.join(
                save_dir,
                REL_OFFSET_FILES[role],
            ),
            prescan["rel_offsets"][role],
        )

    hashes_path = os.path.join(
        save_dir,
        CACHE_HASHES_NAME,
    )

    with open(
        hashes_path,
        "w",
        encoding="utf-8",
    ) as f:
        for idx in range(
            prescan["num_samples"]
        ):
            record = {
                "sample_index": int(idx),
                "input_hash": prescan[
                    "input_hashes"
                ][idx],
                "valid_count": int(
                    prescan["valid_counts"][idx]
                ),
            }

            for role in HIER_ROLES:
                record[
                    f"{role}_rel"
                ] = int(
                    prescan[
                        "rel_offsets"
                    ][role][idx]
                )

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"✅ Prescan metadata 已保存到: {save_dir}"
    )


# ======================================================================
# 7. 保存 Teacher common-vocab lm_head
# ======================================================================

def save_teacher_common_lm_head_pt(
    teacher_model,
    alignment: dict,
    save_dir: str,
    cache_dtype: str,
    row_chunk_size: int,
):
    """
    为兼容后续 distill_wos_paired_tokens.py，
    保存成单个 .pt payload。

    为避免 GPU 上一次性复制全部 common-vocab rows，
    按 chunk 从 GPU -> CPU 写入预分配 CPU Tensor。
    """
    lm_head = (
        teacher_model.get_output_embeddings()
    )

    if lm_head is None:
        raise RuntimeError(
            "❌ Teacher 缺少 lm_head/output embeddings。"
        )

    weight = lm_head.weight
    hidden_size = int(
        weight.shape[1]
    )
    common_size = int(
        alignment["common_token_count"]
    )

    disk_dtype = torch_cache_dtype(
        cache_dtype
    )

    common_weight = torch.empty(
        (
            common_size,
            hidden_size,
        ),
        dtype=disk_dtype,
        device="cpu",
    )

    teacher_common_ids = alignment[
        "teacher_common_ids"
    ]

    print("\n" + "=" * 92)
    print(
        "💾 保存 Teacher common-vocab lm_head"
    )
    print(
        f"common vocab : {common_size:,}"
    )
    print(
        f"hidden size  : {hidden_size}"
    )
    print(
        f"cache dtype  : {cache_dtype}"
    )
    print(
        f"chunk rows   : {row_chunk_size}"
    )
    print("=" * 92)

    with torch.inference_mode():
        for start in tqdm(
            range(
                0,
                common_size,
                row_chunk_size,
            ),
            desc="Save common lm_head",
        ):
            end = min(
                start + row_chunk_size,
                common_size,
            )

            ids = torch.as_tensor(
                teacher_common_ids[
                    start:end
                ],
                dtype=torch.long,
                device=weight.device,
            )

            rows = weight.index_select(
                0,
                ids,
            )

            common_weight[
                start:end
            ].copy_(
                rows.detach()
                .to(
                    device="cpu",
                    dtype=disk_dtype,
                )
            )

    common_bias = None

    if getattr(
        lm_head,
        "bias",
        None,
    ) is not None:
        bias = lm_head.bias
        common_bias = torch.empty(
            (common_size,),
            dtype=disk_dtype,
            device="cpu",
        )

        with torch.inference_mode():
            for start in range(
                0,
                common_size,
                row_chunk_size,
            ):
                end = min(
                    start + row_chunk_size,
                    common_size,
                )

                ids = torch.as_tensor(
                    teacher_common_ids[
                        start:end
                    ],
                    dtype=torch.long,
                    device=bias.device,
                )

                rows = bias.index_select(
                    0,
                    ids,
                )

                common_bias[
                    start:end
                ].copy_(
                    rows.detach()
                    .to(
                        device="cpu",
                        dtype=disk_dtype,
                    )
                )

    payload = {
        "weight": common_weight,
        "bias": common_bias,
        "student_common_ids": (
            torch.from_numpy(
                alignment[
                    "student_common_ids"
                ].copy()
            ).long()
        ),
        "teacher_common_ids": (
            torch.from_numpy(
                alignment[
                    "teacher_common_ids"
                ].copy()
            ).long()
        ),
        "teacher_hidden_size": hidden_size,
    }

    path = os.path.join(
        save_dir,
        CACHE_COMMON_HEAD_NAME,
    )

    torch.save(
        payload,
        path,
    )

    print(
        f"✅ Teacher common lm_head 已保存: {path}"
    )

    return {
        "path": path,
        "hidden_size": hidden_size,
        "common_size": common_size,
    }


# ======================================================================
# 8. Teacher batch padding
# ======================================================================

def pad_teacher_batch(
    samples: List[dict],
    teacher_pad_token_id: int,
    device: torch.device,
):
    max_len = max(
        len(x["teacher_input_ids"])
        for x in samples
    )

    batch_size = len(samples)

    input_ids = torch.full(
        (
            batch_size,
            max_len,
        ),
        int(teacher_pad_token_id),
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.zeros(
        (
            batch_size,
            max_len,
        ),
        dtype=torch.long,
        device=device,
    )

    for row, sample in enumerate(
        samples
    ):
        ids = sample[
            "teacher_input_ids"
        ]
        n = len(ids)

        input_ids[
            row,
            :n,
        ] = torch.tensor(
            ids,
            dtype=torch.long,
            device=device,
        )

        attention_mask[
            row,
            :n,
        ] = 1

    return (
        input_ids,
        attention_mask,
    )


# ======================================================================
# 9. Teacher final hidden 提取
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
    """
    AutoModelForCausalLM(Llama) 的 teacher.model 是 LlamaModel。
    teacher.model(...).last_hidden_state 已经过 Llama final norm，
    正好是 lm_head 的直接输入表示。
    """
    teacher_decoder = getattr(
        teacher_model,
        "model",
        None,
    )

    if teacher_decoder is None:
        raise RuntimeError(
            "❌ 当前 Teacher 不是预期的 LlamaForCausalLM "
            "结构，找不到 teacher.model。"
        )

    hidden_size = int(
        teacher_model.config.hidden_size
    )

    total_positions = int(
        prescan["total_positions"]
    )
    num_samples = int(
        prescan["num_samples"]
    )

    np_dtype = numpy_cache_dtype(
        cache_dtype
    )

    hidden_path = os.path.join(
        save_dir,
        CACHE_HIDDEN_NAME,
    )

    hidden_mmap = (
        np.lib.format.open_memmap(
            hidden_path,
            mode="w+",
            dtype=np_dtype,
            shape=(
                total_positions,
                hidden_size,
            ),
        )
    )

    teacher_pad_id = (
        teacher_tokenizer.pad_token_id
    )

    if teacher_pad_id is None:
        teacher_pad_id = (
            teacher_tokenizer.eos_token_id
        )

    if teacher_pad_id is None:
        teacher_pad_id = 0

    device = next(
        teacher_model.parameters()
    ).device

    print("\n" + "=" * 92)
    print(
        "🧊 第二遍：训练后的 Teacher 真正前向，"
        "写入 Completion final hidden cache"
    )
    print(
        f"样本数              : {num_samples:,}"
    )
    print(
        f"总 KD positions      : {total_positions:,}"
    )
    print(
        f"Teacher hidden size  : {hidden_size}"
    )
    print(
        f"Cache dtype          : {cache_dtype}"
    )
    print(
        f"Extraction batch size: {extraction_batch_size}"
    )
    print(
        f"Cache path           : {hidden_path}"
    )
    print("=" * 92)

    teacher_model.eval()
    teacher_model.requires_grad_(False)
    teacher_model.config.use_cache = False

    trainable_count = sum(
        p.numel()
        for p in teacher_model.parameters()
        if p.requires_grad
    )

    if trainable_count != 0:
        raise RuntimeError(
            f"❌ Teacher 仍有可训练参数: "
            f"{trainable_count:,}"
        )

    with torch.inference_mode():
        for batch_start in tqdm(
            range(
                0,
                num_samples,
                extraction_batch_size,
            ),
            desc="Teacher extract",
        ):
            batch_end = min(
                batch_start
                + extraction_batch_size,
                num_samples,
            )

            batch_samples = []

            for idx in range(
                batch_start,
                batch_end,
            ):
                sample = build_one_sample(
                    item=data[idx],
                    idx=idx,
                    student_tokenizer=student_tokenizer,
                    student_to_teacher=alignment[
                        "student_to_teacher"
                    ],
                    student_token_ids=student_token_ids,
                    system_prompt=system_prompt,
                    max_seq_length=max_seq_length,
                    reserve_tokens=reserve_tokens,
                )

                # 与第一遍 prescan 严格核对。
                if (
                    sample["valid_count"]
                    != int(
                        prescan[
                            "valid_counts"
                        ][idx]
                    )
                ):
                    raise RuntimeError(
                        f"❌ 第 {idx} 条二次构造 valid_count "
                        "与 prescan 不一致。"
                    )

                for role in HIER_ROLES:
                    if (
                        sample[
                            "rel_offsets"
                        ][role]
                        != int(
                            prescan[
                                "rel_offsets"
                            ][role][idx]
                        )
                    ):
                        raise RuntimeError(
                            f"❌ 第 {idx} 条 {role} "
                            "relative offset 与 prescan 不一致。"
                        )

                expected_hash = prescan[
                    "input_hashes"
                ][idx]

                if (
                    sample[
                        "student_input_sha256"
                    ]
                    != expected_hash
                ):
                    raise RuntimeError(
                        f"❌ 第 {idx} 条 Student token hash "
                        "与 prescan 不一致。"
                    )

                batch_samples.append(
                    sample
                )

            (
                teacher_input_ids,
                attention_mask,
            ) = pad_teacher_batch(
                batch_samples,
                teacher_pad_token_id=teacher_pad_id,
                device=device,
            )

            # 只跑 Llama decoder，不需要真的计算 128K logits。
            outputs = teacher_decoder(
                input_ids=teacher_input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

            hidden = (
                outputs.last_hidden_state
            )

            if (
                hidden.size(-1)
                != hidden_size
            ):
                raise RuntimeError(
                    "❌ Teacher last_hidden_state "
                    f"最后一维异常: {hidden.size(-1)} "
                    f"!= {hidden_size}"
                )

            for row, sample in enumerate(
                batch_samples
            ):
                sample_idx = (
                    batch_start + row
                )
                actual_len = len(
                    sample[
                        "teacher_input_ids"
                    ]
                )

                if actual_len < 2:
                    raise RuntimeError(
                        f"❌ 第 {sample_idx} 条序列长度不足 2。"
                    )

                # prediction hidden position p
                # 对应 target label[p+1]
                prediction_hidden = hidden[
                    row,
                    : actual_len - 1,
                    :,
                ]

                valid_positions = torch.tensor(
                    sample[
                        "valid_positions"
                    ],
                    dtype=torch.long,
                    device=prediction_hidden.device,
                )

                selected = (
                    prediction_hidden.index_select(
                        dim=0,
                        index=valid_positions,
                    )
                )

                expected_count = int(
                    prescan[
                        "valid_counts"
                    ][sample_idx]
                )

                if selected.shape != (
                    expected_count,
                    hidden_size,
                ):
                    raise RuntimeError(
                        f"❌ 第 {sample_idx} 条 hidden shape "
                        f"异常: {tuple(selected.shape)} != "
                        f"({expected_count}, {hidden_size})"
                    )

                abs_start = int(
                    prescan[
                        "offsets"
                    ][sample_idx]
                )
                abs_end = int(
                    prescan[
                        "offsets"
                    ][sample_idx + 1]
                )

                selected_cpu = (
                    selected.detach()
                    .float()
                    .cpu()
                    .numpy()
                )

                if np_dtype == np.float16:
                    selected_cpu = (
                        selected_cpu.astype(
                            np.float16,
                            copy=False,
                        )
                    )
                else:
                    selected_cpu = (
                        selected_cpu.astype(
                            np.float32,
                            copy=False,
                        )
                    )

                hidden_mmap[
                    abs_start:abs_end
                ] = selected_cpu

            del (
                outputs,
                hidden,
                teacher_input_ids,
                attention_mask,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    hidden_mmap.flush()
    del hidden_mmap

    return hidden_path


# ======================================================================
# 10. 最终 cache 校验
# ======================================================================

def validate_written_cache(
    save_dir: str,
    prescan: dict,
    hidden_size: int,
    common_vocab_size: int,
):
    hidden = np.load(
        os.path.join(
            save_dir,
            CACHE_HIDDEN_NAME,
        ),
        mmap_mode="r",
    )

    head_payload = torch.load(
        os.path.join(
            save_dir,
            CACHE_COMMON_HEAD_NAME,
        ),
        map_location="cpu",
    )

    lm_head = head_payload[
        "weight"
    ]

    expected_hidden_shape = (
        int(
            prescan[
                "total_positions"
            ]
        ),
        int(hidden_size),
    )

    expected_head_shape = (
        int(common_vocab_size),
        int(hidden_size),
    )

    if hidden.shape != expected_hidden_shape:
        raise RuntimeError(
            f"❌ hidden cache shape 错误: "
            f"{hidden.shape} != "
            f"{expected_hidden_shape}"
        )

    if tuple(lm_head.shape) != (
        expected_head_shape
    ):
        raise RuntimeError(
            f"❌ lm_head cache shape 错误: "
            f"{tuple(lm_head.shape)} != "
            f"{expected_head_shape}"
        )

    # 检查 hidden 开头/中间/结尾。
    probe_rows = sorted(
        set(
            [
                0,
                max(
                    0,
                    hidden.shape[0] // 2,
                ),
                hidden.shape[0] - 1,
            ]
        )
    )

    for row in probe_rows:
        arr = np.asarray(
            hidden[row],
            dtype=np.float32,
        )

        if not np.isfinite(arr).all():
            raise FloatingPointError(
                f"❌ hidden cache 第 {row} 行 "
                "出现 NaN/Inf。"
            )

    # 四个 hierarchy-token hidden 都逐个抽查。
    sample_probes = sorted(
        set(
            [
                0,
                max(
                    0,
                    prescan[
                        "num_samples"
                    ] // 2,
                ),
                prescan[
                    "num_samples"
                ] - 1,
            ]
        )
    )

    for sample_idx in sample_probes:
        start = int(
            prescan[
                "offsets"
            ][sample_idx]
        )

        for role in HIER_ROLES:
            rel = int(
                prescan[
                    "rel_offsets"
                ][role][sample_idx]
            )

            arr = np.asarray(
                hidden[
                    start + rel
                ],
                dtype=np.float32,
            )

            if not np.isfinite(
                arr
            ).all():
                raise FloatingPointError(
                    f"❌ sample={sample_idx}, "
                    f"role={role} cached hidden "
                    "出现 NaN/Inf。"
                )

    # lm_head 抽查。
    head_probe_rows = sorted(
        set(
            [
                0,
                max(
                    0,
                    lm_head.shape[0] // 2,
                ),
                lm_head.shape[0] - 1,
            ]
        )
    )

    for row in head_probe_rows:
        arr = (
            lm_head[row]
            .float()
            .numpy()
        )

        if not np.isfinite(arr).all():
            raise FloatingPointError(
                f"❌ common lm_head 第 {row} 行 "
                "出现 NaN/Inf。"
            )

    print("\n" + "=" * 92)
    print("✅ 离线 Teacher cache 最终校验通过")
    print(
        "teacher_completion_hidden: "
        f"{hidden.shape}, dtype={hidden.dtype}"
    )
    print(
        "teacher_common_lm_head    : "
        f"{tuple(lm_head.shape)}, "
        f"dtype={lm_head.dtype}"
    )
    print(
        "✅ 四个第二次 hierarchy-token hidden "
        "均可通过 relative offset 精确读取"
    )
    print(
        "✅ 后续可直接计算三条 Hier 关系："
        "L1_START→L1_END, "
        "L1_END→L2_START, "
        "L2_START→L2_END"
    )
    print("=" * 92 + "\n")


# ======================================================================
# 11. 主函数
# ======================================================================

def main(args):
    set_all_seeds(
        args.seed
    )

    if args.extraction_batch_size <= 0:
        raise ValueError(
            "❌ --extraction_batch_size 必须 > 0。"
        )

    if args.lm_head_row_chunk_size <= 0:
        raise ValueError(
            "❌ --lm_head_row_chunk_size 必须 > 0。"
        )

    if os.path.exists(
        args.save_dir
    ):
        if not args.overwrite:
            raise FileExistsError(
                f"❌ save_dir 已存在: {args.save_dir}\n"
                "为防止覆盖旧 cache，默认停止。\n"
                "若确认重建，请传 --overwrite true"
            )

        shutil.rmtree(
            args.save_dir
        )

    os.makedirs(
        args.save_dir,
        exist_ok=True,
    )

    print("\n" + "=" * 104)
    print(
        "WOS 双 START/END Token Teacher 离线缓存提取"
    )
    print(
        f"Student raw base/tokenizer : "
        f"{args.student_model_path}"
    )
    print(
        f"Teacher trained best model : "
        f"{args.teacher_model_path}"
    )
    print(
        f"Train data                 : "
        f"{args.train_data_path}"
    )
    print(
        f"Save dir                   : "
        f"{args.save_dir}"
    )
    print(
        f"Device                     : "
        f"{args.device}"
    )
    print("=" * 104 + "\n")

    # --------------------------------------------------------------
    # A. TRAIN 数据
    # --------------------------------------------------------------
    data = load_json_or_jsonl(
        args.train_data_path
    )

    print(
        f"读取训练集: {len(data):,} 条"
    )

    # 先全量检查 clean 字段。
    for idx, item in enumerate(
        tqdm(
            data,
            desc="Validate TRAIN fields",
            leave=False,
        )
    ):
        get_clean_sample_fields(
            item,
            "TRAIN",
            idx,
        )

    print(
        "✅ TRAIN 五字段基本格式检查通过"
    )

    # --------------------------------------------------------------
    # B. Student tokenizer：权威训练序列
    # --------------------------------------------------------------
    student_tokenizer = (
        AutoTokenizer.from_pretrained(
            args.student_model_path
        )
    )

    install_training_chat_template(
        student_tokenizer
    )

    # --------------------------------------------------------------
    # C. Teacher tokenizer：只做 token-string ID 映射
    # --------------------------------------------------------------
    teacher_tokenizer = (
        AutoTokenizer.from_pretrained(
            args.teacher_model_path
        )
    )

    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = (
            teacher_tokenizer.eos_token
        )

    teacher_tokenizer.padding_side = "right"

    student_token_ids = (
        validate_atomic_reserved_tokens(
            student_tokenizer,
            "Student",
        )
    )

    teacher_token_ids = (
        validate_atomic_reserved_tokens(
            teacher_tokenizer,
            "Teacher",
        )
    )

    alignment = (
        build_teacher_student_vocab_alignment(
            student_tokenizer,
            teacher_tokenizer,
        )
    )

    save_alignment_summary(
        args.save_dir,
        alignment,
    )

    # --------------------------------------------------------------
    # D. 第一遍：只构造序列/offset，不跑 Teacher
    # --------------------------------------------------------------
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

    save_prescan_metadata(
        args.save_dir,
        prescan,
    )

    # --------------------------------------------------------------
    # E. 加载“已经训练好的”冻结 Teacher
    # --------------------------------------------------------------
    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "❌ 指定了 CUDA，但 "
            "torch.cuda.is_available()=False。"
        )

    device = torch.device(
        args.device
    )

    if args.model_dtype == "bfloat16":
        model_dtype = torch.bfloat16
    elif args.model_dtype == "float16":
        model_dtype = torch.float16
    elif args.model_dtype == "float32":
        model_dtype = torch.float32
    else:
        raise ValueError(
            "❌ --model_dtype 只支持 "
            "bfloat16/float16/float32"
        )

    teacher_model = (
        AutoModelForCausalLM.from_pretrained(
            args.teacher_model_path,
            torch_dtype=model_dtype,
            attn_implementation=(
                args.attn_implementation
            ),
            device_map=None,
        )
    )

    teacher_model.to(
        device
    )

    teacher_model.eval()
    teacher_model.requires_grad_(
        False
    )
    teacher_model.config.use_cache = (
        False
    )

    teacher_vocab_rows = int(
        teacher_model
        .get_output_embeddings()
        .weight.shape[0]
    )

    if teacher_vocab_rows != len(
        teacher_tokenizer
    ):
        raise RuntimeError(
            "❌ Teacher lm_head 行数与 "
            "Teacher tokenizer 长度不一致: "
            f"{teacher_vocab_rows} != "
            f"{len(teacher_tokenizer)}"
        )

    hidden_size = int(
        teacher_model.config.hidden_size
    )

    print("\n" + "=" * 92)
    print("✅ 训练后的 Teacher 加载完成")
    print(
        f"Teacher hidden size : {hidden_size}"
    )
    print(
        f"Teacher vocab rows  : "
        f"{teacher_vocab_rows:,}"
    )
    print(
        f"Teacher dtype       : "
        f"{next(teacher_model.parameters()).dtype}"
    )
    print(
        f"Teacher device      : "
        f"{next(teacher_model.parameters()).device}"
    )
    print(
        "Teacher trainable   : 0"
    )
    print("=" * 92 + "\n")

    # --------------------------------------------------------------
    # F. 保存 common-vocab Teacher lm_head
    # --------------------------------------------------------------
    head_info = (
        save_teacher_common_lm_head_pt(
            teacher_model=teacher_model,
            alignment=alignment,
            save_dir=args.save_dir,
            cache_dtype=args.cache_dtype,
            row_chunk_size=(
                args.lm_head_row_chunk_size
            ),
        )
    )

    # --------------------------------------------------------------
    # G. Teacher 只完整遍历一次 TRAIN
    # --------------------------------------------------------------
    hidden_path = (
        extract_teacher_hidden_cache(
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
            extraction_batch_size=(
                args.extraction_batch_size
            ),
            cache_dtype=args.cache_dtype,
            save_dir=args.save_dir,
        )
    )

    # --------------------------------------------------------------
    # H. Manifest
    # --------------------------------------------------------------
    manifest = {
        "format_version": 2,
        "dataset": "WOS",
        "split": "train",
        "task": (
            "paired START/END hierarchy-token offline KD"
        ),

        "num_samples": int(
            prescan["num_samples"]
        ),
        "total_completion_prediction_positions": int(
            prescan["total_positions"]
        ),
        "teacher_hidden_size": int(
            hidden_size
        ),
        "common_vocab_size": int(
            alignment["common_token_count"]
        ),
        "hidden_storage_dtype": (
            args.cache_dtype
        ),
        "hidden_storage_shape": [
            int(
                prescan[
                    "total_positions"
                ]
            ),
            int(
                hidden_size
            ),
        ],

        "student_raw_model_path_for_tokenization": (
            os.path.abspath(
                args.student_model_path
            )
        ),
        "teacher_model_path": (
            os.path.abspath(
                args.teacher_model_path
            )
        ),
        "train_data_path": (
            os.path.abspath(
                args.train_data_path
            )
        ),

        "max_seq_length": int(
            args.max_seq_length
        ),
        "reserve_tokens": int(
            args.reserve_tokens
        ),
        "system_prompt": (
            args.system_prompt
        ),
        "chat_template": CHAT_TEMPLATE,

        "student_reserved_token_ids": (
            student_token_ids
        ),
        "teacher_reserved_token_ids": (
            teacher_token_ids
        ),

        "tokens": {
            "ctx_start": CTX_START_TOKEN,
            "ctx_end": CTX_END_TOKEN,
            "l1_start": HIER_START_TOKEN_L1,
            "l1_end": HIER_END_TOKEN_L1,
            "l2_start": HIER_START_TOKEN_L2,
            "l2_end": HIER_END_TOKEN_L2,
        },

        "completion": (
            "L1_START + L1_LABEL + L1_END + "
            "L2_START + L2_LABEL + L2_END + EOT"
        ),
        "completion_has_inter_level_separator": False,
        "l1_end_l2_start_are_token_adjacent": True,
        "l2_end_directly_predicts_eot": True,

        "teacher_forcing": True,

        "token_alignment_rule": (
            "same token string; numeric IDs may differ"
        ),
        "common_token_list_sha256": (
            alignment[
                "common_token_sha256"
            ]
        ),
        "protected_info": (
            alignment[
                "protected_info"
            ]
        ),

        "causal_position_rule": (
            "prediction_hidden = "
            "teacher.model(...).last_hidden_state[:, :-1, :]; "
            "select positions where labels[1:] != -100"
        ),

        "full_kd_mask": (
            "all Completion causal prediction positions"
        ),

        "special_hidden_targets": [
            "l1_start_second_occurrence",
            "l1_end_second_occurrence",
            "l2_start_second_occurrence",
            "l2_end_second_occurrence",
        ],

        "hidden_align_source": (
            "four cached final hidden states at "
            "Completion second occurrences"
        ),

        "hidden_hier_edges": {
            "l1_internal": (
                "h(l1_end) - h(l1_start)"
            ),
            "l1_to_l2": (
                "h(l2_start) - h(l1_end)"
            ),
            "l2_internal": (
                "h(l2_end) - h(l2_start)"
            ),
        },

        "traditional_kd_reconstruction": (
            "teacher_common_logits = "
            "cached_completion_hidden @ "
            "cached_teacher_common_lm_head.T "
            "(+ cached bias if present); "
            "then temperature-scaled KL"
        ),

        "traditional_kd_is_still_logits_kd": True,

        "files": {
            "manifest": (
                CACHE_MANIFEST_NAME
            ),
            "offsets": (
                CACHE_OFFSETS_NAME
            ),
            "valid_counts": (
                CACHE_VALID_COUNTS_NAME
            ),
            "l1_start_rel_offsets": (
                REL_OFFSET_FILES[
                    "l1_start"
                ]
            ),
            "l1_end_rel_offsets": (
                REL_OFFSET_FILES[
                    "l1_end"
                ]
            ),
            "l2_start_rel_offsets": (
                REL_OFFSET_FILES[
                    "l2_start"
                ]
            ),
            "l2_end_rel_offsets": (
                REL_OFFSET_FILES[
                    "l2_end"
                ]
            ),
            "student_input_hashes": (
                CACHE_HASHES_NAME
            ),
            "teacher_completion_hidden": (
                os.path.basename(
                    hidden_path
                )
            ),
            "teacher_common_lm_head": (
                os.path.basename(
                    head_info["path"]
                )
            ),
            "vocab_alignment": (
                CACHE_ALIGNMENT_NAME
            ),
        },
    }

    manifest_path = os.path.join(
        args.save_dir,
        CACHE_MANIFEST_NAME,
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------------
    # I. 最终校验
    # --------------------------------------------------------------
    validate_written_cache(
        save_dir=args.save_dir,
        prescan=prescan,
        hidden_size=hidden_size,
        common_vocab_size=alignment[
            "common_token_count"
        ],
    )

    print("\n" + "=" * 104)
    print(
        "🎉 WOS 双 START/END Token Teacher "
        "离线缓存提取完成"
    )
    print(
        f"Cache dir: {args.save_dir}"
    )
    print("")
    print("后续 Student 蒸馏阶段：")
    print(
        "1) 不再加载 8B Teacher Transformer；"
    )
    print(
        "2) cached hidden × cached common lm_head "
        "恢复 Teacher logits，Traditional 仍是 logits KL；"
    )
    print(
        "3) 四个 cached second-occurrence hidden "
        "用于 4-token Align；"
    )
    print(
        "4) 三个 hidden difference "
        "用于 3-edge Hier；"
    )
    print(
        "5) sample_index + offsets + SHA256 "
        "保证 DataLoader shuffle 后仍读到正确 Teacher 特征。"
    )
    print("=" * 104 + "\n")


# ======================================================================
# 12. 参数
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract offline Teacher cache for current "
            "WOS paired START/END hierarchy tokens."
        )
    )

    parser.add_argument(
        "--student_model_path",
        type=str,
        default=(
            "/data1/lzp/code/"
            "llama-3.2_1b/"
        ),
        help=(
            "未训练 Student raw base/tokenizer；"
            "它定义权威训练 token 序列。"
        ),
    )

    parser.add_argument(
        "--teacher_model_path",
        type=str,
        default=(
            "/data1/lzp/code/"
            "hierarchical_distill/"
            "checkpoints_teacher_wos_2/"
            "best_sft_model/"
        ),
        help=(
            "当前双 START/END Token "
            "训练完成的 merged best 8B Teacher。"
        ),
    )

    parser.add_argument(
        "--train_data_path",
        type=str,
        default=(
            "/data1/lzp/code/data/wos/"
            "train_formatted.jsonl"
        ),
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=(
            "/data1/lzp/code/"
            "hierarchical_distill/"
            "teacher_wos_2token_pair_cache/"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--reserve_tokens",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
    )

    parser.add_argument(
        "--extraction_batch_size",
        type=int,
        default=1,
        help=(
            "8B Teacher 前向 batch；"
            "先用 1，显存足够再尝试 2/4。"
        ),
    )

    parser.add_argument(
        "--lm_head_row_chunk_size",
        type=int,
        default=4096,
        help=(
            "保存 Teacher common lm_head 时"
            "每次从 GPU 复制多少行。"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--model_dtype",
        type=str,
        choices=[
            "bfloat16",
            "float16",
            "float32",
        ],
        default="bfloat16",
        help="Teacher 前向 dtype。",
    )

    parser.add_argument(
        "--cache_dtype",
        type=str,
        choices=[
            "float16",
            "float32",
        ],
        default="float16",
        help=(
            "磁盘缓存 dtype；"
            "float16 约为 float32 一半空间。"
        ),
    )

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
    )

    parser.add_argument(
        "--scan_limit",
        type=int,
        default=-1,
        help=(
            "-1=全量；"
            "可先用 10/100 做独立 sanity cache。"
        ),
    )

    parser.add_argument(
        "--overwrite",
        type=str2bool,
        default=False,
        help=(
            "默认禁止覆盖已有 cache；"
            "确认重建时传 true。"
        ),
    )

    args = parser.parse_args()
    main(args)
