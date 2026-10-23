#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WOS 双 Token KD 严格测试脚本
============================

适用于当前 paired START/END hierarchy-token Student：

  L1_START = <|reserved_special_token_112|>
  L1_END   = <|reserved_special_token_113|>
  L2_START = <|reserved_special_token_114|>
  L2_END   = <|reserved_special_token_115|>

训练 / 测试 Completion 严格为：

  L1_START + L1_LABEL + L1_END + L2_START + L2_LABEL + L2_END + EOT

本脚本支持：
  1) best_kd_model / final_model（merged model）
  2) best_kd_checkpoint / final_kd_checkpoint（PEFT + full IO）

测试阶段：
  - 不加载 Teacher
  - 不读取 Teacher cache
  - 不需要 hidden projector
  - 不计算 KD / Align / Hier
  - 只进行 Student 正常 completion-only Test CE 与严格生成测试

指标与当前双 Token KD 训练 validation 保持一致：
  - Global Micro-F1
  - Global Macro-F1
  - L1 Micro/Macro-F1
  - L2 Micro/Macro-F1
  - Exact Parse Rate
  - Valid Label Pair Rate
  - Hierarchy Consistency Rate

同时额外输出：
  - L1/L2 Accuracy
  - Both Levels Correct Rate
  - Invalid L1/L2 Rate
  - 逐样本原始生成结果
  - Prompt / 正文 token span SHA256 审计
"""

import os

os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import hashlib
import json
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    set_seed as hf_set_seed,
)


# ======================================================================
# 0. 与当前双 Token KD 训练严格一致的固定定义
# ======================================================================

CTX_START_TOKEN = "<|reserved_special_token_110|>"
CTX_END_TOKEN = "<|reserved_special_token_111|>"

HIER_START_TOKEN_L1 = "<|reserved_special_token_112|>"
HIER_END_TOKEN_L1 = "<|reserved_special_token_113|>"
HIER_START_TOKEN_L2 = "<|reserved_special_token_114|>"
HIER_END_TOKEN_L2 = "<|reserved_special_token_115|>"

HIER_TOKEN_ROLES = (
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

FULL_IO_WEIGHTS_NAME = "full_input_output_embeddings.pt"
DISTILL_CONFIG_NAME = "distill_config.json"
BEST_METRICS_NAME = "best_metrics.json"

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
# 1. 随机种子
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

    value = str(v).lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


# ======================================================================
# 2. Prompt / Completion：直接复制当前双 Token KD 训练定义
# ======================================================================

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


def build_completion_text(l1_label: str, l2_label: str) -> str:
    return (
        f"{HIER_START_TOKEN_L1}{l1_label}{HIER_END_TOKEN_L1}"
        f"{HIER_START_TOKEN_L2}{l2_label}{HIER_END_TOKEN_L2}"
    )


def build_prompt_messages(text: str, system_prompt: str) -> List[dict]:
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


def configure_tokenizer(tokenizer):
    tokenizer.chat_template = CHAT_TEMPLATE

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token_id is None:
        raise RuntimeError("❌ tokenizer 没有可用 pad_token_id。")

    tokenizer.padding_side = "left"
    return tokenizer


def encode_prompt_ids(
    tokenizer,
    text: str,
    system_prompt: str,
    source_index: int,
) -> Tuple[str, List[int]]:
    messages = build_prompt_messages(
        text,
        system_prompt,
    )

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
                f"❌ TEST 第 {source_index} 条 Prompt BOS "
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
        common = min(len(direct), len(reencoded))
        mismatch = next(
            (
                i
                for i in range(common)
                if direct[i] != reencoded[i]
            ),
            common,
        )

        raise RuntimeError(
            f"❌ TEST 第 {source_index} 条 Prompt "
            "两种编码入口不一致。\n"
            f"direct_len={len(direct)}, "
            f"reencoded_len={len(reencoded)}, "
            f"first_mismatch={mismatch}"
        )

    return rendered, direct


# ======================================================================
# 3. 六个 reserved token + Completion 布局检查
# ======================================================================

def validate_atomic_reserved_tokens(
    tokenizer,
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
    out = {}

    print("\n" + "=" * 88)
    print("🔎 验证双 Token 测试所需 6 个 reserved tokens")

    for role, token in token_map.items():
        if token not in vocab:
            raise RuntimeError(
                f"❌ tokenizer 缺少 {token}"
            )

        ids = tokenizer.encode(
            token,
            add_special_tokens=False,
        )
        converted = tokenizer.convert_tokens_to_ids(token)

        if len(ids) != 1:
            raise RuntimeError(
                f"❌ {token} 不是单 token: {ids}"
            )

        token_id = int(ids[0])

        if token_id != int(converted):
            raise RuntimeError(
                f"❌ {token} encode/convert ID 不一致："
                f"{token_id} != {converted}"
            )

        if token_id == tokenizer.unk_token_id:
            raise RuntimeError(
                f"❌ {token} 被编码成 UNK。"
            )

        out[role] = token_id
        print(
            f"✅ {role:>8s}: {token} -> {token_id}"
        )

    if len(set(out.values())) != 6:
        raise RuntimeError(
            f"❌ 六个 reserved token ID 未两两不同: {out}"
        )

    print("✅ 测试阶段不 add_tokens、不 resize embedding")
    print("=" * 88 + "\n")
    return out


def _find_all_positions(
    ids: List[int],
    token_id: int,
) -> List[int]:
    return [
        i
        for i, x in enumerate(ids)
        if int(x) == int(token_id)
    ]


def check_prompt_completion_layout(
    tokenizer,
    system_prompt: str,
    token_ids: Dict[str, int],
) -> None:
    _, prompt_ids = encode_prompt_ids(
        tokenizer=tokenizer,
        text="probe check text",
        system_prompt=system_prompt,
        source_index=0,
    )

    expected_prompt_counts = {
        token_ids["ctx_start"]: 1,
        token_ids["ctx_end"]: 1,
        token_ids["l1_start"]: 1,
        token_ids["l1_end"]: 1,
        token_ids["l2_start"]: 1,
        token_ids["l2_end"]: 1,
    }

    for token_id, expected_count in expected_prompt_counts.items():
        actual_count = prompt_ids.count(token_id)
        if actual_count != expected_count:
            raise RuntimeError(
                f"❌ Prompt token_id={token_id} "
                f"应出现 {expected_count} 次，实际={actual_count}"
            )

    completion_ids = tokenizer.encode(
        build_completion_text(
            "CS",
            "Machine learning",
        ),
        add_special_tokens=False,
    )
    completion_ids = [int(x) for x in completion_ids]

    eot_id = tokenizer.convert_tokens_to_ids(
        "<|eot_id|>"
    )
    if eot_id is None or eot_id == tokenizer.unk_token_id:
        raise RuntimeError(
            "❌ tokenizer 缺少原子 <|eot_id|>。"
        )
    eot_id = int(eot_id)

    full_ids = (
        prompt_ids
        + completion_ids
        + [eot_id]
    )

    positions = {
        role: _find_all_positions(
            full_ids,
            token_ids[role],
        )
        for role in HIER_TOKEN_ROLES
    }

    for role in HIER_TOKEN_ROLES:
        if len(positions[role]) != 2:
            raise RuntimeError(
                f"❌ {role} 在 Prompt+Completion "
                f"应出现 2 次，实际={positions[role]}"
            )

    first = {
        role: positions[role][0]
        for role in HIER_TOKEN_ROLES
    }
    second = {
        role: positions[role][1]
        for role in HIER_TOKEN_ROLES
    }

    if not (
        first["l1_start"]
        < first["l1_end"]
        < first["l2_start"]
        < first["l2_end"]
        < second["l1_start"]
        < second["l1_end"]
        < second["l2_start"]
        < second["l2_end"]
    ):
        raise RuntimeError(
            "❌ 四个 hierarchy token occurrence 顺序异常："
            f"first={first}, second={second}"
        )

    if (
        second["l2_start"]
        != second["l1_end"] + 1
    ):
        raise RuntimeError(
            "❌ Completion 中 L1_END 与 L2_START 不相邻。"
        )

    if (
        second["l2_end"] + 1 >= len(full_ids)
        or full_ids[second["l2_end"] + 1] != eot_id
    ):
        raise RuntimeError(
            "❌ Completion 中 L2_END 后不是 EOT。"
        )

    print("\n" + "=" * 88)
    print("✅ 双 Token Prompt / Completion 布局检查通过")
    print(f"Prompt first occurrence : {first}")
    print(f"Completion second       : {second}")
    print("✅ L1_END 与 L2_START 严格相邻")
    print("✅ L2_END 后直接 EOT")
    print("=" * 88 + "\n")


# ======================================================================
# 4. 训练配置元信息（可选审计）
# ======================================================================

def find_metadata_file(
    model_path: str,
    filename: str,
) -> Optional[str]:
    model_abs = os.path.abspath(model_path)
    candidates = [
        os.path.join(model_abs, filename),
        os.path.join(
            os.path.dirname(model_abs),
            filename,
        ),
        os.path.join(
            os.path.dirname(
                os.path.dirname(model_abs)
            ),
            filename,
        ),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    return None


def check_distill_config(
    model_path: str,
) -> Optional[dict]:
    path = find_metadata_file(
        model_path,
        DISTILL_CONFIG_NAME,
    )

    if path is None:
        print(
            "⚠️ 未找到 distill_config.json；"
            "测试仍会通过 Prompt/Token/layout 严格检查继续。"
        )
        return None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        cfg = json.load(f)

    expected_prompt_rule = (
        "exactly identical to current paired "
        "START/END Student/Teacher SFT"
    )

    actual_rule = cfg.get(
        "prompt_completion_definition"
    )
    if (
        actual_rule is not None
        and actual_rule != expected_prompt_rule
    ):
        raise RuntimeError(
            "❌ distill_config 中 Prompt/Completion 定义异常："
            f"{actual_rule!r}"
        )

    print("\n" + "=" * 88)
    print(f"✅ 找到训练配置: {path}")
    print(
        f"experiment_mode: "
        f"{cfg.get('experiment_mode', 'unknown')}"
    )
    print(
        f"kd_weight: "
        f"{cfg.get('kd_weight', 'unknown')}"
    )
    print(
        f"align_weight_effective: "
        f"{cfg.get('align_weight_effective', 'unknown')}"
    )
    print(
        f"hier_weight_effective: "
        f"{cfg.get('hier_weight_effective', 'unknown')}"
    )
    print("=" * 88 + "\n")

    return cfg


# ======================================================================
# 5. 数据 / taxonomy / label2id
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
        obj = json.loads(raw)
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
                    f"❌ JSONL 第 {line_no} 行不是 dict。"
                )

            rows.append(item)

        return rows

    if not isinstance(obj, list):
        raise TypeError(
            "❌ JSON 顶层必须是 list。"
        )

    for idx, item in enumerate(obj):
        if not isinstance(item, dict):
            raise TypeError(
                f"❌ JSON 第 {idx} 条不是 dict。"
            )

    return obj


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
            f"❌ {split_name} 第 {idx} 条缺字段: {missing}"
        )

    text = item["text"]
    l1 = item["l1_label"]
    l2 = item["l2_label"]
    l1_id = item["l1_id"]
    l2_id = item["l2_id"]

    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            f"❌ {split_name} 第 {idx} 条 text 为空。"
        )

    if not isinstance(l1, str) or not l1.strip():
        raise ValueError(
            f"❌ {split_name} 第 {idx} 条 l1_label 非法。"
        )

    if not isinstance(l2, str) or not l2.strip():
        raise ValueError(
            f"❌ {split_name} 第 {idx} 条 l2_label 非法。"
        )

    if isinstance(l1_id, bool) or not isinstance(l1_id, int):
        raise TypeError(
            f"❌ {split_name} 第 {idx} 条 l1_id 必须是 int。"
        )

    if isinstance(l2_id, bool) or not isinstance(l2_id, int):
        raise TypeError(
            f"❌ {split_name} 第 {idx} 条 l2_id 必须是 int。"
        )

    return (
        text.strip(),
        l1.strip(),
        l2.strip(),
        int(l1_id),
        int(l2_id),
    )


def parse_taxonomy(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"❌ taxonomy 不存在: {path}"
        )

    parent_to_children = {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(
            f,
            start=1,
        ):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue

            parts = [
                x.strip()
                for x in line.split("\t")
                if x.strip()
            ]

            if len(parts) < 2:
                raise ValueError(
                    f"❌ taxonomy 第 {line_no} 行异常。"
                )

            parent = parts[0]
            children = parts[1:]

            if parent in parent_to_children:
                raise ValueError(
                    f"❌ taxonomy parent 重复: {parent}"
                )

            if len(children) != len(set(children)):
                raise ValueError(
                    f"❌ taxonomy parent={parent!r} "
                    "存在重复 child。"
                )

            parent_to_children[parent] = children

    if "Root" not in parent_to_children:
        raise ValueError(
            "❌ taxonomy 缺少 Root。"
        )

    l1_labels = parent_to_children["Root"]
    l2_labels = []
    l2_to_l1 = {}

    for l1 in l1_labels:
        if l1 not in parent_to_children:
            raise ValueError(
                f"❌ L1={l1!r} 没有子节点行。"
            )

        for l2 in parent_to_children[l1]:
            if l2 in l2_to_l1:
                raise ValueError(
                    f"❌ L2={l2!r} 重复归属。"
                )
            l2_to_l1[l2] = l1
            l2_labels.append(l2)

    return (
        l1_labels,
        l2_labels,
        l2_to_l1,
    )


def load_label_space(
    dict_path: str,
    taxonomy_path: str,
    expected_num_l1: int,
    expected_num_l2: int,
):
    if not os.path.isfile(dict_path):
        raise FileNotFoundError(
            f"❌ label2id 不存在: {dict_path}"
        )

    with open(
        dict_path,
        "r",
        encoding="utf-8",
    ) as f:
        d = json.load(f)

    l1_map = {
        str(k): int(v)
        for k, v in d["l1_label2id"].items()
    }
    l2_map = {
        str(k): int(v)
        for k, v in d["l2_label2id"].items()
    }

    tax_l1, tax_l2, l2_to_l1 = (
        parse_taxonomy(taxonomy_path)
    )

    if (
        expected_num_l1 > 0
        and len(tax_l1) != expected_num_l1
    ):
        raise ValueError(
            f"❌ L1 数量异常: "
            f"{len(tax_l1)} != {expected_num_l1}"
        )

    if (
        expected_num_l2 > 0
        and len(tax_l2) != expected_num_l2
    ):
        raise ValueError(
            f"❌ L2 数量异常: "
            f"{len(tax_l2)} != {expected_num_l2}"
        )

    expected_l1 = {
        x: i
        for i, x in enumerate(tax_l1)
    }
    expected_l2 = {
        x: i
        for i, x in enumerate(tax_l2)
    }

    if l1_map != expected_l1:
        raise ValueError(
            "❌ L1 label2id 与 taxonomy 顺序不一致。"
        )

    if l2_map != expected_l2:
        raise ValueError(
            "❌ L2 label2id 与 taxonomy 顺序不一致。"
        )

    print("\n" + "=" * 88)
    print("🏷️ WOS 测试标签空间校验完成")
    print(f"L1: {len(l1_map)}")
    print(f"L2: {len(l2_map)}")
    print("✅ label2id 与 taxonomy 完全一致")
    print("=" * 88 + "\n")

    return (
        l1_map,
        l2_map,
        l2_to_l1,
    )


def validate_test_split(
    path: str,
    l1_map: Dict[str, int],
    l2_map: Dict[str, int],
    l2_to_l1: Dict[str, str],
) -> List[dict]:
    data = load_json_or_jsonl(path)

    seen_l1 = set()
    seen_l2 = set()

    for i, item in enumerate(
        tqdm(
            data,
            desc="校验TEST",
            leave=False,
        )
    ):
        _, l1, l2, id1, id2 = (
            get_clean_sample_fields(
                item,
                "TEST",
                i,
            )
        )

        if l1 not in l1_map:
            raise ValueError(
                f"❌ TEST {i} 未知 L1={l1!r}"
            )

        if l2 not in l2_map:
            raise ValueError(
                f"❌ TEST {i} 未知 L2={l2!r}"
            )

        if id1 != l1_map[l1]:
            raise ValueError(
                f"❌ TEST {i} L1 label-ID 错位。"
            )

        if id2 != l2_map[l2]:
            raise ValueError(
                f"❌ TEST {i} L2 label-ID 错位。"
            )

        if l2_to_l1[l2] != l1:
            raise ValueError(
                f"❌ TEST {i} 父子关系错误。"
            )

        seen_l1.add(l1)
        seen_l2.add(l2)

    print("\n" + "=" * 88)
    print("✅ TEST 严格校验通过")
    print(f"样本数: {len(data):,}")
    print(
        f"实际出现 L1: "
        f"{len(seen_l1)}/{len(l1_map)}"
    )
    print(
        f"实际出现 L2: "
        f"{len(seen_l2)}/{len(l2_map)}"
    )
    print("静默跳过: 0")
    print("=" * 88 + "\n")

    return data


# ======================================================================
# 6. 截断 / hash / 正文 span 审计
# ======================================================================

def truncate_text_for_prompt(
    text: str,
    tokenizer,
    max_seq_length: int,
    reserve_tokens: int,
) -> str:
    max_text_len = (
        max_seq_length
        - reserve_tokens
    )

    if max_text_len <= 0:
        raise ValueError(
            "❌ max_seq_length 必须大于 reserve_tokens。"
        )

    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if len(ids) > max_text_len:
        ids = ids[:max_text_len]
        text = tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    return text


def _token_ids_sha256(ids: List[int]) -> str:
    payload = ",".join(
        str(int(x))
        for x in ids
    )
    return hashlib.sha256(
        payload.encode("ascii")
    ).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def audit_prompt_body_span(
    prompt_ids: List[int],
    truncated_text: str,
    tokenizer,
    token_ids: Dict[str, int],
    source_index: int,
) -> dict:
    start_positions = _find_all_positions(
        prompt_ids,
        token_ids["ctx_start"],
    )
    end_positions = _find_all_positions(
        prompt_ids,
        token_ids["ctx_end"],
    )

    if (
        len(start_positions) != 1
        or len(end_positions) != 1
    ):
        raise RuntimeError(
            f"❌ TEST {source_index} CTX occurrence 异常："
            f"START={start_positions}, END={end_positions}"
        )

    start = start_positions[0]
    end = end_positions[0]

    if end <= start + 1:
        raise RuntimeError(
            f"❌ TEST {source_index} 正文边界异常。"
        )

    actual = [
        int(x)
        for x in prompt_ids[
            start + 1:
            end
        ]
    ]

    expected = tokenizer.encode(
        f"\n{truncated_text}\n",
        add_special_tokens=False,
    )
    expected = [
        int(x)
        for x in expected
    ]

    if actual != expected:
        raise RuntimeError(
            f"❌ TEST {source_index} 正文 token span "
            "与截断正文编码不一致。"
        )

    decoded = tokenizer.decode(
        actual,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    expected_text = (
        f"\n{truncated_text}\n"
    )

    if decoded != expected_text:
        raise RuntimeError(
            f"❌ TEST {source_index} 正文解码不一致。"
        )

    return {
        "source_index": int(source_index),
        "ctx_start_position": int(start),
        "ctx_end_position": int(end),
        "body_token_count": int(len(actual)),
        "body_text_sha256": _text_sha256(
            truncated_text
        ),
        "body_token_sha256": (
            _token_ids_sha256(actual)
        ),
        "prompt_token_sha256": (
            _token_ids_sha256(prompt_ids)
        ),
        "token_exact_match": True,
        "decoded_segment_match": True,
    }


def scan_text_truncation(
    data: List[dict],
    tokenizer,
    max_seq_length: int,
    reserve_tokens: int,
) -> None:
    max_text_len = (
        max_seq_length
        - reserve_tokens
    )

    truncated_count = 0
    max_original = 0

    for idx, item in enumerate(
        tqdm(
            data,
            desc="检查测试集截断比例",
            leave=False,
        )
    ):
        text, _, _, _, _ = (
            get_clean_sample_fields(
                item,
                "TEST",
                idx,
            )
        )

        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        max_original = max(
            max_original,
            len(ids),
        )

        if len(ids) > max_text_len:
            truncated_count += 1

    total = len(data)

    print("\n" + "=" * 88)
    print(
        f"正文 token 预算: {max_text_len}"
    )
    print(
        f"测试集: {total:,} 条"
    )
    print(
        f"正文被截断: {truncated_count:,} "
        f"({truncated_count/max(1,total):.2%})"
    )
    print(
        f"最长原始正文: {max_original} tokens"
    )
    print("=" * 88 + "\n")


def prepare_test_prompts(
    test_data: List[dict],
    tokenizer,
    token_ids: Dict[str, int],
    max_seq_length: int,
    reserve_tokens: int,
    system_prompt: str,
    output_dir: str,
) -> Tuple[List[dict], str]:
    prompts = []
    audits = []

    for idx, item in enumerate(
        tqdm(
            test_data,
            desc="准备双Token测试Prompt",
            leave=False,
        )
    ):
        text, l1, l2, id1, id2 = (
            get_clean_sample_fields(
                item,
                "TEST",
                idx,
            )
        )

        text = truncate_text_for_prompt(
            text,
            tokenizer,
            max_seq_length,
            reserve_tokens,
        )

        prompt_text, prompt_ids = (
            encode_prompt_ids(
                tokenizer,
                text,
                system_prompt,
                idx,
            )
        )

        if len(prompt_ids) > max_seq_length:
            raise RuntimeError(
                f"❌ TEST {idx} Prompt 长度 "
                f"{len(prompt_ids)} > {max_seq_length}"
            )

        expected_counts = {
            token_ids["ctx_start"]: 1,
            token_ids["ctx_end"]: 1,
            token_ids["l1_start"]: 1,
            token_ids["l1_end"]: 1,
            token_ids["l2_start"]: 1,
            token_ids["l2_end"]: 1,
        }

        for tid, expected_count in expected_counts.items():
            if prompt_ids.count(tid) != expected_count:
                raise RuntimeError(
                    f"❌ TEST {idx} Prompt token_id={tid} "
                    f"次数异常。"
                )

        audit = audit_prompt_body_span(
            prompt_ids=prompt_ids,
            truncated_text=text,
            tokenizer=tokenizer,
            token_ids=token_ids,
            source_index=idx,
        )
        audits.append(audit)

        prompts.append(
            {
                "source_index": idx,
                "prompt": prompt_text,
                "prompt_ids": prompt_ids,
                "true_l1_str": l1,
                "true_l2_str": l2,
                "true_l1": id1,
                "true_l2": id2,
                **audit,
            }
        )

    audit_path = os.path.join(
        output_dir,
        "test_body_span_audit.jsonl",
    )

    with open(
        audit_path,
        "w",
        encoding="utf-8",
    ) as f:
        for rec in audits:
            f.write(
                json.dumps(
                    rec,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"✅ 测试 Prompt 准备完成: "
        f"{len(prompts):,} 条"
    )
    print(
        "✅ 所有正文 token span / chat-template "
        "编码入口完全一致"
    )
    print(
        f"💾 正文审计: {audit_path}"
    )

    return prompts, audit_path


# ======================================================================
# 7. 严格解析双 Token 连续输出
# ======================================================================

def _strip_trailing_stop_strings(text: str) -> str:
    text = text.replace(
        "\r\n",
        "\n",
    ).strip()

    stop_strings = (
        "<|eot_id|>",
        "<|end_of_text|>",
    )

    changed = True
    while changed:
        changed = False

        for s in stop_strings:
            if text.endswith(s):
                text = text[
                    :-len(s)
                ].rstrip()
                changed = True

    return text


def parse_generation_exact(
    raw_resp: str,
) -> Tuple[str, str]:
    """
    必须与当前双 Token KD 训练 validation 的 strict parser 一致。

    合法格式只能是：

      L1_START + L1 + L1_END + L2_START + L2 + L2_END

    四个 hierarchy token 各出现且只出现一次；
    token 与标签之间不能有额外空格或换行。
    """
    text = _strip_trailing_stop_strings(
        raw_resp
    )

    required = (
        HIER_START_TOKEN_L1,
        HIER_END_TOKEN_L1,
        HIER_START_TOKEN_L2,
        HIER_END_TOKEN_L2,
    )

    if any(
        text.count(t) != 1
        for t in required
    ):
        return "", ""

    prefix = HIER_START_TOKEN_L1
    middle = (
        HIER_END_TOKEN_L1
        + HIER_START_TOKEN_L2
    )
    suffix = HIER_END_TOKEN_L2

    if (
        not text.startswith(prefix)
        or not text.endswith(suffix)
    ):
        return "", ""

    body = text[
        len(prefix):
        -len(suffix)
    ]

    if body.count(middle) != 1:
        return "", ""

    l1, l2 = body.split(
        middle,
        1,
    )

    if not l1 or not l2:
        return "", ""

    # token 和标签之间不允许额外空白。
    if (
        l1 != l1.strip()
        or l2 != l2.strip()
    ):
        return "", ""

    if (
        "\n" in l1
        or "\n" in l2
        or "\r" in l1
        or "\r" in l2
    ):
        return "", ""

    return l1, l2


# ======================================================================
# 8. 指标：与当前双 Token KD validation 一致
# ======================================================================

def compute_metric_bundle(
    true_l1,
    true_l2,
    pred_l1,
    pred_l2,
    num_l1,
    num_l2,
):
    if not (
        len(true_l1)
        == len(true_l2)
        == len(pred_l1)
        == len(pred_l2)
    ):
        raise ValueError(
            "❌ metric 输入长度不一致。"
        )

    all_true = []
    all_pred = []

    for t1, t2, p1, p2 in zip(
        true_l1,
        true_l2,
        pred_l1,
        pred_l2,
    ):
        all_true.extend(
            [
                t1,
                t2 + num_l1,
            ]
        )

        all_pred.extend(
            [
                p1,
                (
                    p2 + num_l1
                    if p2 != 9999
                    else 99999
                ),
            ]
        )

    return {
        "global_micro": float(
            f1_score(
                all_true,
                all_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "global_macro": float(
            f1_score(
                all_true,
                all_pred,
                labels=list(
                    range(
                        num_l1
                        + num_l2
                    )
                ),
                average="macro",
                zero_division=0,
            )
        ),
        "l1_micro": float(
            f1_score(
                true_l1,
                pred_l1,
                average="micro",
                zero_division=0,
            )
        ),
        "l1_macro": float(
            f1_score(
                true_l1,
                pred_l1,
                labels=list(
                    range(num_l1)
                ),
                average="macro",
                zero_division=0,
            )
        ),
        "l2_micro": float(
            f1_score(
                true_l2,
                pred_l2,
                average="micro",
                zero_division=0,
            )
        ),
        "l2_macro": float(
            f1_score(
                true_l2,
                pred_l2,
                labels=list(
                    range(num_l2)
                ),
                average="macro",
                zero_division=0,
            )
        ),
    }


# ======================================================================
# 9. merged / PEFT checkpoint 加载
# ======================================================================

def _torch_load_cpu(path: str):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def restore_full_io_weights(
    base_model,
    checkpoint_path: str,
) -> None:
    weight_path = os.path.join(
        checkpoint_path,
        FULL_IO_WEIGHTS_NAME,
    )

    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            "❌ PEFT checkpoint 缺少完整 "
            f"input/lm_head 权重: {weight_path}"
        )

    payload = _torch_load_cpu(
        weight_path
    )

    if (
        "input_embeddings" not in payload
        or "actually_tied" not in payload
    ):
        raise KeyError(
            f"❌ {weight_path} 内容不完整。"
        )

    saved_input = payload[
        "input_embeddings"
    ]
    saved_tied = bool(
        payload["actually_tied"]
    )

    inp = (
        base_model
        .get_input_embeddings()
        .weight
    )
    out = (
        base_model
        .get_output_embeddings()
    )

    if out is None:
        raise RuntimeError(
            "❌ Base model 缺少 lm_head。"
        )

    if tuple(saved_input.shape) != tuple(inp.shape):
        raise RuntimeError(
            "❌ saved input embedding shape "
            "与 base model 不一致。"
        )

    if saved_tied:
        base_model.config.tie_word_embeddings = True
        base_model.tie_weights()

        inp = (
            base_model
            .get_input_embeddings()
            .weight
        )
        out = (
            base_model
            .get_output_embeddings()
        )

        if (
            inp.data_ptr()
            != out.weight.data_ptr()
        ):
            raise RuntimeError(
                "❌ 无法恢复共享 input/lm_head。"
            )

        with torch.no_grad():
            inp.copy_(
                saved_input.to(
                    dtype=inp.dtype
                )
            )

    else:
        saved_output = payload.get(
            "output_embeddings"
        )

        if saved_output is None:
            raise KeyError(
                "❌ 未共享模型缺少 output_embeddings。"
            )

        if (
            inp.data_ptr()
            == out.weight.data_ptr()
        ):
            new_head = torch.nn.Linear(
                out.in_features,
                out.out_features,
                bias=(out.bias is not None),
                device=out.weight.device,
                dtype=out.weight.dtype,
            )
            base_model.set_output_embeddings(
                new_head
            )
            out = (
                base_model
                .get_output_embeddings()
            )

        if (
            tuple(saved_output.shape)
            != tuple(out.weight.shape)
        ):
            raise RuntimeError(
                "❌ saved lm_head shape "
                "与 base model 不一致。"
            )

        with torch.no_grad():
            inp.copy_(
                saved_input.to(
                    dtype=inp.dtype
                )
            )
            out.weight.copy_(
                saved_output.to(
                    dtype=out.weight.dtype
                )
            )

        base_model.config.tie_word_embeddings = False

    print(
        "✅ 已恢复 PEFT checkpoint 的完整 "
        f"input/lm_head: {weight_path}"
    )


def load_trained_model(
    model_path: str,
    base_model_path: str,
    tokenizer,
    device,
    torch_dtype,
    attn_implementation: str,
):
    adapter_config = os.path.join(
        model_path,
        "adapter_config.json",
    )

    is_peft = os.path.isfile(
        adapter_config
    )

    print("\n" + "=" * 88)

    if is_peft:
        print(
            "📦 检测到 PEFT checkpoint"
        )
        print(
            f"Base: {base_model_path}"
        )
        print(
            f"Checkpoint: {model_path}"
        )

        base = (
            AutoModelForCausalLM
            .from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
                attn_implementation=(
                    attn_implementation
                ),
                device_map=None,
            )
        )

        if (
            base
            .get_input_embeddings()
            .weight.shape[0]
            != len(tokenizer)
        ):
            raise RuntimeError(
                "❌ base embedding 行数与 "
                "tokenizer 长度不一致。"
            )

        restore_full_io_weights(
            base,
            model_path,
        )

        model = PeftModel.from_pretrained(
            base,
            model_path,
            is_trainable=False,
        )

        artifact_type = "peft_checkpoint"

    else:
        print(
            "📦 检测到 merged model"
        )
        print(
            f"Model: {model_path}"
        )

        model = (
            AutoModelForCausalLM
            .from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                attn_implementation=(
                    attn_implementation
                ),
                device_map=None,
            )
        )

        artifact_type = "merged_model"

    model.to(device)
    model.eval()

    model.config.use_cache = True
    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    if getattr(
        model,
        "generation_config",
        None,
    ) is not None:
        model.generation_config.use_cache = True
        model.generation_config.pad_token_id = (
            tokenizer.pad_token_id
        )
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None

    inp = (
        model
        .get_input_embeddings()
        .weight
    )
    out = (
        model
        .get_output_embeddings()
    )

    if out is None:
        raise RuntimeError(
            "❌ 测试模型缺少 lm_head。"
        )

    if (
        inp.shape[0] != len(tokenizer)
        or out.weight.shape[0] != len(tokenizer)
    ):
        raise RuntimeError(
            "❌ tokenizer 与模型输入/输出 "
            "vocab 大小不一致。"
        )

    tied = (
        inp.data_ptr()
        == out.weight.data_ptr()
    )

    print("✅ 模型加载完成")
    print(
        f"Artifact type: {artifact_type}"
    )
    print(
        f"Tokenizer length: {len(tokenizer)}"
    )
    print(
        f"Input vocab rows: {inp.shape[0]}"
    )
    print(
        f"Output vocab rows: {out.weight.shape[0]}"
    )
    print(
        f"Input/lm_head tied: {tied}"
    )
    print("=" * 88 + "\n")

    return model, artifact_type


# ======================================================================
# 10. completion-only Test Loss
# ======================================================================

class WOS2TokenTestLossDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        token_ids,
        max_length,
        reserve_tokens,
        system_prompt,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.token_ids = token_ids
        self.max_length = int(max_length)
        self.reserve_tokens = int(
            reserve_tokens
        )
        self.system_prompt = system_prompt

        eot_id = (
            tokenizer
            .convert_tokens_to_ids(
                "<|eot_id|>"
            )
        )

        if (
            eot_id is None
            or eot_id == tokenizer.unk_token_id
        ):
            raise RuntimeError(
                "❌ tokenizer 缺少 <|eot_id|>。"
            )

        self.eot_id = int(eot_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, l1, l2, _, _ = (
            get_clean_sample_fields(
                self.data[idx],
                "TEST_LOSS",
                idx,
            )
        )

        text = truncate_text_for_prompt(
            text,
            self.tokenizer,
            self.max_length,
            self.reserve_tokens,
        )

        _, prompt_ids = encode_prompt_ids(
            self.tokenizer,
            text,
            self.system_prompt,
            idx,
        )

        completion_ids = (
            self.tokenizer.encode(
                build_completion_text(
                    l1,
                    l2,
                ),
                add_special_tokens=False,
            )
        )
        completion_ids = [
            int(x)
            for x in completion_ids
        ]

        input_ids = (
            prompt_ids
            + completion_ids
            + [self.eot_id]
        )

        labels = (
            [-100] * len(prompt_ids)
            + completion_ids
            + [self.eot_id]
        )

        if len(input_ids) > self.max_length:
            raise RuntimeError(
                f"❌ TEST_LOSS {idx} 总长度 "
                f"{len(input_ids)} > {self.max_length}"
            )

        expected_counts = {
            self.token_ids["ctx_start"]: 1,
            self.token_ids["ctx_end"]: 1,
            self.token_ids["l1_start"]: 2,
            self.token_ids["l1_end"]: 2,
            self.token_ids["l2_start"]: 2,
            self.token_ids["l2_end"]: 2,
        }

        for tid, expected_count in expected_counts.items():
            if input_ids.count(tid) != expected_count:
                raise RuntimeError(
                    f"❌ TEST_LOSS {idx} token_id={tid} "
                    "occurrence 异常。"
                )

        # Completion 中 L1_END / L2_START 必须相邻。
        l1_end_pos = _find_all_positions(
            input_ids,
            self.token_ids["l1_end"],
        )[1]
        l2_start_pos = _find_all_positions(
            input_ids,
            self.token_ids["l2_start"],
        )[1]

        if l2_start_pos != l1_end_pos + 1:
            raise RuntimeError(
                f"❌ TEST_LOSS {idx} "
                "L1_END/L2_START 不相邻。"
            )

        return {
            "input_ids": input_ids,
            "attention_mask": (
                [1] * len(input_ids)
            ),
            "labels": labels,
        }


def compute_test_loss(
    model,
    data,
    tokenizer,
    token_ids,
    device,
    batch_size,
    max_seq_length,
    reserve_tokens,
    system_prompt,
) -> float:
    dataset = WOS2TokenTestLossDataset(
        data=data,
        tokenizer=tokenizer,
        token_ids=token_ids,
        max_length=max_seq_length,
        reserve_tokens=reserve_tokens,
        system_prompt=system_prompt,
    )

    old_padding = tokenizer.padding_side
    old_cache = model.config.use_cache

    tokenizer.padding_side = "right"
    model.config.use_cache = False

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )

    total_loss = 0.0
    total_count = 0

    try:
        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="Computing Test Loss",
            ):
                batch = {
                    k: (
                        v.to(device)
                        if torch.is_tensor(v)
                        else v
                    )
                    for k, v in batch.items()
                }

                outputs = model(
                    **batch,
                    use_cache=False,
                    return_dict=True,
                )

                bs = int(
                    batch["input_ids"].size(0)
                )

                total_loss += (
                    outputs.loss
                    .detach()
                    .float()
                    .item()
                    * bs
                )
                total_count += bs

    finally:
        tokenizer.padding_side = old_padding
        model.config.use_cache = old_cache

    return (
        total_loss
        / max(1, total_count)
    )


# ======================================================================
# 11. Generation stop IDs
# ======================================================================

def build_stop_ids(tokenizer) -> List[int]:
    eot_id = (
        tokenizer
        .convert_tokens_to_ids(
            "<|eot_id|>"
        )
    )

    if (
        eot_id is None
        or eot_id == tokenizer.unk_token_id
    ):
        eot_id = tokenizer.eos_token_id

    stop_ids = []

    for sid in (
        tokenizer.eos_token_id,
        eot_id,
    ):
        if (
            sid is not None
            and sid not in stop_ids
        ):
            stop_ids.append(
                int(sid)
            )

    if not stop_ids:
        raise RuntimeError(
            "❌ generation 没有可用 stop id。"
        )

    print("\n" + "=" * 88)
    print("🛑 Generation stop IDs")
    print(
        f"eos_token_id: {tokenizer.eos_token_id}"
    )
    print(
        f"eot_id: {eot_id}"
    )
    print(
        f"stop_ids: {stop_ids}"
    )
    print(
        f"pad_token_id: {tokenizer.pad_token_id}"
    )
    print("=" * 88 + "\n")

    return stop_ids


# ======================================================================
# 12. Main
# ======================================================================

def main(args):
    set_all_seeds(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "❌ 未检测到 CUDA。"
        )

    if not os.path.isdir(
        args.model_path
    ):
        raise FileNotFoundError(
            f"❌ model_path 不存在: {args.model_path}"
        )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    device = torch.device(
        args.device
    )

    torch_dtype = (
        torch.bfloat16
        if args.bf16
        else torch.float32
    )

    # --------------------------------------------------------------
    # A. 标签空间 / TEST
    # --------------------------------------------------------------
    (
        l1_map,
        l2_map,
        l2_to_l1,
    ) = load_label_space(
        args.dict_path,
        args.taxonomy_path,
        args.expected_num_l1,
        args.expected_num_l2,
    )

    test_data = validate_test_split(
        args.test_data_path,
        l1_map,
        l2_map,
        l2_to_l1,
    )

    # --------------------------------------------------------------
    # B. tokenizer + 训练配置审计
    # --------------------------------------------------------------
    print(
        f"📦 从训练产物加载 tokenizer: "
        f"{args.model_path}"
    )

    tokenizer = configure_tokenizer(
        AutoTokenizer.from_pretrained(
            args.model_path
        )
    )

    token_ids = (
        validate_atomic_reserved_tokens(
            tokenizer
        )
    )

    check_prompt_completion_layout(
        tokenizer,
        args.system_prompt,
        token_ids,
    )

    distill_cfg = check_distill_config(
        args.model_path
    )

    # --------------------------------------------------------------
    # C. TEST Prompt 审计
    # --------------------------------------------------------------
    scan_text_truncation(
        test_data,
        tokenizer,
        args.max_seq_length,
        args.reserve_tokens,
    )

    (
        test_prompts,
        audit_path,
    ) = prepare_test_prompts(
        test_data=test_data,
        tokenizer=tokenizer,
        token_ids=token_ids,
        max_seq_length=args.max_seq_length,
        reserve_tokens=args.reserve_tokens,
        system_prompt=args.system_prompt,
        output_dir=args.output_dir,
    )

    if len(test_prompts) != len(test_data):
        raise RuntimeError(
            "❌ TEST 不允许静默跳过样本。"
        )

    # --------------------------------------------------------------
    # D. Student
    # --------------------------------------------------------------
    (
        model,
        artifact_type,
    ) = load_trained_model(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch_dtype,
        attn_implementation=(
            args.attn_implementation
        ),
    )

    # --------------------------------------------------------------
    # E. Test CE
    # --------------------------------------------------------------
    test_loss = None

    if args.compute_test_loss:
        test_loss = compute_test_loss(
            model=model,
            data=test_data,
            tokenizer=tokenizer,
            token_ids=token_ids,
            device=device,
            batch_size=(
                args.loss_eval_batch_size
            ),
            max_seq_length=(
                args.max_seq_length
            ),
            reserve_tokens=(
                args.reserve_tokens
            ),
            system_prompt=(
                args.system_prompt
            ),
        )

        print(
            f"✅ Test completion-only Loss: "
            f"{test_loss:.6f}"
        )

    # --------------------------------------------------------------
    # F. Strict generation
    # --------------------------------------------------------------
    stop_ids = build_stop_ids(
        tokenizer
    )

    tokenizer.padding_side = "left"
    model.config.use_cache = True

    true_l1 = []
    true_l2 = []
    pred_l1 = []
    pred_l2 = []

    exact_parse_count = 0
    valid_pair_count = 0
    hierarchy_consistent_count = 0
    l1_correct_count = 0
    l2_correct_count = 0
    both_correct_count = 0

    detailed_results = []
    print_counter = 0

    print(
        "\n🚀 开始 WOS 双 Token KD Student 严格测试..."
    )

    with torch.no_grad():
        for start in tqdm(
            range(
                0,
                len(test_prompts),
                args.eval_batch_size,
            ),
            desc="Running Strict Test",
        ):
            batch = test_prompts[
                start:
                start + args.eval_batch_size
            ]

            # 直接使用预先经过双入口一致性校验的 prompt IDs，
            # 左 padding，避免再次 tokenize prompt 字符串。
            max_len = max(
                len(item["prompt_ids"])
                for item in batch
            )

            input_ids = torch.full(
                (len(batch), max_len),
                tokenizer.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros(
                (len(batch), max_len),
                dtype=torch.long,
                device=device,
            )

            for row_idx, item in enumerate(batch):
                ids = item["prompt_ids"]
                n = len(ids)

                input_ids[
                    row_idx,
                    max_len - n:
                ] = torch.tensor(
                    ids,
                    dtype=torch.long,
                    device=device,
                )

                attention_mask[
                    row_idx,
                    max_len - n:
                ] = 1

            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=(
                    args.max_new_tokens
                ),
                eos_token_id=stop_ids,
                pad_token_id=(
                    tokenizer.pad_token_id
                ),
                do_sample=False,
                use_cache=True,
            )

            generated_ids = output_ids[
                :,
                max_len:
            ]

            # hierarchy token 是 special token，不能 skip。
            raw_texts = (
                tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )

            for raw_resp, item in zip(
                raw_texts,
                batch,
            ):
                p1_str, p2_str = (
                    parse_generation_exact(
                        raw_resp
                    )
                )

                exact_parse = bool(
                    p1_str and p2_str
                )

                if exact_parse:
                    exact_parse_count += 1

                p1_id = l1_map.get(
                    p1_str,
                    9999,
                )
                p2_id = l2_map.get(
                    p2_str,
                    9999,
                )

                valid_pair = (
                    p1_id != 9999
                    and p2_id != 9999
                )

                hierarchy_consistent = False

                if valid_pair:
                    valid_pair_count += 1

                    hierarchy_consistent = (
                        l2_to_l1.get(p2_str)
                        == p1_str
                    )

                    if hierarchy_consistent:
                        hierarchy_consistent_count += 1

                correct_l1 = (
                    p1_id
                    == item["true_l1"]
                )
                correct_l2 = (
                    p2_id
                    == item["true_l2"]
                )
                both_correct = (
                    correct_l1
                    and correct_l2
                )

                if correct_l1:
                    l1_correct_count += 1
                if correct_l2:
                    l2_correct_count += 1
                if both_correct:
                    both_correct_count += 1

                true_l1.append(
                    item["true_l1"]
                )
                true_l2.append(
                    item["true_l2"]
                )
                pred_l1.append(
                    p1_id
                )
                pred_l2.append(
                    p2_id
                )

                if (
                    print_counter
                    < args.print_samples
                ):
                    print(
                        "\n"
                        + "-" * 88
                    )
                    print(
                        f"样本 {print_counter + 1} "
                        f"(TEST index="
                        f"{item['source_index']})"
                    )
                    print(
                        "【模型原始输出】"
                    )
                    print(raw_resp)
                    print(
                        f"【严格解析】"
                        f"L1={p1_str!r} | "
                        f"L2={p2_str!r}"
                    )
                    print(
                        f"【Gold】"
                        f"L1={item['true_l1_str']!r} | "
                        f"L2={item['true_l2_str']!r}"
                    )
                    print(
                        f"Parse={exact_parse} | "
                        f"L1正确={correct_l1} | "
                        f"L2正确={correct_l2} | "
                        f"层级一致={hierarchy_consistent}"
                    )
                    print(
                        "-" * 88
                    )
                    print_counter += 1

                detailed_results.append(
                    {
                        "source_index": (
                            item["source_index"]
                        ),
                        "true_l1": (
                            item["true_l1_str"]
                        ),
                        "true_l2": (
                            item["true_l2_str"]
                        ),
                        "true_l1_id": (
                            item["true_l1"]
                        ),
                        "true_l2_id": (
                            item["true_l2"]
                        ),
                        "raw_generated_text_with_special_tokens": (
                            raw_resp
                        ),
                        "extracted_l1_exact": (
                            p1_str
                        ),
                        "extracted_l2_exact": (
                            p2_str
                        ),
                        "pred_l1_id": (
                            p1_id
                        ),
                        "pred_l2_id": (
                            p2_id
                        ),
                        "exact_parse_valid": (
                            exact_parse
                        ),
                        "valid_l1_label": (
                            p1_id != 9999
                        ),
                        "valid_l2_label": (
                            p2_id != 9999
                        ),
                        "valid_label_pair": (
                            valid_pair
                        ),
                        "hierarchy_consistent": (
                            hierarchy_consistent
                        ),
                        "correct_l1": (
                            correct_l1
                        ),
                        "correct_l2": (
                            correct_l2
                        ),
                        "both_levels_correct": (
                            both_correct
                        ),
                        "body_text_sha256": (
                            item[
                                "body_text_sha256"
                            ]
                        ),
                        "body_token_sha256": (
                            item[
                                "body_token_sha256"
                            ]
                        ),
                        "prompt_token_sha256": (
                            item[
                                "prompt_token_sha256"
                            ]
                        ),
                        "body_token_count": (
                            item[
                                "body_token_count"
                            ]
                        ),
                        "ctx_start_position": (
                            item[
                                "ctx_start_position"
                            ]
                        ),
                        "ctx_end_position": (
                            item[
                                "ctx_end_position"
                            ]
                        ),
                    }
                )

    # --------------------------------------------------------------
    # G. Metrics
    # --------------------------------------------------------------
    metrics = compute_metric_bundle(
        true_l1,
        true_l2,
        pred_l1,
        pred_l2,
        len(l1_map),
        len(l2_map),
    )

    n = max(
        1,
        len(true_l1),
    )

    exact_parse_rate = (
        exact_parse_count / n
    )
    valid_pair_rate = (
        valid_pair_count / n
    )
    hierarchy_consistency_rate = (
        hierarchy_consistent_count / n
    )
    l1_accuracy = (
        l1_correct_count / n
    )
    l2_accuracy = (
        l2_correct_count / n
    )
    both_correct_rate = (
        both_correct_count / n
    )
    invalid_l1_rate = (
        sum(x == 9999 for x in pred_l1)
        / n
    )
    invalid_l2_rate = (
        sum(x == 9999 for x in pred_l2)
        / n
    )

    # --------------------------------------------------------------
    # H. Save
    # --------------------------------------------------------------
    generations_path = os.path.join(
        args.output_dir,
        "wos_2token_test_generations.json",
    )

    summary_path = os.path.join(
        args.output_dir,
        "wos_2token_test_summary.json",
    )

    with open(
        generations_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            detailed_results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    summary = {
        "model_path": os.path.abspath(
            args.model_path
        ),
        "model_artifact_type": (
            artifact_type
        ),
        "base_model_path": os.path.abspath(
            args.base_model_path
        ),
        "test_data_path": os.path.abspath(
            args.test_data_path
        ),
        "tested_samples": len(
            test_prompts
        ),
        "skipped_samples": 0,
        "test_completion_only_loss": (
            test_loss
        ),
        "global_micro_f1": (
            metrics["global_micro"]
        ),
        "global_macro_f1": (
            metrics["global_macro"]
        ),
        "l1_micro_f1": (
            metrics["l1_micro"]
        ),
        "l1_macro_f1": (
            metrics["l1_macro"]
        ),
        "l2_micro_f1": (
            metrics["l2_micro"]
        ),
        "l2_macro_f1": (
            metrics["l2_macro"]
        ),
        "l1_accuracy": (
            l1_accuracy
        ),
        "l2_accuracy": (
            l2_accuracy
        ),
        "both_levels_correct_rate": (
            both_correct_rate
        ),
        "exact_parse_rate": (
            exact_parse_rate
        ),
        "valid_label_pair_rate": (
            valid_pair_rate
        ),
        "hierarchy_consistency_rate": (
            hierarchy_consistency_rate
        ),
        "invalid_l1_rate": (
            invalid_l1_rate
        ),
        "invalid_l2_rate": (
            invalid_l2_rate
        ),
        "body_span_audit_path": (
            audit_path
        ),
        "all_body_spans_exactly_matched": True,
        "chat_template_direct_tokenization_all_matched": True,
        "completion_format": (
            "L1_START + L1_LABEL + L1_END + "
            "L2_START + L2_LABEL + L2_END + EOT"
        ),
        "parsing_rule": (
            "strict continuous hierarchy chain; "
            "four hierarchy tokens each exactly once; "
            "no normalization/fuzzy matching/label correction"
        ),
        "tokens": {
            "ctx_start": {
                "token": CTX_START_TOKEN,
                "id": token_ids[
                    "ctx_start"
                ],
            },
            "ctx_end": {
                "token": CTX_END_TOKEN,
                "id": token_ids[
                    "ctx_end"
                ],
            },
            "l1_start": {
                "token": HIER_START_TOKEN_L1,
                "id": token_ids[
                    "l1_start"
                ],
            },
            "l1_end": {
                "token": HIER_END_TOKEN_L1,
                "id": token_ids[
                    "l1_end"
                ],
            },
            "l2_start": {
                "token": HIER_START_TOKEN_L2,
                "id": token_ids[
                    "l2_start"
                ],
            },
            "l2_end": {
                "token": HIER_END_TOKEN_L2,
                "id": token_ids[
                    "l2_end"
                ],
            },
        },
        "metric_rule": (
            "same fixed WOS label-space Macro/Micro "
            "definition as paired-token KD validation"
        ),
        "generation": {
            "eval_batch_size": (
                args.eval_batch_size
            ),
            "max_new_tokens": (
                args.max_new_tokens
            ),
            "do_sample": False,
            "stop_ids": stop_ids,
            "pad_token_id": (
                tokenizer.pad_token_id
            ),
            "use_cache": True,
        },
        "distill_config": (
            distill_cfg
        ),
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------------
    # I. Final print
    # --------------------------------------------------------------
    print("\n" + "=" * 96)
    print(
        "🎉 WOS 双 Token KD Student 严格测试完成"
    )
    print("-" * 96)

    if test_loss is not None:
        print(
            f"Test Completion-only Loss  : "
            f"{test_loss:.6f}"
        )

    print(
        f"Global Micro-F1            : "
        f"{metrics['global_micro']:.6f}"
    )
    print(
        f"Global Macro-F1            : "
        f"{metrics['global_macro']:.6f}"
    )
    print(
        f"L1 Micro-F1                : "
        f"{metrics['l1_micro']:.6f}"
    )
    print(
        f"L1 Macro-F1                : "
        f"{metrics['l1_macro']:.6f}"
    )
    print(
        f"L2 Micro-F1                : "
        f"{metrics['l2_micro']:.6f}"
    )
    print(
        f"L2 Macro-F1                : "
        f"{metrics['l2_macro']:.6f}"
    )
    print(
        f"L1 Accuracy                : "
        f"{l1_accuracy:.2%}"
    )
    print(
        f"L2 Accuracy                : "
        f"{l2_accuracy:.2%}"
    )
    print(
        f"Both Levels Correct Rate   : "
        f"{both_correct_rate:.2%}"
    )
    print(
        f"Exact Parse Rate           : "
        f"{exact_parse_rate:.2%}"
    )
    print(
        f"Valid Label Pair Rate      : "
        f"{valid_pair_rate:.2%}"
    )
    print(
        f"Hierarchy Consistency Rate : "
        f"{hierarchy_consistency_rate:.2%}"
    )
    print(
        f"Invalid L1 Rate            : "
        f"{invalid_l1_rate:.2%}"
    )
    print(
        f"Invalid L2 Rate            : "
        f"{invalid_l2_rate:.2%}"
    )
    print(
        f"Tested Samples             : "
        f"{len(test_prompts):,}"
    )
    print(
        "Skipped Samples            : 0"
    )
    print("=" * 96)

    print(
        f"💾 Generations: {generations_path}"
    )
    print(
        f"💾 Body audit : {audit_path}"
    )
    print(
        f"💾 Summary    : {summary_path}"
    )


# ======================================================================
# 13. CLI
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Strict test for current WOS paired START/END "
            "hierarchy-token KD Student."
        )
    )

    # 默认测试你刚跑的 traditional best merged model。
    parser.add_argument(
        "--model_path",
        type=str,
        default=(
            "/data1/lzp/code/hierarchical_distill/"
            "checkpoints_wos_2token_traditional/"
            "best_kd_model"
        ),
    )

    parser.add_argument(
        "--base_model_path",
        type=str,
        default=(
            "/data1/lzp/code/"
            "llama-3.2_1b/"
        ),
    )

    parser.add_argument(
        "--test_data_path",
        type=str,
        default=(
            "/data1/lzp/code/data/wos/"
            "test_formatted.jsonl"
        ),
    )

    parser.add_argument(
        "--taxonomy_path",
        type=str,
        default=(
            "/data1/lzp/code/data/wos/"
            "wos.taxnomy"
        ),
    )

    parser.add_argument(
        "--dict_path",
        type=str,
        default=(
            "/data1/lzp/code/data/wos/"
            "label2id.json"
        ),
    )

    parser.add_argument(
        "--expected_num_l1",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--expected_num_l2",
        type=int,
        default=134,
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
        "--eval_batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--loss_eval_batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--compute_test_loss",
        type=str2bool,
        default=True,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--bf16",
        type=str2bool,
        default=True,
    )

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
    )

    parser.add_argument(
        "--system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "/data1/lzp/code/hierarchical_distill/"
            "checkpoints_wos_2token_traditional/"
            "test_results_best/"
        ),
    )

    parser.add_argument(
        "--print_samples",
        type=int,
        default=5,
    )

    args = parser.parse_args()
    main(args)
