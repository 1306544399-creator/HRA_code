#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WOS paired hierarchy-token KD
=============================

针对当前“双 Token / 层”设计：
  L1_START = <|reserved_special_token_112|>
  L1_END   = <|reserved_special_token_113|>
  L2_START = <|reserved_special_token_114|>
  L2_END   = <|reserved_special_token_115|>

Completion 严格为：
  L1_START + L1_LABEL + L1_END + L2_START + L2_LABEL + L2_END + EOT

本脚本有两个 stage：

1) cache
   用已经训练好的 Teacher best_sft_model 离线提取：
   - 全 Completion causal KD 位置的 Teacher final hidden
   - 4 个层级 Token 第二次 occurrence 的相对位置
   - common-vocab Teacher lm_head
   之后训练 Student 时不再加载 8B Teacher Transformer。

2) train
   Student 必须从 raw base Llama-3.2-1B 初始化，不读取 Student-SFT。
   支持四种实验：
     traditional : Task CE + Full Completion Logits KD
     align       : traditional + 4-token Hidden Align
     hier        : traditional + 2-relation Hidden Hier
     full        : traditional + Align + Hier

Hidden Align：
  对 L1_START / L1_END / L2_START / L2_END 四个第二次 occurrence
  分别对齐 Teacher final hidden，Student 先通过共享 Linear projector。

Hidden Hier：
  保持旧单 Token WOS 的“L1 -> L2 表示差分”定义，
  在双 Token 结构中扩展为两条平行的跨层关系：

    START relation:
      L1_START -> L2_START

    END relation:
      L1_END   -> L2_END

  每条关系：
    ΔS = P(hS_L2) - P(hS_L1)
    ΔT = hT_L2 - hT_L1
    loss_relation = MSE(ΔS, ΔT)

  最终：
    L_hier = weighted_mean(L_hier_start, L_hier_end)

  这样 Hier 始终只描述 L1 -> L2 的跨层关系，
  不混入 L1_START->L1_END / L2_START->L2_END 的层内状态变化。

重要：
- Teacher = 已训练 best_sft_model
- Student = 未训练 raw base
- Prompt / Completion 与当前 paired-token Student/Teacher SFT 保持一致
- Traditional KD 仍是 common-vocab logits KL，不是 hidden MSE
- Teacher / Student token ID 可不同，按 token 字符串语义建立映射
"""

import os

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import hashlib
import json
import math
import random
import shutil
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import peft
import transformers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed as hf_set_seed,
)

try:
    import swanlab
    from swanlab.integration.transformers import SwanLabCallback
    HAS_SWANLAB = True
except Exception:
    swanlab = None
    SwanLabCallback = None
    HAS_SWANLAB = False


# ======================================================================
# 0. 常量
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
FULL_IO_META_NAME = "full_input_output_embeddings_meta.json"
HIDDEN_PROJECTOR_WEIGHTS_NAME = "hidden_state_projector.pt"
HIDDEN_PROJECTOR_META_NAME = "hidden_state_projector_meta.json"
WOS_RESUME_STATE_NAME = "wos_paired_resume_state.json"
WOS_PENDING_NAME = "wos_paired_save_in_progress.json"
KD_TRAINER_STATE_PREFIX = "wos_paired_kd_trainer_state_rank_"

CACHE_MANIFEST = "cache_manifest.json"
CACHE_HIDDEN = "teacher_completion_hidden.npy"
CACHE_OFFSETS = "offsets.npy"
CACHE_VALID_COUNTS = "valid_counts.npy"
CACHE_HASHES = "student_input_hashes.jsonl"
CACHE_COMMON_HEAD = "teacher_common_lm_head.pt"

CACHE_REL_FILES = {
    "l1_start": "l1_start_rel_offsets.npy",
    "l1_end": "l1_end_rel_offsets.npy",
    "l2_start": "l2_start_rel_offsets.npy",
    "l2_end": "l2_end_rel_offsets.npy",
}

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
    "Follow the required output format exactly. Do not produce explanations or any extra conversational text."
)


# ======================================================================
# 1. 通用工具
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


def unwrap_training_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _reset_directory(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _token_ids_sha256(ids: List[int]) -> str:
    payload = ",".join(str(int(x)) for x in ids)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_final_norm_module(model):
    """
    兼容：
      LlamaForCausalLM
      PeftModelForCausalLM
      DDP(PeftModelForCausalLM)
    """
    root = unwrap_training_model(model)

    if hasattr(root, "get_base_model"):
        try:
            root = root.get_base_model()
        except Exception:
            pass

    candidates = []
    if hasattr(root, "model"):
        candidates.append(getattr(root.model, "norm", None))
        if hasattr(root.model, "model"):
            candidates.append(getattr(root.model.model, "norm", None))

    candidates.append(getattr(root, "norm", None))

    for module in candidates:
        if isinstance(module, nn.Module):
            return module

    raise RuntimeError("❌ 找不到 Llama final norm，无法精确抓取 final hidden。")


def assert_raw_base_student_path(model_path: str) -> None:
    """
    防止误加载训练过的 Student。
    """
    p = os.path.abspath(model_path)
    low = p.lower()

    suspicious = (
        "best_sft_model",
        "best_kd_model",
        "final_model",
        "student_sft",
        "checkpoints_student",
        "best_sft_checkpoint",
        "final_sft_checkpoint",
        "best_kd_checkpoint",
        "final_kd_checkpoint",
    )
    hit = [x for x in suspicious if x in low]
    if hit:
        raise ValueError(
            "❌ Student 必须从未训练 raw base 开始，但 --student_model_path "
            f"看起来是训练输出：{p}\nmatched={hit}"
        )

    adapter_markers = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
    )
    found = [
        name for name in adapter_markers
        if os.path.isfile(os.path.join(p, name))
    ]
    if found:
        raise ValueError(
            "❌ Student 路径是 PEFT adapter/checkpoint，不是 raw base："
            f"{p}\nfound={found}"
        )

    print("\n" + "=" * 84)
    print("🧱 Student 初始化检查")
    print(f"Student raw base: {p}")
    print("✅ 未发现 Student-SFT / KD / PEFT checkpoint 标记")
    print("✅ 本次 KD 不加载任何训练过的 Student 权重")
    print("=" * 84 + "\n")


# ======================================================================
# 2. 当前 paired-token Prompt / Completion
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
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_content(text)},
    ]


def configure_tokenizer(tokenizer):
    tokenizer.chat_template = CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError("❌ tokenizer 没有 pad_token_id。")
    tokenizer.padding_side = "right"
    return tokenizer


def validate_atomic_reserved_tokens(tokenizer) -> Dict[str, int]:
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

    print("\n" + "=" * 84)
    print("🔎 验证 6 个 Llama reserved tokens")

    for role, token in token_map.items():
        if token not in vocab:
            raise RuntimeError(f"❌ tokenizer 缺少 {token}")
        ids = tokenizer.encode(token, add_special_tokens=False)
        tid = tokenizer.convert_tokens_to_ids(token)
        if len(ids) != 1 or int(ids[0]) != int(tid):
            raise RuntimeError(
                f"❌ {token} 必须严格是单 token：encode={ids}, convert={tid}"
            )
        if tid == tokenizer.unk_token_id:
            raise RuntimeError(f"❌ {token} 被编码为 UNK")
        out[role] = int(tid)
        print(f"✅ {role:8s}: {token} -> {tid}")

    if len(set(out.values())) != 6:
        raise RuntimeError(f"❌ 6 个 reserved token ID 未两两不同：{out}")

    print("=" * 84 + "\n")
    return out


def encode_prompt_ids(tokenizer, text: str, system_prompt: str) -> List[int]:
    messages = build_prompt_messages(text, system_prompt)
    direct = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    direct = [int(x) for x in direct]

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
        raise RuntimeError("❌ chat template 两种编码入口不一致。")

    return direct


def truncate_text_for_prompt(
    text: str,
    tokenizer,
    max_seq_length: int,
    reserve_tokens: int,
) -> str:
    max_text_len = max_seq_length - reserve_tokens
    if max_text_len <= 0:
        raise ValueError("❌ max_seq_length 必须大于 reserve_tokens")

    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_text_len:
        ids = ids[:max_text_len]
        text = tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return text


def find_second_occurrence(ids: List[int], token_id: int, name: str) -> int:
    positions = [i for i, x in enumerate(ids) if int(x) == int(token_id)]
    if len(positions) != 2:
        raise RuntimeError(
            f"❌ {name} 在完整样本中必须出现 2 次，实际={positions}"
        )
    return int(positions[1])


def _strip_trailing_stop_strings(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    stop_strings = ("<|eot_id|>", "<|end_of_text|>")
    changed = True
    while changed:
        changed = False
        for s in stop_strings:
            if text.endswith(s):
                text = text[:-len(s)].rstrip()
                changed = True
    return text


def parse_generation_exact(raw_resp: str) -> Tuple[str, str]:
    text = _strip_trailing_stop_strings(raw_resp)

    required = (
        HIER_START_TOKEN_L1,
        HIER_END_TOKEN_L1,
        HIER_START_TOKEN_L2,
        HIER_END_TOKEN_L2,
    )
    if any(text.count(t) != 1 for t in required):
        return "", ""

    prefix = HIER_START_TOKEN_L1
    middle = HIER_END_TOKEN_L1 + HIER_START_TOKEN_L2
    suffix = HIER_END_TOKEN_L2

    if not text.startswith(prefix) or not text.endswith(suffix):
        return "", ""

    body = text[len(prefix):-len(suffix)]
    if body.count(middle) != 1:
        return "", ""

    l1, l2 = body.split(middle, 1)
    if not l1 or not l2:
        return "", ""
    if l1 != l1.strip() or l2 != l2.strip():
        return "", ""
    if "\n" in l1 or "\n" in l2 or "\r" in l1 or "\r" in l2:
        return "", ""
    return l1, l2


# ======================================================================
# 3. 数据 / taxonomy
# ======================================================================

REQUIRED_FIELDS = ("text", "l1_label", "l2_label", "l1_id", "l2_id")


def load_json_or_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        raise ValueError(f"❌ 空数据文件: {path}")

    try:
        obj = json.loads(raw)
        if not isinstance(obj, list):
            raise TypeError("JSON 顶层必须是 list")
        return obj
    except json.JSONDecodeError:
        rows = []
        for line_no, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise TypeError(f"第 {line_no} 行不是 dict")
            rows.append(item)
        return rows


def parse_taxonomy(path: str):
    parent_to_children = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = [x.strip() for x in line.split("\t") if x.strip()]
            if len(parts) < 2:
                raise ValueError(f"❌ taxonomy 第 {line_no} 行异常")
            parent = parts[0]
            if parent in parent_to_children:
                raise ValueError(f"❌ taxonomy parent 重复: {parent}")
            parent_to_children[parent] = parts[1:]

    if "Root" not in parent_to_children:
        raise ValueError("❌ taxonomy 缺少 Root")

    l1_labels = parent_to_children["Root"]
    l2_labels = []
    l2_to_l1 = {}
    for l1 in l1_labels:
        if l1 not in parent_to_children:
            raise ValueError(f"❌ taxonomy L1 无子节点行: {l1}")
        for l2 in parent_to_children[l1]:
            if l2 in l2_to_l1:
                raise ValueError(f"❌ L2 重复归属: {l2}")
            l2_to_l1[l2] = l1
            l2_labels.append(l2)
    return l1_labels, l2_labels, l2_to_l1


def load_label_space(dict_path: str, taxonomy_path: str):
    with open(dict_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    l1_map = {str(k): int(v) for k, v in d["l1_label2id"].items()}
    l2_map = {str(k): int(v) for k, v in d["l2_label2id"].items()}

    tax_l1, tax_l2, l2_to_l1 = parse_taxonomy(taxonomy_path)
    expected_l1 = {x: i for i, x in enumerate(tax_l1)}
    expected_l2 = {x: i for i, x in enumerate(tax_l2)}

    if l1_map != expected_l1:
        raise ValueError("❌ L1 label2id 与 taxonomy 顺序不一致")
    if l2_map != expected_l2:
        raise ValueError("❌ L2 label2id 与 taxonomy 顺序不一致")

    return l1_map, l2_map, l2_to_l1


def validate_split(
    path: str,
    split_name: str,
    l1_map: Dict[str, int],
    l2_map: Dict[str, int],
    l2_to_l1: Dict[str, str],
) -> List[dict]:
    data = load_json_or_jsonl(path)

    for i, item in enumerate(tqdm(data, desc=f"校验{split_name}", leave=False)):
        missing = [k for k in REQUIRED_FIELDS if k not in item]
        if missing:
            raise KeyError(f"❌ {split_name} {i} 缺字段: {missing}")

        text = item["text"]
        l1 = item["l1_label"]
        l2 = item["l2_label"]
        id1 = item["l1_id"]
        id2 = item["l2_id"]

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"❌ {split_name} {i} text 空")
        if l1 not in l1_map or l2 not in l2_map:
            raise ValueError(f"❌ {split_name} {i} 未知标签")
        if int(id1) != l1_map[l1] or int(id2) != l2_map[l2]:
            raise ValueError(f"❌ {split_name} {i} label-ID 错位")
        if l2_to_l1[l2] != l1:
            raise ValueError(f"❌ {split_name} {i} L1-L2 父子关系错误")

    print(f"✅ {split_name}: {len(data):,} 条严格校验通过")
    return data


# ======================================================================
# 4. Dataset / Collator
# ======================================================================

class PairedTokenWOSDataset(Dataset):
    def __init__(
        self,
        data: List[dict],
        tokenizer,
        token_ids: Dict[str, int],
        max_length: int,
        reserve_tokens: int,
        system_prompt: str,
        split_name: str,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.token_ids = token_ids
        self.max_length = int(max_length)
        self.reserve_tokens = int(reserve_tokens)
        self.system_prompt = system_prompt
        self.split_name = split_name

        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is None or eot_id == tokenizer.unk_token_id:
            raise RuntimeError("❌ tokenizer 缺少原子 <|eot_id|>")
        self.eot_id = int(eot_id)

    def __len__(self):
        return len(self.data)

    def encode_item(self, idx: int) -> dict:
        item = self.data[idx]
        text = str(item["text"]).strip()
        l1 = str(item["l1_label"]).strip()
        l2 = str(item["l2_label"]).strip()

        text = truncate_text_for_prompt(
            text,
            self.tokenizer,
            self.max_length,
            self.reserve_tokens,
        )

        prompt_ids = encode_prompt_ids(
            self.tokenizer,
            text,
            self.system_prompt,
        )
        completion_text = build_completion_text(l1, l2)
        completion_ids = self.tokenizer.encode(
            completion_text,
            add_special_tokens=False,
        )
        completion_ids = [int(x) for x in completion_ids]

        input_ids = prompt_ids + completion_ids + [self.eot_id]
        labels = [-100] * len(prompt_ids) + completion_ids + [self.eot_id]

        if len(input_ids) > self.max_length:
            raise RuntimeError(
                f"❌ {self.split_name} sample={idx} 完整序列超长: "
                f"{len(input_ids)} > {self.max_length}。"
                "请增大 reserve_tokens 或 max_seq_length。"
            )

        positions = {
            role: find_second_occurrence(
                input_ids,
                self.token_ids[role],
                role,
            )
            for role in HIER_TOKEN_ROLES
        }

        if not (
            positions["l1_start"]
            < positions["l1_end"]
            < positions["l2_start"]
            < positions["l2_end"]
        ):
            raise RuntimeError(
                f"❌ sample={idx} 四个层级 token 顺序错误: {positions}"
            )

        if positions["l2_start"] != positions["l1_end"] + 1:
            raise RuntimeError(
                f"❌ sample={idx} L1_END 与 L2_START 不相邻"
            )

        valid_mask = [
            labels[p + 1] != -100
            for p in range(len(labels) - 1)
        ]
        valid_positions = [p for p, yes in enumerate(valid_mask) if yes]
        pos_to_rel = {p: r for r, p in enumerate(valid_positions)}

        for role, pos in positions.items():
            if pos not in pos_to_rel:
                raise RuntimeError(
                    f"❌ sample={idx} {role} 第二次 occurrence "
                    "不在 causal KD mask"
                )

        rel_positions = {
            role: int(pos_to_rel[pos])
            for role, pos in positions.items()
        }

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "sample_index": int(idx),
            "positions": positions,
            "rel_positions": rel_positions,
            "valid_count": len(valid_positions),
            "input_hash": _token_ids_sha256(input_ids),
            "truncated_text": text,
            "l1_label": l1,
            "l2_label": l2,
            "l1_id": int(item["l1_id"]),
            "l2_id": int(item["l2_id"]),
        }

    def __getitem__(self, idx):
        x = self.encode_item(idx)
        return {
            "input_ids": x["input_ids"],
            "attention_mask": x["attention_mask"],
            "labels": x["labels"],
            "sample_index": x["sample_index"],
            "l1_start_pos": x["positions"]["l1_start"],
            "l1_end_pos": x["positions"]["l1_end"],
            "l2_start_pos": x["positions"]["l2_start"],
            "l2_end_pos": x["positions"]["l2_end"],
        }


class PairedTokenCollator:
    def __init__(self, tokenizer):
        self.pad_id = int(tokenizer.pad_token_id)

    def __call__(self, features: List[dict]) -> dict:
        bsz = len(features)
        max_len = max(len(x["input_ids"]) for x in features)

        input_ids = torch.full(
            (bsz, max_len),
            self.pad_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (bsz, max_len),
            dtype=torch.long,
        )
        labels = torch.full(
            (bsz, max_len),
            -100,
            dtype=torch.long,
        )

        sample_index = torch.zeros(bsz, dtype=torch.long)
        positions = {
            role: torch.zeros(bsz, dtype=torch.long)
            for role in HIER_TOKEN_ROLES
        }

        key_map = {
            "l1_start": "l1_start_pos",
            "l1_end": "l1_end_pos",
            "l2_start": "l2_start_pos",
            "l2_end": "l2_end_pos",
        }

        for row, x in enumerate(features):
            n = len(x["input_ids"])
            input_ids[row, :n] = torch.tensor(x["input_ids"], dtype=torch.long)
            attention_mask[row, :n] = 1
            labels[row, :n] = torch.tensor(x["labels"], dtype=torch.long)
            sample_index[row] = int(x["sample_index"])
            for role in HIER_TOKEN_ROLES:
                positions[role][row] = int(x[key_map[role]])

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sample_index": sample_index,
        }
        for role in HIER_TOKEN_ROLES:
            batch[f"{role}_pos"] = positions[role]
        return batch


# ======================================================================
# 5. Teacher / Student vocab 对齐
# ======================================================================

def build_teacher_student_vocab_alignment(
    student_tokenizer,
    teacher_tokenizer,
) -> dict:
    sv = student_tokenizer.get_vocab()
    tv = teacher_tokenizer.get_vocab()

    common = set(sv) & set(tv)
    if not common:
        raise RuntimeError("❌ Teacher/Student 没有共同 token")

    protected = [
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

    print("\n" + "=" * 92)
    print("🔐 Teacher / Student protected tokens 检查")
    protected_info = {}

    for tok in protected:
        if tok not in sv or tok not in tv:
            raise RuntimeError(f"❌ 两边至少一边缺少 protected token: {tok}")
        sids = student_tokenizer.encode(tok, add_special_tokens=False)
        tids = teacher_tokenizer.encode(tok, add_special_tokens=False)
        if len(sids) != 1 or len(tids) != 1:
            raise RuntimeError(
                f"❌ protected token 不是两边原子 token: {tok}; "
                f"S={sids}, T={tids}"
            )
        if tok not in common:
            raise RuntimeError(f"❌ protected token 未进入 common vocab: {tok}")
        protected_info[tok] = {
            "student_id": int(sv[tok]),
            "teacher_id": int(tv[tok]),
        }
        print(
            f"✅ {tok}: S={sv[tok]}, T={tv[tok]}"
        )

    common_sorted = sorted(common, key=lambda t: int(sv[t]))
    student_common_ids = [int(sv[t]) for t in common_sorted]
    teacher_common_ids = [int(tv[t]) for t in common_sorted]

    student_to_teacher = np.full(
        (len(student_tokenizer),),
        -1,
        dtype=np.int64,
    )
    for tok in common_sorted:
        student_to_teacher[int(sv[tok])] = int(tv[tok])

    print("\n📚 Common vocab")
    print(f"Student vocab: {len(sv):,}")
    print(f"Teacher vocab: {len(tv):,}")
    print(f"Common:        {len(common_sorted):,}")
    print(f"Student-only:  {len(set(sv)-set(tv)):,}")
    print(f"Teacher-only:  {len(set(tv)-set(sv)):,}")
    print("✅ KD logits 按 token 字符串语义对齐，不按数字 ID 硬对齐")
    print("=" * 92 + "\n")

    return {
        "student_common_ids": student_common_ids,
        "teacher_common_ids": teacher_common_ids,
        "student_to_teacher": student_to_teacher,
        "protected_info": protected_info,
        "common_vocab_size": len(common_sorted),
    }


# ======================================================================
# 6. CACHE stage：训练后的 Teacher -> 离线 cache
# ======================================================================

def translate_student_batch_to_teacher(
    student_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mapping: torch.Tensor,
    teacher_pad_id: int,
) -> torch.Tensor:
    if student_input_ids.min().item() < 0:
        raise RuntimeError("❌ Student input_ids 出现负数")
    if student_input_ids.max().item() >= mapping.numel():
        raise RuntimeError("❌ Student input_ids 超出映射表")

    teacher_ids = mapping[student_input_ids]
    bad = attention_mask.bool() & teacher_ids.lt(0)
    if bad.any():
        bad_student_ids = torch.unique(student_input_ids[bad]).tolist()
        raise RuntimeError(
            "❌ 实际输入中出现 Student-only token，无法做严格 Teacher 对齐。\n"
            f"student_ids={bad_student_ids[:30]}"
        )

    teacher_ids = teacher_ids.clone()
    teacher_ids[~attention_mask.bool()] = int(teacher_pad_id)
    return teacher_ids


@torch.no_grad()
def build_teacher_cache(args) -> None:
    set_all_seeds(args.seed)

    student_tokenizer = configure_tokenizer(
        AutoTokenizer.from_pretrained(args.student_model_path)
    )
    teacher_tokenizer = configure_tokenizer(
        AutoTokenizer.from_pretrained(args.teacher_model_path)
    )

    student_token_ids = validate_atomic_reserved_tokens(student_tokenizer)
    validate_atomic_reserved_tokens(teacher_tokenizer)

    l1_map, l2_map, l2_to_l1 = load_label_space(
        args.dict_path,
        args.taxonomy_path,
    )
    train_data = validate_split(
        args.train_data_path,
        "TRAIN",
        l1_map,
        l2_map,
        l2_to_l1,
    )

    dataset = PairedTokenWOSDataset(
        data=train_data,
        tokenizer=student_tokenizer,
        token_ids=student_token_ids,
        max_length=args.max_seq_length,
        reserve_tokens=args.reserve_tokens,
        system_prompt=args.system_prompt,
        split_name="CACHE_TRAIN",
    )

    alignment = build_teacher_student_vocab_alignment(
        student_tokenizer,
        teacher_tokenizer,
    )

    os.makedirs(args.cache_dir, exist_ok=True)

    # --------------------------------------------------------------
    # 第一遍：只编码，得到总 valid count / offsets / 4 个 rel offset / hash
    # --------------------------------------------------------------
    n = len(dataset)
    valid_counts = np.zeros(n, dtype=np.int32)
    offsets = np.zeros(n + 1, dtype=np.int64)
    rels = {
        role: np.zeros(n, dtype=np.int32)
        for role in HIER_TOKEN_ROLES
    }
    hash_records = []

    print("\n🔎 Cache 第一遍：审计所有训练样本布局")
    for i in tqdm(range(n), desc="cache-layout"):
        x = dataset.encode_item(i)
        valid_counts[i] = int(x["valid_count"])
        offsets[i + 1] = offsets[i] + int(x["valid_count"])
        for role in HIER_TOKEN_ROLES:
            rels[role][i] = int(x["rel_positions"][role])
        hash_records.append({
            "sample_index": i,
            "input_hash": x["input_hash"],
            "valid_count": int(x["valid_count"]),
            **{
                f"{role}_rel": int(x["rel_positions"][role])
                for role in HIER_TOKEN_ROLES
            },
        })

    total_valid = int(offsets[-1])
    print(f"✅ Samples={n:,}")
    print(f"✅ Total Completion KD prediction positions={total_valid:,}")

    np.save(os.path.join(args.cache_dir, CACHE_OFFSETS), offsets)
    np.save(os.path.join(args.cache_dir, CACHE_VALID_COUNTS), valid_counts)
    for role in HIER_TOKEN_ROLES:
        np.save(
            os.path.join(args.cache_dir, CACHE_REL_FILES[role]),
            rels[role],
        )

    with open(
        os.path.join(args.cache_dir, CACHE_HASHES),
        "w",
        encoding="utf-8",
    ) as f:
        for rec in hash_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --------------------------------------------------------------
    # Teacher：必须是已经训练并合并后的 best_sft_model
    # --------------------------------------------------------------
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        device_map=None,
    )
    teacher.to(args.device)
    teacher.eval()
    teacher.requires_grad_(False)
    teacher.config.use_cache = False

    if sum(p.numel() for p in teacher.parameters() if p.requires_grad) != 0:
        raise RuntimeError("❌ Teacher 未完全冻结")

    teacher_hidden_size = int(teacher.config.hidden_size)

    if teacher.get_output_embeddings().weight.shape[0] != len(teacher_tokenizer):
        raise RuntimeError(
            "❌ Teacher tokenizer 长度与 lm_head 行数不一致"
        )

    # common-vocab Teacher lm_head
    teacher_common_ids_t = torch.tensor(
        alignment["teacher_common_ids"],
        dtype=torch.long,
        device=teacher.get_output_embeddings().weight.device,
    )
    head = teacher.get_output_embeddings()
    common_weight = head.weight.index_select(
        0, teacher_common_ids_t
    ).detach().cpu()

    common_bias = None
    if getattr(head, "bias", None) is not None:
        common_bias = head.bias.index_select(
            0, teacher_common_ids_t
        ).detach().cpu()

    torch.save(
        {
            "weight": common_weight.to(torch.bfloat16),
            "bias": (
                common_bias.to(torch.bfloat16)
                if common_bias is not None
                else None
            ),
            "student_common_ids": torch.tensor(
                alignment["student_common_ids"],
                dtype=torch.long,
            ),
            "teacher_common_ids": torch.tensor(
                alignment["teacher_common_ids"],
                dtype=torch.long,
            ),
            "teacher_hidden_size": teacher_hidden_size,
        },
        os.path.join(args.cache_dir, CACHE_COMMON_HEAD),
    )

    # final hidden mmap：float16，减少磁盘
    hidden_mm = np.lib.format.open_memmap(
        os.path.join(args.cache_dir, CACHE_HIDDEN),
        mode="w+",
        dtype=np.float16,
        shape=(total_valid, teacher_hidden_size),
    )

    collator = PairedTokenCollator(student_tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.cache_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collator,
    )

    mapping_t = torch.from_numpy(
        alignment["student_to_teacher"]
    ).to(args.device)
    teacher_pad_id = teacher_tokenizer.pad_token_id

    final_norm = get_final_norm_module(teacher)

    print("\n🧊 Cache 第二遍：运行训练后的 Teacher")
    write_cursor = 0

    for batch in tqdm(loader, desc="teacher-cache"):
        input_ids = batch["input_ids"].to(args.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(
            args.device, non_blocking=True
        )
        labels = batch["labels"].to(args.device, non_blocking=True)

        teacher_ids = translate_student_batch_to_teacher(
            input_ids,
            attention_mask,
            mapping_t,
            teacher_pad_id,
        )

        captured = {}

        def hook(_module, _inputs, output):
            captured["hidden"] = output

        handle = final_norm.register_forward_hook(hook)
        try:
            _ = teacher(
                input_ids=teacher_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        finally:
            handle.remove()

        if "hidden" not in captured:
            raise RuntimeError("❌ 未抓到 Teacher final hidden")
        hidden = captured["hidden"]

        shift_labels = labels[:, 1:]
        valid_mask = shift_labels.ne(-100)

        for row in range(input_ids.size(0)):
            actual_len = int(attention_mask[row].sum().item())
            row_mask = valid_mask[row, :max(0, actual_len - 1)]
            selected = hidden[
                row,
                :max(0, actual_len - 1),
                :
            ][row_mask]
            k = int(selected.size(0))

            hidden_mm[
                write_cursor:write_cursor + k
            ] = (
                selected.detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float16)
            )
            write_cursor += k

    hidden_mm.flush()

    if write_cursor != total_valid:
        raise RuntimeError(
            f"❌ Cache 写入数量不一致: wrote={write_cursor}, expected={total_valid}"
        )

    manifest = {
        "format_version": 2,
        "task": "WOS paired START-END hierarchy token KD",
        "teacher_model_path": os.path.abspath(args.teacher_model_path),
        "student_raw_model_path_for_tokenization": os.path.abspath(
            args.student_model_path
        ),
        "train_data_path": os.path.abspath(args.train_data_path),
        "taxonomy_path": os.path.abspath(args.taxonomy_path),
        "dict_path": os.path.abspath(args.dict_path),
        "num_samples": n,
        "teacher_hidden_size": teacher_hidden_size,
        "common_vocab_size": int(alignment["common_vocab_size"]),
        "total_completion_prediction_positions": total_valid,
        "hidden_storage_dtype": "float16",
        "hidden_storage_shape": [total_valid, teacher_hidden_size],
        "max_seq_length": args.max_seq_length,
        "reserve_tokens": args.reserve_tokens,
        "system_prompt": args.system_prompt,
        "chat_template": CHAT_TEMPLATE,
        "completion": (
            "L1_START + L1_LABEL + L1_END + "
            "L2_START + L2_LABEL + L2_END + EOT"
        ),
        "teacher_forcing": True,
        "full_kd_mask": "labels[:,1:] != -100",
        "special_hidden_targets": [
            f"{r}_second_occurrence" for r in HIER_TOKEN_ROLES
        ],
        "tokens": {
            "ctx_start": CTX_START_TOKEN,
            "ctx_end": CTX_END_TOKEN,
            **ROLE_TO_TOKEN,
        },
        "protected_info": alignment["protected_info"],
    }

    with open(
        os.path.join(args.cache_dir, CACHE_MANIFEST),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 92)
    print("✅ Teacher cache 完成")
    print(f"Teacher: {args.teacher_model_path}")
    print(f"Cache:   {args.cache_dir}")
    print(f"Hidden:  {total_valid:,} x {teacher_hidden_size}")
    print("✅ 后续 train stage 不再加载 8B Teacher Transformer")
    print("=" * 92 + "\n")


# ======================================================================
# 7. Offline cache reader
# ======================================================================

class OfflineTeacherCache:
    def __init__(self, cache_dir: str, expected_num_samples: int):
        self.cache_dir = cache_dir

        with open(
            os.path.join(cache_dir, CACHE_MANIFEST),
            "r",
            encoding="utf-8",
        ) as f:
            self.manifest = json.load(f)

        self.offsets = np.load(
            os.path.join(cache_dir, CACHE_OFFSETS),
            mmap_mode="r",
        )
        self.valid_counts = np.load(
            os.path.join(cache_dir, CACHE_VALID_COUNTS),
            mmap_mode="r",
        )
        self.rels = {
            role: np.load(
                os.path.join(cache_dir, CACHE_REL_FILES[role]),
                mmap_mode="r",
            )
            for role in HIER_TOKEN_ROLES
        }
        self.hidden = np.load(
            os.path.join(cache_dir, CACHE_HIDDEN),
            mmap_mode="r",
        )
        self.head_payload = _torch_load_cpu(
            os.path.join(cache_dir, CACHE_COMMON_HEAD)
        )

        self.num_samples = int(self.manifest["num_samples"])
        self.teacher_hidden_size = int(
            self.manifest["teacher_hidden_size"]
        )
        self.common_vocab_size = int(
            self.manifest["common_vocab_size"]
        )

        if self.num_samples != expected_num_samples:
            raise RuntimeError(
                "❌ Cache 样本数与当前训练集不一致："
                f"cache={self.num_samples}, train={expected_num_samples}"
            )

        self.hash_records = []
        with open(
            os.path.join(cache_dir, CACHE_HASHES),
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                if line.strip():
                    self.hash_records.append(json.loads(line))

        if len(self.hash_records) != self.num_samples:
            raise RuntimeError("❌ Cache hash 记录数异常")

        expected_total = int(
            self.manifest["total_completion_prediction_positions"]
        )
        if tuple(self.hidden.shape) != (
            expected_total,
            self.teacher_hidden_size,
        ):
            raise RuntimeError(
                f"❌ Cache hidden shape 异常: {self.hidden.shape}"
            )
        if self.offsets.shape != (self.num_samples + 1,):
            raise RuntimeError("❌ offsets shape 异常")
        if self.valid_counts.shape != (self.num_samples,):
            raise RuntimeError("❌ valid_counts shape 异常")
        for role in HIER_TOKEN_ROLES:
            if self.rels[role].shape != (self.num_samples,):
                raise RuntimeError(f"❌ {role} rel shape 异常")

        sw = self.head_payload["student_common_ids"]
        tw = self.head_payload["teacher_common_ids"]
        cw = self.head_payload["weight"]

        if sw.numel() != self.common_vocab_size:
            raise RuntimeError("❌ student_common_ids shape 异常")
        if tw.numel() != self.common_vocab_size:
            raise RuntimeError("❌ teacher_common_ids shape 异常")
        if tuple(cw.shape) != (
            self.common_vocab_size,
            self.teacher_hidden_size,
        ):
            raise RuntimeError("❌ teacher common lm_head shape 异常")

        print("\n" + "=" * 92)
        print("✅ Offline Teacher cache 校验通过")
        print(f"Cache: {cache_dir}")
        print(f"Samples: {self.num_samples:,}")
        print(f"Teacher hidden: {self.teacher_hidden_size}")
        print(f"Common vocab: {self.common_vocab_size:,}")
        print("✅ 4 个第二次 hierarchy-token occurrence 均有独立 rel offset")
        print("=" * 92 + "\n")

    def verify_sample(
        self,
        sample_idx: int,
        current_hash: str,
        current_valid_count: int,
        current_rel: Dict[str, int],
    ):
        rec = self.hash_records[sample_idx]
        if rec["input_hash"] != current_hash:
            raise RuntimeError(
                f"❌ sample={sample_idx} input hash 与 cache 不一致。"
                "Prompt/Completion/截断/tokenizer 发生了变化。"
            )
        if int(self.valid_counts[sample_idx]) != int(current_valid_count):
            raise RuntimeError(
                f"❌ sample={sample_idx} valid_count 与 cache 不一致"
            )
        for role in HIER_TOKEN_ROLES:
            if int(self.rels[role][sample_idx]) != int(current_rel[role]):
                raise RuntimeError(
                    f"❌ sample={sample_idx} {role} rel offset 与 cache 不一致"
                )

    def fetch_batch(
        self,
        sample_indices: List[int],
    ):
        kd_chunks = []
        specials = {
            role: [] for role in HIER_TOKEN_ROLES
        }

        for idx in sample_indices:
            start = int(self.offsets[idx])
            end = int(self.offsets[idx + 1])
            arr = np.asarray(self.hidden[start:end])
            kd_chunks.append(arr)

            for role in HIER_TOKEN_ROLES:
                rel = int(self.rels[role][idx])
                specials[role].append(
                    np.asarray(self.hidden[start + rel])
                )

        kd = np.concatenate(kd_chunks, axis=0)
        sp = {
            role: np.stack(values, axis=0)
            for role, values in specials.items()
        }
        return kd, sp


# ======================================================================
# 8. 保存 / 导出 Student
# ======================================================================

def enable_full_input_output_training(model) -> bool:
    model = unwrap_training_model(model)
    inp = model.get_input_embeddings().weight
    out = model.get_output_embeddings()

    if out is None:
        raise RuntimeError("❌ Student 缺少 output embeddings")

    inp.requires_grad_(True)
    out.weight.requires_grad_(True)

    tied = inp.data_ptr() == out.weight.data_ptr()
    print("\n🎯 Student full input/lm_head training")
    print(f"actually tied: {tied}")
    print(f"input requires_grad: {inp.requires_grad}")
    print(f"head requires_grad:  {out.weight.requires_grad}")
    return tied


def get_hidden_projector(model):
    model = unwrap_training_model(model)
    p = getattr(model, "hidden_projector", None)
    if p is not None and not isinstance(p, nn.Module):
        raise TypeError("hidden_projector 不是 nn.Module")
    return p


def save_full_input_output_weights(model, output_dir: str):
    model = unwrap_training_model(model)
    os.makedirs(output_dir, exist_ok=True)

    inp = model.get_input_embeddings().weight
    out = model.get_output_embeddings()
    tied = inp.data_ptr() == out.weight.data_ptr()

    payload = {
        "input_embeddings": inp.detach().cpu(),
        "actually_tied": tied,
    }
    if not tied:
        payload["output_embeddings"] = out.weight.detach().cpu()

    torch.save(
        payload,
        os.path.join(output_dir, FULL_IO_WEIGHTS_NAME),
    )

    with open(
        os.path.join(output_dir, FULL_IO_META_NAME),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "actually_tied": tied,
                "input_shape": list(inp.shape),
                "output_shape": list(out.weight.shape),
                "contains_output_tensor_separately": not tied,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def save_hidden_projector_weights(model, output_dir: str):
    p = get_hidden_projector(model)
    if p is None:
        return

    torch.save(
        {
            "state_dict": {
                k: v.detach().cpu()
                for k, v in p.state_dict().items()
            }
        },
        os.path.join(output_dir, HIDDEN_PROJECTOR_WEIGHTS_NAME),
    )

    with open(
        os.path.join(output_dir, HIDDEN_PROJECTOR_META_NAME),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "type": p.__class__.__name__,
                "in_features": int(p.in_features),
                "out_features": int(p.out_features),
                "bias": p.bias is not None,
                "shared_across_all_four_hierarchy_tokens": True,
                "used_only_for_training_losses": True,
                "needed_for_inference": False,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def save_peft_checkpoint_with_full_io(
    model, tokenizer, output_dir: str, overwrite: bool = True, checkpoint_metadata=None,
) -> None:
    """写完临时目录再替换，并保留上一份模型；包含原格式的 IO/Projector。"""
    raw = unwrap_training_model(model)
    target = output_dir + f".saving-{os.getpid()}-{time.time_ns()}" if overwrite else output_dir
    os.makedirs(target, exist_ok=not overwrite)
    raw.save_pretrained(target)
    save_full_input_output_weights(raw, target)
    save_hidden_projector_weights(raw, target)
    tokenizer.save_pretrained(target)
    if checkpoint_metadata is not None:
        atomic_write_json(os.path.join(target, "best_kd_metadata.json"), checkpoint_metadata)
    if overwrite:
        previous = output_dir + ".previous"
        if os.path.isdir(output_dir):
            if os.path.isdir(previous):
                shutil.rmtree(previous)
            os.replace(output_dir, previous)
        os.replace(target, output_dir)
    print(f"✅ 已保存 PEFT + 完整 IO + 可选 Projector: {output_dir}")


def distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def atomic_write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + f".tmp-{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def read_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def checkpoint_file_sizes(checkpoint: str, world_size: int,
                          projector_required: bool = False, require_kd_state: bool = False) -> dict:
    """当前脚本的普通单卡 / DDP checkpoint；不适用于 DeepSpeed/FSDP。"""
    required = [
        "adapter_config.json", FULL_IO_WEIGHTS_NAME, FULL_IO_META_NAME,
        "trainer_state.json", "optimizer.pt", "scheduler.pt", "training_args.bin",
    ]
    if projector_required:
        required.extend([HIDDEN_PROJECTOR_WEIGHTS_NAME, HIDDEN_PROJECTOR_META_NAME])
    elif os.path.isfile(os.path.join(checkpoint, HIDDEN_PROJECTOR_WEIGHTS_NAME)):
        raise RuntimeError("断点包含 Projector，但当前模式不创建它，请沿用原模式与权重。")
    if require_kd_state:
        required.extend(f"{KD_TRAINER_STATE_PREFIX}{rank}.json" for rank in range(world_size))
    adapters = ["adapter_model.safetensors", "adapter_model.bin"]
    adapter = next(
        (name for name in adapters if os.path.isfile(os.path.join(checkpoint, name))),
        None,
    )
    if adapter is None:
        raise RuntimeError(f"❌ 断点缺少 LoRA 权重: {checkpoint}")
    required.append(adapter)
    rng_names = (["rng_state.pth"] if world_size == 1 else
                 [f"rng_state_{rank}.pth" for rank in range(world_size)])
    actual_rng_names = {
        name for name in os.listdir(checkpoint)
        if re.fullmatch(r"rng_state(?:_\d+)?\.pth", name)
    }
    if actual_rng_names != set(rng_names):
        raise RuntimeError("❌ 断点 RNG 文件与当前进程数不一致，请使用原来的 GPU/进程数。")
    required.extend(rng_names)
    sizes = {}
    for name in required:
        path = os.path.join(checkpoint, name)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise RuntimeError(f"❌ 断点不完整，缺少文件或文件为空: {path}")
        sizes[name] = os.path.getsize(path)
    return sizes


def validate_checkpoint(checkpoint: str, world_size: int, projector_required: bool = False) -> int:
    if os.path.exists(os.path.join(checkpoint, WOS_PENDING_NAME)):
        raise RuntimeError(f"❌ 该断点保存时被中断，不能用于续训: {checkpoint}")
    custom_path = os.path.join(checkpoint, WOS_RESUME_STATE_NAME)
    sizes = checkpoint_file_sizes(checkpoint, world_size, projector_required,
                                  require_kd_state=os.path.isfile(custom_path))
    state = read_json_file(os.path.join(checkpoint, "trainer_state.json"))
    step = int(state["global_step"])
    if step <= 0:
        raise RuntimeError(f"❌ 断点 global_step 无效: {checkpoint}")
    folder_match = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(checkpoint))
    if folder_match is None or int(folder_match.group(1)) != step:
        raise RuntimeError("❌ 续训必须指定与 global_step 一致的周期 checkpoint-N 目录。")
    custom_path = os.path.join(checkpoint, WOS_RESUME_STATE_NAME)
    if os.path.isfile(custom_path):
        custom = read_json_file(custom_path)
        if int(custom["global_step"]) != step:
            raise RuntimeError("❌ Trainer 与 WOS 回调保存步数不一致。")
        if custom.get("checkpoint_files") != sizes:
            raise RuntimeError("❌ checkpoint 文件大小与完成保存时不一致，可能文件不完整。")
    return step


def resolve_resume_checkpoint(args) -> Optional[str]:
    requested = args.resume_from_checkpoint
    root = os.path.abspath(os.path.join(args.save_dir, "checkpoints"))
    candidates = []
    if os.path.isdir(root):
        for name in os.listdir(root):
            match = re.fullmatch(r"checkpoint-(\d+)", name)
            path = os.path.join(root, name)
            if match and os.path.isdir(path):
                candidates.append((int(match.group(1)), path))
    if requested is None or str(requested).strip().lower() in ("", "none", "false"):
        history = os.path.join(args.save_dir, "eval_history.jsonl")
        if candidates or (os.path.isfile(history) and os.path.getsize(history)):
            raise RuntimeError(
                "❌ save_dir 已有实验数据。续训请加 --resume_from_checkpoint auto；"
                "新实验请使用新的 --save_dir，不要清空旧目录。"
            )
        return None
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if str(requested).strip().lower() in ("auto", "true", "latest"):
        for _, path in sorted(candidates, reverse=True):
            try:
                validate_checkpoint(path, world_size, wos_projector_required(args))
                return path
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                print(f"⚠️ 跳过未完成/不完整断点 {path}: {exc}")
        raise RuntimeError(f"❌ 未找到可恢复的完整周期断点: {root}")
    path = os.path.abspath(os.path.expanduser(requested))
    if os.path.normcase(os.path.dirname(path)) != os.path.normcase(root):
        raise RuntimeError(
            "❌ --resume_from_checkpoint 必须位于原实验 --save_dir/checkpoints 下；"
            "请同时传入原实验的 --save_dir。"
        )
    validate_checkpoint(path, world_size, wos_projector_required(args))
    return path


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wos_projector_required(args) -> bool:
    mode = str(args.experiment_mode).strip().lower()
    return ((mode in ("align", "full") and args.align_weight > 0)
            or (mode in ("hier", "full") and args.hier_weight > 0))


def cache_file_fingerprint(path: str) -> dict:
    stat = os.stat(path)
    result = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size <= 64 * 1024 * 1024:
        result.update(sha256=sha256_file(path), hash_policy="full")
    else:
        chunk = 1024 * 1024
        offsets = sorted({0, max(0, stat.st_size // 2 - chunk // 2), max(0, stat.st_size - chunk)})
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for offset in offsets:
                f.seek(offset)
                digest.update(str(offset).encode("ascii"))
                digest.update(f.read(chunk))
        result.update(sha256=digest.hexdigest(), hash_policy="head_middle_tail_1MiB")
    return result


def build_run_signature(args) -> dict:
    fields = (
        "experiment_mode", "seed",
        "max_seq_length", "reserve_tokens", "system_prompt", "num_epochs",
        "batch_size", "accumulation_steps", "lr", "io_lr", "projector_lr",
        "kd_temperature", "kd_weight", "align_weight", "hier_weight", "align_alpha",
        "l1_start_weight", "l1_end_weight", "l2_start_weight", "l2_end_weight",
        "hier_start_weight", "hier_end_weight", "weight_decay", "logging_steps", "lr_scheduler_type", "warmup_ratio",
        "lora_rank", "lora_alpha", "lora_dropout", "eval_steps", "eval_batch_size",
        "loss_eval_batch_size", "max_new_tokens", "bf16", "gradient_checkpointing",
        "num_workers", "pin_memory", "attn_implementation",
    )
    signature = {name: getattr(args, name) for name in fields}
    signature.update(dataset="WOS", training_stage="paired_token_offline_distillation", max_depth=2,
                     hierarchy_policy="four_second_occurrence_align_targets;START_and_END_L2_minus_L1_relation_mse",
                     projector_required=wos_projector_required(args),
                     world_size=int(os.environ.get("WORLD_SIZE", "1")))
    signature["model_path"] = os.path.realpath(args.student_model_path)
    signature["teacher_cache_dir"] = os.path.realpath(args.cache_dir)
    signature["data_sha256"] = {name: sha256_file(getattr(args, name))
                               for name in ("train_data_path", "val_data_path", "taxonomy_path", "dict_path")}
    signature["base_configuration_sha256"] = {
        name: sha256_file(os.path.join(args.student_model_path, name))
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        if os.path.isfile(os.path.join(args.student_model_path, name))
    }
    signature["token_roles"] = dict(ROLE_TO_TOKEN)
    signature["chat_template_sha256"] = hashlib.sha256(CHAT_TEMPLATE.encode("utf-8")).hexdigest()
    manifest_path = os.path.join(args.cache_dir, CACHE_MANIFEST)
    names = {
        "hidden": CACHE_HIDDEN, "offsets": CACHE_OFFSETS, "valid_counts": CACHE_VALID_COUNTS,
        "hashes": CACHE_HASHES, "common_head": CACHE_COMMON_HEAD,
        **{role: filename for role, filename in CACHE_REL_FILES.items()},
    }
    signature["teacher_cache_fingerprint"] = {
        "manifest_sha256": sha256_file(manifest_path),
        "files": {key: cache_file_fingerprint(os.path.join(args.cache_dir, name))
                  for key, name in sorted(names.items())},
    }
    signature["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                             "peft": peft.__version__}
    return signature


def validate_resume_configuration(checkpoint: str, signature: dict) -> None:
    path = os.path.join(checkpoint, WOS_RESUME_STATE_NAME)
    if os.path.isfile(path):
        saved = read_json_file(path)["run_signature"]
        changed = [key for key in sorted(set(saved) | set(signature)) if saved.get(key) != signature.get(key)]
        if changed:
            raise RuntimeError("WOS 配对蒸馏续训配置与断点不一致: " + ", ".join(changed))
        return
    experiment = os.path.dirname(os.path.dirname(checkpoint))
    config_path = os.path.join(experiment, "distill_config.json")
    if not os.path.isfile(config_path):
        raise RuntimeError("旧配对蒸馏实验缺少 distill_config.json，无法核对蒸馏模式和权重。")
    config = read_json_file(config_path)
    effective_align = signature["align_weight"] if signature["experiment_mode"] in ("align", "full") else 0.0
    effective_hier = signature["hier_weight"] if signature["experiment_mode"] in ("hier", "full") else 0.0
    expected = {
        "experiment_mode": signature["experiment_mode"], "student_init": "raw_base_only_no_student_sft",
        "kd_temperature": signature["kd_temperature"], "kd_weight": signature["kd_weight"],
        "align_weight_effective": effective_align, "hier_weight_effective": effective_hier,
        "align_alpha": signature["align_alpha"],
        "four_align_targets": [f"{role}_second_occurrence" for role in HIER_TOKEN_ROLES],
        "special_weights": {role: signature[role + "_weight"] for role in HIER_TOKEN_ROLES},
        "hier_relations": {"start_relation": "L1_START -> L2_START", "end_relation": "L1_END -> L2_END"},
        "relation_weights": {"start_relation": signature["hier_start_weight"], "end_relation": signature["hier_end_weight"]},
    }
    changed = [key for key, value in expected.items() if config.get(key) != value]
    for key, current in (("student_model_path", "model_path"), ("teacher_cache_dir", "teacher_cache_dir")):
        if not config.get(key) or os.path.realpath(config[key]) != signature[current]:
            changed.append(key)
    try:
        saved_args = torch.load(os.path.join(checkpoint, "training_args.bin"), map_location="cpu", weights_only=False)
    except TypeError:
        saved_args = torch.load(os.path.join(checkpoint, "training_args.bin"), map_location="cpu")
    mapping = {"seed": "seed", "num_epochs": "num_train_epochs", "batch_size": "per_device_train_batch_size",
               "accumulation_steps": "gradient_accumulation_steps", "lr": "learning_rate",
               "weight_decay": "weight_decay", "lr_scheduler_type": "lr_scheduler_type", "warmup_ratio": "warmup_ratio",
               "bf16": "bf16", "gradient_checkpointing": "gradient_checkpointing", "eval_steps": "save_steps",
               "num_workers": "dataloader_num_workers", "pin_memory": "dataloader_pin_memory",
               "logging_steps": "logging_steps"}
    for key, saved_key in mapping.items():
        value = getattr(saved_args, saved_key, None)
        if getattr(value, "value", value) != signature[key]:
            changed.append(key)
    adapter = read_json_file(os.path.join(checkpoint, "adapter_config.json"))
    for key, saved_key in (("lora_rank", "r"), ("lora_alpha", "lora_alpha"), ("lora_dropout", "lora_dropout")):
        if adapter.get(saved_key) != signature[key]:
            changed.append(key)
    if changed:
        raise RuntimeError("旧 WOS 配对蒸馏断点配置不一致: " + ", ".join(changed))
    print("⚠️ 兼容旧配对蒸馏断点：已核对 distill_config、原生参数和 LoRA。旧版未记录数据/缓存内容指纹、io_lr、projector_lr、环境版本；请沿用原实验设置。")


def load_full_input_output_weights_for_resume(model, checkpoint: str) -> None:
    raw = unwrap_training_model(model)
    payload = _torch_load_cpu(os.path.join(checkpoint, FULL_IO_WEIGHTS_NAME))
    input_weight = raw.get_input_embeddings().weight
    output_weight = raw.get_output_embeddings().weight
    tied = input_weight.data_ptr() == output_weight.data_ptr()
    if bool(payload["actually_tied"]) != tied:
        raise RuntimeError("❌ 基础模型的 input/lm_head 共享关系与断点不一致。")
    tensors = [(input_weight, payload["input_embeddings"], "input embedding")]
    if not tied:
        if "output_embeddings" not in payload:
            raise RuntimeError("❌ 断点缺少完整 lm_head 权重。")
        tensors.append((output_weight, payload["output_embeddings"], "lm_head"))
    for parameter, saved, name in tensors:
        if tuple(parameter.shape) != tuple(saved.shape):
            raise RuntimeError(f"❌ {name} 形状与断点不一致。")
    with torch.no_grad():
        for parameter, saved, _ in tensors:
            # copy_ 支持跨设备与 dtype 转换，避免先创建整张 GPU 临时矩阵。
            parameter.copy_(saved)
    del payload, tensors
    print(f"✅ 已恢复完整 input embedding / lm_head: {checkpoint}")


def read_eval_history_for_resume(path: str, resume_step: int) -> List[dict]:
    if not os.path.isfile(path):
        raise RuntimeError("❌ 旧断点缺少 eval_history.jsonl，无法恢复验证与历史最佳状态。")
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    records = []
    previous_step = -1
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                print("⚠️ 检测到中断留下的不完整历史末行，恢复时将备份并移除。")
                break
            raise RuntimeError(f"❌ 验证历史第 {index + 1} 条损坏。")
        step = int(record["step"])
        if step <= previous_step:
            raise RuntimeError("❌ 验证历史 step 不严格递增，无法安全恢复。")
        previous_step = step
        if step <= resume_step:
            records.append(record)
    return records


def write_history_snapshot(path: str, records: List[dict]) -> None:
    content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    old = None
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            old = f.read()
    if old == content:
        return
    if old:
        backup = path + f".before-resume-{time.time_ns()}.bak"
        shutil.copy2(path, backup)
        print(f"✅ 已备份原验证历史: {backup}")
    temporary = path + f".tmp-{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def select_best_record(records: List[dict]) -> Optional[dict]:
    if not records:
        return None
    selected = records[0]
    for record in records[1:]:
        ma, mi = float(record["global_macro"]), float(record["global_micro"])
        best_ma, best_mi = float(selected["global_macro"]), float(selected["global_micro"])
        if ma > best_ma or (math.isclose(ma, best_ma, abs_tol=1e-12) and mi > best_mi):
            selected = record
    return selected


def archive_future_checkpoints(save_dir: str, resume_step: int) -> None:
    root = os.path.join(save_dir, "checkpoints")
    if not os.path.isdir(root):
        return
    destination = os.path.join(save_dir, "resume_backups", str(time.time_ns()), "checkpoints")
    for name in os.listdir(root):
        match = re.fullmatch(r"checkpoint-(\d+)", name)
        source = os.path.join(root, name)
        if match and int(match.group(1)) > resume_step and os.path.isdir(source):
            os.makedirs(destination, exist_ok=True)
            shutil.move(source, os.path.join(destination, name))
            print(f"✅ 已备份恢复步数之后的旧断点: {name} -> {destination}")


def ensure_wos_historical_best_available(save_dir: str, records: List[dict],
                                         projector_required: bool, allow_legacy_best: bool) -> None:
    best = select_best_record(records)
    if best is None:
        return
    online = os.path.join(save_dir, "best_kd_checkpoint")
    required = [FULL_IO_WEIGHTS_NAME, FULL_IO_META_NAME, "adapter_config.json"]
    if projector_required:
        required.extend([HIDDEN_PROJECTOR_WEIGHTS_NAME, HIDDEN_PROJECTOR_META_NAME])

    def complete_model(path):
        return (all(os.path.isfile(os.path.join(path, name)) and os.path.getsize(os.path.join(path, name)) > 0
                    for name in required)
                and any(os.path.isfile(os.path.join(path, name)) for name in ("adapter_model.safetensors", "adapter_model.bin"))
                and not os.path.exists(os.path.join(path, WOS_PENDING_NAME)))

    metrics_path = os.path.join(save_dir, "best_metrics.json")
    source = None
    for candidate in (online, online + ".previous"):
        metadata_path = os.path.join(candidate, "best_kd_metadata.json")
        if complete_model(candidate) and os.path.isfile(metadata_path):
            meta = read_json_file(metadata_path)
            if all(meta.get(key) == best[key] for key in ("step", "global_macro", "global_micro")):
                if candidate == online:
                    atomic_write_json(metrics_path, best)
                    return
                source = candidate
                break
    if source is None:
        periodic = os.path.join(save_dir, "checkpoints", f"checkpoint-{int(best['step'])}")
        if complete_model(periodic):
            source = periodic
        elif (allow_legacy_best and complete_model(online)
              and not os.path.isfile(os.path.join(online, "best_kd_metadata.json"))
              and os.path.isfile(metrics_path)
              and all(read_json_file(metrics_path).get(key) == best[key]
                      for key in ("step", "global_macro", "global_micro"))):
            atomic_write_json(os.path.join(online, "best_kd_metadata.json"), best)
            atomic_write_json(metrics_path, best)
            print("✅ 已核对旧 best_metrics.json 与配对蒸馏历史最佳模型。")
            return
        else:
            raise RuntimeError(
                f"无法取得恢复历史对应的最佳模型 step={best['step']}。"
                "请恢复最近完整断点，或还原对应历史最佳模型备份。"
            )
    temporary = online + f".restore-{time.time_ns()}"
    shutil.copytree(source, temporary)
    atomic_write_json(os.path.join(temporary, "best_kd_metadata.json"), best)
    if os.path.isdir(online):
        os.replace(online, online + f".before-resume-{time.time_ns()}")
    os.replace(temporary, online)

    atomic_write_json(metrics_path, best)


def load_projector_weights_for_resume(model, checkpoint: str) -> None:
    projector = getattr(unwrap_training_model(model), "hidden_projector", None)
    path = os.path.join(checkpoint, HIDDEN_PROJECTOR_WEIGHTS_NAME)
    if projector is None:
        if os.path.isfile(path):
            raise RuntimeError("断点含 Projector，但当前实验未创建它，不能完整续训。")
        return
    if not os.path.isfile(path):
        raise RuntimeError("断点缺少 hidden_state_projector.pt，无法恢复 Align/Hier 蒸馏。")
    payload = _torch_load_cpu(path)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError("Projector 断点格式不正确。")
    current = projector.state_dict()
    saved = payload["state_dict"]
    if set(current) != set(saved):
        raise RuntimeError("Projector 参数名称与断点不一致。")
    for key in current:
        if tuple(current[key].shape) != tuple(saved[key].shape):
            raise RuntimeError(f"Projector {key} 形状与断点不一致。")
    # load_state_dict 默认原位复制，不重新创建 Linear 或 Parameter。
    projector.load_state_dict(saved, strict=True)
    print(f"✅ 已恢复WOS 四个 START/END token 共享 Projector: {checkpoint}")




def export_merged_full_model(
    base_model_path: str,
    peft_checkpoint_dir: str,
    output_dir: str,
    tokenizer,
    torch_dtype,
):
    weight_path = os.path.join(
        peft_checkpoint_dir,
        FULL_IO_WEIGHTS_NAME,
    )
    payload = _torch_load_cpu(weight_path)

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map="cpu",
    )

    inp = base.get_input_embeddings().weight
    out = base.get_output_embeddings()

    saved_inp = payload["input_embeddings"]
    tied = bool(payload["actually_tied"])

    if tuple(inp.shape) != tuple(saved_inp.shape):
        raise RuntimeError("❌ 导出时 input embedding shape 不一致")

    with torch.no_grad():
        inp.copy_(saved_inp.to(dtype=inp.dtype))

    if tied:
        if inp.data_ptr() != out.weight.data_ptr():
            base.config.tie_word_embeddings = True
            base.tie_weights()
    else:
        saved_out = payload["output_embeddings"]
        if inp.data_ptr() == out.weight.data_ptr():
            new_head = nn.Linear(
                out.in_features,
                out.out_features,
                bias=(out.bias is not None),
                dtype=out.weight.dtype,
            )
            base.set_output_embeddings(new_head)
            out = base.get_output_embeddings()
        with torch.no_grad():
            out.weight.copy_(saved_out.to(dtype=out.weight.dtype))
        base.config.tie_word_embeddings = False

    peft_model = PeftModel.from_pretrained(
        base,
        peft_checkpoint_dir,
        is_trainable=False,
    )
    merged = peft_model.merge_and_unload()
    merged.config.use_cache = True

    _reset_directory(output_dir)
    merged.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(output_dir)
    print(f"✅ 导出 merged KD model: {output_dir}")


# ======================================================================
# 9. Trainer
# ======================================================================

class FullIOKDTrainer(Trainer):
    """
    与旧单 Token WOS 蒸馏保持一致的分组优化器：

      - LoRA 参数：args.learning_rate
      - 完整 input embedding / lm_head：io_learning_rate
      - shared hidden projector：projector_learning_rate

    同时严格区分 decay / no_decay 参数组。
    """

    def __init__(
        self,
        *args,
        io_learning_rate: float,
        projector_learning_rate: Optional[float] = None,
        **kwargs,
    ):
        if io_learning_rate <= 0:
            raise ValueError(
                f"❌ io_learning_rate 必须 > 0，实际={io_learning_rate}"
            )
        if (
            projector_learning_rate is not None
            and projector_learning_rate <= 0
        ):
            raise ValueError(
                "❌ projector_learning_rate 必须 > 0，"
                f"实际={projector_learning_rate}"
            )

        self.io_learning_rate = float(io_learning_rate)
        self.projector_learning_rate = (
            None
            if projector_learning_rate is None
            else float(projector_learning_rate)
        )
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        raw_model = unwrap_training_model(opt_model)

        input_module = raw_model.get_input_embeddings()
        output_module = raw_model.get_output_embeddings()
        if input_module is None or output_module is None:
            raise RuntimeError(
                "❌ 创建优化器时找不到 input embedding 或 lm_head。"
            )

        io_param_ids = {
            id(input_module.weight),
            id(output_module.weight),
        }

        projector = get_hidden_projector(raw_model)
        projector_param_ids = set()
        if projector is not None:
            projector_param_ids = {
                id(p)
                for p in projector.parameters()
                if p.requires_grad
            }
            if not projector_param_ids:
                raise RuntimeError(
                    "❌ hidden_projector 存在但没有可训练参数。"
                )

        decay_parameter_names = set(
            self.get_decay_parameter_names(opt_model)
        )

        grouped = {
            "lora_decay": [],
            "lora_no_decay": [],
            "io_decay": [],
            "io_no_decay": [],
            "projector_decay": [],
            "projector_no_decay": [],
        }
        seen_param_ids = set()

        for name, parameter in opt_model.named_parameters():
            if not parameter.requires_grad:
                continue

            pid = id(parameter)
            if pid in seen_param_ids:
                continue
            seen_param_ids.add(pid)

            is_io = pid in io_param_ids
            is_projector = pid in projector_param_ids
            if is_io and is_projector:
                raise RuntimeError(
                    "❌ 参数同时被识别为 IO 与 Projector。"
                )

            use_decay = name in decay_parameter_names

            if is_io:
                key = "io_decay" if use_decay else "io_no_decay"
            elif is_projector:
                key = (
                    "projector_decay"
                    if use_decay
                    else "projector_no_decay"
                )
            else:
                key = (
                    "lora_decay"
                    if use_decay
                    else "lora_no_decay"
                )

            grouped[key].append(parameter)

        io_count = sum(
            p.numel()
            for key in ("io_decay", "io_no_decay")
            for p in grouped[key]
        )
        projector_count = sum(
            p.numel()
            for key in (
                "projector_decay",
                "projector_no_decay",
            )
            for p in grouped[key]
        )
        lora_count = sum(
            p.numel()
            for key in ("lora_decay", "lora_no_decay")
            for p in grouped[key]
        )

        if io_count == 0:
            raise RuntimeError(
                "❌ 参数分组失败：没有找到可训练 input embedding/lm_head。"
            )
        if lora_count == 0:
            raise RuntimeError(
                "❌ 参数分组失败：没有找到可训练 LoRA 参数。"
            )
        if projector is not None and projector_count == 0:
            raise RuntimeError(
                "❌ 参数分组失败：没有找到 Projector 参数。"
            )

        projector_lr = (
            self.projector_learning_rate
            if self.projector_learning_rate is not None
            else self.args.learning_rate
        )

        optimizer_grouped_parameters = []

        def add_group(
            key: str,
            learning_rate: float,
            weight_decay: float,
        ):
            if grouped[key]:
                optimizer_grouped_parameters.append(
                    {
                        "params": grouped[key],
                        "lr": learning_rate,
                        "weight_decay": weight_decay,
                        "group_name": key,
                    }
                )

        add_group(
            "lora_decay",
            self.args.learning_rate,
            self.args.weight_decay,
        )
        add_group(
            "lora_no_decay",
            self.args.learning_rate,
            0.0,
        )
        add_group(
            "io_decay",
            self.io_learning_rate,
            self.args.weight_decay,
        )
        add_group(
            "io_no_decay",
            self.io_learning_rate,
            0.0,
        )
        add_group(
            "projector_decay",
            projector_lr,
            self.args.weight_decay,
        )
        add_group(
            "projector_no_decay",
            projector_lr,
            0.0,
        )

        try:
            optimizer_cls, optimizer_kwargs = (
                Trainer.get_optimizer_cls_and_kwargs(
                    self.args,
                    opt_model,
                )
            )
        except TypeError:
            optimizer_cls, optimizer_kwargs = (
                Trainer.get_optimizer_cls_and_kwargs(
                    self.args
                )
            )

        optimizer_kwargs = dict(optimizer_kwargs)
        optimizer_kwargs.pop("params", None)
        optimizer_kwargs["lr"] = self.args.learning_rate

        self.optimizer = optimizer_cls(
            optimizer_grouped_parameters,
            **optimizer_kwargs,
        )

        tied = (
            input_module.weight.data_ptr()
            == output_module.weight.data_ptr()
        )

        print("\n" + "=" * 84)
        print("🎛️ 分组学习率优化器创建完成")
        print(
            f"LoRA 初始 LR: {self.args.learning_rate:.8g}"
        )
        print(
            f"Embedding/lm_head 初始 LR: "
            f"{self.io_learning_rate:.8g}"
        )
        if projector is not None:
            print(
                f"Shared Projector 初始 LR: "
                f"{projector_lr:.8g}"
            )
        print(f"LoRA 可训练参数量: {lora_count:,}")
        print(f"IO 可训练参数量: {io_count:,}")
        if projector is not None:
            print(
                f"Projector 可训练参数量: "
                f"{projector_count:,}"
            )
        print(f"input/lm_head 实际共享: {tied}")
        print(
            "✅ 与旧单 Token WOS 蒸馏使用相同的 "
            "decay/no_decay 分组逻辑"
        )
        print("=" * 84 + "\n")

        return self.optimizer

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        target = self.model if model is None else model
        load_full_input_output_weights_for_resume(target, resume_from_checkpoint)
        load_projector_weights_for_resume(target, resume_from_checkpoint)
        path = os.path.join(resume_from_checkpoint,
                            f"{KD_TRAINER_STATE_PREFIX}{self.args.process_index}.json")
        if os.path.isfile(path):
            payload = read_json_file(path)
            expected = int(read_json_file(os.path.join(resume_from_checkpoint, "trainer_state.json"))["global_step"])
            if int(payload["global_step"]) != expected:
                raise RuntimeError("WOS 损失日志状态与断点步数不一致。")
            meters = payload["meters"]
            count = int(payload["meter_n"])
            if not isinstance(meters, dict) or count < 0:
                raise RuntimeError("配对蒸馏损失日志状态格式错误。")
            self._meters = dict(meters)
            self._meter_n = count
        else:
            print("⚠️ 旧断点没有损失日志累计状态，将从零重新累计训练日志；已恢复模型权重。")

    def _save_checkpoint(self, model, trial, *args, **kwargs):
        output_dir = os.path.join(self._get_output_dir(trial), f"checkpoint-{self.state.global_step}")
        if self.is_world_process_zero():
            os.makedirs(output_dir, exist_ok=True)
            atomic_write_json(os.path.join(output_dir, WOS_PENDING_NAME),
                              {"global_step": int(self.state.global_step)})
        distributed_barrier()
        result = super()._save_checkpoint(model, trial, *args, **kwargs)
        atomic_write_json(
            os.path.join(output_dir, f"{KD_TRAINER_STATE_PREFIX}{self.args.process_index}.json"),
            {"global_step": int(self.state.global_step),
             "meters": dict(self._meters), "meter_n": int(self._meter_n)},
        )
        return result

    def _save(
        self,
        output_dir=None,
        state_dict=None,
    ):
        super()._save(
            output_dir=output_dir,
            state_dict=state_dict,
        )
        if output_dir is None:
            output_dir = self.args.output_dir
        if self.is_world_process_zero():
            save_full_input_output_weights(
                self.model,
                output_dir,
            )
            save_hidden_projector_weights(
                self.model,
                output_dir,
            )


class OfflinePairedTokenKDTrainer(FullIOKDTrainer):
    def __init__(
        self,
        *args,
        teacher_cache: OfflineTeacherCache,
        dataset_for_verification: PairedTokenWOSDataset,
        kd_temperature: float,
        kd_weight: float,
        align_weight: float,
        hier_weight: float,
        align_alpha: float,
        special_weights: Dict[str, float],
        relation_weights: Dict[str, float],
        **kwargs,
    ):
        self.teacher_cache = teacher_cache
        self.verify_dataset = dataset_for_verification

        self.kd_temperature = float(kd_temperature)
        self.kd_weight = float(kd_weight)
        self.align_weight = float(align_weight)
        self.hier_weight = float(hier_weight)
        self.align_alpha = float(align_alpha)

        self.special_weights = {
            k: float(v) for k, v in special_weights.items()
        }
        self.relation_weights = {
            k: float(v) for k, v in relation_weights.items()
        }

        if self.kd_temperature <= 0:
            raise ValueError("kd_temperature 必须 > 0")
        if min(
            self.kd_weight,
            self.align_weight,
            self.hier_weight,
        ) < 0:
            raise ValueError("loss weight 不能为负")
        if not 0 <= self.align_alpha <= 1:
            raise ValueError("align_alpha 必须在 [0,1]")

        for name, value in self.special_weights.items():
            if value < 0:
                raise ValueError(
                    f"special weight 不能为负: {name}={value}"
                )
        for name, value in self.relation_weights.items():
            if value < 0:
                raise ValueError(
                    f"relation weight 不能为负: {name}={value}"
                )

        self._meters = {}
        self._meter_n = 0
        self._printed_sanity = False

        super().__init__(*args, **kwargs)

        payload = teacher_cache.head_payload
        self.student_common_ids = payload[
            "student_common_ids"
        ].long().to(self.args.device)

        # 与旧单 Token WOS offline KD 保持一致：
        # 磁盘 cache 可以是 float16，但训练恢复 Teacher logits 时，
        # bf16 训练统一转 bf16；否则转 float32。
        target_dtype = (
            torch.bfloat16
            if self.args.bf16
            else torch.float32
        )

        self.teacher_common_lm_head = payload[
            "weight"
        ].to(
            device=self.args.device,
            dtype=target_dtype,
        )
        self.teacher_common_lm_head.requires_grad_(False)

        self.teacher_common_lm_head_bias = payload.get("bias")
        if self.teacher_common_lm_head_bias is not None:
            self.teacher_common_lm_head_bias = (
                self.teacher_common_lm_head_bias.to(
                    device=self.args.device,
                    dtype=target_dtype,
                )
            )
            self.teacher_common_lm_head_bias.requires_grad_(False)

    @staticmethod
    def _temperature_kl(student_logits, teacher_logits, temperature):
        s = F.log_softmax(
            student_logits.float() / temperature,
            dim=-1,
        )
        t = F.softmax(
            teacher_logits.float() / temperature,
            dim=-1,
        )
        return F.kl_div(
            s,
            t,
            reduction="batchmean",
        ) * (temperature ** 2)

    def _align_one(self, projected_student, teacher_hidden):
        teacher_hidden = teacher_hidden.float()
        projected_student = projected_student.float()

        mse = F.mse_loss(
            projected_student,
            teacher_hidden,
            reduction="mean",
        )
        cos = (
            1.0
            - F.cosine_similarity(
                projected_student,
                teacher_hidden,
                dim=-1,
                eps=1e-8,
            )
        ).mean()
        loss = (
            self.align_alpha * mse
            + (1.0 - self.align_alpha) * cos
        )
        return loss, mse, cos

    def _acc(self, name, value):
        self._meters[name] = self._meters.get(name, 0.0) + float(
            value.detach().float().item()
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        del num_items_in_batch

        # ----------------------------------------------------------
        # A. 取出 metadata，避免传进 CausalLM.forward
        # ----------------------------------------------------------
        sample_index = inputs.pop("sample_index")
        pos = {
            role: inputs.pop(f"{role}_pos").to(self.args.device)
            for role in HIER_TOKEN_ROLES
        }

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        # ----------------------------------------------------------
        # B. 只在需要 Align/Hier 时抓 Student final hidden
        # ----------------------------------------------------------
        need_hidden = self.align_weight > 0 or self.hier_weight > 0
        captured = {}
        handle = None

        if need_hidden:
            final_norm = get_final_norm_module(model)

            def hook(_module, _inputs, output):
                captured["hidden"] = output

            handle = final_norm.register_forward_hook(hook)

        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
        finally:
            if handle is not None:
                handle.remove()

        task_loss = outputs.loss
        if task_loss is None:
            raise RuntimeError("❌ Student 没有 task loss")

        shift_labels = labels[:, 1:]
        valid_mask = shift_labels.ne(-100)
        valid_count = int(valid_mask.sum().item())
        if valid_count <= 0:
            raise RuntimeError("❌ 当前 batch 无 Completion KD 位置")

        row_ids = torch.arange(
            input_ids.size(0),
            device=input_ids.device,
        )

        for role in HIER_TOKEN_ROLES:
            if not valid_mask[row_ids, pos[role]].all():
                raise RuntimeError(
                    f"❌ {role} 第二次 occurrence 不在 causal KD mask"
                )

        # ----------------------------------------------------------
        # C. 每个 batch 严格验证 cache 对齐
        # ----------------------------------------------------------
        sample_indices = [
            int(x) for x in sample_index.detach().cpu().tolist()
        ]
        for row, idx in enumerate(sample_indices):
            actual_len = int(attention_mask[row].sum().item())
            current_ids = input_ids[
                row, :actual_len
            ].detach().cpu().tolist()
            current_hash = _token_ids_sha256(current_ids)

            row_valid_positions = torch.nonzero(
                valid_mask[row, :max(0, actual_len - 1)],
                as_tuple=False,
            ).flatten().detach().cpu().tolist()
            pos_to_rel = {
                int(p): r for r, p in enumerate(row_valid_positions)
            }
            current_rel = {
                role: int(pos_to_rel[int(pos[role][row].item())])
                for role in HIER_TOKEN_ROLES
            }

            self.teacher_cache.verify_sample(
                idx,
                current_hash,
                len(row_valid_positions),
                current_rel,
            )

        # ----------------------------------------------------------
        # D. Full Completion Logits KD
        # ----------------------------------------------------------
        student_shift_logits = outputs.logits[:, :-1, :]
        student_selected = student_shift_logits[valid_mask]
        student_common_logits = student_selected.index_select(
            dim=-1,
            index=self.student_common_ids,
        )

        cached_kd_np, cached_special_np = (
            self.teacher_cache.fetch_batch(sample_indices)
        )

        teacher_dtype = self.teacher_common_lm_head.dtype
        cached_kd_hidden = torch.from_numpy(
            cached_kd_np.copy()
        ).to(
            device=self.args.device,
            dtype=teacher_dtype,
        )

        teacher_common_logits = F.linear(
            cached_kd_hidden,
            self.teacher_common_lm_head,
            self.teacher_common_lm_head_bias,
        )

        if student_common_logits.shape != teacher_common_logits.shape:
            raise RuntimeError(
                "❌ Student / Teacher common logits shape 不一致："
                f"S={tuple(student_common_logits.shape)}, "
                f"T={tuple(teacher_common_logits.shape)}"
            )

        full_kd_loss = self._temperature_kl(
            student_common_logits,
            teacher_common_logits,
            self.kd_temperature,
        )

        zero = task_loss.new_zeros(())
        align_loss = zero
        hier_loss = zero

        # ----------------------------------------------------------
        # E. 4-token Hidden Align
        # ----------------------------------------------------------
        align_parts = {}
        projected = {}
        teacher_special = {}

        if need_hidden:
            if "hidden" not in captured:
                raise RuntimeError("❌ 未抓到 Student final hidden")
            student_hidden = captured["hidden"]

            projector = get_hidden_projector(model)
            if projector is None:
                raise RuntimeError("❌ 需要 hidden loss 但 projector 不存在")

            for role in HIER_TOKEN_ROLES:
                hs = student_hidden[
                    row_ids,
                    pos[role],
                    :,
                ]
                projected[role] = projector(hs.float())

                teacher_special[role] = torch.from_numpy(
                    cached_special_np[role].copy()
                ).to(
                    device=self.args.device,
                    dtype=torch.float32,
                )

        if self.align_weight > 0:
            numerator = zero
            denom = 0.0

            for role in HIER_TOKEN_ROLES:
                loss_r, mse_r, cos_r = self._align_one(
                    projected[role],
                    teacher_special[role],
                )
                align_parts[role] = (loss_r, mse_r, cos_r)

                w = self.special_weights[role]
                numerator = numerator + w * loss_r
                denom += w

            if denom <= 0:
                raise RuntimeError("❌ 4 个 special align weights 不能全为 0")
            align_loss = numerator / denom

        # ----------------------------------------------------------
        # F. 双关系 Hidden Hier
        #
        # 保持旧单 Token 的 L1 -> L2 差分定义：
        #   START: L2_START - L1_START
        #   END  : L2_END   - L1_END
        #
        # 两条都是跨层 L1 -> L2 relation，不混入层内 START->END 变化。
        # ----------------------------------------------------------
        relation_losses = {}
        if self.hier_weight > 0:
            relations = {
                "start_relation": (
                    "l1_start",
                    "l2_start",
                ),
                "end_relation": (
                    "l1_end",
                    "l2_end",
                ),
            }

            numerator = zero
            denom = 0.0

            for relation_name, (src, dst) in relations.items():
                delta_s = (
                    projected[dst]
                    - projected[src]
                )
                delta_t = (
                    teacher_special[dst]
                    - teacher_special[src]
                )

                relation_loss = F.mse_loss(
                    delta_s.float(),
                    delta_t.float(),
                    reduction="mean",
                )
                relation_losses[
                    relation_name
                ] = relation_loss

                w = self.relation_weights[
                    relation_name
                ]
                numerator = (
                    numerator
                    + w * relation_loss
                )
                denom += w

            if denom <= 0:
                raise RuntimeError(
                    "❌ START/END 两个 hierarchy "
                    "relation weights 不能全为 0"
                )

            hier_loss = numerator / denom

        total_loss = (
            task_loss
            + self.kd_weight * full_kd_loss
            + self.align_weight * align_loss
            + self.hier_weight * hier_loss
        )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "❌ total loss 出现 NaN/Inf："
                f"task={task_loss.detach().float().item()}, "
                f"kd={full_kd_loss.detach().float().item()}, "
                f"align={align_loss.detach().float().item()}, "
                f"hier={hier_loss.detach().float().item()}"
            )

        # ----------------------------------------------------------
        # G. 首 batch sanity
        # ----------------------------------------------------------
        if not self._printed_sanity:
            print("\n" + "=" * 96)
            print("🧪 Paired-token KD 首 batch sanity")
            print(f"Batch={input_ids.size(0)}")
            print(f"Full KD positions={valid_count}")
            for role in HIER_TOKEN_ROLES:
                print(
                    f"{role:8s} positions: "
                    f"{pos[role].detach().cpu().tolist()}"
                )
            print(
                "✅ Distill targets = 4 个 Completion 第二次 occurrence"
            )
            print(
                "✅ Hier relations = "
                "L1_START→L2_START | L1_END→L2_END"
            )
            print(
                f"Loss weights: task=1, kd={self.kd_weight}, "
                f"align={self.align_weight}, hier={self.hier_weight}"
            )

            if self.align_weight > 0:
                print(
                    "📈 Align 分项监控: "
                    "L1_START / L1_END / L2_START / L2_END，"
                    "每个均记录 Total + MSE + CosLoss"
                )

            if self.hier_weight > 0:
                print(
                    "📈 Hier 分项监控: "
                    "START relation(L1_START→L2_START) + "
                    "END relation(L1_END→L2_END)"
                )

            print("=" * 96 + "\n")
            self._printed_sanity = True

        self._acc("task", task_loss)
        self._acc("kd", full_kd_loss)
        self._acc("align", align_loss)
        self._acc("hier", hier_loss)
        self._acc("total", total_loss)

        for role, vals in align_parts.items():
            self._acc(f"align_{role}", vals[0])
            self._acc(f"align_{role}_mse", vals[1])
            self._acc(f"align_{role}_cos", vals[2])
        for name, v in relation_losses.items():
            self._acc(f"hier_{name}", v)

        self._meter_n += 1

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def log(self, logs, *args, **kwargs):
        """
        这些 key 会由 Hugging Face 的 SwanLabCallback 自动上传。

        Align 模式 / Full 模式会额外出现：
          kd/align_l1_start
          kd/align_l1_start_mse
          kd/align_l1_start_cos

          kd/align_l1_end
          kd/align_l1_end_mse
          kd/align_l1_end_cos

          kd/align_l2_start
          kd/align_l2_start_mse
          kd/align_l2_start_cos

          kd/align_l2_end
          kd/align_l2_end_mse
          kd/align_l2_end_cos

        Hier 模式 / Full 模式会额外出现：
          kd/hier_start_relation
          kd/hier_end_relation

        另外始终记录：
          kd/task
          kd/kd
          kd/align
          kd/hier
          kd/total
        """
        if self._meter_n > 0:
            n = self._meter_n
            for k, v in self._meters.items():
                logs[f"kd/{k}"] = v / n
            self._meters = {}
            self._meter_n = 0

        return super().log(
            logs,
            *args,
            **kwargs,
        )


# ======================================================================
# 10. 验证与 best 保存
# ======================================================================

def compute_metric_bundle(
    true_l1,
    true_l2,
    pred_l1,
    pred_l2,
    num_l1,
    num_l2,
):
    all_true = []
    all_pred = []

    for t1, t2, p1, p2 in zip(
        true_l1, true_l2, pred_l1, pred_l2
    ):
        all_true.extend([t1, t2 + num_l1])
        all_pred.extend([
            p1,
            p2 + num_l1 if p2 != 9999 else 99999,
        ])

    return {
        "global_micro": float(f1_score(
            all_true, all_pred,
            average="micro",
            zero_division=0,
        )),
        "global_macro": float(f1_score(
            all_true, all_pred,
            labels=list(range(num_l1 + num_l2)),
            average="macro",
            zero_division=0,
        )),
        "l1_micro": float(f1_score(
            true_l1, pred_l1,
            average="micro",
            zero_division=0,
        )),
        "l1_macro": float(f1_score(
            true_l1, pred_l1,
            labels=list(range(num_l1)),
            average="macro",
            zero_division=0,
        )),
        "l2_micro": float(f1_score(
            true_l2, pred_l2,
            average="micro",
            zero_division=0,
        )),
        "l2_macro": float(f1_score(
            true_l2, pred_l2,
            labels=list(range(num_l2)),
            average="macro",
            zero_division=0,
        )),
    }


class GenerativeF1EvalCallback(TrainerCallback):
    """
    与旧单 Token WOS 蒸馏保持同一验证口径，同时适配双 Token 输出：

      - teacher-forcing Eval Loss
      - Global Micro / Macro F1
      - L1 Micro / Macro F1
      - L2 Micro / Macro F1
      - Exact Parse Rate
      - Valid Label Pair Rate
      - Hierarchy Consistency Rate
      - eval_history.jsonl
      - Global Macro 优先，Micro 作为 tie-break 保存 best
    """

    def __init__(
        self,
        val_data,
        l1_map,
        l2_map,
        l2_to_l1,
        tokenizer,
        max_seq_length,
        reserve_tokens,
        system_prompt,
        eval_steps,
        eval_batch_size,
        loss_eval_batch_size,
        max_new_tokens,
        save_dir,
        use_swanlab,
        resume_checkpoint: Optional[str] = None,
        run_signature: Optional[dict] = None,
    ):
        self.val_data = val_data
        self.l1_map = l1_map
        self.l2_map = l2_map
        self.l2_to_l1 = l2_to_l1
        self.tokenizer = tokenizer
        self.max_seq_length = int(max_seq_length)
        self.reserve_tokens = int(reserve_tokens)
        self.system_prompt = system_prompt
        self.eval_steps = int(eval_steps)
        self.eval_batch_size = int(eval_batch_size)
        self.loss_eval_batch_size = int(
            loss_eval_batch_size
        )
        self.max_new_tokens = int(max_new_tokens)
        self.save_dir = save_dir
        self.use_swanlab = bool(use_swanlab)
        self.resume_checkpoint = resume_checkpoint
        self.run_signature = run_signature or {}
        self.history_records = []
        self._force_final_eval = False
        if self.eval_steps <= 0:
            raise ValueError("eval_steps 必须大于 0。")

        self.best_macro = -1.0
        self.best_micro = -1.0
        self.last_eval_step = -1

        self.history_path = os.path.join(
            save_dir,
            "eval_history.jsonl",
        )

        self.eot_id = tokenizer.convert_tokens_to_ids(
            "<|eot_id|>"
        )
        if (
            self.eot_id is None
            or self.eot_id == tokenizer.unk_token_id
        ):
            raise RuntimeError(
                "❌ validation 缺少原子 <|eot_id|>。"
            )
        self.eot_id = int(self.eot_id)

    def _state_dict(self):
        return {"best_macro": self.best_macro, "best_micro": self.best_micro,
                "last_eval_step": self.last_eval_step}

    def _restore_state(self, payload):
        for key in self._state_dict():
            if key not in payload:
                raise RuntimeError(f"WOS 回调断点缺少字段: {key}")
            setattr(self, key, payload[key])

    def _root_action(self, state, action):
        result = [None]
        if state.is_world_process_zero:
            try:
                result[0] = {"value": action(), "error": None}
            except Exception as exc:
                result[0] = {"value": None, "error": f"{type(exc).__name__}: {exc}"}
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(result, src=0)
        if result[0] is None:
            raise RuntimeError("非主进程未初始化分布式通信。")
        if result[0]["error"]:
            raise RuntimeError(result[0]["error"])
        return result[0]["value"]

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.resume_checkpoint:
            return control

        def restore():
            step = int(read_json_file(os.path.join(self.resume_checkpoint, "trainer_state.json"))["global_step"])
            if int(state.global_step) != step:
                raise RuntimeError("Trainer 未正确恢复 WOS global_step。")
            if step >= int(state.max_steps):
                raise RuntimeError("该 WOS 断点已完成原训练计划，无剩余训练步数。")
            custom_path = os.path.join(self.resume_checkpoint, WOS_RESUME_STATE_NAME)
            legacy = not os.path.isfile(custom_path)
            full_legacy_history = []
            if not legacy:
                saved = read_json_file(custom_path)
                records = saved["eval_history"]
                self._restore_state(saved["callback_state"])
            else:
                full_legacy_history = read_eval_history_for_resume(self.history_path, 10 ** 30)
                records = [record for record in full_legacy_history if int(record["step"]) <= step]
                if step >= self.eval_steps and (not records or int(records[-1]["step"]) != step):
                    raise RuntimeError("旧 WOS 断点缺少对应步数的验证记录。")
                best = select_best_record(records)
                self.best_macro = float(best["global_macro"]) if best else -1.0
                self.best_micro = float(best["global_micro"]) if best else -1.0
                self.last_eval_step = int(records[-1]["step"]) if records else -1
            previous = -1
            for record in records:
                current = int(record["step"])
                if current <= previous or current > step:
                    raise RuntimeError("WOS 验证历史与断点步数不一致。")
                previous = current
            if self.last_eval_step != (int(records[-1]["step"]) if records else -1):
                raise RuntimeError("WOS 回调验证步数与历史不一致。")
            allow_legacy_best = legacy
            ensure_wos_historical_best_available(
                self.save_dir, records, bool(self.run_signature.get("projector_required", False)), allow_legacy_best
            )
            write_history_snapshot(self.history_path, records)
            archive_future_checkpoints(self.save_dir, step)
            self.history_records = records
            print(f"✅ 已恢复 WOS step={step}, last_eval={self.last_eval_step}, best_Macro={self.best_macro:.6f}")
            return {"callback_state": self._state_dict(), "eval_history": records}

        payload = self._root_action(state, restore)
        self._restore_state(payload["callback_state"])
        self.history_records = payload["eval_history"]
        return control

    def on_save(self, args, state, control, **kwargs):
        distributed_barrier()

        def save():
            checkpoint = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            self.tokenizer.save_pretrained(checkpoint)
            atomic_write_json(os.path.join(checkpoint, WOS_RESUME_STATE_NAME), {
                "format_version": 1, "global_step": int(state.global_step),
                "run_signature": self.run_signature,
                "callback_state": self._state_dict(), "eval_history": self.history_records,
                "checkpoint_files": checkpoint_file_sizes(
                    checkpoint, int(args.world_size), bool(self.run_signature.get("projector_required", False)),
                    require_kd_state=True,
                ),
            })
            pending = os.path.join(checkpoint, WOS_PENDING_NAME)
            if os.path.isfile(pending):
                os.remove(pending)
            print(f"✅ 完整 WOS 续训断点保存完成: {checkpoint}")

        self._root_action(state, save)
        return control


    def _build_supervised_item(self, row):
        text = truncate_text_for_prompt(
            str(row["text"]).strip(),
            self.tokenizer,
            self.max_seq_length,
            self.reserve_tokens,
        )

        prompt_ids = encode_prompt_ids(
            self.tokenizer,
            text,
            self.system_prompt,
        )

        completion_ids = self.tokenizer.encode(
            build_completion_text(
                str(row["l1_label"]).strip(),
                str(row["l2_label"]).strip(),
            ),
            add_special_tokens=False,
        )
        completion_ids = [
            int(x) for x in completion_ids
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

        if len(input_ids) > self.max_seq_length:
            raise RuntimeError(
                "❌ validation supervised sequence 超长："
                f"{len(input_ids)} > "
                f"{self.max_seq_length}"
            )

        return input_ids, labels

    @torch.no_grad()
    def _compute_eval_loss(
        self,
        model,
        device,
    ) -> float:
        """
        与旧单 Token 代码一致：
        completion-only teacher-forcing validation CE。
        """
        old_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"

        total_loss = 0.0
        total_count = 0

        try:
            for start in tqdm(
                range(
                    0,
                    len(self.val_data),
                    self.loss_eval_batch_size,
                ),
                desc="Eval Loss",
                leave=False,
            ):
                rows = self.val_data[
                    start:
                    start + self.loss_eval_batch_size
                ]

                items = [
                    self._build_supervised_item(row)
                    for row in rows
                ]
                max_len = max(
                    len(ids)
                    for ids, _ in items
                )

                input_ids = torch.full(
                    (len(items), max_len),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.zeros(
                    (len(items), max_len),
                    dtype=torch.long,
                    device=device,
                )
                labels = torch.full(
                    (len(items), max_len),
                    -100,
                    dtype=torch.long,
                    device=device,
                )

                for i, (ids, labs) in enumerate(items):
                    n = len(ids)
                    input_ids[i, :n] = torch.tensor(
                        ids,
                        dtype=torch.long,
                        device=device,
                    )
                    attention_mask[i, :n] = 1
                    labels[i, :n] = torch.tensor(
                        labs,
                        dtype=torch.long,
                        device=device,
                    )

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                    return_dict=True,
                )

                bs = len(items)
                total_loss += (
                    outputs.loss.detach()
                    .float()
                    .item()
                    * bs
                )
                total_count += bs

        finally:
            self.tokenizer.padding_side = old_padding

        return total_loss / max(
            1,
            total_count,
        )

    def _append_history(self, payload):
        with open(
            self.history_path,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
            os.fsync(f.fileno())
        self.history_records.append(dict(payload))

    @torch.no_grad()
    def _evaluate(self, model, step):
        root = unwrap_training_model(model)
        was_training = root.training
        old_cache = root.config.use_cache

        root.eval()
        device = next(root.parameters()).device

        try:
            root.config.use_cache = False
            eval_loss = self._compute_eval_loss(
                root,
                device,
            )

            root.config.use_cache = True

            true_l1 = []
            true_l2 = []
            pred_l1 = []
            pred_l2 = []

            exact_parse_count = 0
            valid_label_pair_count = 0
            hierarchy_consistent_count = 0

            old_padding = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"

            try:
                for start in tqdm(
                    range(
                        0,
                        len(self.val_data),
                        self.eval_batch_size,
                    ),
                    desc=f"val@{step}",
                    leave=False,
                ):
                    rows = self.val_data[
                        start:
                        start + self.eval_batch_size
                    ]

                    prompt_ids_list = []

                    for row in rows:
                        text = truncate_text_for_prompt(
                            str(row["text"]).strip(),
                            self.tokenizer,
                            self.max_seq_length,
                            self.reserve_tokens,
                        )

                        prompt_ids_list.append(
                            encode_prompt_ids(
                                self.tokenizer,
                                text,
                                self.system_prompt,
                            )
                        )

                    max_len = max(
                        len(x)
                        for x in prompt_ids_list
                    )

                    input_ids = torch.full(
                        (len(rows), max_len),
                        self.tokenizer.pad_token_id,
                        dtype=torch.long,
                        device=device,
                    )
                    attention_mask = torch.zeros(
                        (len(rows), max_len),
                        dtype=torch.long,
                        device=device,
                    )

                    for i, ids in enumerate(
                        prompt_ids_list
                    ):
                        n = len(ids)
                        input_ids[
                            i,
                            max_len - n:
                        ] = torch.tensor(
                            ids,
                            dtype=torch.long,
                            device=device,
                        )
                        attention_mask[
                            i,
                            max_len - n:
                        ] = 1

                    generated = root.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=(
                            self.tokenizer.pad_token_id
                        ),
                        eos_token_id=self.eot_id,
                    )

                    new_tokens = generated[
                        :,
                        max_len:
                    ]

                    decoded = (
                        self.tokenizer.batch_decode(
                            new_tokens,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    )

                    for row, resp in zip(
                        rows,
                        decoded,
                    ):
                        l1_pred, l2_pred = (
                            parse_generation_exact(resp)
                        )

                        if l1_pred and l2_pred:
                            exact_parse_count += 1

                        p1 = self.l1_map.get(
                            l1_pred,
                            9999,
                        )
                        p2 = self.l2_map.get(
                            l2_pred,
                            9999,
                        )

                        if (
                            p1 != 9999
                            and p2 != 9999
                        ):
                            valid_label_pair_count += 1

                            if (
                                self.l2_to_l1.get(
                                    l2_pred
                                )
                                == l1_pred
                            ):
                                hierarchy_consistent_count += 1

                        true_l1.append(
                            int(row["l1_id"])
                        )
                        true_l2.append(
                            int(row["l2_id"])
                        )
                        pred_l1.append(p1)
                        pred_l2.append(p2)

            finally:
                self.tokenizer.padding_side = old_padding

            metrics = compute_metric_bundle(
                true_l1,
                true_l2,
                pred_l1,
                pred_l2,
                len(self.l1_map),
                len(self.l2_map),
            )

            n = max(
                1,
                len(true_l1),
            )
            exact_parse_rate = (
                exact_parse_count / n
            )
            valid_label_pair_rate = (
                valid_label_pair_count / n
            )
            hierarchy_consistency_rate = (
                hierarchy_consistent_count / n
            )

            print("\n" + "=" * 92)
            print(
                f"📊 Validation @ step {step}"
            )
            print(
                f"eval_loss: "
                f"{eval_loss:.6f}"
            )
            for k, v in metrics.items():
                print(
                    f"{k}: {v:.6f}"
                )
            print(
                "exact_parse_rate: "
                f"{exact_parse_rate:.6f}"
            )
            print(
                "valid_label_pair_rate: "
                f"{valid_label_pair_rate:.6f}"
            )
            print(
                "hierarchy_consistency_rate: "
                f"{hierarchy_consistency_rate:.6f}"
            )
            print("=" * 92 + "\n")

            history_payload = {
                "step": int(step),
                "eval_loss": float(eval_loss),
                **metrics,
                "exact_parse_rate": float(
                    exact_parse_rate
                ),
                "valid_label_pair_rate": float(
                    valid_label_pair_rate
                ),
                "hierarchy_consistency_rate": float(
                    hierarchy_consistency_rate
                ),
            }
            self._append_history(
                history_payload
            )

            # ------------------------------------------------------
            # SwanLab validation metrics
            #
            # 自定义生成式 validation 不会自动进入 Trainer/SwanLab，
            # 因此这里显式写入，与旧单 Token WOS 代码保持一致。
            # ------------------------------------------------------
            if self.use_swanlab:
                if not HAS_SWANLAB or swanlab is None:
                    raise RuntimeError(
                        "❌ --use_swanlab=True，但当前环境无法导入 swanlab。"
                    )

                swanlab.log(
                    {
                        "Eval/Loss": float(eval_loss),

                        "Eval/Global_Micro_F1": float(
                            metrics["global_micro"]
                        ),
                        "Eval/Global_Macro_F1": float(
                            metrics["global_macro"]
                        ),

                        "Eval/L1_Micro_F1": float(
                            metrics["l1_micro"]
                        ),
                        "Eval/L1_Macro_F1": float(
                            metrics["l1_macro"]
                        ),

                        "Eval/L2_Micro_F1": float(
                            metrics["l2_micro"]
                        ),
                        "Eval/L2_Macro_F1": float(
                            metrics["l2_macro"]
                        ),

                        "Eval/Exact_Parse_Rate": float(
                            exact_parse_rate
                        ),
                        "Eval/Valid_Label_Pair_Rate": float(
                            valid_label_pair_rate
                        ),
                        "Eval/Hierarchy_Consistency_Rate": float(
                            hierarchy_consistency_rate
                        ),
                    },
                    step=int(step),
                )

            macro = metrics[
                "global_macro"
            ]
            micro = metrics[
                "global_micro"
            ]

            better = (
                macro > self.best_macro
                or (
                    math.isclose(
                        macro,
                        self.best_macro,
                        abs_tol=1e-12,
                    )
                    and micro > self.best_micro
                )
            )

            if better:
                self.best_macro = macro
                self.best_micro = micro

                best_dir = os.path.join(
                    self.save_dir,
                    "best_kd_checkpoint",
                )

                save_peft_checkpoint_with_full_io(
                    root,
                    self.tokenizer,
                    best_dir,
                    overwrite=True,
                    checkpoint_metadata=history_payload,
                )

                with open(
                    os.path.join(
                        self.save_dir,
                        "best_metrics.json",
                    ),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        history_payload,
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                print(
                    f"🏆 更新 best KD checkpoint: "
                    f"{best_dir}"
                )

            self.last_eval_step = int(step)

        finally:
            root.config.use_cache = old_cache
            if was_training:
                root.train()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if step <= 0 or step == self.last_eval_step or (not self._force_final_eval and step % self.eval_steps != 0):
            return control

        def evaluate():
            if model is None:
                raise RuntimeError("WOS 配对蒸馏验证回调未取得模型。")
            self._evaluate(unwrap_training_model(model), step)
            return self._state_dict()

        self._restore_state(self._root_action(state, evaluate))
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if int(state.global_step) <= 0 or int(state.global_step) == self.last_eval_step:
            return control
        self._force_final_eval = True
        try:
            return self.on_step_end(args, state, control, model=model, **kwargs)
        finally:
            self._force_final_eval = False


# ======================================================================
# 11. TRAIN stage
# ======================================================================

def resolve_mode_weights(args):
    mode = args.experiment_mode
    if mode == "traditional":
        return 0.0, 0.0
    if mode == "align":
        return args.align_weight, 0.0
    if mode == "hier":
        return 0.0, args.hier_weight
    if mode == "full":
        return args.align_weight, args.hier_weight
    raise ValueError(mode)


def save_distill_config(args, effective_align, effective_hier, run_signature=None):
    payload = {
        "experiment_mode": args.experiment_mode,
        "student_init": "raw_base_only_no_student_sft",
        "student_model_path": os.path.abspath(args.student_model_path),
        "teacher_source": "offline cache extracted from trained best_sft_model",
        "teacher_cache_dir": os.path.abspath(args.cache_dir),
        "loss_total": (
            "TaskCE + kd_weight*FullCompletionLogitsKD + "
            "effective_align*FourTokenHiddenAlign + "
            "effective_hier*TwoRelationHiddenHier"
        ),
        "kd_temperature": args.kd_temperature,
        "kd_weight": args.kd_weight,
        "align_weight_effective": effective_align,
        "hier_weight_effective": effective_hier,
        "align_alpha": args.align_alpha,
        "four_align_targets": [
            f"{r}_second_occurrence" for r in HIER_TOKEN_ROLES
        ],
        "special_weights": {
            "l1_start": args.l1_start_weight,
            "l1_end": args.l1_end_weight,
            "l2_start": args.l2_start_weight,
            "l2_end": args.l2_end_weight,
        },
        "hier_relations": {
            "start_relation": "L1_START -> L2_START",
            "end_relation": "L1_END -> L2_END",
        },
        "relation_weights": {
            "start_relation": args.hier_start_weight,
            "end_relation": args.hier_end_weight,
        },
        "prompt_completion_definition": (
            "exactly identical to current paired START/END Student/Teacher SFT"
        ),
    }
    if run_signature is not None:
        payload["run_signature"] = run_signature
    config_path = os.path.join(args.save_dir, "distill_config.json")
    if os.path.isfile(config_path):
        shutil.copy2(config_path, config_path + f".before-resume-{time.time_ns()}.bak")
    atomic_write_json(config_path, payload)


def train_student(args) -> None:
    set_all_seeds(args.seed)
    assert_raw_base_student_path(args.student_model_path)
    args.save_dir = os.path.abspath(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    resume_checkpoint = resolve_resume_checkpoint(args)
    run_signature = build_run_signature(args)
    if resume_checkpoint:
        validate_resume_configuration(resume_checkpoint, run_signature)
        print(f"🔄 将恢复完整 WOS 配对蒸馏断点: {resume_checkpoint}")
    else:
        print("🆕 开始新的 WOS 配对蒸馏实验。")

    l1_map, l2_map, l2_to_l1 = load_label_space(
        args.dict_path,
        args.taxonomy_path,
    )
    train_data = validate_split(
        args.train_data_path,
        "TRAIN",
        l1_map,
        l2_map,
        l2_to_l1,
    )
    val_data = validate_split(
        args.val_data_path,
        "VALIDATION",
        l1_map,
        l2_map,
        l2_to_l1,
    )

    tokenizer = configure_tokenizer(
        AutoTokenizer.from_pretrained(args.student_model_path)
    )
    token_ids = validate_atomic_reserved_tokens(tokenizer)

    train_dataset = PairedTokenWOSDataset(
        train_data,
        tokenizer,
        token_ids,
        args.max_seq_length,
        args.reserve_tokens,
        args.system_prompt,
        "TRAIN",
    )

    # 启动前至少完整走一遍 layout/hash，保证与 cache 一致
    cache = OfflineTeacherCache(
        args.cache_dir,
        expected_num_samples=len(train_dataset),
    )

    # 与旧单 Token offline KD 一样，启动前先核对 cache 提取定义。
    manifest = cache.manifest

    if int(
        manifest.get(
            "max_seq_length",
            -1,
        )
    ) != int(args.max_seq_length):
        raise RuntimeError(
            "❌ 当前 max_seq_length 与 Teacher cache 不一致："
            f"train={args.max_seq_length}, "
            f"cache={manifest.get('max_seq_length')}"
        )

    if int(
        manifest.get(
            "reserve_tokens",
            -1,
        )
    ) != int(args.reserve_tokens):
        raise RuntimeError(
            "❌ 当前 reserve_tokens 与 Teacher cache 不一致："
            f"train={args.reserve_tokens}, "
            f"cache={manifest.get('reserve_tokens')}"
        )

    if str(
        manifest.get(
            "system_prompt",
            "",
        )
    ) != str(args.system_prompt):
        raise RuntimeError(
            "❌ 当前 system_prompt 与 Teacher cache 提取时不一致。"
        )

    cached_template = manifest.get(
        "chat_template"
    )
    if (
        cached_template is not None
        and str(cached_template) != str(CHAT_TEMPLATE)
    ):
        raise RuntimeError(
            "❌ 当前 chat_template 与 Teacher cache 提取时不一致。"
        )

    cached_completion = str(
        manifest.get(
            "completion",
            "",
        )
    )
    expected_completion_rule = (
        "L1_START + L1_LABEL + L1_END + "
        "L2_START + L2_LABEL + L2_END + EOT"
    )
    if (
        cached_completion
        and cached_completion != expected_completion_rule
    ):
        raise RuntimeError(
            "❌ Teacher cache Completion 定义与当前双 Token KD 不一致："
            f"cache={cached_completion!r}"
        )

    # 四个 hierarchy token 必须在 cached common vocab 中。
    common_id_set = set(
        int(x)
        for x in cache.head_payload[
            "student_common_ids"
        ].tolist()
    )
    for role in HIER_TOKEN_ROLES:
        if int(token_ids[role]) not in common_id_set:
            raise RuntimeError(
                f"❌ {role} token ID={token_ids[role]} "
                "未进入 Teacher cache common vocab。"
            )

    print("\n🔎 训练前全量验证 Student 编码与 Teacher cache")
    for i in tqdm(range(len(train_dataset)), desc="verify-cache", leave=False):
        x = train_dataset.encode_item(i)
        cache.verify_sample(
            i,
            x["input_hash"],
            x["valid_count"],
            x["rel_positions"],
        )
    print("✅ 全部训练样本与 cache 的 hash / valid positions / 4-token offsets 完全一致")

    model_dtype = torch.bfloat16 if args.bf16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.student_model_path,
        torch_dtype=model_dtype,
        attn_implementation=args.attn_implementation,
        device_map=None,
    )

    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        raise RuntimeError(
            "❌ Student tokenizer 长度与 embedding 行数不一致；本脚本不 resize"
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = (
            tokenizer.pad_token_id
        )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    actually_tied = enable_full_input_output_training(model)

    # 与旧单 Token 代码保持一致。
    # gradient checkpointing + PEFT 下显式要求输入 embedding 参与梯度图。
    model.enable_input_require_grads()

    if not model.get_input_embeddings().weight.requires_grad:
        raise RuntimeError(
            "❌ input embedding 未进入训练。"
        )
    if not model.get_output_embeddings().weight.requires_grad:
        raise RuntimeError(
            "❌ lm_head 未进入训练。"
        )

    effective_align, effective_hier = resolve_mode_weights(args)
    needs_projector = effective_align > 0 or effective_hier > 0

    if needs_projector:
        student_dim = int(
            unwrap_training_model(model).config.hidden_size
        )
        teacher_dim = int(cache.teacher_hidden_size)
        projector = nn.Linear(
            student_dim,
            teacher_dim,
            bias=True,
            dtype=torch.float32,
        )
        unwrap_training_model(model).add_module(
            "hidden_projector",
            projector,
        )

    special_weights = {
        "l1_start": args.l1_start_weight,
        "l1_end": args.l1_end_weight,
        "l2_start": args.l2_start_weight,
        "l2_end": args.l2_end_weight,
    }
    relation_weights = {
        "start_relation": args.hier_start_weight,
        "end_relation": args.hier_end_weight,
    }

    print("\n" + "=" * 96)
    print("🎯 WOS Paired-Token KD 配置")
    print(f"Mode: {args.experiment_mode}")
    print(f"Student raw base: {args.student_model_path}")
    print(f"Teacher cache:    {args.cache_dir}")
    print("Teacher 8B Transformer during training: 不加载")
    print(f"input/lm_head tied: {actually_tied}")
    print(
        f"Loss = TaskCE + {args.kd_weight}*FullLogitsKD + "
        f"{effective_align}*4TokenAlign + "
        f"{effective_hier}*2RelationHier"
    )
    print(
        "4 Align targets: "
        "L1_START, L1_END, L2_START, L2_END 第二次 occurrence"
    )
    print(
        "2 Hier relations: "
        "L1_START→L2_START | L1_END→L2_END"
    )

    if effective_align > 0:
        print(
            "SwanLab Align metrics: "
            "4 tokens × (Total/MSE/CosLoss)"
        )

    if effective_hier > 0:
        print(
            "SwanLab Hier metrics: "
            "START relation MSE + END relation MSE"
        )

    print(
        "SwanLab validation metrics: "
        "Loss + Global/L1/L2 Micro/Macro + "
        "ExactParse + ValidPair + HierConsistency"
    )

    print("=" * 96 + "\n")

    training_args = TrainingArguments(
        output_dir=os.path.join(args.save_dir, "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=False,
        ignore_data_skip=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=args.pin_memory,
        seed=args.seed,
        data_seed=args.seed,
    )

    callbacks = []

    if HAS_SWANLAB and args.use_swanlab:
        callbacks.append(
            SwanLabCallback(
                project=args.project_name,
                experiment_name=args.run_name,
            )
        )

    callbacks.append(
        GenerativeF1EvalCallback(
            val_data=val_data,
            l1_map=l1_map,
            l2_map=l2_map,
            l2_to_l1=l2_to_l1,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            reserve_tokens=args.reserve_tokens,
            system_prompt=args.system_prompt,
            eval_steps=args.eval_steps,
            eval_batch_size=args.eval_batch_size,
            loss_eval_batch_size=args.loss_eval_batch_size,
            max_new_tokens=args.max_new_tokens,
            save_dir=args.save_dir,
            use_swanlab=args.use_swanlab,
            resume_checkpoint=resume_checkpoint,
            run_signature=run_signature,
        )
    )

    trainer = OfflinePairedTokenKDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=PairedTokenCollator(tokenizer),
        callbacks=callbacks,
        io_learning_rate=args.io_lr,
        projector_learning_rate=(
            args.projector_lr if needs_projector else None
        ),
        teacher_cache=cache,
        dataset_for_verification=train_dataset,
        kd_temperature=args.kd_temperature,
        kd_weight=args.kd_weight,
        align_weight=effective_align,
        hier_weight=effective_hier,
        align_alpha=args.align_alpha,
        special_weights=special_weights,
        relation_weights=relation_weights,
    )

    model.print_trainable_parameters()
    if trainer.is_world_process_zero():
        save_distill_config(args, effective_align, effective_hier, run_signature=run_signature)
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    if trainer.is_world_process_zero():
        final_ckpt = os.path.join(
            args.save_dir,
            "final_kd_checkpoint",
        )
        save_peft_checkpoint_with_full_io(
            model,
            tokenizer,
            final_ckpt,
            overwrite=True,
        )

        best_ckpt = os.path.join(
            args.save_dir,
            "best_kd_checkpoint",
        )
        if not os.path.isdir(best_ckpt):
            print("⚠️ 未触发 best 保存，复制 final 为 best")
            save_peft_checkpoint_with_full_io(
                model,
                tokenizer,
                best_ckpt,
                overwrite=True,
            )

        if args.export_merged_model:
            export_merged_full_model(
                args.student_model_path,
                final_ckpt,
                os.path.join(args.save_dir, "final_model"),
                tokenizer,
                model_dtype,
            )
            export_merged_full_model(
                args.student_model_path,
                best_ckpt,
                os.path.join(args.save_dir, "best_kd_model"),
                tokenizer,
                model_dtype,
            )

    print("\n🎉 Paired-token KD 完成")


# ======================================================================
# 12. CLI
# ======================================================================

def add_common_data_args(p):
    p.add_argument(
        "--train_data_path",
        type=str,
        default="/data1/lzp/code/data/wos/train_formatted.jsonl",
    )
    p.add_argument(
        "--val_data_path",
        type=str,
        default="/data1/lzp/code/data/wos/val_formatted.jsonl",
    )
    p.add_argument(
        "--taxonomy_path",
        type=str,
        default="/data1/lzp/code/data/wos/wos.taxnomy",
    )
    p.add_argument(
        "--dict_path",
        type=str,
        default="/data1/lzp/code/data/wos/label2id.json",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--reserve_tokens", type=int, default=300)
    p.add_argument(
        "--system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "WOS paired START/END hierarchy-token offline KD: "
            "trained Teacher -> raw Student"
        )
    )
    sub = parser.add_subparsers(
        dest="stage",
        required=True,
    )

    # ---------------- cache ----------------
    pc = sub.add_parser(
        "cache",
        help="从训练好的 Teacher best_sft_model 提取离线 cache",
    )
    add_common_data_args(pc)

    pc.add_argument(
        "--teacher_model_path",
        type=str,
        default=(
            "/data1/lzp/code/hierarchical_distill/"
            "checkpoints_teacher_wos_2/best_sft_model"
        ),
    )
    pc.add_argument(
        "--student_model_path",
        type=str,
        default="/data1/lzp/code/llama-3.2_1b/",
        help="仅用于构造与 KD Student 完全一致的 tokenization/position。",
    )
    pc.add_argument(
        "--cache_dir",
        type=str,
        default=(
            "/data1/lzp/code/hierarchical_distill/"
            "teacher_wos_2token_hiddenstates/"
        ),
    )
    pc.add_argument("--cache_batch_size", type=int, default=1)
    pc.add_argument("--device", type=str, default="cuda:0")
    pc.add_argument("--bf16", type=str2bool, default=True)
    pc.add_argument("--num_workers", type=int, default=0)
    pc.add_argument("--pin_memory", type=str2bool, default=True)
    pc.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
    )

    # ---------------- train ----------------
    pt = sub.add_parser(
        "train",
        help="从 raw Student 开始，用离线训练 Teacher cache 蒸馏",
    )
    add_common_data_args(pt)

    pt.add_argument(
        "--student_model_path",
        type=str,
        default="/data1/lzp/code/llama-3.2_1b/",
    )
    pt.add_argument(
        "--cache_dir",
        type=str,
        default=(
            "/data1/lzp/code/hierarchical_distill/"
            "teacher_wos_2token_hiddenstates/"
        ),
    )
    pt.add_argument(
        "--save_dir",
        type=str,
        default=(
            "/data1/lzp/code/hierarchical_distill/"
            "checkpoints_wos_2token_kd"
        ),
    )

    pt.add_argument(
        "--experiment_mode",
        type=str,
        choices=["traditional", "align", "hier", "full"],
        default="traditional",
    )

    pt.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="不传则新实验；auto/true/latest 恢复最近完整周期断点；也可指定原 save_dir/checkpoints/checkpoint-N。",
    )
    pt.add_argument("--num_epochs", type=float, default=3.0)
    pt.add_argument("--batch_size", type=int, default=4)
    pt.add_argument("--accumulation_steps", type=int, default=4)
    pt.add_argument("--lr", type=float, default=1.0e-4)
    pt.add_argument("--io_lr", type=float, default=1.0e-5)
    pt.add_argument("--projector_lr", type=float, default=1.0e-4)
    pt.add_argument("--weight_decay", type=float, default=0.0)

    pt.add_argument("--kd_temperature", type=float, default=2.0)
    pt.add_argument("--kd_weight", type=float, default=1.0)

    pt.add_argument("--align_weight", type=float, default=1.0)
    pt.add_argument("--hier_weight", type=float, default=1.0)
    pt.add_argument("--align_alpha", type=float, default=0.5)

    # 4 个 Align target 的权重
    pt.add_argument("--l1_start_weight", type=float, default=1.0)
    pt.add_argument("--l1_end_weight", type=float, default=1.0)
    pt.add_argument("--l2_start_weight", type=float, default=1.0)
    pt.add_argument("--l2_end_weight", type=float, default=1.0)

    # 两条 L1 -> L2 Hier relation 的权重
    pt.add_argument(
        "--hier_start_weight",
        type=float,
        default=1.0,
        help="L1_START -> L2_START relation 权重",
    )
    pt.add_argument(
        "--hier_end_weight",
        type=float,
        default=1.0,
        help="L1_END -> L2_END relation 权重",
    )

    pt.add_argument("--lr_scheduler_type", type=str, default="linear")
    pt.add_argument("--warmup_ratio", type=float, default=0.0)

    pt.add_argument("--lora_rank", type=int, default=16)
    pt.add_argument("--lora_alpha", type=int, default=32)
    pt.add_argument("--lora_dropout", type=float, default=0.05)

    pt.add_argument("--eval_steps", type=int, default=500)
    pt.add_argument("--eval_batch_size", type=int, default=16)
    pt.add_argument(
        "--loss_eval_batch_size",
        type=int,
        default=4,
        help="teacher-forcing validation loss batch size",
    )
    pt.add_argument("--max_new_tokens", type=int, default=32)
    pt.add_argument("--save_total_limit", type=int, default=8)
    pt.add_argument("--logging_steps", type=int, default=10)
    pt.add_argument("--export_merged_model", type=str2bool, default=True)

    pt.add_argument("--num_workers", type=int, default=0)
    pt.add_argument("--pin_memory", type=str2bool, default=True)
    pt.add_argument("--bf16", type=str2bool, default=True)
    pt.add_argument(
        "--gradient_checkpointing",
        type=str2bool,
        default=True,
    )
    pt.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
    )

    pt.add_argument("--use_swanlab", type=str2bool, default=True)
    pt.add_argument(
        "--project_name",
        type=str,
        default="WOS-PairedToken-Offline-KD",
    )
    pt.add_argument(
        "--run_name",
        type=str,
        default="",
    )

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.stage == "cache":
        build_teacher_cache(args)
    elif args.stage == "train":
        if not args.run_name:
            args.run_name = (
                f"Llama3.2-1B-WOS-PairedToken-KD-"
                f"{args.experiment_mode}-Seed{args.seed}"
            )
        train_student(args)
    else:
        raise ValueError(args.stage)
