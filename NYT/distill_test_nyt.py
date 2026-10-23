import os

# 必须在 import torch 前设置。
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import json
import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import f1_score
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


def clear_cuda_memory(stage: str = "") -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if stage:
        print(f"🧹 CUDA 临时显存清理完成: {stage}")


def is_cuda_oom_error(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    type_match = oom_type is not None and isinstance(exc, oom_type)
    message = str(exc).lower()
    return bool(type_match or "cuda out of memory" in message)


# ======================================================================
# 1. 与教师训练完全一致的 Llama 预留特殊 Token
# ======================================================================
CTX_START_TOKEN = "<|reserved_special_token_110|>"
CTX_END_TOKEN = "<|reserved_special_token_111|>"

DEPTH_PROBE_TOKENS: Dict[int, str] = {
    depth: f"<|reserved_special_token_{111 + depth}|>"
    for depth in range(1, 9)
}

FULL_IO_WEIGHTS_NAME = "full_input_output_embeddings.pt"


def validate_atomic_reserved_tokens(
    tokenizer,
    expected_max_depth: int,
) -> Dict[str, object]:
    if expected_max_depth != len(DEPTH_PROBE_TOKENS):
        raise ValueError(
            "❌ 当前脚本固定为 NYT 8 层 token 设计，"
            f"expected_max_depth={expected_max_depth}，"
            f"但 DEPTH_PROBE_TOKENS={len(DEPTH_PROBE_TOKENS)}。"
        )

    token_map = {
        "ctx_start": CTX_START_TOKEN,
        "ctx_end": CTX_END_TOKEN,
    }
    token_map.update(
        {
            f"depth_{depth}": DEPTH_PROBE_TOKENS[depth]
            for depth in range(1, expected_max_depth + 1)
        }
    )

    vocab = tokenizer.get_vocab()
    flat_ids: Dict[str, int] = {}

    print("\n" + "=" * 78)
    print("🔎 验证 Llama 预留特殊 Token...")

    for role, token in token_map.items():
        if token not in vocab:
            raise RuntimeError(
                f"❌ {token} 不在当前 tokenizer 词表中。"
            )

        ids = tokenizer.encode(token, add_special_tokens=False)
        converted_id = tokenizer.convert_tokens_to_ids(token)

        if len(ids) != 1:
            raise RuntimeError(
                f"❌ {token} 不是严格单 token，实际 ids={ids}"
            )

        token_id = int(ids[0])
        if converted_id != token_id:
            raise RuntimeError(
                f"❌ {token} encode ID 与 convert ID 不一致："
                f"encode={token_id}, convert={converted_id}"
            )
        if token_id == tokenizer.unk_token_id:
            raise RuntimeError(f"❌ {token} 被编码成 unk_token。")

        flat_ids[role] = token_id
        print(f"✅ {role}: {token} -> ID={token_id}")

    if len(set(flat_ids.values())) != len(flat_ids):
        raise RuntimeError(
            f"❌ 预留特殊 Token ID 未两两区分：{flat_ids}"
        )

    print("✅ 正文边界 token 与 8 个层级 token 均为不可分割原子 token")
    print("=" * 78 + "\n")

    return {
        "ctx_start": flat_ids["ctx_start"],
        "ctx_end": flat_ids["ctx_end"],
        "depth": {
            depth: flat_ids[f"depth_{depth}"]
            for depth in range(1, expected_max_depth + 1)
        },
    }


# ======================================================================
# 2. 与教师训练完全一致的 Prompt / Completion 逻辑
# ======================================================================
def build_user_content(text: str, max_depth: int) -> str:
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


def build_prompt_messages(
    text: str,
    system_prompt: str,
    max_depth: int,
) -> List[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": build_user_content(text, max_depth=max_depth),
        },
    ]


def encode_chat_prompt_consistently(
    tokenizer,
    messages: List[dict],
    split_name: str,
    source_index: int,
) -> Tuple[str, List[int]]:
    prompt_ids_direct = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_ids_direct = [int(x) for x in prompt_ids_direct]

    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is not None:
        bos_count = prompt_ids_direct.count(int(bos_token_id))
        if bos_count != 1:
            raise RuntimeError(
                f"❌ {split_name} 第 {source_index} 条 Prompt 中 BOS 应出现1次，"
                f"实际={bos_count}。"
            )

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids_from_text = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    prompt_ids_from_text = [int(x) for x in prompt_ids_from_text]

    if prompt_ids_direct != prompt_ids_from_text:
        raise RuntimeError(
            f"❌ {split_name} 第 {source_index} 条 Prompt 两种编码入口不一致。"
        )

    return prompt_text, prompt_ids_direct


def truncate_text_for_prompt(
    text: str,
    tokenizer,
    max_seq_length: int = 3072,
    reserve_tokens: int = 1024,
) -> str:
    max_text_len = max_seq_length - reserve_tokens
    if max_text_len <= 0:
        raise ValueError(
            f"❌ max_seq_length={max_seq_length} 必须大于 "
            f"reserve_tokens={reserve_tokens}"
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
# 3. NYT clean 数据与 label_space
# ======================================================================
REQUIRED_CLEAN_FIELDS = (
    "text",
    "labels",
    "label_ids",
    "top_labels",
    "top_label_ids",
    "leaf_labels",
    "leaf_label_ids",
)


def load_json_or_jsonl(path: str) -> List[dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"❌ 数据文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        raise ValueError(f"❌ 数据文件为空: {path}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        rows = []
        for line_no, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"❌ JSONL 解析失败: {path}, line={line_no}: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise TypeError(
                    f"❌ JSONL 每行必须是 dict: line={line_no}"
                )
            rows.append(item)
        return rows

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise TypeError(f"❌ JSON 顶层必须是 list: {path}")
    if not all(isinstance(item, dict) for item in data):
        raise TypeError("❌ JSON 中每条样本必须是 dict。")
    return data


def _normalize_string_list(value, field_name: str, split_name: str, idx: int):
    if not isinstance(value, list):
        raise TypeError(
            f"{split_name} 第 {idx} 条 {field_name} 必须是 list。"
        )

    normalized = []
    for position, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise TypeError(
                f"{split_name} 第 {idx} 条 {field_name}[{position}] "
                "必须是非空 str。"
            )
        normalized.append(item.strip())
    return normalized


def _normalize_int_list(value, field_name: str, split_name: str, idx: int):
    if not isinstance(value, list):
        raise TypeError(
            f"{split_name} 第 {idx} 条 {field_name} 必须是 list。"
        )

    normalized = []
    for position, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(
                f"{split_name} 第 {idx} 条 {field_name}[{position}] "
                f"必须是 int，实际={item!r}"
            )
        normalized.append(int(item))
    return normalized


def get_clean_sample_fields(
    item: dict,
    split_name: str,
    idx: int,
) -> Tuple[
    str,
    List[str],
    List[int],
    List[str],
    List[int],
    List[str],
    List[int],
]:
    missing = [key for key in REQUIRED_CLEAN_FIELDS if key not in item]
    if missing:
        raise KeyError(
            f"{split_name} 第 {idx} 条缺少字段: {missing}"
        )

    text = item["text"]
    if not isinstance(text, str):
        raise TypeError(
            f"{split_name} 第 {idx} 条 text 必须是 str。"
        )
    text = text.strip()
    if not text:
        raise ValueError(f"{split_name} 第 {idx} 条 text 为空。")

    labels = _normalize_string_list(item["labels"], "labels", split_name, idx)
    label_ids = _normalize_int_list(item["label_ids"], "label_ids", split_name, idx)
    top_labels = _normalize_string_list(item["top_labels"], "top_labels", split_name, idx)
    top_label_ids = _normalize_int_list(item["top_label_ids"], "top_label_ids", split_name, idx)
    leaf_labels = _normalize_string_list(item["leaf_labels"], "leaf_labels", split_name, idx)
    leaf_label_ids = _normalize_int_list(item["leaf_label_ids"], "leaf_label_ids", split_name, idx)

    if not labels:
        raise ValueError(f"{split_name} 第 {idx} 条 labels 为空。")

    return (
        text,
        labels,
        label_ids,
        top_labels,
        top_label_ids,
        leaf_labels,
        leaf_label_ids,
    )


def _normalize_label2id(raw_mapping, name: str) -> Dict[str, int]:
    if not isinstance(raw_mapping, dict):
        raise TypeError(f"❌ {name} 必须是 dict。")

    normalized: Dict[str, int] = {}
    for raw_label, raw_id in raw_mapping.items():
        label = str(raw_label).strip()
        if not label:
            raise ValueError(f"❌ {name} 中存在空标签。")
        if isinstance(raw_id, bool):
            raise TypeError(f"❌ {name}[{label!r}] 不能是 bool。")
        normalized[label] = int(raw_id)
    return normalized


def _normalize_children(raw_children) -> Dict[str, List[str]]:
    if not isinstance(raw_children, dict):
        raise TypeError("❌ label_space.children 必须是 dict。")

    normalized: Dict[str, List[str]] = {}
    for raw_parent, raw_list in raw_children.items():
        parent = str(raw_parent).strip()
        if not isinstance(raw_list, list):
            raise TypeError(f"❌ children[{parent!r}] 必须是 list。")
        children = [str(x).strip() for x in raw_list]
        if any(not x for x in children):
            raise ValueError(f"❌ children[{parent!r}] 中存在空子节点。")
        if len(children) != len(set(children)):
            raise ValueError(f"❌ children[{parent!r}] 中存在重复子节点。")
        normalized[parent] = children
    return normalized


def load_and_validate_label_space(
    label_space_path: str,
    expected_num_labels: int,
    expected_max_depth: int,
) -> dict:
    if not os.path.isfile(label_space_path):
        raise FileNotFoundError(
            f"❌ label_space.json 不存在: {label_space_path}"
        )

    with open(label_space_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    required = (
        "num_labels",
        "max_depth",
        "label2id",
        "id2label",
        "top_labels",
        "parent",
        "children",
        "depth",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"❌ label_space.json 缺少字段: {missing}")

    label2id = _normalize_label2id(raw["label2id"], "label2id")
    num_labels = int(raw["num_labels"])
    max_depth = int(raw["max_depth"])

    if num_labels != len(label2id):
        raise ValueError("❌ num_labels 与 label2id 数量不一致。")
    if expected_num_labels > 0 and num_labels != expected_num_labels:
        raise ValueError(
            f"❌ NYT 标签数异常: {num_labels} != {expected_num_labels}"
        )
    if expected_max_depth > 0 and max_depth != expected_max_depth:
        raise ValueError(
            f"❌ NYT 最大深度异常: {max_depth} != {expected_max_depth}"
        )

    ids = sorted(label2id.values())
    if ids != list(range(num_labels)):
        raise ValueError("❌ label2id 必须是连续 0-based ID。")

    id2label = {
        int(raw_id): str(raw_label).strip()
        for raw_id, raw_label in raw["id2label"].items()
    }
    expected_id2label = {
        label_id: label for label, label_id in label2id.items()
    }
    if id2label != expected_id2label:
        raise ValueError("❌ id2label 与 label2id 不完全互逆。")

    parent = {
        str(child).strip(): str(raw_parent).strip()
        for child, raw_parent in raw["parent"].items()
    }
    children = _normalize_children(raw["children"])
    depth = {
        str(label).strip(): int(raw_depth)
        for label, raw_depth in raw["depth"].items()
    }
    top_labels = [str(label).strip() for label in raw["top_labels"]]

    all_task_labels = set(label2id)
    all_nodes = all_task_labels | {"Root"}

    if set(parent) != all_task_labels:
        raise ValueError("❌ parent 的键必须恰好覆盖全部任务标签。")
    if set(children) != all_nodes:
        raise ValueError("❌ children 必须覆盖 Root 和全部任务标签。")
    if set(depth) != all_nodes:
        raise ValueError("❌ depth 必须覆盖 Root 和全部任务标签。")
    if depth["Root"] != 0:
        raise ValueError("❌ Root 深度必须是 0。")
    if children["Root"] != top_labels:
        raise ValueError("❌ top_labels 与 children['Root'] 不一致。")

    for label in sorted(all_task_labels, key=label2id.get):
        current_depth = depth[label]
        if current_depth < 1 or current_depth > max_depth:
            raise ValueError(f"❌ 标签 {label!r} 深度越界。")

        current_parent = parent[label]
        if current_parent not in all_nodes:
            raise ValueError(f"❌ 标签 {label!r} 父节点未知。")
        if label not in children[current_parent]:
            raise ValueError("❌ parent / children 不一致。")
        if depth[current_parent] != current_depth - 1:
            raise ValueError("❌ parent / child 深度关系错误。")

    ancestors: Dict[str, List[str]] = {}
    for label in sorted(all_task_labels, key=label2id.get):
        chain = []
        current = label
        seen = set()
        while current != "Root":
            if current in seen:
                raise ValueError(f"❌ taxonomy 存在环: {label!r}")
            seen.add(current)
            chain.append(current)
            current = parent[current]
        chain.reverse()
        if len(chain) != depth[label]:
            raise ValueError("❌ 祖先链长度与 depth 不一致。")
        ancestors[label] = chain

    depth_to_labels: Dict[int, List[str]] = {
        d: [] for d in range(1, max_depth + 1)
    }
    for label in sorted(all_task_labels, key=label2id.get):
        depth_to_labels[depth[label]].append(label)

    label_ids_by_depth = {
        d: [label2id[label] for label in labels]
        for d, labels in depth_to_labels.items()
    }

    print("\n" + "=" * 78)
    print("🏷️ NYT 标签空间严格校验完成")
    print(f"任务标签数: {num_labels}")
    print(f"最大深度: {max_depth}")
    for d in range(1, max_depth + 1):
        print(f"Depth {d}: {len(depth_to_labels[d])} 个标签")
    print("=" * 78 + "\n")

    return {
        **raw,
        "num_labels": num_labels,
        "max_depth": max_depth,
        "label2id": label2id,
        "id2label": id2label,
        "top_labels": top_labels,
        "parent": parent,
        "children": children,
        "depth": depth,
        "ancestors": ancestors,
        "depth_to_labels": depth_to_labels,
        "label_ids_by_depth": label_ids_by_depth,
    }


def _is_ancestor_closed(label_set: Set[str], label_space: dict) -> bool:
    parent = label_space["parent"]
    for label in label_set:
        current = label
        while parent[current] != "Root":
            current = parent[current]
            if current not in label_set:
                return False
    return True


def validate_clean_split(
    data_path: str,
    split_name: str,
    label_space: dict,
) -> List[dict]:
    data = load_json_or_jsonl(data_path)
    label2id = label_space["label2id"]
    depth = label_space["depth"]
    children = label_space["children"]

    seen_labels = set()
    per_depth_occurrence = defaultdict(int)

    for idx, item in enumerate(tqdm(data, desc=f"校验{split_name}", leave=False)):
        (
            _,
            labels,
            label_ids,
            top_labels,
            top_label_ids,
            leaf_labels,
            leaf_label_ids,
        ) = get_clean_sample_fields(item, split_name, idx)

        if len(labels) != len(label_ids):
            raise ValueError(f"❌ {split_name} 第 {idx} 条 labels/IDs 长度不一致。")
        if len(labels) != len(set(labels)):
            raise ValueError(f"❌ {split_name} 第 {idx} 条 labels 重复。")

        unknown = set(labels) - set(label2id)
        if unknown:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条出现未知标签: {sorted(unknown)}"
            )

        expected_labels = sorted(labels, key=label2id.get)
        expected_ids = [label2id[label] for label in expected_labels]
        if labels != expected_labels:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条 labels 未按全局 label ID 排序。"
            )
        if label_ids != expected_ids:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条 label_ids 与 label2id 不一致。"
            )

        label_set = set(labels)
        if not _is_ancestor_closed(label_set, label_space):
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条 labels 不是祖先闭包。"
            )

        expected_top = [label for label in labels if depth[label] == 1]
        expected_top_ids = [label2id[label] for label in expected_top]
        if top_labels != expected_top or top_label_ids != expected_top_ids:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条 top_* 与 D1 不一致。"
            )

        expected_leaf = [
            label
            for label in labels
            if not any(child in label_set for child in children[label])
        ]
        expected_leaf_ids = [label2id[label] for label in expected_leaf]
        if leaf_labels != expected_leaf or leaf_label_ids != expected_leaf_ids:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条 leaf_* 与样本端点不一致。"
            )

        seen_labels.update(labels)
        for label in labels:
            per_depth_occurrence[depth[label]] += 1

    print("\n" + "=" * 78)
    print(f"✅ {split_name} 严格校验通过")
    print(f"样本数: {len(data)}")
    print(f"实际出现标签: {len(seen_labels)} / {label_space['num_labels']}")
    for d in range(1, label_space["max_depth"] + 1):
        print(f"Depth {d} 标签实例数: {per_depth_occurrence[d]}")
    print("=" * 78 + "\n")

    return data


# ======================================================================
# 4. 与训练完全一致的层内顺序规则
# ======================================================================
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


# ======================================================================
# 5. 与训练完全一致的生成解析与语义校验
# ======================================================================
def _strip_trailing_stop_strings(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    stop_strings = ("<|eot_id|>", "<|end_of_text|>")

    changed = True
    while changed:
        changed = False
        for stop_string in stop_strings:
            if text.endswith(stop_string):
                text = text[: -len(stop_string)].rstrip()
                changed = True
    return text


def trim_generated_at_first_stop(
    generated_ids: torch.Tensor,
    stop_ids: Sequence[int],
) -> List[List[int]]:
    stop_id_set = {int(token_id) for token_id in stop_ids}
    trimmed_rows: List[List[int]] = []

    for row in generated_ids.detach().cpu().tolist():
        end_position = len(row)
        for position, token_id in enumerate(row):
            if int(token_id) in stop_id_set:
                end_position = position + 1
                break
        trimmed_rows.append([int(x) for x in row[:end_position]])
    return trimmed_rows


def parse_generation_exact(
    raw_resp: str,
    max_depth: int,
) -> Tuple[Optional[Dict[int, List[str]]], str]:
    text = _strip_trailing_stop_strings(raw_resp)
    lines = text.split("\n") if text else []

    if len(lines) != max_depth:
        return None, f"expected_{max_depth}_lines_got_{len(lines)}"

    parsed: Dict[int, List[str]] = {}

    for current_depth, line in enumerate(lines, start=1):
        prefix = DEPTH_PROBE_TOKENS[current_depth] + " "
        if not line.startswith(prefix):
            return None, f"depth_{current_depth}_prefix_mismatch"

        payload = line[len(prefix) :]
        try:
            labels = json.loads(payload)
        except json.JSONDecodeError:
            return None, f"depth_{current_depth}_invalid_json"

        if not isinstance(labels, list):
            return None, f"depth_{current_depth}_not_json_array"
        if not all(isinstance(label, str) for label in labels):
            return None, f"depth_{current_depth}_contains_non_string"

        parsed[current_depth] = list(labels)

    return parsed, "ok"


def validate_parsed_prediction(
    parsed_layers: Optional[Dict[int, List[str]]],
    label_space: dict,
) -> dict:
    if parsed_layers is None:
        return {
            "pred_ids": set(),
            "pred_labels": set(),
            "parse_ok": False,
            "all_labels_valid": False,
            "hierarchy_closed": False,
            "invalid_for_micro": True,
            "duplicate_count": 0,
            "unknown_labels": [],
            "wrong_depth_labels": [],
            "whitespace_error_labels": [],
            "order_error_depths": [],
            "empty_prediction": True,
            "missing_top_prediction": True,
        }

    label2id = label_space["label2id"]
    depth = label_space["depth"]
    max_depth = int(label_space["max_depth"])

    pred_labels: Set[str] = set()
    duplicate_count = 0
    unknown_labels = []
    wrong_depth_labels = []
    whitespace_error_labels = []
    order_error_depths = []

    for current_depth in range(1, max_depth + 1):
        layer_labels = parsed_layers[current_depth]
        local_seen = set()

        can_check_hierarchical_order = all(
            label in label2id and int(depth[label]) == current_depth
            for label in layer_labels
        )
        if can_check_hierarchical_order:
            previous_depth_labels = (
                parsed_layers[current_depth - 1]
                if current_depth > 1
                else []
            )
            expected_order = sort_depth_labels_by_parent_order(
                labels=layer_labels,
                current_depth=current_depth,
                previous_depth_labels=previous_depth_labels,
                label_space=label_space,
            )
            if layer_labels != expected_order:
                order_error_depths.append(current_depth)

        for label in layer_labels:
            if label != label.strip():
                whitespace_error_labels.append(label)

            if not label:
                unknown_labels.append(label)
                continue

            if label in local_seen or label in pred_labels:
                duplicate_count += 1
            local_seen.add(label)

            if label not in label2id:
                unknown_labels.append(label)
                continue

            pred_labels.add(label)

            if int(depth[label]) != current_depth:
                wrong_depth_labels.append(
                    {
                        "label": label,
                        "predicted_depth": current_depth,
                        "true_depth": int(depth[label]),
                    }
                )

    empty_prediction = len(pred_labels) == 0
    missing_top_prediction = len(parsed_layers.get(1, [])) == 0

    structural_and_label_valid = (
        duplicate_count == 0
        and not unknown_labels
        and not wrong_depth_labels
        and not whitespace_error_labels
        and not order_error_depths
        and not empty_prediction
        and not missing_top_prediction
    )

    hierarchy_closed = (
        _is_ancestor_closed(pred_labels, label_space)
        if structural_and_label_valid
        else False
    )

    pred_ids = {label2id[label] for label in pred_labels}
    invalid_for_micro = not (
        structural_and_label_valid and hierarchy_closed
    )

    return {
        "pred_ids": pred_ids,
        "pred_labels": pred_labels,
        "parse_ok": True,
        "all_labels_valid": structural_and_label_valid,
        "hierarchy_closed": hierarchy_closed,
        "invalid_for_micro": invalid_for_micro,
        "duplicate_count": duplicate_count,
        "unknown_labels": unknown_labels,
        "wrong_depth_labels": wrong_depth_labels,
        "whitespace_error_labels": whitespace_error_labels,
        "order_error_depths": order_error_depths,
        "empty_prediction": empty_prediction,
        "missing_top_prediction": missing_top_prediction,
    }


# ======================================================================
# 6. 与训练完全一致的多标签指标
# ======================================================================
def _sets_to_indicator(
    id_sets: Sequence[Set[int]],
    num_labels: int,
) -> np.ndarray:
    matrix = np.zeros((len(id_sets), num_labels), dtype=np.int32)
    for row, label_ids in enumerate(id_sets):
        for label_id in label_ids:
            if label_id < 0 or label_id >= num_labels:
                raise ValueError(
                    f"❌ multilabel ID 越界: {label_id}, num_labels={num_labels}"
                )
            matrix[row, label_id] = 1
    return matrix


def compute_multilabel_metric_bundle(
    true_id_sets: Sequence[Set[int]],
    pred_id_sets: Sequence[Set[int]],
    invalid_flags: Sequence[bool],
    label_space: dict,
) -> dict:
    if not (
        len(true_id_sets) == len(pred_id_sets) == len(invalid_flags)
    ):
        raise ValueError("❌ multilabel metric 输入长度不一致。")
    if len(true_id_sets) == 0:
        raise ValueError("❌ multilabel metric 输入不能为空。")

    num_labels = int(label_space["num_labels"])
    y_true = _sets_to_indicator(true_id_sets, num_labels)
    y_pred = _sets_to_indicator(pred_id_sets, num_labels)

    global_micro = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    present_label_ids = np.flatnonzero(y_true.sum(axis=0) > 0)
    if len(present_label_ids) == 0:
        raise RuntimeError("❌ 当前 test split 没有任何真实标签。")

    global_macro = f1_score(
        y_true[:, present_label_ids],
        y_pred[:, present_label_ids],
        average="macro",
        zero_division=0,
    )

    global_macro_all_labels = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    global_samples = f1_score(
        y_true,
        y_pred,
        average="samples",
        zero_division=0,
    )

    invalid_true = np.zeros((len(true_id_sets), 1), dtype=np.int32)
    invalid_pred = np.asarray(invalid_flags, dtype=np.int32).reshape(-1, 1)
    y_true_strict = np.concatenate([y_true, invalid_true], axis=1)
    y_pred_strict = np.concatenate([y_pred, invalid_pred], axis=1)

    strict_micro = f1_score(
        y_true_strict,
        y_pred_strict,
        average="micro",
        zero_division=0,
    )
    strict_samples = f1_score(
        y_true_strict,
        y_pred_strict,
        average="samples",
        zero_division=0,
    )

    metrics = {
        "global_micro": float(global_micro),
        "global_macro": float(global_macro),
        "global_macro_all_labels": float(global_macro_all_labels),
        "global_samples": float(global_samples),
        "present_label_count": int(len(present_label_ids)),
        "strict_micro": float(strict_micro),
        "strict_samples": float(strict_samples),
    }

    for current_depth in range(1, label_space["max_depth"] + 1):
        depth_ids = label_space["label_ids_by_depth"][current_depth]
        if not depth_ids:
            metrics[f"depth_{current_depth}_micro"] = 0.0
            metrics[f"depth_{current_depth}_macro"] = 0.0
            metrics[f"depth_{current_depth}_present_labels"] = 0
            continue

        true_depth = y_true[:, depth_ids]
        pred_depth = y_pred[:, depth_ids]

        depth_micro = f1_score(
            true_depth,
            pred_depth,
            average="micro",
            zero_division=0,
        )
        present_depth_columns = np.flatnonzero(
            true_depth.sum(axis=0) > 0
        )
        if len(present_depth_columns) == 0:
            depth_macro = 0.0
        else:
            depth_macro = f1_score(
                true_depth[:, present_depth_columns],
                pred_depth[:, present_depth_columns],
                average="macro",
                zero_division=0,
            )

        metrics[f"depth_{current_depth}_micro"] = float(depth_micro)
        metrics[f"depth_{current_depth}_macro"] = float(depth_macro)
        metrics[f"depth_{current_depth}_present_labels"] = int(
            len(present_depth_columns)
        )

    return metrics


# ======================================================================
# 7. 测试 Prompt 预处理 + 安全生成
# ======================================================================
def build_test_prompts(
    test_data: List[dict],
    tokenizer,
    label_space: dict,
    system_prompt: str,
    max_seq_length: int,
    reserve_tokens: int,
    token_ids: Dict[str, object],
) -> List[dict]:
    max_depth = int(label_space["max_depth"])
    depth_ids = token_ids["depth"]
    prepared = []
    truncated_count = 0

    for idx, item in enumerate(
        tqdm(test_data, desc="构造 TEST Prompt", leave=False)
    ):
        text, labels, label_ids = get_clean_sample_fields(
            item,
            "TEST",
            idx,
        )[:3]

        raw_body_len = len(tokenizer.encode(text, add_special_tokens=False))
        truncated_text = truncate_text_for_prompt(
            text,
            tokenizer,
            max_seq_length=max_seq_length,
            reserve_tokens=reserve_tokens,
        )
        truncated_body_len = len(
            tokenizer.encode(truncated_text, add_special_tokens=False)
        )
        if truncated_body_len < raw_body_len:
            truncated_count += 1

        messages = build_prompt_messages(
            text=truncated_text,
            system_prompt=system_prompt,
            max_depth=max_depth,
        )
        prompt_text, prompt_ids = encode_chat_prompt_consistently(
            tokenizer=tokenizer,
            messages=messages,
            split_name="TEST",
            source_index=idx,
        )

        if len(prompt_ids) > max_seq_length:
            raise ValueError(
                f"❌ TEST 第 {idx} 条 Prompt 超过 max_seq_length: "
                f"{len(prompt_ids)} > {max_seq_length}"
            )

        if prompt_ids.count(int(token_ids["ctx_start"])) != 1:
            raise RuntimeError(
                f"❌ TEST 第 {idx} 条 CTX_START 出现次数异常。"
            )
        if prompt_ids.count(int(token_ids["ctx_end"])) != 1:
            raise RuntimeError(
                f"❌ TEST 第 {idx} 条 CTX_END 出现次数异常。"
            )

        for d in range(1, max_depth + 1):
            current_id = int(depth_ids[d])
            if prompt_ids.count(current_id) != 1:
                raise RuntimeError(
                    f"❌ TEST 第 {idx} 条 Prompt 中 D{d} token "
                    "应恰好出现 1 次。"
                )

        prepared.append(
            {
                "source_index": int(idx),
                "prompt": prompt_text,
                "prompt_ids": prompt_ids,
                "prompt_len": int(len(prompt_ids)),
                "raw_body_len": int(raw_body_len),
                "truncated_body_len": int(truncated_body_len),
                "true_label_ids": set(int(x) for x in label_ids),
                "true_labels": list(labels),
            }
        )

    # 只按模型输入 Prompt 长度排序，减少 padding；不使用 test 真值安排 batch。
    prepared.sort(key=lambda x: x["prompt_len"])

    print("\n" + "=" * 78)
    print("✅ TEST Prompt 构造完成")
    print(f"样本数: {len(prepared)}")
    print(f"正文发生 2048-token 截断: {truncated_count}/{len(prepared)}")
    print(f"最短 Prompt: {min(x['prompt_len'] for x in prepared)} tokens")
    print(f"最长 Prompt: {max(x['prompt_len'] for x in prepared)} tokens")
    print("说明：只按 Prompt 长度排序，测试真值不参与模型输入或 batch 分组。")
    print("=" * 78 + "\n")

    return prepared


def _manual_left_pad_prompt_ids(
    batch: List[dict],
    pad_token_id: int,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(item["prompt_ids"]) for item in batch)
    batch_size = len(batch)

    input_ids = torch.full(
        (batch_size, max_len),
        fill_value=int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
    )

    for row_idx, item in enumerate(batch):
        ids = torch.tensor(item["prompt_ids"], dtype=torch.long)
        current_len = int(ids.numel())
        input_ids[row_idx, max_len - current_len :] = ids
        attention_mask[row_idx, max_len - current_len :] = 1

    return (
        input_ids.to(device, non_blocking=False),
        attention_mask.to(device, non_blocking=False),
    )


def _generate_attempt_without_leaking_oom(
    model,
    tokenizer,
    batch: List[dict],
    device,
    stop_ids: Sequence[int],
    max_new_tokens: int,
) -> Optional[List[str]]:
    """
    单次生成尝试。

    OOM 在本函数内部被吞掉；函数返回后异常对象、调用栈和失败张量引用都
    离开作用域，再由外层执行更小 batch 重试，避免 OOM 递归累积显存。
    """
    input_ids = None
    attention_mask = None
    output_ids = None
    generated_ids = None
    trimmed_generated_ids = None

    try:
        input_ids, attention_mask = _manual_left_pad_prompt_ids(
            batch=batch,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
        )

        input_len = int(input_ids.shape[1])

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                eos_token_id=list(stop_ids),
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                use_cache=True,
            )

        generated_ids = output_ids[:, input_len:]
        trimmed_generated_ids = trim_generated_at_first_stop(
            generated_ids=generated_ids,
            stop_ids=stop_ids,
        )

        raw_texts = [
            tokenizer.decode(
                row_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for row_ids in trimmed_generated_ids
        ]
        return raw_texts

    except RuntimeError as exc:
        if not is_cuda_oom_error(exc):
            raise
        return None

    finally:
        if input_ids is not None:
            del input_ids
        if attention_mask is not None:
            del attention_mask
        if output_ids is not None:
            del output_ids
        if generated_ids is not None:
            del generated_ids
        if trimmed_generated_ids is not None:
            del trimmed_generated_ids


def generate_with_safe_batch_fallback(
    model,
    tokenizer,
    batch: List[dict],
    device,
    stop_ids: Sequence[int],
    max_new_tokens: int,
) -> List[str]:
    """
    迭代式 OOM 回退：B -> B/2 -> ... -> 1。

    每次 OOM 尝试都先完整退出 _generate_attempt_without_leaking_oom，
    再清理缓存后处理更小 batch，不在异常 traceback 内递归。
    """
    outputs: List[Optional[str]] = [None] * len(batch)
    pending: List[Tuple[int, List[dict]]] = [(0, batch)]

    while pending:
        offset, current_batch = pending.pop()
        raw_texts = _generate_attempt_without_leaking_oom(
            model=model,
            tokenizer=tokenizer,
            batch=current_batch,
            device=device,
            stop_ids=stop_ids,
            max_new_tokens=max_new_tokens,
        )

        if raw_texts is not None:
            for local_idx, raw_text in enumerate(raw_texts):
                outputs[offset + local_idx] = raw_text
            continue

        failed_size = len(current_batch)
        clear_cuda_memory(
            f"TEST 生成 batch={failed_size} OOM，失败尝试已退出"
        )

        if failed_size <= 1:
            raise RuntimeError(
                "❌ TEST 生成在 batch=1 时仍然 CUDA OOM。"
                "请检查是否有其他 GPU 进程，或进一步降低序列/生成上限。"
            )

        split_at = failed_size // 2
        left_batch = current_batch[:split_at]
        right_batch = current_batch[split_at:]

        print(
            f"⚠️ TEST 生成 batch={failed_size} 显存不足，"
            f"自动按 {len(left_batch)} + {len(right_batch)} 重试。"
        )

        # 栈后进先出：先压右，再压左，以保持处理顺序自然。
        pending.append((offset + split_at, right_batch))
        pending.append((offset, left_batch))

    if any(text is None for text in outputs):
        raise RuntimeError("❌ OOM 回退结束后仍有样本缺少生成结果。")

    return [str(text) for text in outputs]


# ======================================================================
# 8. 模型加载
# ======================================================================
def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def restore_full_input_output_weights(base_model, checkpoint_path: str) -> None:
    """恢复蒸馏训练时单独保存的完整 input embedding / lm_head。"""
    weight_path = os.path.join(checkpoint_path, FULL_IO_WEIGHTS_NAME)
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            "❌ 当前 model_path 被识别为 PEFT checkpoint，但缺少完整 IO 权重: "
            f"{weight_path}"
        )

    payload = _torch_load_cpu(weight_path)
    if "input_embeddings" not in payload or "actually_tied" not in payload:
        raise KeyError(f"❌ {weight_path} 内容不完整。")

    saved_input = payload["input_embeddings"]
    saved_tied = bool(payload["actually_tied"])

    input_weight = base_model.get_input_embeddings().weight
    output_module = base_model.get_output_embeddings()
    if output_module is None:
        raise RuntimeError("❌ Base model 缺少 lm_head。")

    if tuple(saved_input.shape) != tuple(input_weight.shape):
        raise RuntimeError(
            "❌ 保存的 input embedding 与 base model 形状不一致: "
            f"saved={tuple(saved_input.shape)}, base={tuple(input_weight.shape)}"
        )

    if saved_tied:
        base_model.config.tie_word_embeddings = True
        base_model.tie_weights()
        input_weight = base_model.get_input_embeddings().weight
        output_weight = base_model.get_output_embeddings().weight
        if input_weight.data_ptr() != output_weight.data_ptr():
            raise RuntimeError("❌ 无法恢复训练时共享的 input/lm_head。")
        with torch.no_grad():
            input_weight.copy_(saved_input.to(dtype=input_weight.dtype))
    else:
        saved_output = payload.get("output_embeddings")
        if saved_output is None:
            raise KeyError("❌ 训练时 input/lm_head 未共享，但缺少 output_embeddings。")

        if input_weight.data_ptr() == output_module.weight.data_ptr():
            new_head = torch.nn.Linear(
                output_module.in_features,
                output_module.out_features,
                bias=(output_module.bias is not None),
                device=output_module.weight.device,
                dtype=output_module.weight.dtype,
            )
            base_model.set_output_embeddings(new_head)
            output_module = base_model.get_output_embeddings()

        if tuple(saved_output.shape) != tuple(output_module.weight.shape):
            raise RuntimeError(
                "❌ 保存的 lm_head 与 base model 形状不一致: "
                f"saved={tuple(saved_output.shape)}, base={tuple(output_module.weight.shape)}"
            )

        with torch.no_grad():
            input_weight.copy_(saved_input.to(dtype=input_weight.dtype))
            output_module.weight.copy_(saved_output.to(dtype=output_module.weight.dtype))
        base_model.config.tie_word_embeddings = False

    print(f"✅ 已恢复蒸馏训练时完整 input embedding/lm_head: {weight_path}")
    print(f"训练时实际共享 input/lm_head: {saved_tied}")


def load_distilled_model_and_tokenizer(args):
    """
    支持两种蒸馏产物：
      1) best_kd_model / final_model：完整 merged model；
      2) best_kd_checkpoint / best_macro_checkpoint：PEFT + full IO checkpoint。

    Align/Hier 的 hidden_projector 只参与训练辅助损失，生成测试不需要加载。
    """
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"❌ 蒸馏模型目录不存在: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError("❌ tokenizer 没有可用 pad_token_id。")
    tokenizer.padding_side = "left"

    model_dtype = torch.bfloat16 if args.bf16 else torch.float32
    adapter_config = os.path.join(args.model_path, "adapter_config.json")
    is_peft_checkpoint = os.path.isfile(adapter_config)

    print("\n" + "=" * 78)
    if is_peft_checkpoint:
        print("📦 检测到 NYT 蒸馏 PEFT checkpoint + 完整 IO 权重")
        print(f"Base model: {args.base_model_path}")
        print(f"Checkpoint: {args.model_path}")
        if not os.path.isdir(args.base_model_path):
            raise FileNotFoundError(f"❌ base_model_path 不存在: {args.base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=model_dtype,
            attn_implementation=args.attn_implementation,
            device_map=None,
        )
        if base_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            raise RuntimeError(
                "❌ Base model embedding 行数与蒸馏 tokenizer 长度不一致。"
            )
        restore_full_input_output_weights(base_model, args.model_path)
        model = PeftModel.from_pretrained(
            base_model,
            args.model_path,
            is_trainable=False,
        )
        artifact_type = "peft_checkpoint"
    else:
        print("📦 检测到 NYT 蒸馏完整 merged model")
        print(f"Model: {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=model_dtype,
            attn_implementation=args.attn_implementation,
            device_map=None,
        )
        artifact_type = "merged_model"

    if not torch.cuda.is_available():
        raise RuntimeError("❌ 当前测试脚本要求 CUDA GPU。")
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.use_cache = True

    input_rows = int(model.get_input_embeddings().weight.shape[0])
    output_module = model.get_output_embeddings()
    if output_module is None:
        raise RuntimeError("❌ 测试模型缺少 lm_head。")
    output_rows = int(output_module.weight.shape[0])
    if input_rows != len(tokenizer) or output_rows != len(tokenizer):
        raise RuntimeError(
            "❌ tokenizer 与模型词表大小不一致: "
            f"tokenizer={len(tokenizer)}, input={input_rows}, output={output_rows}"
        )

    print(f"Artifact type: {artifact_type}")
    print(f"dtype: {model_dtype}")
    print(f"attention: {args.attn_implementation}")
    print(f"device: {device}")
    print("✅ NYT 蒸馏模型加载完成；测试阶段不读取 Teacher cache，不计算 KD/Align/Hier。")
    print("=" * 78 + "\n")
    return model, tokenizer, device, artifact_type


# ======================================================================
# 9. 正式测试
# ======================================================================
def run_test(args) -> None:
    set_all_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    label_space = load_and_validate_label_space(
        label_space_path=args.label_space_path,
        expected_num_labels=args.expected_num_labels,
        expected_max_depth=args.expected_max_depth,
    )

    test_data = validate_clean_split(
        data_path=args.test_data_path,
        split_name="TEST",
        label_space=label_space,
    )

    model, tokenizer, device, model_artifact_type = load_distilled_model_and_tokenizer(args)

    token_ids = validate_atomic_reserved_tokens(
        tokenizer,
        expected_max_depth=args.expected_max_depth,
    )

    test_prompts = build_test_prompts(
        test_data=test_data,
        tokenizer=tokenizer,
        label_space=label_space,
        system_prompt=args.system_prompt,
        max_seq_length=args.max_seq_length,
        reserve_tokens=args.reserve_tokens,
        token_ids=token_ids,
    )

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is None or eot_id == tokenizer.unk_token_id:
        eot_id = tokenizer.eos_token_id
    if eot_id is None:
        raise RuntimeError("❌ tokenizer 缺少 EOT/EOS stop token。")

    stop_ids: List[int] = []
    for stop_id in [tokenizer.eos_token_id, int(eot_id)]:
        if stop_id is not None and int(stop_id) not in stop_ids:
            stop_ids.append(int(stop_id))

    max_depth = int(label_space["max_depth"])

    true_id_sets: List[Set[int]] = []
    pred_id_sets: List[Set[int]] = []
    invalid_flags: List[bool] = []
    result_rows: List[dict] = []

    exact_parse_count = 0
    valid_output_count = 0
    hierarchy_closed_count = 0
    exact_match_count = 0
    total_unknown_labels = 0
    total_wrong_depth_labels = 0
    total_duplicate_labels = 0
    total_whitespace_error_labels = 0
    total_order_error_depths = 0
    empty_prediction_count = 0
    missing_top_prediction_count = 0

    clear_cuda_memory("正式 TEST 生成前")

    total_batches = math.ceil(len(test_prompts) / args.eval_batch_size)
    progress = tqdm(total=total_batches, desc="NYT Distill TEST")

    for start in range(0, len(test_prompts), args.eval_batch_size):
        batch = test_prompts[start : start + args.eval_batch_size]

        raw_texts = generate_with_safe_batch_fallback(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            device=device,
            stop_ids=stop_ids,
            max_new_tokens=args.max_new_tokens,
        )

        for raw_resp, item in zip(raw_texts, batch):
            parsed_layers, parse_reason = parse_generation_exact(
                raw_resp,
                max_depth=max_depth,
            )
            prediction = validate_parsed_prediction(
                parsed_layers,
                label_space,
            )

            if prediction["parse_ok"]:
                exact_parse_count += 1
            if prediction["all_labels_valid"]:
                valid_output_count += 1
            if (
                prediction["all_labels_valid"]
                and prediction["hierarchy_closed"]
            ):
                hierarchy_closed_count += 1

            true_ids = set(item["true_label_ids"])
            pred_ids = set(prediction["pred_ids"])
            invalid = bool(prediction["invalid_for_micro"])

            if not invalid and pred_ids == true_ids:
                exact_match_count += 1

            total_unknown_labels += len(prediction["unknown_labels"])
            total_wrong_depth_labels += len(prediction["wrong_depth_labels"])
            total_duplicate_labels += int(prediction["duplicate_count"])
            total_whitespace_error_labels += len(
                prediction["whitespace_error_labels"]
            )
            total_order_error_depths += len(prediction["order_error_depths"])
            empty_prediction_count += int(prediction["empty_prediction"])
            missing_top_prediction_count += int(
                prediction["missing_top_prediction"]
            )

            true_id_sets.append(true_ids)
            pred_id_sets.append(pred_ids)
            invalid_flags.append(invalid)

            parsed_for_json = None
            if parsed_layers is not None:
                parsed_for_json = {
                    str(d): parsed_layers[d]
                    for d in range(1, max_depth + 1)
                }

            pred_labels_sorted = sorted(
                prediction["pred_labels"],
                key=label_space["label2id"].get,
            )

            result_rows.append(
                {
                    "source_index": int(item["source_index"]),
                    "prompt_len": int(item["prompt_len"]),
                    "raw_body_len": int(item["raw_body_len"]),
                    "truncated_body_len": int(item["truncated_body_len"]),
                    "true_labels": list(item["true_labels"]),
                    "true_label_ids": sorted(int(x) for x in true_ids),
                    "raw_generation": raw_resp,
                    "parse_reason": parse_reason,
                    "parsed_layers": parsed_for_json,
                    "pred_labels": pred_labels_sorted,
                    "pred_label_ids": sorted(int(x) for x in pred_ids),
                    "parse_ok": bool(prediction["parse_ok"]),
                    "all_labels_valid": bool(prediction["all_labels_valid"]),
                    "hierarchy_closed": bool(prediction["hierarchy_closed"]),
                    "invalid_for_micro": invalid,
                    "exact_set_match": bool((not invalid) and pred_ids == true_ids),
                    "duplicate_count": int(prediction["duplicate_count"]),
                    "unknown_labels": prediction["unknown_labels"],
                    "wrong_depth_labels": prediction["wrong_depth_labels"],
                    "whitespace_error_labels": prediction[
                        "whitespace_error_labels"
                    ],
                    "order_error_depths": prediction["order_error_depths"],
                    "empty_prediction": bool(prediction["empty_prediction"]),
                    "missing_top_prediction": bool(
                        prediction["missing_top_prediction"]
                    ),
                }
            )

        progress.update(1)

    progress.close()
    clear_cuda_memory("TEST 生成完成")

    metrics = compute_multilabel_metric_bundle(
        true_id_sets=true_id_sets,
        pred_id_sets=pred_id_sets,
        invalid_flags=invalid_flags,
        label_space=label_space,
    )

    sample_count = len(true_id_sets)
    if sample_count != len(test_data):
        raise RuntimeError(
            f"❌ 测试样本数不一致: 预测={sample_count}, 数据={len(test_data)}"
        )

    exact_parse_rate = exact_parse_count / sample_count
    valid_output_rate = valid_output_count / sample_count
    hierarchy_closed_rate = hierarchy_closed_count / sample_count
    exact_match_rate = exact_match_count / sample_count
    invalid_output_rate = sum(invalid_flags) / sample_count

    summary = {
        "model_path": args.model_path,
        "model_artifact_type": model_artifact_type,
        "base_model_path": args.base_model_path,
        "test_data_path": args.test_data_path,
        "label_space_path": args.label_space_path,
        "sample_count": sample_count,
        "max_seq_length": args.max_seq_length,
        "reserve_tokens": args.reserve_tokens,
        "body_token_limit": args.max_seq_length - args.reserve_tokens,
        "max_new_tokens": args.max_new_tokens,
        "eval_batch_size_configured": args.eval_batch_size,
        "generation": {
            "do_sample": False,
            "use_cache": True,
            "attention_implementation": args.attn_implementation,
        },
        "metrics": metrics,
        "generation_quality": {
            "exact_parse_rate": float(exact_parse_rate),
            "valid_output_rate": float(valid_output_rate),
            "ancestor_closure_rate": float(hierarchy_closed_rate),
            "exact_set_match_rate": float(exact_match_rate),
            "invalid_output_rate": float(invalid_output_rate),
            "unknown_label_count": int(total_unknown_labels),
            "wrong_depth_label_count": int(total_wrong_depth_labels),
            "duplicate_label_count": int(total_duplicate_labels),
            "whitespace_error_label_count": int(
                total_whitespace_error_labels
            ),
            "canonical_order_error_depth_count": int(
                total_order_error_depths
            ),
            "empty_prediction_count": int(empty_prediction_count),
            "missing_top_prediction_count": int(
                missing_top_prediction_count
            ),
        },
        "metric_rule": {
            "global_micro": "complete 166-label space",
            "global_macro": "only labels present in TEST ground truth",
            "global_macro_all_labels": "all 166 labels diagnostic",
            "strict_metrics": "one invalid-output sentinel false-positive dimension",
        },
    }

    # 恢复原始 TEST 顺序后保存逐样本结果。
    result_rows.sort(key=lambda x: x["source_index"])

    predictions_path = os.path.join(
        args.output_dir,
        "distill_test_predictions.jsonl",
    )
    with open(predictions_path, "w", encoding="utf-8") as f:
        for row in result_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics_path = os.path.join(
        args.output_dir,
        "distill_test_metrics.json",
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    text_path = os.path.join(
        args.output_dir,
        "distill_test_summary.txt",
    )
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("NYT Distilled Student Test Summary\n")
        f.write("=" * 78 + "\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Samples: {sample_count}\n")
        f.write(f"Global Micro-F1: {metrics['global_micro']:.6f}\n")
        f.write(f"Global Macro-F1(Present): {metrics['global_macro']:.6f}\n")
        f.write(
            f"Global Macro-F1-All166: "
            f"{metrics['global_macro_all_labels']:.6f}\n"
        )
        f.write(f"Samples-F1: {metrics['global_samples']:.6f}\n")
        f.write(f"Strict Micro-F1: {metrics['strict_micro']:.6f}\n")
        f.write(f"Strict Samples-F1: {metrics['strict_samples']:.6f}\n")
        f.write(f"Present labels: {metrics['present_label_count']}/166\n")
        for d in range(1, max_depth + 1):
            f.write(
                f"D{d} Micro/Macro: "
                f"{metrics[f'depth_{d}_micro']:.6f}/"
                f"{metrics[f'depth_{d}_macro']:.6f} | "
                f"Present={metrics[f'depth_{d}_present_labels']}\n"
            )
        f.write(f"Exact Parse Rate: {exact_parse_rate:.6f}\n")
        f.write(f"Valid Output Rate: {valid_output_rate:.6f}\n")
        f.write(f"Ancestor Closure Rate: {hierarchy_closed_rate:.6f}\n")
        f.write(f"Exact Set Match Rate: {exact_match_rate:.6f}\n")
        f.write(f"Invalid Output Rate: {invalid_output_rate:.6f}\n")

    print("\n" + "=" * 90)
    print("🎯 NYT 蒸馏学生模型 TEST 结果")
    print(f"Model: {args.model_path}")
    print(f"TEST samples: {sample_count}")
    print(
        f"Global Micro-F1: {metrics['global_micro']:.4f} | "
        f"Global Macro-F1(Present): {metrics['global_macro']:.4f} | "
        f"Macro-All166: {metrics['global_macro_all_labels']:.4f} | "
        f"Samples-F1: {metrics['global_samples']:.4f}"
    )
    print(
        f"Strict Micro-F1: {metrics['strict_micro']:.4f} | "
        f"Strict Samples-F1: {metrics['strict_samples']:.4f} | "
        f"PresentLabels: {metrics['present_label_count']}/166"
    )

    for d in range(1, max_depth + 1):
        print(
            f"D{d} Micro/Macro: "
            f"{metrics[f'depth_{d}_micro']:.4f}/"
            f"{metrics[f'depth_{d}_macro']:.4f} | "
            f"PresentLabels={metrics[f'depth_{d}_present_labels']}"
        )

    print(
        f"ExactParse={exact_parse_rate:.4f} | "
        f"ValidOutput={valid_output_rate:.4f} | "
        f"AncestorClosed={hierarchy_closed_rate:.4f} | "
        f"ExactSetMatch={exact_match_rate:.4f} | "
        f"InvalidOutput={invalid_output_rate:.4f}"
    )
    print(
        f"UnknownLabels={total_unknown_labels} | "
        f"WrongDepthLabels={total_wrong_depth_labels} | "
        f"DuplicateLabels={total_duplicate_labels} | "
        f"WhitespaceErrors={total_whitespace_error_labels} | "
        f"OrderErrorDepths={total_order_error_depths}"
    )
    print(
        f"EmptyPredictions={empty_prediction_count} | "
        f"MissingTopPredictions={missing_top_prediction_count}"
    )
    print(f"\n💾 Metrics:     {metrics_path}")
    print(f"💾 Predictions: {predictions_path}")
    print(f"💾 Summary:     {text_path}")
    print("=" * 90 + "\n")


# ======================================================================
# 10. 参数：集中在最底部，便于直接修改
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
            "NYT Distilled Student Generative Test: exact training prompt + strict "
            "8-depth JSON parsing + present-class Macro-F1"
        )
    )

    # ================= 路径参数 =================
    # 默认测试 Traditional logits KD 按验证集 Global Macro-F1 选出的 best merged model。
    # Align/Hier/Full 只需运行时改 --model_path / --output_dir，不需要改测试逻辑。
    parser.add_argument(
        "--model_path",
        type=str,
        default=(
            "/data1/lzp/code/hierarchical_distill/"
            "checkpoints_nyt_logits_kd/best_kd_model"
        ),
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/data1/lzp/code/llama-3.2_1b/",
        help="仅当 model_path 是 PEFT checkpoint 时使用。",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="/data1/lzp/code/data/nyt/test_formatted.jsonl",
    )
    parser.add_argument(
        "--label_space_path",
        type=str,
        default="/data1/lzp/code/data/nyt/label_space.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "/data1/lzp/code/hierarchical_distill/checkpoints_nyt_logits_kd/"
            "test_results_best_kd_model/"
        ),
    )

    # ================= 数据与随机种子 =================
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_num_labels", type=int, default=166)
    parser.add_argument("--expected_max_depth", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=3072)
    parser.add_argument("--reserve_tokens", type=int, default=1024)

    # ================= 生成测试参数 =================
    # 1B 蒸馏学生独立测试显存明显低于训练中验证；默认 16。
    # 若个别长 batch OOM，会安全自动二分到 8/4/2/1。
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=520)

    # ================= 性能参数 =================
    # CUDA_VISIBLE_DEVICES 已屏蔽物理卡号时，这里通常保持 cuda:0。
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")

    # ================= 与训练完全相同的 system prompt =================
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are a specialized classifier for the NYT hierarchical "
            "multi-label taxonomy. Follow the required eight-line JSON-array "
            "output format exactly. Do not produce explanations or any extra "
            "conversational text."
        ),
    )

    args = parser.parse_args()
    run_test(args)
