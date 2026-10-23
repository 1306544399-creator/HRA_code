import os

# 在导入 torch 前设置，尽量保证可复现性。
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import hashlib
import json
import random
import re
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
# 0. 与最新学生训练脚本完全一致的固定定义
# ======================================================================
CTX_START_TOKEN = "<|reserved_special_token_110|>"
CTX_END_TOKEN = "<|reserved_special_token_111|>"
PROBE_TOKEN_L1 = "<|reserved_special_token_112|>"
PROBE_TOKEN_L2 = "<|reserved_special_token_113|>"

FULL_IO_WEIGHTS_NAME = "full_input_output_embeddings.pt"
RESERVED_TOKEN_INFO_NAME = "reserved_token_info.json"
RESOLVED_LABEL_SPACE_NAME = "resolved_label_space.json"

REQUIRED_CLEAN_FIELDS = (
    "text",
    "l1_label",
    "l2_label",
    "l1_id",
    "l2_id",
)

# 与训练脚本中的 tokenizer.chat_template 完全一致。
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


# ======================================================================
# 2. 最新 Prompt / Completion：必须与训练脚本一致
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


# ======================================================================
# 3. Llama 预留特殊 Token 严格检查
# ======================================================================
def validate_atomic_reserved_tokens(tokenizer) -> Dict[str, int]:
    token_map = {
        "ctx_start": CTX_START_TOKEN,
        "ctx_end": CTX_END_TOKEN,
        "l1": PROBE_TOKEN_L1,
        "l2": PROBE_TOKEN_L2,
    }

    vocab = tokenizer.get_vocab()
    token_ids: Dict[str, int] = {}

    print("\n" + "=" * 72)
    print("🔎 验证最新训练使用的 Llama 预留特殊 Token...")

    for role, token in token_map.items():
        if token not in vocab:
            raise RuntimeError(
                f"❌ {token} 不在测试 tokenizer 词表中。"
                "请确认加载的是最新训练保存的 tokenizer。"
            )

        ids = tokenizer.encode(token, add_special_tokens=False)
        converted_id = tokenizer.convert_tokens_to_ids(token)

        if len(ids) != 1:
            raise RuntimeError(f"❌ {token} 不是单 token，实际 ids={ids}")

        token_id = ids[0]
        if converted_id != token_id:
            raise RuntimeError(
                f"❌ {token} 的 encode ID 与 convert ID 不一致："
                f"encode={token_id}, convert={converted_id}"
            )

        if token_id == tokenizer.unk_token_id:
            raise RuntimeError(f"❌ {token} 被编码为 unk_token。")

        token_ids[role] = token_id
        print(f"✅ {role}: {token} -> ID={token_id}")

    if len(set(token_ids.values())) != 4:
        raise RuntimeError(f"❌ 四个预留 Token ID 未两两区分：{token_ids}")

    print("✅ 测试阶段不 add_tokens、不 resize_token_embeddings")
    print("=" * 72 + "\n")
    return token_ids


def find_metadata_file(model_path: str, filename: str) -> Optional[str]:
    model_abs = os.path.abspath(model_path)
    candidates = [
        os.path.join(model_abs, filename),
        os.path.join(os.path.dirname(model_abs), filename),
        os.path.join(os.path.dirname(os.path.dirname(model_abs)), filename),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def check_reserved_token_metadata(
    model_path: str,
    tokenizer,
    token_ids: Dict[str, int],
) -> None:
    path = find_metadata_file(model_path, RESERVED_TOKEN_INFO_NAME)

    if path is None:
        print(
            "⚠️ 未找到 reserved_token_info.json；"
            "已通过 tokenizer 原子性与 Prompt 布局检查继续测试。"
        )
        return

    with open(path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    expected = {
        "ctx_start_token": CTX_START_TOKEN,
        "ctx_start_token_id": token_ids["ctx_start"],
        "ctx_end_token": CTX_END_TOKEN,
        "ctx_end_token_id": token_ids["ctx_end"],
        "lvl1_token": PROBE_TOKEN_L1,
        "lvl1_token_id": token_ids["l1"],
        "lvl2_token": PROBE_TOKEN_L2,
        "lvl2_token_id": token_ids["l2"],
        "tokenizer_length": len(tokenizer),
    }

    for key, expected_value in expected.items():
        if key not in saved:
            raise KeyError(f"❌ {path} 缺少字段 {key!r}。")
        actual = saved[key]
        if isinstance(expected_value, int):
            actual = int(actual)
        if actual != expected_value:
            raise RuntimeError(
                f"❌ 训练/测试特殊 Token 元信息不一致：{key}，"
                f"saved={actual!r}, runtime={expected_value!r}"
            )

    print(f"✅ 训练/测试预留 Token 元信息完全一致: {path}")


def check_prompt_layout(
    tokenizer,
    system_prompt: str,
    token_ids: Dict[str, int],
) -> None:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_content("probe check text")},
    ]

    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    expected = {
        token_ids["ctx_start"]: 1,
        token_ids["ctx_end"]: 1,
        token_ids["l1"]: 1,
        token_ids["l2"]: 1,
    }

    for token_id, expected_count in expected.items():
        actual_count = prompt_ids.count(token_id)
        if actual_count != expected_count:
            raise RuntimeError(
                f"❌ 测试 Prompt 中 token_id={token_id} 应出现 "
                f"{expected_count} 次，实际 {actual_count} 次。"
            )

    completion_ids = tokenizer.encode(
        build_completion_text("CS", "Machine learning"),
        add_special_tokens=False,
    )
    full_ids = prompt_ids + completion_ids

    l1_positions = [
        i for i, token_id in enumerate(full_ids) if token_id == token_ids["l1"]
    ]
    l2_positions = [
        i for i, token_id in enumerate(full_ids) if token_id == token_ids["l2"]
    ]

    if len(l1_positions) != 2 or len(l2_positions) != 2:
        raise RuntimeError(
            "❌ Prompt + Completion 中层级 Token 次数异常："
            f"L1={l1_positions}, L2={l2_positions}"
        )

    if not (
        l1_positions[0] < l2_positions[0] < l1_positions[1] < l2_positions[1]
    ):
        raise RuntimeError(
            "❌ Prompt + Completion 层级 Token 顺序异常："
            f"L1={l1_positions}, L2={l2_positions}"
        )

    print("\n" + "=" * 72)
    print("✅ 测试 Prompt / Completion 与最新训练布局一致")
    print(f"L1 Token 两次位置: {l1_positions}")
    print(f"L2 Token 两次位置: {l2_positions}")
    print("生成时 Prompt 中各出现一次；模型应在答案中再次生成一次。")
    print("=" * 72 + "\n")


# ======================================================================
# 4. 严格读取 clean JSON / JSONL
# ======================================================================
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
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"❌ JSONL 解析失败: {path}, line={line_no}: {e}"
                ) from e

            if not isinstance(item, dict):
                raise TypeError(
                    f"❌ JSONL 每行必须是 dict: {path}, line={line_no}"
                )
            rows.append(item)
        return rows

    if not isinstance(data, list):
        raise TypeError(f"❌ JSON 顶层必须是 list: {path}")

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"❌ JSON 第 {idx} 条必须是 dict。")

    return data


def get_clean_sample_fields(
    item: dict,
    split_name: str,
    idx: int,
) -> Tuple[str, str, str, int, int]:
    missing = [k for k in REQUIRED_CLEAN_FIELDS if k not in item]
    if missing:
        raise KeyError(
            f"{split_name} 第 {idx} 条缺少字段: {missing}。"
            f"必须包含 {list(REQUIRED_CLEAN_FIELDS)}"
        )

    text = item["text"]
    l1_label = item["l1_label"]
    l2_label = item["l2_label"]
    l1_id = item["l1_id"]
    l2_id = item["l2_id"]

    if not isinstance(text, str) or not text.strip():
        raise TypeError(f"{split_name} 第 {idx} 条 text 必须是非空 str。")
    if not isinstance(l1_label, str) or not l1_label.strip():
        raise TypeError(f"{split_name} 第 {idx} 条 l1_label 必须是非空 str。")
    if not isinstance(l2_label, str) or not l2_label.strip():
        raise TypeError(f"{split_name} 第 {idx} 条 l2_label 必须是非空 str。")

    if isinstance(l1_id, bool) or not isinstance(l1_id, int):
        raise TypeError(f"{split_name} 第 {idx} 条 l1_id 必须是 int。")
    if isinstance(l2_id, bool) or not isinstance(l2_id, int):
        raise TypeError(f"{split_name} 第 {idx} 条 l2_id 必须是 int。")

    return (
        text.strip(),
        l1_label.strip(),
        l2_label.strip(),
        l1_id,
        l2_id,
    )


# ======================================================================
# 5. taxonomy + label2id 严格一致性检查
# ======================================================================
def parse_wos_taxonomy(taxonomy_path: str):
    if not os.path.isfile(taxonomy_path):
        raise FileNotFoundError(f"❌ taxonomy 文件不存在: {taxonomy_path}")

    parent_to_children: Dict[str, List[str]] = {}

    with open(taxonomy_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.rstrip("\n\r")
            if not line.strip():
                continue

            parts = [part.strip() for part in line.split("\t") if part.strip()]
            if len(parts) < 2:
                raise ValueError(
                    f"❌ taxonomy 第 {line_num} 行格式异常: {line!r}"
                )

            parent = parts[0]
            children = parts[1:]

            if parent in parent_to_children:
                raise ValueError(f"❌ taxonomy 中父节点重复: {parent!r}")
            if len(children) != len(set(children)):
                raise ValueError(
                    f"❌ taxonomy 第 {line_num} 行存在重复子节点: {parent!r}"
                )

            parent_to_children[parent] = children

    if "Root" not in parent_to_children:
        raise ValueError("❌ taxonomy 中未找到 Root 行。")

    l1_labels = parent_to_children["Root"]
    if len(l1_labels) != len(set(l1_labels)):
        raise ValueError("❌ taxonomy Root 行中存在重复 L1。")

    l2_labels: List[str] = []
    l2_to_l1: Dict[str, str] = {}

    for l1 in l1_labels:
        if l1 not in parent_to_children:
            raise ValueError(f"❌ taxonomy 中 L1 {l1!r} 没有子节点行。")

        for l2 in parent_to_children[l1]:
            if l2 in l2_to_l1:
                raise ValueError(
                    f"❌ L2 {l2!r} 重复归属于 "
                    f"{l2_to_l1[l2]!r} 和 {l1!r}"
                )
            l2_to_l1[l2] = l1
            l2_labels.append(l2)

    expected_parents = {"Root"} | set(l1_labels)
    unexpected_parents = set(parent_to_children) - expected_parents
    if unexpected_parents:
        raise ValueError(
            "❌ taxonomy 存在 WOS 两级结构之外的额外父节点: "
            f"{sorted(unexpected_parents)[:20]}"
        )

    overlap = set(l1_labels) & set(l2_labels)
    if overlap:
        raise ValueError(f"❌ L1/L2 标签名重叠: {sorted(overlap)[:20]}")

    return l1_labels, l2_labels, l2_to_l1


def _normalize_label2id_mapping(raw_mapping, name: str) -> Dict[str, int]:
    if not isinstance(raw_mapping, dict):
        raise TypeError(f"❌ {name} 必须是 dict。")

    normalized: Dict[str, int] = {}
    for label, idx in raw_mapping.items():
        label = str(label)
        if isinstance(idx, bool):
            raise TypeError(f"❌ {name}[{label!r}] 不能是 bool。")
        try:
            idx_int = int(idx)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"❌ {name}[{label!r}] 不是合法整数 ID: {idx!r}"
            ) from e
        normalized[label] = idx_int

    return normalized


def load_and_validate_label_space(
    dict_path: str,
    taxonomy_path: str,
    expected_num_l1: int = 7,
    expected_num_l2: int = 134,
):
    if not os.path.isfile(dict_path):
        raise FileNotFoundError(f"❌ label2id 文件不存在: {dict_path}")

    with open(dict_path, "r", encoding="utf-8") as f:
        raw_dict = json.load(f)

    if not isinstance(raw_dict, dict):
        raise TypeError("❌ label2id.json 顶层必须是 dict。")

    for key in ("l1_label2id", "l2_label2id"):
        if key not in raw_dict:
            raise KeyError(f"❌ label2id.json 缺少字段 {key!r}。")

    l1_label2id = _normalize_label2id_mapping(
        raw_dict["l1_label2id"], "l1_label2id"
    )
    l2_label2id = _normalize_label2id_mapping(
        raw_dict["l2_label2id"], "l2_label2id"
    )

    tax_l1, tax_l2, l2_to_l1 = parse_wos_taxonomy(taxonomy_path)

    if expected_num_l1 > 0 and len(tax_l1) != expected_num_l1:
        raise ValueError(
            f"❌ taxonomy L1 数量异常: {len(tax_l1)} != {expected_num_l1}"
        )
    if expected_num_l2 > 0 and len(tax_l2) != expected_num_l2:
        raise ValueError(
            f"❌ taxonomy L2 数量异常: {len(tax_l2)} != {expected_num_l2}"
        )

    expected_l1 = {label: idx for idx, label in enumerate(tax_l1)}
    expected_l2 = {label: idx for idx, label in enumerate(tax_l2)}

    if l1_label2id != expected_l1:
        raise ValueError("❌ label2id.json 的 L1 映射与 taxonomy 顺序不一致。")
    if l2_label2id != expected_l2:
        raise ValueError("❌ label2id.json 的 L2 映射与 taxonomy 顺序不一致。")

    if "l1_id2label" in raw_dict:
        expected = {str(v): k for k, v in l1_label2id.items()}
        actual = {str(k): str(v) for k, v in raw_dict["l1_id2label"].items()}
        if actual != expected:
            raise ValueError("❌ l1_id2label 与 l1_label2id 不互逆。")

    if "l2_id2label" in raw_dict:
        expected = {str(v): k for k, v in l2_label2id.items()}
        actual = {str(k): str(v) for k, v in raw_dict["l2_id2label"].items()}
        if actual != expected:
            raise ValueError("❌ l2_id2label 与 l2_label2id 不互逆。")

    print("\n" + "=" * 72)
    print("🏷️ 测试标签空间严格校验完成")
    print(f"L1: {len(l1_label2id)} 类")
    print(f"L2: {len(l2_label2id)} 类")
    print("✅ label2id.json 与 taxonomy 标签及 ID 顺序完全一致")
    print("=" * 72 + "\n")

    return l1_label2id, l2_label2id, l2_to_l1


def check_training_label_space_consistency(
    model_path: str,
    l1_label2id: Dict[str, int],
    l2_label2id: Dict[str, int],
    l2_to_l1: Dict[str, str],
) -> None:
    path = find_metadata_file(model_path, RESOLVED_LABEL_SPACE_NAME)

    if path is None:
        print(
            "⚠️ 未找到训练时 resolved_label_space.json；"
            "当前测试仍会严格校验 taxonomy + label2id.json。"
        )
        return

    with open(path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    saved_l1 = _normalize_label2id_mapping(
        saved.get("l1_label2id"), "saved l1_label2id"
    )
    saved_l2 = _normalize_label2id_mapping(
        saved.get("l2_label2id"), "saved l2_label2id"
    )

    if saved_l1 != l1_label2id:
        raise ValueError("❌ 当前测试 L1 标签空间与训练时不一致。")
    if saved_l2 != l2_label2id:
        raise ValueError("❌ 当前测试 L2 标签空间与训练时不一致。")

    if "l2_parent" in saved:
        saved_parent = {str(k): str(v) for k, v in saved["l2_parent"].items()}
        if saved_parent != l2_to_l1:
            raise ValueError("❌ 当前测试 taxonomy 父子关系与训练时不一致。")

    print(f"✅ 当前测试标签空间与训练时完全一致: {path}")


def validate_clean_split(
    data_path: str,
    split_name: str,
    l1_label2id: Dict[str, int],
    l2_label2id: Dict[str, int],
    l2_to_l1: Dict[str, str],
) -> List[dict]:
    data = load_json_or_jsonl(data_path)
    seen_l1 = set()
    seen_l2 = set()

    for idx, item in enumerate(tqdm(data, desc=f"校验{split_name}", leave=False)):
        text, l1_label, l2_label, l1_id, l2_id = get_clean_sample_fields(
            item, split_name, idx
        )
        del text

        if l1_label not in l1_label2id:
            raise ValueError(f"❌ {split_name} 第 {idx} 条未知 L1: {l1_label!r}")
        if l2_label not in l2_label2id:
            raise ValueError(f"❌ {split_name} 第 {idx} 条未知 L2: {l2_label!r}")

        if l1_id != l1_label2id[l1_label]:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条 L1 ID 错位: "
                f"label={l1_label!r}, sample_id={l1_id}, "
                f"expected={l1_label2id[l1_label]}"
            )

        if l2_id != l2_label2id[l2_label]:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条 L2 ID 错位: "
                f"label={l2_label!r}, sample_id={l2_id}, "
                f"expected={l2_label2id[l2_label]}"
            )

        expected_parent = l2_to_l1.get(l2_label)
        if expected_parent != l1_label:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条父子关系错误: "
                f"L1={l1_label!r}, L2={l2_label!r}, "
                f"taxonomy parent={expected_parent!r}"
            )

        seen_l1.add(l1_label)
        seen_l2.add(l2_label)

    print("\n" + "=" * 72)
    print(f"✅ {split_name} 严格校验通过")
    print(f"样本数: {len(data)}")
    print(f"实际出现 L1: {len(seen_l1)} / {len(l1_label2id)}")
    print(f"实际出现 L2: {len(seen_l2)} / {len(l2_label2id)}")
    print("缺失字段、未知标签、ID 错位、父子关系错误、静默跳过: 0")
    print("=" * 72 + "\n")
    return data


# ======================================================================
# 6. 截断、生成解析、指标
# ======================================================================
def truncate_text_for_prompt(
    text: str,
    tokenizer,
    max_seq_length: int = 2048,
    reserve_tokens: int = 300,
) -> str:
    max_text_len = max_seq_length - reserve_tokens
    if max_text_len <= 0:
        raise ValueError(
            f"❌ max_seq_length={max_seq_length} "
            f"必须大于 reserve_tokens={reserve_tokens}"
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


def parse_generation_exact(raw_resp: str) -> Tuple[str, str]:
    """
    严格解析最新 Completion：
      <|reserved_special_token_112|> L1
      <|reserved_special_token_113|> L2

    注意：必须对 skip_special_tokens=False 的原始输出调用本函数。
    """
    text = _strip_trailing_stop_strings(raw_resp)

    pattern = (
        rf"^{re.escape(PROBE_TOKEN_L1)}[ \t]+([^\n]+)\n"
        rf"{re.escape(PROBE_TOKEN_L2)}[ \t]+([^\n]+)$"
    )
    match = re.match(pattern, text)

    if not match:
        return "", ""

    return match.group(1).strip(), match.group(2).strip()


def compute_metric_bundle(
    true_l1: List[int],
    true_l2: List[int],
    pred_l1: List[int],
    pred_l2: List[int],
    num_l1: int,
    num_l2: int,
):
    """
    与 Traditional KD 训练阶段 validation 完全一致的指标口径：
      - L1 使用局部 ID [0, num_l1)；
      - Global 中 L2 ID = local_l2_id + num_l1；
      - 非法预测保留 sentinel 并完整计错；
      - Macro-F1 固定在完整 WOS 任务标签空间上计算，
        不把 9999 / 99999 当作额外伪类别。

    注意：这与原学生测试脚本的“present-only Macro”不同。
    为了让 Student SFT、Traditional KD、后续 Align/Hier 消融严格可比，
    测试必须复用蒸馏训练 validation 的同一口径。
    """
    if not (
        len(true_l1)
        == len(true_l2)
        == len(pred_l1)
        == len(pred_l2)
    ):
        raise ValueError("❌ metric 输入长度不一致。")

    if not true_l1:
        raise ValueError("❌ metric 输入不能为空。")

    if num_l1 <= 0 or num_l2 <= 0:
        raise ValueError(
            f"❌ num_l1/num_l2 必须为正数，"
            f"实际 num_l1={num_l1}, num_l2={num_l2}"
        )

    invalid_true_l1 = [x for x in true_l1 if x < 0 or x >= num_l1]
    invalid_true_l2 = [x for x in true_l2 if x < 0 or x >= num_l2]
    if invalid_true_l1 or invalid_true_l2:
        raise ValueError(
            "❌ metric 真实标签超出合法范围："
            f"invalid_true_l1={invalid_true_l1[:20]}, "
            f"invalid_true_l2={invalid_true_l2[:20]}"
        )

    all_true: List[int] = []
    all_pred: List[int] = []

    for t1, t2, p1, p2 in zip(true_l1, true_l2, pred_l1, pred_l2):
        all_true.extend([t1, t2 + num_l1])
        all_pred.extend([
            p1,
            p2 + num_l1 if p2 != 9999 else 99999,
        ])

    # Micro：非法预测 sentinel 保留在 y_pred 中，因此完整计错。
    global_micro = f1_score(
        all_true,
        all_pred,
        average="micro",
        zero_division=0,
    )

    # Macro：严格固定到 WOS 完整任务标签空间。
    global_valid_labels = list(range(num_l1 + num_l2))
    global_macro = f1_score(
        all_true,
        all_pred,
        labels=global_valid_labels,
        average="macro",
        zero_division=0,
    )

    l1_micro = f1_score(
        true_l1,
        pred_l1,
        average="micro",
        zero_division=0,
    )
    l1_macro = f1_score(
        true_l1,
        pred_l1,
        labels=list(range(num_l1)),
        average="macro",
        zero_division=0,
    )

    l2_micro = f1_score(
        true_l2,
        pred_l2,
        average="micro",
        zero_division=0,
    )
    l2_macro = f1_score(
        true_l2,
        pred_l2,
        labels=list(range(num_l2)),
        average="macro",
        zero_division=0,
    )

    return {
        "global_micro": float(global_micro),
        "global_macro": float(global_macro),
        "l1_micro": float(l1_micro),
        "l1_macro": float(l1_macro),
        "l2_micro": float(l2_micro),
        "l2_macro": float(l2_macro),
    }


# ======================================================================
# 7. 测试正文截断扫描与 Prompt 构造
# ======================================================================
def scan_text_truncation(
    data: List[dict],
    tokenizer,
    max_seq_length: int,
    reserve_tokens: int,
) -> None:
    max_text_len = max_seq_length - reserve_tokens
    if max_text_len <= 0:
        raise ValueError("❌ 正文 token 预算必须大于 0。")

    truncated_count = 0
    max_original_tokens = 0
    first_kept_text = None
    first_dropped_text = None

    for idx, item in enumerate(tqdm(data, desc="检查测试集截断比例")):
        text, _, _, _, _ = get_clean_sample_fields(item, "TEST", idx)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        max_original_tokens = max(max_original_tokens, len(token_ids))

        if len(token_ids) > max_text_len:
            truncated_count += 1
            if first_kept_text is None:
                first_kept_text = tokenizer.decode(
                    token_ids[:max_text_len],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                first_dropped_text = tokenizer.decode(
                    token_ids[max_text_len:],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )

    total = len(data)
    ratio = truncated_count / max(1, total)

    print("\n" + "=" * 72)
    print(f"正文阈值: {max_text_len} tokens")
    print(
        f"测试集共 {total} 条，正文被截断 {truncated_count} 条 "
        f"({ratio:.2%})"
    )
    print(f"最长原始正文: {max_original_tokens} tokens")

    if truncated_count > 0:
        print("第一条被截断样本，保留文本末尾 200 字符：")
        print("......" + first_kept_text[-200:])
        print("真正丢弃文本开头 200 字符：")
        print(first_dropped_text[:200] + "......")

    print("=" * 72 + "\n")



def _token_ids_sha256(token_ids: List[int]) -> str:
    """为一段 token ID 生成稳定签名，便于学生/教师/蒸馏三条流程核对。"""
    payload = ",".join(str(int(token_id)) for token_id in token_ids)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _text_sha256(text: str) -> str:
    """文本级签名；即使教师/学生 tokenizer 不同，也可核对截断正文是否相同。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _decode_ids(tokenizer, token_ids: List[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _find_first_non_whitespace_token_offset(tokenizer, token_ids: List[int]):
    """
    返回正文边界内部第一个开始贡献非空白字符的 token 相对位置。
    仅用于打印调试，不参与正文 span 定义。
    """
    for end in range(1, len(token_ids) + 1):
        decoded_prefix = _decode_ids(tokenizer, token_ids[:end])
        if decoded_prefix.strip():
            return end - 1
    return None


def _print_token_window(
    tokenizer,
    prompt_ids: List[int],
    left: int,
    right: int,
    title: str,
) -> None:
    left = max(0, left)
    right = min(len(prompt_ids), right)

    print(title)
    print("index | token_id | tokenizer token | 单 token decode")
    for index in range(left, right):
        token_id = int(prompt_ids[index])
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        decoded_piece = _decode_ids(tokenizer, [token_id])
        print(
            f"{index:5d} | {token_id:8d} | "
            f"{token_str!r} | {decoded_piece!r}"
        )


def audit_prompt_body_span(
    prompt_ids: List[int],
    truncated_text: str,
    tokenizer,
    token_ids: Dict[str, int],
    source_index: int,
    print_detail: bool,
    preview_chars: int,
) -> dict:
    """
    按 token ID 严格提取正文边界内部的 span，并核对：
      1. CTX_START / CTX_END 各出现一次且顺序正确；
      2. 两个边界 token 之间的 token ID，必须与编码
         ``"\\n" + truncated_text + "\\n"`` 完全一致；
      3. 解码后去掉模板添加的一对换行，必须逐字符等于截断正文；
      4. 保存正文 token ID 的 SHA-256，便于学生/教师/蒸馏逐样本比对。

    由于正文边界是 Llama 原生单 token，若第 2 项通过，就不存在旧
    ``<CTX>`` 被拆分后遗留 ``>`` 并与正文首 token 黏连的问题。
    """
    start_positions = [
        index
        for index, token_id in enumerate(prompt_ids)
        if token_id == token_ids["ctx_start"]
    ]
    end_positions = [
        index
        for index, token_id in enumerate(prompt_ids)
        if token_id == token_ids["ctx_end"]
    ]

    if len(start_positions) != 1 or len(end_positions) != 1:
        raise RuntimeError(
            f"❌ TEST 第 {source_index} 条正文边界次数异常："
            f"CTX_START={start_positions}, CTX_END={end_positions}"
        )

    start_pos = start_positions[0]
    end_pos = end_positions[0]
    if end_pos <= start_pos + 1:
        raise RuntimeError(
            f"❌ TEST 第 {source_index} 条正文边界顺序错误或正文为空："
            f"start={start_pos}, end={end_pos}"
        )

    # 这是后续教师/学生/蒸馏应共同采用的权威正文边界 span。
    between_ids = [int(x) for x in prompt_ids[start_pos + 1 : end_pos]]

    # build_user_content 在正文前后各放置一个换行。
    expected_between_text = f"\n{truncated_text}\n"
    expected_between_ids = tokenizer.encode(
        expected_between_text,
        add_special_tokens=False,
    )
    expected_between_ids = [int(x) for x in expected_between_ids]

    token_exact_match = between_ids == expected_between_ids
    decoded_between_text = _decode_ids(tokenizer, between_ids)
    decoded_segment_match = decoded_between_text == expected_between_text

    # 去除模板额外加入的一前一后两个换行，得到人可读的正文。
    if decoded_between_text.startswith("\n") and decoded_between_text.endswith("\n"):
        extracted_text = decoded_between_text[1:-1]
    else:
        extracted_text = decoded_between_text

    extracted_text_match = extracted_text == truncated_text

    if not token_exact_match or not decoded_segment_match or not extracted_text_match:
        raise RuntimeError(
            f"❌ TEST 第 {source_index} 条正文 token span 与构造正文不一致。\n"
            f"token_exact_match={token_exact_match}\n"
            f"decoded_segment_match={decoded_segment_match}\n"
            f"extracted_text_match={extracted_text_match}\n"
            f"start_pos={start_pos}, end_pos={end_pos}\n"
            f"actual_ids_head={between_ids[:20]}\n"
            f"expected_ids_head={expected_between_ids[:20]}\n"
            f"actual_text_head={decoded_between_text[:200]!r}\n"
            f"expected_text_head={expected_between_text[:200]!r}"
        )

    first_non_ws_offset = _find_first_non_whitespace_token_offset(
        tokenizer,
        between_ids,
    )
    first_non_ws_abs_pos = (
        start_pos + 1 + first_non_ws_offset
        if first_non_ws_offset is not None
        else None
    )

    first_after_start_id = between_ids[0]
    last_before_end_id = between_ids[-1]

    record = {
        "source_index": int(source_index),
        "ctx_start_position": int(start_pos),
        "ctx_end_position": int(end_pos),
        "prompt_token_count": int(len(prompt_ids)),
        "between_boundary_token_count": int(len(between_ids)),
        "expected_between_token_count": int(len(expected_between_ids)),
        "token_exact_match": bool(token_exact_match),
        "decoded_segment_match": bool(decoded_segment_match),
        "extracted_text_match": bool(extracted_text_match),
        "unexpected_gt_or_boundary_fusion": False,
        "body_text_sha256": _text_sha256(truncated_text),
        "body_token_sha256": _token_ids_sha256(between_ids),
        "first_token_after_ctx_start_id": int(first_after_start_id),
        "first_token_after_ctx_start": tokenizer.convert_ids_to_tokens(
            first_after_start_id
        ),
        "first_token_after_ctx_start_decoded": _decode_ids(
            tokenizer,
            [first_after_start_id],
        ),
        "last_token_before_ctx_end_id": int(last_before_end_id),
        "last_token_before_ctx_end": tokenizer.convert_ids_to_tokens(
            last_before_end_id
        ),
        "last_token_before_ctx_end_decoded": _decode_ids(
            tokenizer,
            [last_before_end_id],
        ),
        "first_non_whitespace_token_absolute_position": (
            int(first_non_ws_abs_pos)
            if first_non_ws_abs_pos is not None
            else None
        ),
    }

    if print_detail:
        preview_chars = max(50, int(preview_chars))
        head = extracted_text[:preview_chars]
        tail = extracted_text[-preview_chars:]

        print("\n" + "=" * 88)
        print(f"🔬 TEST 正文 token span 审计：source_index={source_index}")
        print(f"CTX_START 位置/ID: {start_pos}/{token_ids['ctx_start']}")
        print(f"CTX_END   位置/ID: {end_pos}/{token_ids['ctx_end']}")
        print(f"边界内部 token 数: {len(between_ids)}")
        print(f"token ID 与预期完全一致: {token_exact_match}")
        print(f"解码正文与截断正文逐字符一致: {extracted_text_match}")
        print("出现额外 '>' 或边界与正文首 token 黏连: False")
        print(f"正文文本 SHA-256: {record['body_text_sha256']}")
        print(f"正文 token SHA-256: {record['body_token_sha256']}")

        print("\n【截断后正文开头】")
        print(repr(head))
        print("【截断后正文结尾】")
        print(repr(tail))
        print("【边界内部完整解码的开头（包含模板换行）】")
        print(repr(decoded_between_text[:preview_chars]))

        _print_token_window(
            tokenizer,
            prompt_ids,
            start_pos - 1,
            start_pos + 8,
            "\n【CTX_START 附近 token】",
        )
        _print_token_window(
            tokenizer,
            prompt_ids,
            end_pos - 7,
            end_pos + 2,
            "\n【CTX_END 附近 token】",
        )

        if first_non_ws_abs_pos is not None:
            first_id = int(prompt_ids[first_non_ws_abs_pos])
            print("\n【正文第一个贡献非空白字符的 token】")
            print(f"absolute_position: {first_non_ws_abs_pos}")
            print(f"token_id: {first_id}")
            print(f"tokenizer token: {tokenizer.convert_ids_to_tokens(first_id)!r}")
            print(f"single-token decode: {_decode_ids(tokenizer, [first_id])!r}")
            context_left = max(start_pos, first_non_ws_abs_pos - 2)
            context_right = min(len(prompt_ids), first_non_ws_abs_pos + 3)
            print(
                "从 CTX_START 到正文首个非空白 token 的联合解码: "
                f"{_decode_ids(tokenizer, prompt_ids[start_pos:first_non_ws_abs_pos + 1])!r}"
            )
            _print_token_window(
                tokenizer,
                prompt_ids,
                context_left,
                context_right,
                "正文首 token 附近：",
            )

        print("=" * 88 + "\n")

    return record


def prepare_test_prompts(
    test_data: List[dict],
    tokenizer,
    token_ids: Dict[str, int],
    max_seq_length: int,
    reserve_tokens: int,
    system_prompt: str,
    output_dir: str,
    body_audit_samples: int,
    body_audit_preview_chars: int,
) -> Tuple[List[dict], str]:
    test_prompts = []
    body_audit_records = []

    for idx, item in enumerate(
        tqdm(test_data, desc="准备测试 Prompt", leave=False)
    ):
        text, l1_str, l2_str, l1_id, l2_id = get_clean_sample_fields(
            item, "TEST", idx
        )

        text = truncate_text_for_prompt(
            text,
            tokenizer,
            max_seq_length=max_seq_length,
            reserve_tokens=reserve_tokens,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_content(text)},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 已经由 chat template 写入 BOS/EOT 等特殊 token，重新编码时必须
        # add_special_tokens=False，避免再次添加 BOS。
        prompt_ids = tokenizer(
            prompt,
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]

        # 与训练 Dataset 使用的 apply_chat_template(tokenize=True) 逐 token 核对。
        # 两条入口必须完全一致，否则学生训练、测试和后续蒸馏会产生偏移。
        prompt_ids_direct = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if prompt_ids != prompt_ids_direct:
            mismatch_at = next(
                (
                    i
                    for i, (left_id, right_id) in enumerate(
                        zip(prompt_ids, prompt_ids_direct)
                    )
                    if left_id != right_id
                ),
                min(len(prompt_ids), len(prompt_ids_direct)),
            )
            raise RuntimeError(
                f"❌ TEST 第 {idx} 条 chat template 两种编码入口不一致。\n"
                f"render-then-tokenize长度={len(prompt_ids)}, "
                f"direct长度={len(prompt_ids_direct)}\n"
                f"首个不一致位置={mismatch_at}\n"
                "请检查是否在已经模板化的字符串上重复添加了 BOS/EOS。"
            )

        if len(prompt_ids) > max_seq_length:
            raise ValueError(
                f"❌ TEST 第 {idx} 条 Prompt 超过 max_seq_length: "
                f"{len(prompt_ids)} > {max_seq_length}"
            )

        expected_counts = {
            token_ids["ctx_start"]: 1,
            token_ids["ctx_end"]: 1,
            token_ids["l1"]: 1,
            token_ids["l2"]: 1,
        }
        for token_id, expected_count in expected_counts.items():
            actual_count = prompt_ids.count(token_id)
            if actual_count != expected_count:
                raise RuntimeError(
                    f"❌ TEST 第 {idx} 条 Prompt 中 token_id={token_id} "
                    f"应出现 {expected_count} 次，实际 {actual_count} 次。"
                )

        body_audit = audit_prompt_body_span(
            prompt_ids=prompt_ids,
            truncated_text=text,
            tokenizer=tokenizer,
            token_ids=token_ids,
            source_index=idx,
            print_detail=(idx < body_audit_samples),
            preview_chars=body_audit_preview_chars,
        )
        body_audit["chat_template_direct_tokenization_match"] = True
        body_audit["prompt_token_sha256"] = _token_ids_sha256(
            [int(x) for x in prompt_ids]
        )
        body_audit_records.append(body_audit)

        test_prompts.append(
            {
                "source_index": idx,
                "prompt": prompt,
                "true_l1_str": l1_str,
                "true_l2_str": l2_str,
                "true_l1": l1_id,
                "true_l2": l2_id,
                "body_text_sha256": body_audit["body_text_sha256"],
                "body_token_sha256": body_audit["body_token_sha256"],
                "prompt_token_sha256": body_audit["prompt_token_sha256"],
                "body_token_count": body_audit["between_boundary_token_count"],
                "ctx_start_position": body_audit["ctx_start_position"],
                "ctx_end_position": body_audit["ctx_end_position"],
            }
        )

    body_audit_path = os.path.join(output_dir, "test_body_span_audit.jsonl")
    with open(body_audit_path, "w", encoding="utf-8") as f:
        for record in body_audit_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ 测试生成 Prompt 准备完成：{len(test_prompts)} 条")
    print("✅ 所有样本正文边界 token span 与截断正文完全一致")
    print("✅ chat template 直接编码与模板字符串重编码逐 token 完全一致")
    print("✅ 未发现额外 '>' 或边界与正文首 token 黏连")
    print(f"💾 正文 token span 审计日志: {body_audit_path}")
    return test_prompts, body_audit_path


# ======================================================================
# 8. 同时支持两种最新训练产物：merged model 或 PEFT checkpoint
# ======================================================================
def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def restore_full_input_output_weights(base_model, checkpoint_path: str) -> None:
    weight_path = os.path.join(checkpoint_path, FULL_IO_WEIGHTS_NAME)
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            "❌ 当前路径是 PEFT checkpoint，但缺少训练时额外保存的完整 "
            f"input/lm_head 权重: {weight_path}"
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
            "❌ 保存的 input embedding 与 base model 形状不一致："
            f"saved={tuple(saved_input.shape)}, base={tuple(input_weight.shape)}"
        )

    if saved_tied:
        # 训练时共享：恢复共享结构，再复制一份共享权重。
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

        # 若当前 base 意外共享，先创建独立 lm_head。
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
                "❌ 保存的 lm_head 与 base model 形状不一致："
                f"saved={tuple(saved_output.shape)}, "
                f"base={tuple(output_module.weight.shape)}"
            )

        with torch.no_grad():
            input_weight.copy_(saved_input.to(dtype=input_weight.dtype))
            output_module.weight.copy_(
                saved_output.to(dtype=output_module.weight.dtype)
            )

        base_model.config.tie_word_embeddings = False

    print(f"✅ 已恢复训练时完整 input embedding/lm_head: {weight_path}")
    print(f"训练时实际共享: {saved_tied}")


def load_trained_model(
    model_path: str,
    base_model_path: str,
    tokenizer,
    device,
    torch_dtype,
    attn_implementation: str,
):
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_peft_checkpoint = os.path.isfile(adapter_config_path)

    print("\n" + "=" * 72)
    if is_peft_checkpoint:
        print("📦 检测到 PEFT checkpoint + 完整 input/lm_head 权重")
        print(f"Base model: {base_model_path}")
        print(f"PEFT checkpoint: {model_path}")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=None,
        )

        if base_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            raise RuntimeError(
                "❌ Base model embedding 行数与 tokenizer 长度不一致。"
                "最新方案不新增 token，不应发生 resize。"
            )

        restore_full_input_output_weights(base_model, model_path)

        model = PeftModel.from_pretrained(
            base_model,
            model_path,
            is_trainable=False,
        )
    else:
        print("📦 检测到已导出的完整 merged model")
        print(f"Merged model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=None,
        )

    model.to(device)
    model.eval()

    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None

    input_weight = model.get_input_embeddings().weight
    output_module = model.get_output_embeddings()
    if output_module is None:
        raise RuntimeError("❌ 测试模型缺少 lm_head。")

    input_rows = input_weight.shape[0]
    output_rows = output_module.weight.shape[0]
    actually_tied = input_weight.data_ptr() == output_module.weight.data_ptr()

    print("✅ 测试模型加载完成")
    print(f"Tokenizer 长度: {len(tokenizer)}")
    print(f"Input embedding 行数: {input_rows}")
    print(f"Output vocab 行数: {output_rows}")
    print(f"Input/lm_head 实际共享: {actually_tied}")
    print("=" * 72 + "\n")

    if input_rows != len(tokenizer) or output_rows != len(tokenizer):
        raise RuntimeError("❌ tokenizer 与模型输入/输出词表大小不一致。")

    return model, "peft_checkpoint" if is_peft_checkpoint else "merged_model"


# ======================================================================
# 9. 与 Traditional KD 训练 validation 完全一致的 completion-only Test Loss
# ======================================================================
class GenerativeWOSTestLossDataset(Dataset):
    def __init__(
        self,
        data: List[dict],
        tokenizer,
        token_ids: Dict[str, int],
        max_length: int,
        reserve_tokens: int,
        system_prompt: str,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.token_ids = token_ids
        self.max_length = max_length
        self.reserve_tokens = reserve_tokens
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, l1_str, l2_str, _, _ = get_clean_sample_fields(
            self.data[idx], "TEST_LOSS", idx
        )

        text = truncate_text_for_prompt(
            text,
            self.tokenizer,
            self.max_length,
            self.reserve_tokens,
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": build_user_content(text)},
        ]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        completion_ids = self.tokenizer.encode(
            build_completion_text(l1_str, l2_str),
            add_special_tokens=False,
        )

        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is None or eot_id == self.tokenizer.unk_token_id:
            eot_id = self.tokenizer.eos_token_id
        if eot_id is None:
            raise RuntimeError("❌ tokenizer 缺少 EOT/EOS。")

        completion_ids.append(eot_id)
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids

        expected_counts = {
            self.token_ids["ctx_start"]: 1,
            self.token_ids["ctx_end"]: 1,
            self.token_ids["l1"]: 2,
            self.token_ids["l2"]: 2,
        }
        for token_id, expected_count in expected_counts.items():
            if input_ids.count(token_id) != expected_count:
                raise RuntimeError(
                    f"❌ TEST_LOSS 第 {idx} 条 token_id={token_id} 次数异常。"
                )

        if len(input_ids) > self.max_length:
            raise ValueError(
                f"❌ TEST_LOSS 第 {idx} 条总长度 "
                f"{len(input_ids)} > {self.max_length}"
            )

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


def compute_test_loss(
    model,
    data: List[dict],
    tokenizer,
    token_ids: Dict[str, int],
    device,
    batch_size: int,
    max_seq_length: int,
    reserve_tokens: int,
    system_prompt: str,
) -> float:
    dataset = GenerativeWOSTestLossDataset(
        data=data,
        tokenizer=tokenizer,
        token_ids=token_ids,
        max_length=max_seq_length,
        reserve_tokens=reserve_tokens,
        system_prompt=system_prompt,
    )

    old_padding_side = tokenizer.padding_side
    old_config_use_cache = model.config.use_cache
    old_gen_use_cache = getattr(model.generation_config, "use_cache", None)

    tokenizer.padding_side = "right"
    model.config.use_cache = False
    if old_gen_use_cache is not None:
        model.generation_config.use_cache = False

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
            for batch in tqdm(loader, desc="Computing Test Loss"):
                batch = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                outputs = model(**batch)

                # 与 Traditional KD 训练阶段 validation callback 完全一致：
                # 每个 batch 的 outputs.loss 先乘 batch size，再对样本数求平均。
                # 这里的 Test Loss 只评估学生的 completion-only task CE。
                # 测试阶段不读取 Teacher cache、不恢复 Teacher logits，
                # 也不计算 KD / Align / Hier 辅助训练损失。
                bs = batch["input_ids"].shape[0]
                total_loss += outputs.loss.detach().float().item() * bs
                total_count += bs
    finally:
        tokenizer.padding_side = old_padding_side
        model.config.use_cache = old_config_use_cache
        if old_gen_use_cache is not None:
            model.generation_config.use_cache = old_gen_use_cache

    return total_loss / max(1, total_count)


# ======================================================================
# 10. Generation 停止 Token
# ======================================================================
def build_stop_ids(tokenizer) -> List[int]:
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is None or eot_id == tokenizer.unk_token_id:
        eot_id = tokenizer.eos_token_id

    stop_ids: List[int] = []
    for sid in [tokenizer.eos_token_id, eot_id]:
        if sid is not None and sid not in stop_ids:
            stop_ids.append(sid)

    if not stop_ids:
        raise RuntimeError("❌ generation 没有可用 eos/eot stop id。")

    print("\n" + "=" * 72)
    print("🛑 Generation 停止设置")
    print(f"eos_token_id: {tokenizer.eos_token_id}")
    print(f"eot_id: {eot_id}")
    print(f"stop_ids: {stop_ids}")
    print(f"pad_token_id: {tokenizer.pad_token_id}")
    print("=" * 72 + "\n")
    return stop_ids


# ======================================================================
# 11. Main
# ======================================================================
def main(args) -> None:
    set_all_seeds(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("❌ 当前脚本按 GPU 推理设计，但未检测到 CUDA。")

    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"❌ model_path 不存在: {args.model_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32

    # ------------------------------------------------------------------
    # A. 标签空间与测试集严格校验
    # ------------------------------------------------------------------
    l1_label2id, l2_label2id, l2_to_l1 = load_and_validate_label_space(
        dict_path=args.dict_path,
        taxonomy_path=args.taxonomy_path,
        expected_num_l1=args.expected_num_l1,
        expected_num_l2=args.expected_num_l2,
    )

    check_training_label_space_consistency(
        model_path=args.model_path,
        l1_label2id=l1_label2id,
        l2_label2id=l2_label2id,
        l2_to_l1=l2_to_l1,
    )

    test_data = validate_clean_split(
        data_path=args.test_data_path,
        split_name="TEST",
        l1_label2id=l1_label2id,
        l2_label2id=l2_label2id,
        l2_to_l1=l2_to_l1,
    )

    # ------------------------------------------------------------------
    # B. 加载最新训练保存的 tokenizer，不新增、不 resize
    # ------------------------------------------------------------------
    print(f"📦 从训练产物加载 tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.chat_template = CHAT_TEMPLATE

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError("❌ tokenizer 没有可用 pad_token_id。")

    tokenizer.padding_side = "left"

    token_ids = validate_atomic_reserved_tokens(tokenizer)
    check_reserved_token_metadata(args.model_path, tokenizer, token_ids)
    check_prompt_layout(tokenizer, args.system_prompt, token_ids)

    # ------------------------------------------------------------------
    # C. 测试 Prompt
    # ------------------------------------------------------------------
    scan_text_truncation(
        data=test_data,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        reserve_tokens=args.reserve_tokens,
    )

    test_prompts, body_audit_path = prepare_test_prompts(
        test_data=test_data,
        tokenizer=tokenizer,
        token_ids=token_ids,
        max_seq_length=args.max_seq_length,
        reserve_tokens=args.reserve_tokens,
        system_prompt=args.system_prompt,
        output_dir=args.output_dir,
        body_audit_samples=args.body_audit_samples,
        body_audit_preview_chars=args.body_audit_preview_chars,
    )

    if len(test_prompts) != len(test_data):
        raise RuntimeError("❌ 严格测试不允许静默跳过样本。")

    # ------------------------------------------------------------------
    # D. 加载最新训练模型
    # ------------------------------------------------------------------
    model, model_artifact_type = load_trained_model(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    )

    # ------------------------------------------------------------------
    # E. 可选：与训练 validation callback 一致的 completion-only loss
    # ------------------------------------------------------------------
    test_loss = None
    if args.compute_test_loss:
        test_loss = compute_test_loss(
            model=model,
            data=test_data,
            tokenizer=tokenizer,
            token_ids=token_ids,
            device=device,
            batch_size=args.loss_eval_batch_size,
            max_seq_length=args.max_seq_length,
            reserve_tokens=args.reserve_tokens,
            system_prompt=args.system_prompt,
        )
        print(f"✅ Test completion-only Loss: {test_loss:.6f}")

    stop_ids = build_stop_ids(tokenizer)

    # ------------------------------------------------------------------
    # F. 严格生成、解析与评价
    # ------------------------------------------------------------------
    tokenizer.padding_side = "left"
    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True

    true_l1: List[int] = []
    true_l2: List[int] = []
    pred_l1: List[int] = []
    pred_l2: List[int] = []

    exact_parse_count = 0
    valid_label_pair_count = 0
    hierarchy_consistent_count = 0
    both_levels_correct_count = 0
    l1_correct_count = 0
    l2_correct_count = 0

    detailed_results = []
    print_counter = 0

    print("\n🚀 开始 WOS Offline Traditional Logits KD 学生模型严格测试...")

    with torch.no_grad():
        for start in tqdm(
            range(0, len(test_prompts), args.eval_batch_size),
            desc="Running Strict Test",
        ):
            batch = test_prompts[start : start + args.eval_batch_size]
            prompts = [item["prompt"] for item in batch]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            ).to(device)

            input_len = inputs.input_ids.shape[1]

            output_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                use_cache=True,
            )

            generated_ids = output_ids[:, input_len:]

            # 关键：层级 Token 本身是 special token，解析时不能 skip。
            raw_texts = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            for raw_resp, item in zip(raw_texts, batch):
                p1_str, p2_str = parse_generation_exact(raw_resp)
                exact_parse_valid = bool(p1_str and p2_str)
                if exact_parse_valid:
                    exact_parse_count += 1

                p1_id = l1_label2id.get(p1_str, 9999)
                p2_id = l2_label2id.get(p2_str, 9999)

                valid_pair = p1_id != 9999 and p2_id != 9999
                hierarchy_consistent = False

                if valid_pair:
                    valid_label_pair_count += 1
                    hierarchy_consistent = l2_to_l1.get(p2_str) == p1_str
                    if hierarchy_consistent:
                        hierarchy_consistent_count += 1

                correct_l1 = p1_id == item["true_l1"]
                correct_l2 = p2_id == item["true_l2"]
                both_correct = correct_l1 and correct_l2

                if correct_l1:
                    l1_correct_count += 1
                if correct_l2:
                    l2_correct_count += 1
                if both_correct:
                    both_levels_correct_count += 1

                true_l1.append(item["true_l1"])
                true_l2.append(item["true_l2"])
                pred_l1.append(p1_id)
                pred_l2.append(p2_id)

                if print_counter < args.print_samples:
                    print(
                        f"\n--- 样本 {print_counter + 1} "
                        f"(TEST index={item['source_index']}) ---"
                    )
                    print("【模型原始输出，保留特殊 Token】")
                    print(raw_resp)
                    print(f"【严格提取】L1: {p1_str!r} | L2: {p2_str!r}")
                    print(
                        f"【真实标签】L1: {item['true_l1_str']!r} | "
                        f"L2: {item['true_l2_str']!r}"
                    )
                    print(
                        f"【结果】Parse={exact_parse_valid}, "
                        f"L1正确={correct_l1}, L2正确={correct_l2}, "
                        f"层级一致={hierarchy_consistent}"
                    )
                    print("-" * 72)
                    print_counter += 1

                detailed_results.append(
                    {
                        "source_index": item["source_index"],
                        "true_l1": item["true_l1_str"],
                        "true_l2": item["true_l2_str"],
                        "true_l1_id": item["true_l1"],
                        "true_l2_id": item["true_l2"],
                        "body_text_sha256": item["body_text_sha256"],
                        "body_token_sha256": item["body_token_sha256"],
                        "prompt_token_sha256": item["prompt_token_sha256"],
                        "body_token_count": item["body_token_count"],
                        "ctx_start_position": item["ctx_start_position"],
                        "ctx_end_position": item["ctx_end_position"],
                        "raw_generated_text_with_special_tokens": raw_resp,
                        "extracted_l1_exact": p1_str,
                        "extracted_l2_exact": p2_str,
                        "pred_l1_id": p1_id,
                        "pred_l2_id": p2_id,
                        "exact_parse_valid": exact_parse_valid,
                        "valid_l1_label": p1_id != 9999,
                        "valid_l2_label": p2_id != 9999,
                        "valid_label_pair": valid_pair,
                        "hierarchy_consistent": hierarchy_consistent,
                        "correct_l1": correct_l1,
                        "correct_l2": correct_l2,
                        "both_levels_correct": both_correct,
                    }
                )

    # ------------------------------------------------------------------
    # G. 指标汇总
    # ------------------------------------------------------------------
    metrics = compute_metric_bundle(
        true_l1=true_l1,
        true_l2=true_l2,
        pred_l1=pred_l1,
        pred_l2=pred_l2,
        num_l1=len(l1_label2id),
        num_l2=len(l2_label2id),
    )

    n = max(1, len(true_l1))
    exact_parse_rate = exact_parse_count / n
    valid_label_pair_rate = valid_label_pair_count / n
    hierarchy_consistency_rate = hierarchy_consistent_count / n
    both_levels_correct_rate = both_levels_correct_count / n
    l1_accuracy = l1_correct_count / n
    l2_accuracy = l2_correct_count / n
    invalid_l1_rate = sum(x == 9999 for x in pred_l1) / n
    invalid_l2_rate = sum(x == 9999 for x in pred_l2) / n

    # ------------------------------------------------------------------
    # H. 保存详细生成结果和汇总
    # ------------------------------------------------------------------
    generation_log_path = os.path.join(
        args.output_dir, "offline_traditional_kd_test_generations.json"
    )
    summary_path = os.path.join(args.output_dir, "offline_traditional_kd_test_summary.json")

    with open(generation_log_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

    summary = {
        "model_path": args.model_path,
        "model_artifact_type": model_artifact_type,
        "base_model_path": args.base_model_path,
        "test_data_path": args.test_data_path,
        "taxonomy_path": args.taxonomy_path,
        "dict_path": args.dict_path,
        "tested_samples": len(test_prompts),
        "skipped_samples": 0,
        "body_span_audit_path": body_audit_path,
        "body_span_rule": (
            "token IDs strictly between reserved_special_token_110 and "
            "reserved_special_token_111; this segment must equal encode('\n' + "
            "truncated_text + '\n') exactly"
        ),
        "all_body_spans_exactly_matched": True,
        "chat_template_direct_tokenization_all_matched": True,
        "already_templated_prompts_retokenized_with_add_special_tokens": False,
        "unexpected_gt_or_boundary_fusion_found": False,
        "test_completion_only_loss": test_loss,
        "global_micro_f1": metrics["global_micro"],
        "global_macro_f1": metrics["global_macro"],
        "l1_micro_f1": metrics["l1_micro"],
        "l1_macro_f1": metrics["l1_macro"],
        "l2_micro_f1": metrics["l2_micro"],
        "l2_macro_f1": metrics["l2_macro"],
        "l1_accuracy": l1_accuracy,
        "l2_accuracy": l2_accuracy,
        "exact_parse_rate": exact_parse_rate,
        "valid_label_pair_rate": valid_label_pair_rate,
        "hierarchy_consistency_rate": hierarchy_consistency_rate,
        "both_levels_correct_rate": both_levels_correct_rate,
        "invalid_l1_rate": invalid_l1_rate,
        "invalid_l2_rate": invalid_l2_rate,
        "ctx_start_token": CTX_START_TOKEN,
        "ctx_start_token_id": token_ids["ctx_start"],
        "ctx_end_token": CTX_END_TOKEN,
        "ctx_end_token_id": token_ids["ctx_end"],
        "level_token_l1": PROBE_TOKEN_L1,
        "level_token_l1_id": token_ids["l1"],
        "level_token_l2": PROBE_TOKEN_L2,
        "level_token_l2_id": token_ids["l2"],
        "tokenizer_length": len(tokenizer),
        "parsing_rule": (
            "exactly two lines; preserve reserved special tokens; "
            "no normalization, fuzzy matching, or label correction"
        ),
        "metric_id_rule": (
            "L1 uses local ids; L2 global metric ids = local_l2_id + num_l1"
        ),
        "macro_label_space_rule": (
            "Macro-F1 is computed on the complete fixed WOS task label space, exactly as in Traditional KD validation; "
            "invalid sentinel predictions are errors but are not extra macro classes"
        ),
        "generation": {
            "eval_batch_size": args.eval_batch_size,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "stop_ids": stop_ids,
            "pad_token_id": tokenizer.pad_token_id,
            "use_cache": True,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # I. 最终输出
    # ------------------------------------------------------------------
    print("\n" + "=" * 84)
    print("🎉 WOS Offline Traditional Logits KD 学生模型严格测试结束")
    print("Prompt、预留 Token、截断、生成格式和指标口径均与 Traditional KD 训练一致")
    print("-" * 84)
    if test_loss is not None:
        print(f"Test Completion-only Loss   : {test_loss:.6f}")
    print(f"Global Micro-F1             : {metrics['global_micro']:.4f}")
    print(f"Global Macro-F1             : {metrics['global_macro']:.4f}")
    print(f"L1 Micro-F1                 : {metrics['l1_micro']:.4f}")
    print(f"L1 Macro-F1                 : {metrics['l1_macro']:.4f}")
    print(f"L2 Micro-F1                 : {metrics['l2_micro']:.4f}")
    print(f"L2 Macro-F1                 : {metrics['l2_macro']:.4f}")
    print(f"L1 Accuracy                 : {l1_accuracy:.2%}")
    print(f"L2 Accuracy                 : {l2_accuracy:.2%}")
    print(f"Exact Parse Rate            : {exact_parse_rate:.2%}")
    print(f"Valid Label Pair Rate       : {valid_label_pair_rate:.2%}")
    print(f"Hierarchy Consistency Rate  : {hierarchy_consistency_rate:.2%}")
    print(f"Both Levels Correct Rate    : {both_levels_correct_rate:.2%}")
    print(f"Invalid L1 Rate             : {invalid_l1_rate:.2%}")
    print(f"Invalid L2 Rate             : {invalid_l2_rate:.2%}")
    print(f"Tested Samples              : {len(test_prompts)}")
    print("Skipped Samples             : 0")
    print("=" * 84)
    print(f"💾 完整生成日志: {generation_log_path}")
    print(f"💾 正文 span 审计: {body_audit_path}")
    print(f"💾 测试汇总: {summary_path}")


# ======================================================================
# 12. 参数解析：常改参数集中在最底部
# ======================================================================
def str2bool(v):
    if isinstance(v, bool):
        return v

    value = str(v).lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Strict Test for WOS Offline Traditional Logits KD Student: reserved hierarchy tokens + "
            "full input/output tuning + exact generation metrics"
        )
    )

    # ================= 模型 =================
    # 推荐直接测试 Traditional KD 训练脚本导出的 best_kd_model（完整 merged model）。
    # 也支持改成 best_kd_checkpoint；脚本会自动加载 base + LoRA + full IO。
    parser.add_argument(
        "--model_path",
        type=str,
        default=(
            "/data1/lzp/code/hierarchical_distill/"
            "checkpoints_wos_offline_logits_kd_seed42_lora16/"
            "best_kd_model"
        ),
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/data1/lzp/code/llama-3.2_1b/",
    )

    # ================= 数据与标签空间 =================
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="/data1/lzp/code/data/wos/test_formatted.jsonl",
    )
    parser.add_argument(
        "--taxonomy_path",
        type=str,
        default="/data1/lzp/code/data/wos/wos.taxnomy",
    )
    parser.add_argument(
        "--dict_path",
        type=str,
        default="/data1/lzp/code/data/wos/label2id.json",
    )
    parser.add_argument("--expected_num_l1", type=int, default=7)
    parser.add_argument("--expected_num_l2", type=int, default=134)

    # ================= 与训练一致的序列参数 =================
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--reserve_tokens", type=int, default=300)

    # ================= 生成与 Loss =================
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--loss_eval_batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--compute_test_loss", type=str2bool, default=True)

    # ================= 性能 =================
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")

    # ================= 与最新训练完全一致的 System Prompt =================
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are a specialized academic classifier for the WOS "
            "two-level taxonomy. Follow the required output format exactly. "
            "Do not produce explanations or any extra conversational text."
        ),
    )

    # ================= 正文边界审计 =================
    # 打印前 N 条样本的正文 token 边界；所有样本都会严格核验并保存 hash。
    parser.add_argument("--body_audit_samples", type=int, default=5)
    parser.add_argument("--body_audit_preview_chars", type=int, default=500)

    # ================= 输出 =================
    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "/data1/lzp/code/hierarchical_distill/"
            "checkpoints_wos_offline_logits_kd_seed42_lora16/"
            "test_results_best"
        ),
    )
    parser.add_argument("--print_samples", type=int, default=5)

    args = parser.parse_args()
    main(args)
