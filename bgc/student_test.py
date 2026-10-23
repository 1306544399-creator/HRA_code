import os

# 在导入 torch 前设置，尽量保证可复现性。
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# 减少长序列生成时的大块显存碎片；必须在 import torch 前设置。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import gc
import json
import random
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import f1_score, precision_score, recall_score
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
# 1. 与训练完全一致的特殊 Token 和 Prompt
# ======================================================================
CTX_START_TOKEN = "<|reserved_special_token_110|>"
CTX_END_TOKEN = "<|reserved_special_token_111|>"

DEPTH_PROBE_TOKENS: Dict[int, str] = {
    depth: f"<|reserved_special_token_{111 + depth}|>"
    for depth in range(1, 5)
}


def find_all_token_positions(
    input_ids: Sequence[int],
    token_id: int,
) -> List[int]:
    """返回指定 token ID 在序列中的全部位置。"""
    return [
        index
        for index, current_id in enumerate(input_ids)
        if int(current_id) == int(token_id)
    ]

FULL_IO_WEIGHTS_NAME = "full_input_output_embeddings.pt"

LLAMA3_CHAT_TEMPLATE = (
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


def build_user_content(text: str, max_depth: int) -> str:
    """
    学生训练、教师训练、测试、层级探测、特征提取和后续蒸馏
    必须逐字复用同一套 Prompt 逻辑。

    框架与 WOS 保持一致：正文边界 -> 任务说明 -> 层级定义 ->
    层级 token 角色绑定 -> 固定输出格式 -> 立即停止 -> 禁止额外文本。
    每个深度 token 被明确指定为对应 taxonomy 深度的层级信息表示标记。
    """
    assignments = "\n".join(
        f"D{depth}={DEPTH_PROBE_TOKENS[depth]}"
        for depth in range(1, max_depth + 1)
    )

    return (
        f"{CTX_START_TOKEN}\n"
        f"{text}\n"
        f"{CTX_END_TOKEN}\n\n"
        "Classify the document according to the BGC hierarchical "
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
    """
    返回一层标签的规范生成顺序。

    排序规则：
      - D1：沿用预处理阶段固定的全局 label ID 顺序；
      - D2-D4：先按父节点在上一层中的生成顺序分组；
      - 同一父节点下的多个子标签，仍按预处理阶段固定的 label ID 顺序。

    正常 clean 数据是祖先闭包，因此深层标签的父节点都会出现在上一层。
    对生成预测中的异常情况，若父节点缺失，则把该组放到已有父节点组之后，
    并按父节点 ID、子节点 ID 保持确定性；祖先闭包检查仍会将其判为 invalid。
    """
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

        # 只用于异常预测的确定性回退；正常祖先闭包数据不会进入这里。
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
    """
    按 taxonomy 深度分组，但仍保持“按层级生成”，不改成按路径生成。

    每一层内部使用父节点优先的层级顺序：上一层父节点先生成，其下一层
    子标签组就先生成；同一父节点下的子标签按固定 label ID 排序。
    """
    max_depth = int(label_space["max_depth"])
    depth = label_space["depth"]
    label2id = label_space["label2id"]

    grouped: Dict[int, List[str]] = {
        current_depth: []
        for current_depth in range(1, max_depth + 1)
    }

    seen = set()
    for raw_label in labels:
        label = str(raw_label).strip()
        if not label:
            raise ValueError("❌ 构造 Completion 时遇到空标签字符串。")
        if label in seen:
            raise ValueError(f"❌ 构造 Completion 时标签重复: {label!r}")
        if label not in label2id:
            raise ValueError(f"❌ 构造 Completion 时遇到未知标签: {label!r}")
        seen.add(label)

        current_depth = int(depth[label])
        if current_depth < 1 or current_depth > max_depth:
            raise ValueError(
                f"❌ 标签 {label!r} 的深度异常: {current_depth}"
            )
        grouped[current_depth].append(label)

    for current_depth in range(1, max_depth + 1):
        previous_depth_labels = (
            grouped[current_depth - 1]
            if current_depth > 1
            else []
        )
        grouped[current_depth] = sort_depth_labels_by_parent_order(
            labels=grouped[current_depth],
            current_depth=current_depth,
            previous_depth_labels=previous_depth_labels,
            label_space=label_space,
        )

    return grouped


def build_completion_text(
    labels: Sequence[str],
    label_space: dict,
) -> str:
    """
    固定输出 4 行：一个特殊 token 预测一整层。

    使用 JSON 数组而不是逗号切分，因为 BGC 标签名称本身可能包含逗号。
    空层明确输出 []，保证 4 个层级 token 在每条样本中都有第二次 occurrence。
    仍然逐层输出；层内先按上一层父节点顺序分组，同父节点子标签按固定
    label ID 排序。所有有效 Completion token（深度 token、JSON 结构、
    标签文本、换行和 EOT）在普通 Completion-only CE 中等权参与监督。
    """
    max_depth = int(label_space["max_depth"])
    grouped = group_labels_by_depth(labels, label_space)

    lines: List[str] = []
    for current_depth in range(1, max_depth + 1):
        labels_json = json.dumps(
            grouped[current_depth],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        lines.append(
            f"{DEPTH_PROBE_TOKENS[current_depth]} {labels_json}"
        )

    return "\n".join(lines) + "\n"


def build_prompt_messages(
    text: str,
    system_prompt: str,
    max_depth: int,
) -> List[dict]:
    """统一构造训练、验证、测试、探测、提取和蒸馏所用 messages。"""
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
    max_seq_length: int = 3072,
    reserve_tokens: int = 1024,
) -> str:
    """
    train / test generation / test loss 共用同一截断逻辑。

    默认配置下正文上限为 3072 - 1024 = 2048 tokens。
    reserve_tokens 是为 chat template、正文边界、任务 Prompt、4 行
    Completion 和 EOT 预留的预算；后面的全量序列预算检查会验证
    每条 Prompt + Completion 是否真正能装入 max_seq_length。
    """
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


def encode_chat_prompt_consistently(
    tokenizer,
    messages: List[dict],
    split_name: str,
    source_index: int,
) -> Tuple[str, List[int]]:
    """
    用两条等价入口构造 Prompt，并逐 token 验证完全一致：

      1. apply_chat_template(tokenize=True)；
      2. apply_chat_template(tokenize=False) 后，再以
         add_special_tokens=False 编码。

    第二条入口必须关闭 add_special_tokens，否则模板字符串中已有的 BOS
    可能被再次添加。返回模板字符串和权威 token IDs。
    """
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
                f"❌ {split_name} 第 {source_index} 条 Prompt 中 BOS 应出现 1 次，"
                f"实际出现 {bos_count} 次。Prompt 开头 IDs={prompt_ids_direct[:12]}"
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
        common = min(len(prompt_ids_direct), len(prompt_ids_from_text))
        mismatch_at = next(
            (
                i
                for i in range(common)
                if prompt_ids_direct[i] != prompt_ids_from_text[i]
            ),
            common,
        )
        raise RuntimeError(
            f"❌ {split_name} 第 {source_index} 条 Prompt 两种编码入口不一致。\n"
            f"direct长度={len(prompt_ids_direct)}, "
            f"render-then-tokenize长度={len(prompt_ids_from_text)}\n"
            f"首个不一致位置={mismatch_at}\n"
            f"direct附近={prompt_ids_direct[max(0, mismatch_at-5):mismatch_at+6]}\n"
            f"reencoded附近={prompt_ids_from_text[max(0, mismatch_at-5):mismatch_at+6]}\n"
            "请检查是否对已经由 chat template 写入 BOS/EOT 的字符串再次使用了 "
            "add_special_tokens=True。"
        )

    return prompt_text, prompt_ids_direct


# ======================================================================
# 2. 数据和标签空间读取、严格校验
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
                    f"❌ JSONL 每行必须是 dict: {path}, line={line_no}, "
                    f"实际={type(item).__name__}"
                )
            rows.append(item)
        return rows

    # 单行 JSONL 只有一个 object 时，json.loads(raw) 会成功得到 dict。
    # 将它视为一条 JSONL 记录，而不是误判为非法顶层。
    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise TypeError(
            f"❌ JSON 顶层必须是 list，或文件为每行一个 object 的 JSONL: "
            f"{path}, 实际={type(data).__name__}"
        )

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(
                f"❌ JSON 第 {idx} 条必须是 dict: "
                f"实际={type(item).__name__}"
            )

    return data


def _normalize_string_list(value, field_name: str, split_name: str, idx: int):
    if not isinstance(value, list):
        raise TypeError(
            f"{split_name} 第 {idx} 条 {field_name} 必须是 list，"
            f"实际={type(value).__name__}"
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
            f"{split_name} 第 {idx} 条 {field_name} 必须是 list，"
            f"实际={type(value).__name__}"
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
    """严格读取 BGC 最终预处理输出，不兼容旧字段，不猜字段名。"""
    missing = [key for key in REQUIRED_CLEAN_FIELDS if key not in item]
    if missing:
        raise KeyError(
            f"{split_name} 第 {idx} 条缺少字段: {missing}。"
            f"必须包含 {list(REQUIRED_CLEAN_FIELDS)}"
        )

    text = item["text"]
    if not isinstance(text, str):
        raise TypeError(
            f"{split_name} 第 {idx} 条 text 必须是 str，"
            f"实际={type(text).__name__}"
        )
    text = text.strip()
    if not text:
        raise ValueError(f"{split_name} 第 {idx} 条 text 为空。")

    labels = _normalize_string_list(
        item["labels"], "labels", split_name, idx
    )
    label_ids = _normalize_int_list(
        item["label_ids"], "label_ids", split_name, idx
    )
    top_labels = _normalize_string_list(
        item["top_labels"], "top_labels", split_name, idx
    )
    top_label_ids = _normalize_int_list(
        item["top_label_ids"], "top_label_ids", split_name, idx
    )
    leaf_labels = _normalize_string_list(
        item["leaf_labels"], "leaf_labels", split_name, idx
    )
    leaf_label_ids = _normalize_int_list(
        item["leaf_label_ids"], "leaf_label_ids", split_name, idx
    )

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
        try:
            label_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"❌ {name}[{label!r}] 不是合法整数 ID: {raw_id!r}"
            ) from exc
        normalized[label] = label_id

    return normalized


def _normalize_children(raw_children) -> Dict[str, List[str]]:
    if not isinstance(raw_children, dict):
        raise TypeError("❌ label_space.children 必须是 dict。")

    normalized: Dict[str, List[str]] = {}
    for raw_parent, raw_list in raw_children.items():
        parent = str(raw_parent).strip()
        if not isinstance(raw_list, list):
            raise TypeError(
                f"❌ children[{parent!r}] 必须是 list。"
            )
        children = []
        for raw_child in raw_list:
            child = str(raw_child).strip()
            if not child:
                raise ValueError(
                    f"❌ children[{parent!r}] 中存在空子节点。"
                )
            children.append(child)
        if len(children) != len(set(children)):
            raise ValueError(
                f"❌ children[{parent!r}] 中存在重复子节点。"
            )
        normalized[parent] = children

    return normalized


def load_and_validate_label_space(
    label_space_path: str,
    expected_num_labels: int,
    expected_max_depth: int,
) -> dict:
    """
    直接读取预处理脚本保存的 label_space.json，并严格校验：
      - 146 个任务标签 ID 连续且从 0 开始；
      - id2label 与 label2id 完全互逆；
      - Root 深度为 0，任务标签深度为 1...4；
      - parent / children / depth 完全一致；
      - top_labels 与 Root 直接子节点一致；
      - 每个标签的祖先链可回溯到 Root。
    """
    if not os.path.isfile(label_space_path):
        raise FileNotFoundError(
            f"❌ label_space.json 不存在: {label_space_path}"
        )

    with open(label_space_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise TypeError("❌ label_space.json 顶层必须是 dict。")

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
        raise KeyError(
            f"❌ label_space.json 缺少字段: {missing}"
        )

    label2id = _normalize_label2id(raw["label2id"], "label2id")
    num_labels = int(raw["num_labels"])
    max_depth = int(raw["max_depth"])

    if num_labels != len(label2id):
        raise ValueError(
            f"❌ num_labels={num_labels}，但 label2id 有 {len(label2id)} 项。"
        )
    if expected_num_labels > 0 and num_labels != expected_num_labels:
        raise ValueError(
            f"❌ BGC 标签数异常: {num_labels} != {expected_num_labels}"
        )
    if expected_max_depth > 0 and max_depth != expected_max_depth:
        raise ValueError(
            f"❌ BGC 最大深度异常: {max_depth} != {expected_max_depth}"
        )

    ids = sorted(label2id.values())
    if ids != list(range(num_labels)):
        raise ValueError(
            "❌ label2id 的 ID 必须是连续 0-based："
            f"实际前后范围={ids[:10]} ... {ids[-10:]}"
        )

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
        missing_parent = all_task_labels - set(parent)
        extra_parent = set(parent) - all_task_labels
        raise ValueError(
            "❌ parent 的键必须恰好覆盖所有任务标签。\n"
            f"missing={sorted(missing_parent)[:20]}\n"
            f"extra={sorted(extra_parent)[:20]}"
        )

    if set(children) != all_nodes:
        missing_nodes = all_nodes - set(children)
        extra_nodes = set(children) - all_nodes
        raise ValueError(
            "❌ children 的键必须覆盖 Root 和全部任务标签。\n"
            f"missing={sorted(missing_nodes)[:20]}\n"
            f"extra={sorted(extra_nodes)[:20]}"
        )

    if set(depth) != all_nodes:
        missing_depth = all_nodes - set(depth)
        extra_depth = set(depth) - all_nodes
        raise ValueError(
            "❌ depth 的键必须覆盖 Root 和全部任务标签。\n"
            f"missing={sorted(missing_depth)[:20]}\n"
            f"extra={sorted(extra_depth)[:20]}"
        )

    if depth["Root"] != 0:
        raise ValueError(
            f"❌ Root 深度必须为 0，实际={depth['Root']}"
        )

    if children["Root"] != top_labels:
        raise ValueError(
            "❌ top_labels 与 children['Root'] 顺序或内容不一致。"
        )

    if len(top_labels) != len(set(top_labels)):
        raise ValueError("❌ top_labels 中存在重复标签。")

    for label in top_labels:
        if label not in all_task_labels:
            raise ValueError(f"❌ 未知 top label: {label!r}")
        if depth[label] != 1:
            raise ValueError(
                f"❌ top label {label!r} 深度不是 1: {depth[label]}"
            )
        if parent[label] != "Root":
            raise ValueError(
                f"❌ top label {label!r} 的父节点不是 Root。"
            )

    for label in sorted(all_task_labels, key=label2id.get):
        current_depth = depth[label]
        if current_depth < 1 or current_depth > max_depth:
            raise ValueError(
                f"❌ 标签 {label!r} 深度越界: {current_depth}"
            )

        current_parent = parent[label]
        if current_parent not in all_nodes:
            raise ValueError(
                f"❌ 标签 {label!r} 的父节点未知: {current_parent!r}"
            )
        if label not in children[current_parent]:
            raise ValueError(
                f"❌ parent/children 不一致: {label!r} 未出现在 "
                f"children[{current_parent!r}] 中。"
            )
        if depth[current_parent] != current_depth - 1:
            raise ValueError(
                f"❌ 深度关系错误: parent={current_parent!r} "
                f"depth={depth[current_parent]}, child={label!r} "
                f"depth={current_depth}"
            )

        for child in children[label]:
            if child not in all_task_labels:
                raise ValueError(
                    f"❌ children[{label!r}] 中出现未知任务标签 {child!r}。"
                )
            if parent[child] != label:
                raise ValueError(
                    f"❌ children/parent 不一致: {child!r} 的 parent "
                    f"是 {parent[child]!r}，不是 {label!r}。"
                )

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
            raise ValueError(
                f"❌ 标签 {label!r} 的祖先链长度 {len(chain)} "
                f"与 depth={depth[label]} 不一致。"
            )
        ancestors[label] = chain

    depth_to_labels: Dict[int, List[str]] = {
        current_depth: []
        for current_depth in range(1, max_depth + 1)
    }
    for label in sorted(all_task_labels, key=label2id.get):
        depth_to_labels[depth[label]].append(label)

    label_ids_by_depth = {
        current_depth: [label2id[label] for label in labels]
        for current_depth, labels in depth_to_labels.items()
    }

    resolved = {
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

    print("\n" + "=" * 78)
    print("🏷️ BGC 标签空间严格校验完成")
    print(f"任务标签数: {num_labels}")
    print(f"最大深度: {max_depth}")
    print(f"Root 直接子节点: {len(top_labels)} -> {top_labels}")
    for current_depth in range(1, max_depth + 1):
        print(
            f"Depth {current_depth}: "
            f"{len(depth_to_labels[current_depth])} 个标签"
        )
    print("✅ label2id / id2label / parent / children / depth 完全一致")
    print("=" * 78 + "\n")

    return resolved


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
    """
    训练开始前一次性严格校验 BGC clean JSONL：
      - 七个字段完整；
      - labels / label_ids 与全局 label_space 一一对应且顺序稳定；
      - labels 是祖先闭包；
      - top_* 与深度 1 标签完全一致；
      - leaf_* 与当前样本端点完全一致。

    任何错误直接停止，不静默跳过。
    """
    data = load_json_or_jsonl(data_path)
    label2id = label_space["label2id"]
    depth = label_space["depth"]
    children = label_space["children"]

    seen_labels = set()
    per_depth_occurrence = defaultdict(int)
    label_count_values = []
    leaf_count_values = []

    for idx, item in enumerate(
        tqdm(data, desc=f"校验{split_name}", leave=False)
    ):
        try:
            (
                _,
                labels,
                label_ids,
                top_labels,
                top_label_ids,
                leaf_labels,
                leaf_label_ids,
            ) = get_clean_sample_fields(item, split_name, idx)
        except Exception as exc:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条样本格式错误: {exc}\n"
                f"样本={item}"
            ) from exc

        if len(labels) != len(label_ids):
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条 labels 与 label_ids 长度不一致。"
            )
        if len(top_labels) != len(top_label_ids):
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条 top_labels 与 "
                "top_label_ids 长度不一致。"
            )
        if len(leaf_labels) != len(leaf_label_ids):
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条 leaf_labels 与 "
                "leaf_label_ids 长度不一致。"
            )

        if len(labels) != len(set(labels)):
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条 labels 中存在重复标签。"
            )
        if len(label_ids) != len(set(label_ids)):
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条 label_ids 中存在重复 ID。"
            )

        unknown = set(labels) - set(label2id)
        if unknown:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条出现未知标签: "
                f"{sorted(unknown)[:20]}"
            )

        expected_labels = sorted(labels, key=label2id.get)
        expected_ids = [label2id[label] for label in expected_labels]

        if labels != expected_labels:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条 labels 未按全局 label ID 排序。"
            )
        if label_ids != expected_ids:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条 label_ids 与 label2id 不一致。\n"
                f"sample={label_ids}\nexpected={expected_ids}"
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
                f"❌ {split_name} 第 {idx} 条 top_* 与深度 1 标签不一致。"
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

        if not top_labels:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条没有深度 1 标签。"
            )
        if not leaf_labels:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条没有样本端点标签。"
            )

        seen_labels.update(labels)
        label_count_values.append(len(labels))
        leaf_count_values.append(len(leaf_labels))
        for label in labels:
            per_depth_occurrence[depth[label]] += 1

    print("\n" + "=" * 78)
    print(f"✅ {split_name} 严格校验通过")
    print(f"文件: {data_path}")
    print(f"样本数: {len(data)}")
    print(
        f"实际出现标签: {len(seen_labels)} / {label_space['num_labels']}"
    )
    print(
        f"平均完整标签数: {sum(label_count_values) / max(1, len(data)):.4f}"
    )
    print(
        f"平均样本端点数: {sum(leaf_count_values) / max(1, len(data)):.4f}"
    )
    print(f"最大完整标签数: {max(label_count_values)}")
    print(f"最大样本端点数: {max(leaf_count_values)}")
    for current_depth in range(1, label_space["max_depth"] + 1):
        print(
            f"Depth {current_depth} 标签实例数: "
            f"{per_depth_occurrence[current_depth]}"
        )
    print("七字段缺失: 0")
    print("未知标签: 0")
    print("标签-ID 错位: 0")
    print("祖先闭包错误: 0")
    print("top/leaf 派生字段错误: 0")
    print("静默跳过: 0")
    print("=" * 78 + "\n")

    return data



def clear_cuda_memory(stage: str = "") -> None:
    """清理失去引用的临时张量与 CUDA 缓存，不影响模型权重。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if stage:
        print(f"🧹 CUDA 临时显存清理完成: {stage}")


def is_cuda_oom_error(exc: BaseException) -> bool:
    """兼容不同 PyTorch 版本的 CUDA OOM 判断。"""
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    type_match = oom_type is not None and isinstance(exc, oom_type)
    return bool(type_match or "cuda out of memory" in str(exc).lower())


def _get_eot_id(tokenizer) -> int:
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is None or eot_id == tokenizer.unk_token_id:
        eot_id = tokenizer.eos_token_id
    if eot_id is None:
        raise RuntimeError(
            "❌ tokenizer 同时缺少 <|eot_id|> 和 eos_token_id。"
        )
    return int(eot_id)


def trim_generated_at_first_stop(
    generated_ids: torch.Tensor,
    stop_ids: Sequence[int],
) -> List[List[int]]:
    """逐样本截到第一个 EOT/EOS，去掉批量生成产生的尾部 PAD。"""
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


# ======================================================================
# 3. Token、模型和训练权重加载
# ======================================================================
def validate_atomic_reserved_tokens(
    tokenizer,
    expected_max_depth: int,
) -> Dict[str, object]:
    """
    严格验证正文边界 token 与 4 个层级 token：
      - 已存在于 tokenizer 词表；
      - add_special_tokens=False 时均严格编码为单 token；
      - 不等于 unk；
      - 所有 token ID 两两不同。

    本脚本不新增 token，也不 resize embedding。
    """
    if expected_max_depth != len(DEPTH_PROBE_TOKENS):
        raise ValueError(
            "❌ 当前脚本固定为 BGC 4 层 token 设计，"
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
                "请检查本地 Llama 模型/tokenizer 版本。"
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
                f"❌ {token} 的 encode ID 与 convert ID 不一致："
                f"encode={token_id}, convert={converted_id}"
            )

        if token_id == tokenizer.unk_token_id:
            raise RuntimeError(f"❌ {token} 被编码成 unk_token。")

        flat_ids[role] = token_id
        print(f"✅ {role}: {token} -> ID={token_id}")

    if len(set(flat_ids.values())) != len(flat_ids):
        raise RuntimeError(
            f"❌ 预留特殊 Token 的 ID 未两两区分：{flat_ids}"
        )

    token_ids: Dict[str, object] = {
        "ctx_start": flat_ids["ctx_start"],
        "ctx_end": flat_ids["ctx_end"],
        "depth": {
            depth: flat_ids[f"depth_{depth}"]
            for depth in range(1, expected_max_depth + 1)
        },
    }

    print("✅ 正文边界 token 与 4 个层级 token 均为不可分割原子 token")
    print("=" * 78 + "\n")
    return token_ids


def check_reserved_token_layout(
    tokenizer,
    system_prompt: str,
    token_ids: Dict[str, object],
    label_space: dict,
) -> None:
    """
    检查：
      - Prompt 中正文起止 Token 各 1 次；
      - Prompt 中每个深度 Token 各 1 次；
      - 拼接 Completion 后每个深度 Token 各 2 次；
      - 第二次 occurrence 按 D1...D4 顺序出现。
    """
    max_depth = int(label_space["max_depth"])
    sample_labels = []
    for current_depth in range(1, max_depth + 1):
        candidates = label_space["depth_to_labels"][current_depth]
        if candidates:
            sample_labels.append(candidates[0])

    messages = build_prompt_messages(
        text="probe check text",
        system_prompt=system_prompt,
        max_depth=max_depth,
    )

    _, prompt_ids = encode_chat_prompt_consistently(
        tokenizer=tokenizer,
        messages=messages,
        split_name="LAYOUT_CHECK",
        source_index=0,
    )

    if prompt_ids.count(int(token_ids["ctx_start"])) != 1:
        raise RuntimeError("❌ Prompt 中 CTX_START 应恰好出现 1 次。")
    if prompt_ids.count(int(token_ids["ctx_end"])) != 1:
        raise RuntimeError("❌ Prompt 中 CTX_END 应恰好出现 1 次。")

    depth_ids = token_ids["depth"]
    for current_depth in range(1, max_depth + 1):
        current_id = int(depth_ids[current_depth])
        if prompt_ids.count(current_id) != 1:
            raise RuntimeError(
                f"❌ Prompt 中 D{current_depth} token_id={current_id} "
                "应恰好出现 1 次。"
            )

    completion_ids = tokenizer.encode(
        build_completion_text(sample_labels, label_space),
        add_special_tokens=False,
    )
    full_ids = prompt_ids + [int(x) for x in completion_ids]

    second_positions = []
    for current_depth in range(1, max_depth + 1):
        current_id = int(depth_ids[current_depth])
        positions = find_all_token_positions(full_ids, current_id)
        if len(positions) != 2:
            raise RuntimeError(
                f"❌ 完整样本中 D{current_depth} Token 应出现 2 次，"
                f"实际位置={positions}"
            )
        second_positions.append(positions[1])

    if second_positions != sorted(second_positions):
        raise RuntimeError(
            "❌ Completion 中 D1-D4 token 的第二次 occurrence 顺序异常："
            f"{second_positions}"
        )

    ctx_start_pos = full_ids.index(int(token_ids["ctx_start"]))
    ctx_end_pos = full_ids.index(int(token_ids["ctx_end"]))

    if ctx_end_pos <= ctx_start_pos + 1:
        raise RuntimeError("❌ 正文起止 Token 之间没有有效正文 token。")

    print("\n" + "=" * 78)
    print("✅ Prompt / Completion 特殊 Token 布局检查通过")
    print(f"正文开始 Token 位置: {ctx_start_pos}")
    print(f"正文结束 Token 位置: {ctx_end_pos}")
    print(f"D1-D4 Token 第二次位置: {second_positions}")
    print("后续层级探测/蒸馏应提取各深度 Token 的第二次 occurrence。")
    print("=" * 78 + "\n")


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def restore_full_io_weights(base_model, checkpoint_dir: str) -> None:
    weight_path = os.path.join(checkpoint_dir, FULL_IO_WEIGHTS_NAME)
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            f"❌ PEFT checkpoint 缺少 {FULL_IO_WEIGHTS_NAME}: {weight_path}"
        )

    payload = _torch_load_cpu(weight_path)
    saved_input = payload["input_embeddings"]
    saved_tied = bool(payload["actually_tied"])

    input_weight = base_model.get_input_embeddings().weight
    output_module = base_model.get_output_embeddings()
    if output_module is None:
        raise RuntimeError("❌ 基础模型缺少 lm_head。")

    if tuple(saved_input.shape) != tuple(input_weight.shape):
        raise RuntimeError(
            "❌ 保存的 input embedding 与基础模型形状不一致："
            f"saved={tuple(saved_input.shape)}, base={tuple(input_weight.shape)}"
        )

    with torch.no_grad():
        input_weight.copy_(saved_input.to(dtype=input_weight.dtype))

        if saved_tied:
            if input_weight.data_ptr() != output_module.weight.data_ptr():
                base_model.config.tie_word_embeddings = True
                base_model.tie_weights()

            if (
                base_model.get_input_embeddings().weight.data_ptr()
                != base_model.get_output_embeddings().weight.data_ptr()
            ):
                raise RuntimeError("❌ 无法恢复共享 input/lm_head 权重。")
        else:
            saved_output = payload.get("output_embeddings")
            if saved_output is None:
                raise RuntimeError("❌ checkpoint 缺少 output_embeddings。")

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
                raise RuntimeError("❌ 保存的 lm_head 与基础模型形状不一致。")

            with torch.no_grad():
                output_module.weight.copy_(
                    saved_output.to(dtype=output_module.weight.dtype)
                )
            base_model.config.tie_word_embeddings = False


def detect_checkpoint_type(model_path: str, checkpoint_type: str) -> str:
    if checkpoint_type != "auto":
        return checkpoint_type

    if os.path.isfile(os.path.join(model_path, "adapter_config.json")):
        return "peft"
    if os.path.isfile(os.path.join(model_path, "config.json")):
        return "merged"

    raise FileNotFoundError(
        f"❌ 无法判断模型类型: {model_path}。"
        "目录中既没有 adapter_config.json，也没有 config.json。"
    )


def load_model_and_tokenizer(args):
    checkpoint_type = detect_checkpoint_type(
        args.model_path,
        args.checkpoint_type,
    )

    # 周期 PEFT checkpoint 通常不包含 tokenizer 文件。训练没有新增 token，
    # 没有 resize embedding，110-119 也都是 Llama 原生 reserved tokens，
    # 因此测试 PEFT checkpoint 时从基础模型目录加载 tokenizer 是正确做法。
    # 完整合并模型优先从自身目录加载；缺少 tokenizer 时回退到基础模型目录。
    if checkpoint_type == "peft":
        if not args.base_model_path:
            raise ValueError(
                "❌ 加载 PEFT checkpoint 时必须提供 --base_model_path，"
                "因为 tokenizer 将从基础模型目录加载。"
            )
        tokenizer_source = args.base_model_path
    else:
        tokenizer_markers = (
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
        )
        has_local_tokenizer = any(
            os.path.isfile(os.path.join(args.model_path, name))
            for name in tokenizer_markers
        )
        if has_local_tokenizer:
            tokenizer_source = args.model_path
        elif args.base_model_path:
            tokenizer_source = args.base_model_path
            print(
                "⚠️ 完整模型目录中未检测到 tokenizer 文件，"
                "回退到基础模型 tokenizer。"
            )
        else:
            raise FileNotFoundError(
                "❌ 完整模型目录中没有 tokenizer 文件，且未提供 "
                "--base_model_path。"
            )

    print(f"📚 Tokenizer 加载目录: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
    )
    tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError("❌ tokenizer 没有 pad_token_id。")
    tokenizer.padding_side = "left"

    model_dtype = torch.bfloat16 if args.bf16 else torch.float32

    if checkpoint_type == "merged":
        print(f"📦 按完整合并模型加载: {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=model_dtype,
            attn_implementation=args.attn_implementation,
            device_map=None,
        )
    else:
        if not args.base_model_path:
            raise ValueError(
                "❌ 加载 PEFT checkpoint 时必须提供 --base_model_path。"
            )

        print(f"📦 基础模型: {args.base_model_path}")
        print(f"📦 PEFT checkpoint: {args.model_path}")

        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=model_dtype,
            attn_implementation=args.attn_implementation,
            device_map=None,
        )
        restore_full_io_weights(base_model, args.model_path)

        model = PeftModel.from_pretrained(
            base_model,
            args.model_path,
            is_trainable=False,
        )

        if args.merge_peft_for_test:
            print("🔗 测试前合并 LoRA adapter...")
            model = model.merge_and_unload()

    embedding_rows = model.get_input_embeddings().weight.shape[0]
    if embedding_rows != len(tokenizer):
        raise RuntimeError(
            "❌ tokenizer 长度与模型 embedding 行数不一致："
            f"tokenizer={len(tokenizer)}, model={embedding_rows}"
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.do_sample = False
        # 清除基础模型中与贪心解码无关的采样配置，避免 warning。
        for name in ("temperature", "top_p", "top_k"):
            if hasattr(model.generation_config, name):
                setattr(model.generation_config, name, None)

    if not torch.cuda.is_available():
        raise RuntimeError("❌ 当前环境没有可用 CUDA GPU。")

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    print(f"✅ 模型已加载到 {device}，dtype={model_dtype}")
    return model, tokenizer, device, checkpoint_type


# ======================================================================
# 4. 严格解析和预测语义检查
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


def parse_generation_exact(
    raw_resp: str,
    max_depth: int,
) -> Tuple[Optional[Dict[int, List[str]]], str]:
    """
    严格解析固定 4 行输出：
      <D1 token> ["label_a","label_b"]
      ...
      <D4 token> []

    这里只判断行数、前缀、JSON 结构与元素类型。
    标签字符串保持模型原始输出，不自动 strip；空格、未知标签、错误深度、
    重复标签、规范顺序和祖先闭包均在后续语义校验中处理。
    """
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

        # 保留模型生成的原始字符串，严格执行 exact taxonomy label strings。
        parsed[current_depth] = list(labels)

    return parsed, "ok"


def validate_parsed_prediction(
    parsed_layers: Optional[Dict[int, List[str]]],
    label_space: dict,
) -> dict:
    """
    将严格解析结果转换为 146 维扁平预测集合，并记录生成语义错误。

    正式 Global/Depth F1 使用模型生成的全部“精确匹配到词表的已知标签”。
    因此，已知标签即使被放在错误深度，也仍会进入扁平预测集合，避免普通
    Global F1 因丢弃错误深度标签而虚高；错误深度同时会令输出变为 invalid。

    invalid_for_micro=True 时，Strict Micro/Samples-F1 会额外加入一个
    “非法输出”哨兵假阳性维度。Global Macro-F1 仍按当前 split 中真实出现的
    标签计算，Global Macro-F1-All146 则固定平均全部 146 个标签。
    """
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

    # pred_labels 是正式扁平 F1 使用的集合：所有精确匹配的已知标签均加入。
    pred_labels: Set[str] = set()
    duplicate_count = 0
    unknown_labels = []
    wrong_depth_labels = []
    whitespace_error_labels = []
    order_error_depths = []

    for current_depth in range(1, max_depth + 1):
        layer_labels = parsed_layers[current_depth]
        local_seen = set()

        # 只有该层所有标签均为精确已知且位于正确深度时，才检查
        # 新的层级规范顺序：父节点顺序优先，同父节点内按固定 label ID。
        can_check_hierarchical_order = all(
            label in label2id
            and int(depth[label]) == current_depth
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

            # 已知标签无论是否重复、是否放错深度，都只在正式扁平
            # 预测集合中保留一次；重复与错误深度另行严格记录。
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
    """
    计算 BGC 层次多标签指标。

    正式指标：
      - Global Micro-F1：完整 146 标签空间。模型错误预测当前 split
        未出现的标签仍会产生 FP。
      - Global Macro-F1：只平均当前 split 的真实标签中实际出现的类别，
        与当前 WOS 代码的 present-class Macro 口径一致。
      - Samples-F1：逐样本标签集合 F1 后取平均。

    辅助指标：
      - Global Macro-F1-All146：固定平均全部 146 个标签，便于诊断。
      - Strict Micro/Samples-F1：额外加入非法输出哨兵维度。
      - D1-D4 Micro：该层完整标签空间；D1-D4 Macro：只平均该层
        当前 split 真实出现的标签。
    """
    if not (
        len(true_id_sets) == len(pred_id_sets) == len(invalid_flags)
    ):
        raise ValueError("❌ multilabel metric 输入长度不一致。")
    if len(true_id_sets) == 0:
        raise ValueError("❌ multilabel metric 输入不能为空。")

    num_labels = int(label_space["num_labels"])
    y_true = _sets_to_indicator(true_id_sets, num_labels)
    y_pred = _sets_to_indicator(pred_id_sets, num_labels)

    # Global Micro 保留完整 146 维，确保对未出现类别的错误预测仍计 FP。
    global_micro = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    present_label_ids = np.flatnonzero(y_true.sum(axis=0) > 0)
    if len(present_label_ids) == 0:
        raise RuntimeError("❌ 当前 split 中没有任何真实标签。")

    # 正式 Macro：只平均当前 split 真实出现的类别。
    global_macro = f1_score(
        y_true[:, present_label_ids],
        y_pred[:, present_label_ids],
        average="macro",
        zero_division=0,
    )

    # 诊断 Macro：固定全部 146 类。
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

    # 严格生成指标：非法输出在额外维度产生一个假阳性。
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
# 6.1 测试集详细错误诊断
# ======================================================================
def _safe_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return float(precision), float(recall), float(f1)


def _subset_metric_bundle(
    indices: Sequence[int],
    true_id_sets: Sequence[Set[int]],
    pred_id_sets: Sequence[Set[int]],
    invalid_flags: Sequence[bool],
    label_space: dict,
) -> Optional[dict]:
    if not indices:
        return None
    return compute_multilabel_metric_bundle(
        true_id_sets=[true_id_sets[i] for i in indices],
        pred_id_sets=[pred_id_sets[i] for i in indices],
        invalid_flags=[invalid_flags[i] for i in indices],
        label_space=label_space,
    )


def compute_detailed_error_analysis(
    true_id_sets: Sequence[Set[int]],
    pred_id_sets: Sequence[Set[int]],
    invalid_flags: Sequence[bool],
    test_records: Sequence[dict],
    label_space: dict,
    output_dir: str,
    top_k: int = 20,
) -> dict:
    """
    在测试集上补充纯生成分类最需要的诊断：
      1. Global/D1-D4 Precision、Recall、F1；
      2. 真实与预测标签数量，判断偏漏标签还是偏多预测；
      3. 每个标签 TP/FP/FN/F1/support；
      4. FP/FN 最多与低 F1 标签；
      5. 单路径、多路径样本分别计分；
      6. 父节点正确但子节点漏预测，以及真实父节点下错选兄弟标签；
      7. 高频兄弟标签替换混淆对。

    这里全部使用已经生成出的已知 BGC 标签集合，不改变正式指标口径。
    """
    if not (
        len(true_id_sets)
        == len(pred_id_sets)
        == len(invalid_flags)
        == len(test_records)
    ):
        raise ValueError("❌ 详细诊断输入长度不一致。")
    if not true_id_sets:
        raise ValueError("❌ 详细诊断输入不能为空。")

    os.makedirs(output_dir, exist_ok=True)
    top_k = max(1, int(top_k))

    num_labels = int(label_space["num_labels"])
    label2id = label_space["label2id"]
    id2label = label_space["id2label"]
    depth_map = label_space["depth"]
    parent_map = label_space["parent"]

    y_true = _sets_to_indicator(true_id_sets, num_labels)
    y_pred = _sets_to_indicator(pred_id_sets, num_labels)

    tp_by_label = np.logical_and(y_true == 1, y_pred == 1).sum(axis=0)
    fp_by_label = np.logical_and(y_true == 0, y_pred == 1).sum(axis=0)
    fn_by_label = np.logical_and(y_true == 1, y_pred == 0).sum(axis=0)
    support_by_label = y_true.sum(axis=0)
    pred_count_by_label = y_pred.sum(axis=0)

    global_tp = int(tp_by_label.sum())
    global_fp = int(fp_by_label.sum())
    global_fn = int(fn_by_label.sum())
    global_precision, global_recall, global_f1 = _safe_prf(
        global_tp, global_fp, global_fn
    )

    present_ids = np.flatnonzero(support_by_label > 0)
    macro_precision = precision_score(
        y_true[:, present_ids],
        y_pred[:, present_ids],
        average="macro",
        zero_division=0,
    )
    macro_recall = recall_score(
        y_true[:, present_ids],
        y_pred[:, present_ids],
        average="macro",
        zero_division=0,
    )
    macro_f1 = f1_score(
        y_true[:, present_ids],
        y_pred[:, present_ids],
        average="macro",
        zero_division=0,
    )

    total_true_labels = int(y_true.sum())
    total_pred_labels = int(y_pred.sum())
    sample_count = len(true_id_sets)
    true_cardinalities = y_true.sum(axis=1)
    pred_cardinalities = y_pred.sum(axis=1)
    cardinality_delta = pred_cardinalities - true_cardinalities

    under_predict_samples = int((cardinality_delta < 0).sum())
    equal_count_samples = int((cardinality_delta == 0).sum())
    over_predict_samples = int((cardinality_delta > 0).sum())

    per_depth: Dict[str, dict] = {}
    for current_depth in range(1, int(label_space["max_depth"]) + 1):
        depth_ids = list(label_space["label_ids_by_depth"][current_depth])
        true_depth = y_true[:, depth_ids]
        pred_depth = y_pred[:, depth_ids]

        depth_tp = int(np.logical_and(true_depth == 1, pred_depth == 1).sum())
        depth_fp = int(np.logical_and(true_depth == 0, pred_depth == 1).sum())
        depth_fn = int(np.logical_and(true_depth == 1, pred_depth == 0).sum())
        depth_p, depth_r, depth_f1 = _safe_prf(
            depth_tp, depth_fp, depth_fn
        )

        present_depth_columns = np.flatnonzero(true_depth.sum(axis=0) > 0)
        if len(present_depth_columns) > 0:
            depth_macro_p = precision_score(
                true_depth[:, present_depth_columns],
                pred_depth[:, present_depth_columns],
                average="macro",
                zero_division=0,
            )
            depth_macro_r = recall_score(
                true_depth[:, present_depth_columns],
                pred_depth[:, present_depth_columns],
                average="macro",
                zero_division=0,
            )
            depth_macro_f1 = f1_score(
                true_depth[:, present_depth_columns],
                pred_depth[:, present_depth_columns],
                average="macro",
                zero_division=0,
            )
        else:
            depth_macro_p = depth_macro_r = depth_macro_f1 = 0.0

        per_depth[str(current_depth)] = {
            "label_count": len(depth_ids),
            "present_label_count": int(len(present_depth_columns)),
            "true_label_instances": int(true_depth.sum()),
            "predicted_label_instances": int(pred_depth.sum()),
            "tp": depth_tp,
            "fp": depth_fp,
            "fn": depth_fn,
            "micro_precision": float(depth_p),
            "micro_recall": float(depth_r),
            "micro_f1": float(depth_f1),
            "macro_precision": float(depth_macro_p),
            "macro_recall": float(depth_macro_r),
            "macro_f1": float(depth_macro_f1),
        }

    per_label_rows: List[dict] = []
    for label_id in range(num_labels):
        label = id2label[label_id]
        tp = int(tp_by_label[label_id])
        fp = int(fp_by_label[label_id])
        fn = int(fn_by_label[label_id])
        precision, recall, f1 = _safe_prf(tp, fp, fn)
        per_label_rows.append(
            {
                "label_id": int(label_id),
                "label": label,
                "depth": int(depth_map[label]),
                "parent": parent_map[label],
                "support": int(support_by_label[label_id]),
                "predicted_count": int(pred_count_by_label[label_id]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    per_label_csv_path = os.path.join(output_dir, "per_label_metrics.csv")
    with open(per_label_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_label_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_label_rows)

    # 统计父节点已预测正确、但真实子标签漏掉的情况；以及父节点真实正确，
    # 模型却在该父节点下面多预测了错误子标签的情况。
    parent_correct_child_fn_count = 0
    parent_correct_child_fn_samples = 0
    true_parent_wrong_child_fp_count = 0
    true_parent_wrong_child_fp_samples = 0
    sibling_confusion_counter: Counter = Counter()

    for true_ids, pred_ids in zip(true_id_sets, pred_id_sets):
        true_labels = {id2label[label_id] for label_id in true_ids}
        pred_labels = {id2label[label_id] for label_id in pred_ids}
        fn_labels = true_labels - pred_labels
        fp_labels = pred_labels - true_labels

        sample_has_parent_correct_fn = False
        for true_label in fn_labels:
            if int(depth_map[true_label]) <= 1:
                continue
            parent_label = parent_map[true_label]
            if parent_label in pred_labels:
                parent_correct_child_fn_count += 1
                sample_has_parent_correct_fn = True
        parent_correct_child_fn_samples += int(sample_has_parent_correct_fn)

        sample_has_true_parent_fp = False
        for pred_label in fp_labels:
            if int(depth_map[pred_label]) <= 1:
                continue
            parent_label = parent_map[pred_label]
            if parent_label in true_labels:
                true_parent_wrong_child_fp_count += 1
                sample_has_true_parent_fp = True
        true_parent_wrong_child_fp_samples += int(sample_has_true_parent_fp)

        # 同一样本中，一个真实子标签漏掉，同时多预测了同一父节点下的
        # 另一个子标签，记录为兄弟标签替换混淆。
        for true_label in fn_labels:
            if int(depth_map[true_label]) <= 1:
                continue
            for pred_label in fp_labels:
                if (
                    int(depth_map[pred_label]) == int(depth_map[true_label])
                    and parent_map[pred_label] == parent_map[true_label]
                ):
                    sibling_confusion_counter[(true_label, pred_label)] += 1

    sibling_rows = [
        {
            "true_label": true_label,
            "predicted_sibling": pred_label,
            "parent": parent_map[true_label],
            "depth": int(depth_map[true_label]),
            "count": int(count),
        }
        for (true_label, pred_label), count in sibling_confusion_counter.most_common()
    ]
    sibling_csv_path = os.path.join(output_dir, "sibling_confusions.csv")
    with open(sibling_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "true_label",
            "predicted_sibling",
            "parent",
            "depth",
            "count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sibling_rows)

    single_path_indices = [
        i
        for i, record in enumerate(test_records)
        if int(record.get("true_leaf_count", 0)) == 1
    ]
    multi_path_indices = [
        i
        for i, record in enumerate(test_records)
        if int(record.get("true_leaf_count", 0)) > 1
    ]

    single_path_metrics = _subset_metric_bundle(
        single_path_indices,
        true_id_sets,
        pred_id_sets,
        invalid_flags,
        label_space,
    )
    multi_path_metrics = _subset_metric_bundle(
        multi_path_indices,
        true_id_sets,
        pred_id_sets,
        invalid_flags,
        label_space,
    )

    top_fn = sorted(
        per_label_rows,
        key=lambda row: (row["fn"], row["support"], -row["f1"]),
        reverse=True,
    )[:top_k]
    top_fp = sorted(
        per_label_rows,
        key=lambda row: (row["fp"], row["predicted_count"], -row["f1"]),
        reverse=True,
    )[:top_k]
    lowest_f1 = sorted(
        [row for row in per_label_rows if row["support"] > 0],
        key=lambda row: (row["f1"], -row["support"], row["label_id"]),
    )[:top_k]
    lowest_recall = sorted(
        [row for row in per_label_rows if row["support"] > 0],
        key=lambda row: (row["recall"], -row["support"], row["label_id"]),
    )[:top_k]
    lowest_precision = sorted(
        [row for row in per_label_rows if row["predicted_count"] > 0],
        key=lambda row: (
            row["precision"],
            -row["predicted_count"],
            row["label_id"],
        ),
    )[:top_k]

    analysis = {
        "sample_count": sample_count,
        "global": {
            "tp": global_tp,
            "fp": global_fp,
            "fn": global_fn,
            "micro_precision": global_precision,
            "micro_recall": global_recall,
            "micro_f1": global_f1,
            "macro_precision_present": float(macro_precision),
            "macro_recall_present": float(macro_recall),
            "macro_f1_present": float(macro_f1),
            "present_label_count": int(len(present_ids)),
        },
        "label_cardinality": {
            "total_true_label_instances": total_true_labels,
            "total_predicted_known_label_instances": total_pred_labels,
            "average_true_labels_per_sample": float(
                total_true_labels / sample_count
            ),
            "average_predicted_known_labels_per_sample": float(
                total_pred_labels / sample_count
            ),
            "mean_pred_minus_true": float(cardinality_delta.mean()),
            "under_predict_sample_count": under_predict_samples,
            "equal_label_count_sample_count": equal_count_samples,
            "over_predict_sample_count": over_predict_samples,
            "under_predict_sample_rate": float(
                under_predict_samples / sample_count
            ),
            "equal_label_count_sample_rate": float(
                equal_count_samples / sample_count
            ),
            "over_predict_sample_rate": float(
                over_predict_samples / sample_count
            ),
        },
        "per_depth": per_depth,
        "path_groups": {
            "single_path_sample_count": len(single_path_indices),
            "multi_path_sample_count": len(multi_path_indices),
            "single_path_metrics": single_path_metrics,
            "multi_path_metrics": multi_path_metrics,
        },
        "hierarchical_error_analysis": {
            "parent_correct_but_true_child_missed_count": int(
                parent_correct_child_fn_count
            ),
            "samples_with_parent_correct_but_child_missed": int(
                parent_correct_child_fn_samples
            ),
            "true_parent_but_wrong_child_fp_count": int(
                true_parent_wrong_child_fp_count
            ),
            "samples_with_true_parent_but_wrong_child_fp": int(
                true_parent_wrong_child_fp_samples
            ),
            "sibling_replacement_pair_count": int(
                sum(sibling_confusion_counter.values())
            ),
            "top_sibling_confusions": sibling_rows[:top_k],
        },
        "top_error_labels": {
            "top_false_negative_labels": top_fn,
            "top_false_positive_labels": top_fp,
            "lowest_f1_labels": lowest_f1,
            "lowest_recall_labels": lowest_recall,
            "lowest_precision_labels": lowest_precision,
        },
        "output_files": {
            "per_label_metrics_csv": per_label_csv_path,
            "sibling_confusions_csv": sibling_csv_path,
        },
    }

    analysis_path = os.path.join(
        output_dir, "test_detailed_error_analysis.json"
    )
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    analysis["output_files"]["analysis_json"] = analysis_path

    summary_txt_path = os.path.join(
        output_dir, "test_detailed_error_summary.txt"
    )
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("BGC TEST DETAILED ERROR DIAGNOSTICS\n")
        f.write("=" * 88 + "\n")
        f.write(
            f"Micro Precision/Recall/F1: "
            f"{global_precision:.6f}/{global_recall:.6f}/{global_f1:.6f}\n"
        )
        f.write(
            f"Macro Precision/Recall/F1(Present): "
            f"{macro_precision:.6f}/{macro_recall:.6f}/{macro_f1:.6f}\n"
        )
        f.write(
            f"Average true/pred labels per sample: "
            f"{total_true_labels / sample_count:.4f}/"
            f"{total_pred_labels / sample_count:.4f}\n"
        )
        f.write(
            f"Under/Equal/Over predicted-label-count samples: "
            f"{under_predict_samples}/{equal_count_samples}/"
            f"{over_predict_samples}\n\n"
        )
        f.write("Per-depth Micro P/R/F1 and Macro P/R/F1\n")
        for depth_key, row in per_depth.items():
            f.write(
                f"D{depth_key}: micro="
                f"{row['micro_precision']:.6f}/"
                f"{row['micro_recall']:.6f}/"
                f"{row['micro_f1']:.6f}; macro="
                f"{row['macro_precision']:.6f}/"
                f"{row['macro_recall']:.6f}/"
                f"{row['macro_f1']:.6f}; "
                f"true/pred={row['true_label_instances']}/"
                f"{row['predicted_label_instances']}\n"
            )
        f.write("\nHierarchical errors\n")
        f.write(
            f"Parent correct but true child missed: "
            f"{parent_correct_child_fn_count} events in "
            f"{parent_correct_child_fn_samples} samples\n"
        )
        f.write(
            f"True parent but wrong child predicted: "
            f"{true_parent_wrong_child_fp_count} events in "
            f"{true_parent_wrong_child_fp_samples} samples\n"
        )
        f.write(
            f"Sibling replacement pairs: "
            f"{sum(sibling_confusion_counter.values())}\n"
        )
        f.write("\nTop FN labels\n")
        for row in top_fn:
            f.write(
                f"D{row['depth']} | FN={row['fn']} | support={row['support']} "
                f"| recall={row['recall']:.4f} | {row['label']}\n"
            )
        f.write("\nTop FP labels\n")
        for row in top_fp:
            f.write(
                f"D{row['depth']} | FP={row['fp']} | predicted="
                f"{row['predicted_count']} | precision={row['precision']:.4f} "
                f"| {row['label']}\n"
            )

    analysis["output_files"]["summary_txt"] = summary_txt_path

    print("\n" + "=" * 100)
    print("🔍 BGC TEST 详细错误诊断")
    print(
        "Global Micro Precision/Recall/F1: "
        f"{global_precision:.6f}/{global_recall:.6f}/{global_f1:.6f}"
    )
    print(
        "Global Macro Precision/Recall/F1(Present): "
        f"{macro_precision:.6f}/{macro_recall:.6f}/{macro_f1:.6f}"
    )
    print(
        "平均真实/预测已知标签数: "
        f"{total_true_labels / sample_count:.4f}/"
        f"{total_pred_labels / sample_count:.4f}"
    )
    print(
        "预测标签数量偏少/相同/偏多的样本: "
        f"{under_predict_samples}/{equal_count_samples}/"
        f"{over_predict_samples}"
    )
    print("\n各层 Micro Precision/Recall/F1")
    for depth_key, row in per_depth.items():
        print(
            f"D{depth_key}: {row['micro_precision']:.6f}/"
            f"{row['micro_recall']:.6f}/{row['micro_f1']:.6f} | "
            f"true/pred={row['true_label_instances']}/"
            f"{row['predicted_label_instances']}"
        )
    print("\n层级错误")
    print(
        "父节点已预测正确但真实子标签漏掉: "
        f"{parent_correct_child_fn_count} 次 / "
        f"{parent_correct_child_fn_samples} 条样本"
    )
    print(
        "真实父节点存在但预测了错误子标签: "
        f"{true_parent_wrong_child_fp_count} 次 / "
        f"{true_parent_wrong_child_fp_samples} 条样本"
    )
    print(
        "兄弟标签替换混淆总数: "
        f"{sum(sibling_confusion_counter.values())}"
    )
    print(f"每标签明细: {per_label_csv_path}")
    print(f"兄弟混淆明细: {sibling_csv_path}")
    print(f"完整诊断 JSON: {analysis_path}")
    print(f"可读摘要: {summary_txt_path}")
    print("=" * 100 + "\n")

    return analysis


# ======================================================================
# 6. 准备测试 Prompt，并检查长度和正文截断
# ======================================================================
def prepare_test_records(
    test_data: List[dict],
    tokenizer,
    label_space: dict,
    system_prompt: str,
    max_seq_length: int,
    reserve_tokens: int,
    max_new_tokens: int,
    token_ids: Dict[str, object],
) -> Tuple[List[dict], dict]:
    max_depth = int(label_space["max_depth"])
    max_raw_body_len = 0
    max_truncated_body_len = 0
    max_prompt_len = 0
    max_completion_len = 0
    max_full_len = 0
    truncated_count = 0
    records = []

    for idx, item in enumerate(
        tqdm(test_data, desc="准备 TEST Prompt", leave=False)
    ):
        (
            text,
            labels,
            label_ids,
            top_labels,
            top_label_ids,
            leaf_labels,
            leaf_label_ids,
        ) = get_clean_sample_fields(item, "TEST", idx)

        raw_body_ids = tokenizer.encode(text, add_special_tokens=False)
        truncated_text = truncate_text_for_prompt(
            text,
            tokenizer,
            max_seq_length=max_seq_length,
            reserve_tokens=reserve_tokens,
        )
        truncated_body_ids = tokenizer.encode(
            truncated_text,
            add_special_tokens=False,
        )

        if len(truncated_body_ids) < len(raw_body_ids):
            truncated_count += 1

        messages = build_prompt_messages(
            text=truncated_text,
            system_prompt=system_prompt,
            max_depth=max_depth,
        )
        prompt_text, prompt_ids = encode_chat_prompt_consistently(
            tokenizer=tokenizer,
            messages=messages,
            split_name="TEST_PROMPT",
            source_index=idx,
        )

        if prompt_ids.count(int(token_ids["ctx_start"])) != 1:
            raise RuntimeError(f"❌ TEST 第 {idx} 条 CTX_START 次数异常。")
        if prompt_ids.count(int(token_ids["ctx_end"])) != 1:
            raise RuntimeError(f"❌ TEST 第 {idx} 条 CTX_END 次数异常。")
        for current_depth in range(1, max_depth + 1):
            current_id = int(token_ids["depth"][current_depth])
            if prompt_ids.count(current_id) != 1:
                raise RuntimeError(
                    f"❌ TEST 第 {idx} 条 Prompt 中 D{current_depth} "
                    "token 次数异常。"
                )

        completion_ids = tokenizer.encode(
            build_completion_text(labels, label_space),
            add_special_tokens=False,
        )
        completion_with_eot_len = len(completion_ids) + 1
        full_len = len(prompt_ids) + completion_with_eot_len

        if len(prompt_ids) > max_seq_length:
            raise ValueError(
                f"❌ TEST 第 {idx} 条 Prompt 超过 max_seq_length："
                f"{len(prompt_ids)} > {max_seq_length}"
            )
        if completion_with_eot_len > max_new_tokens:
            raise ValueError(
                f"❌ TEST 第 {idx} 条正确答案长度 "
                f"{completion_with_eot_len} > max_new_tokens={max_new_tokens}"
            )
        if full_len > max_seq_length:
            raise ValueError(
                f"❌ TEST 第 {idx} 条 Prompt+正确答案长度 "
                f"{full_len} > max_seq_length={max_seq_length}"
            )

        records.append(
            {
                "source_index": idx,
                "prompt": prompt_text,
                "prompt_ids": prompt_ids,
                "true_labels": labels,
                "true_label_ids": set(int(x) for x in label_ids),
                "true_layers": group_labels_by_depth(labels, label_space),
                "true_top_labels": top_labels,
                "true_top_label_ids": list(top_label_ids),
                "true_leaf_labels": leaf_labels,
                "true_leaf_label_ids": list(leaf_label_ids),
                "true_leaf_count": len(leaf_labels),
                "is_multi_path": len(leaf_labels) > 1,
                "raw_body_token_count": len(raw_body_ids),
                "truncated_body_token_count": len(truncated_body_ids),
                "body_was_truncated": (
                    len(truncated_body_ids) < len(raw_body_ids)
                ),
            }
        )

        max_raw_body_len = max(max_raw_body_len, len(raw_body_ids))
        max_truncated_body_len = max(
            max_truncated_body_len, len(truncated_body_ids)
        )
        max_prompt_len = max(max_prompt_len, len(prompt_ids))
        max_completion_len = max(
            max_completion_len, completion_with_eot_len
        )
        max_full_len = max(max_full_len, full_len)

    summary = {
        "num_samples": len(records),
        "max_raw_body_len": max_raw_body_len,
        "max_truncated_body_len": max_truncated_body_len,
        "max_prompt_len": max_prompt_len,
        "max_completion_with_eot_len": max_completion_len,
        "max_full_true_sequence_len": max_full_len,
        "truncated_count": truncated_count,
        "truncated_rate": truncated_count / max(1, len(records)),
    }

    print("\n" + "=" * 88)
    print("📏 TEST 序列预算检查")
    print(f"样本数: {summary['num_samples']}")
    print(f"正文上限: {max_seq_length - reserve_tokens}")
    print(f"最大原始正文 token 数: {max_raw_body_len}")
    print(f"最大截断后正文 token 数: {max_truncated_body_len}")
    print(f"最大 Prompt token 数: {max_prompt_len}")
    print(f"最大正确答案+EOT token 数: {max_completion_len}")
    print(f"最大 Prompt+正确答案 token 数: {max_full_len}")
    print(
        f"正文发生截断: {truncated_count}/{len(records)} "
        f"({summary['truncated_rate']:.2%})"
    )
    print("=" * 88 + "\n")

    # 与训练验证一致：完整测试集不抽样，仅按 Prompt 长度排序减少 padding。
    records.sort(key=lambda item: len(item["prompt_ids"]))
    print("✅ TEST Prompt 已按长度排序，source_index 仍保留原始样本编号")

    return records, summary


# ======================================================================
# 7. 批量生成、解析、保存逐样本结果
# ======================================================================
def _generate_raw_texts_once(
    model,
    tokenizer,
    batch: List[dict],
    device,
    stop_ids: Sequence[int],
    max_new_tokens: int,
) -> List[str]:
    """用当前 batch 大小执行一次与训练验证相同的贪心生成。"""
    prompts = [item["prompt"] for item in batch]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    ).to(device)

    for row_idx, item in enumerate(batch):
        mask = inputs.attention_mask[row_idx].bool()
        actual_prompt_ids = (
            inputs.input_ids[row_idx][mask]
            .detach()
            .cpu()
            .tolist()
        )
        expected_prompt_ids = item["prompt_ids"]
        if actual_prompt_ids != expected_prompt_ids:
            raise RuntimeError(
                "❌ TEST 批量编码后的 Prompt 与训练入口不一致："
                f"source_index={item['source_index']}, "
                f"actual_len={len(actual_prompt_ids)}, "
                f"expected_len={len(expected_prompt_ids)}"
            )

    input_len = inputs.input_ids.shape[1]
    output_ids = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
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

    del inputs
    del output_ids
    del generated_ids
    del trimmed_generated_ids
    return raw_texts


def _generate_raw_texts_with_oom_fallback(
    model,
    tokenizer,
    batch: List[dict],
    device,
    stop_ids: Sequence[int],
    max_new_tokens: int,
) -> List[str]:
    """
    首先按配置 batch_size 生成；若某个长 batch CUDA OOM，则只将该批
    自动二分重试。完整测试集不减少，贪心解码结果与指标口径不改变。
    """
    try:
        return _generate_raw_texts_once(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            device=device,
            stop_ids=stop_ids,
            max_new_tokens=max_new_tokens,
        )
    except RuntimeError as exc:
        if not is_cuda_oom_error(exc):
            raise

        failed_size = len(batch)
        clear_cuda_memory(
            f"TEST 生成 batch={failed_size} OOM 后准备自动二分重试"
        )

        if failed_size <= 1:
            raise RuntimeError(
                "❌ TEST 生成在 batch=1 时仍然 CUDA OOM。"
                "需要缩短 max_seq_length/max_new_tokens，或检查其他进程占用。"
            ) from exc

        split_at = failed_size // 2
        left_batch = batch[:split_at]
        right_batch = batch[split_at:]
        print(
            f"⚠️ TEST 生成 batch={failed_size} 显存不足，自动按 "
            f"{len(left_batch)} + {len(right_batch)} 重试；"
            "batch_size 配置值保持不变。"
        )

        left_texts = _generate_raw_texts_with_oom_fallback(
            model=model,
            tokenizer=tokenizer,
            batch=left_batch,
            device=device,
            stop_ids=stop_ids,
            max_new_tokens=max_new_tokens,
        )
        right_texts = _generate_raw_texts_with_oom_fallback(
            model=model,
            tokenizer=tokenizer,
            batch=right_batch,
            device=device,
            stop_ids=stop_ids,
            max_new_tokens=max_new_tokens,
        )
        return left_texts + right_texts


def run_test_generation(
    model,
    tokenizer,
    device,
    test_records: List[dict],
    label_space: dict,
    batch_size: int,
    max_new_tokens: int,
    predictions_path: str,
    preview_samples: int,
) -> Tuple[List[Set[int]], List[Set[int]], List[bool], dict]:
    """使用与最新训练验证完全一致的生成、解析和指标输入逻辑。"""
    max_depth = int(label_space["max_depth"])
    label2id = label_space["label2id"]

    true_id_sets: List[Set[int]] = []
    pred_id_sets: List[Set[int]] = []
    invalid_flags: List[bool] = []

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
    actual_empty_raw_generation_count = 0
    parse_reason_counter = Counter()

    eot_id = _get_eot_id(tokenizer)
    stop_ids = []
    for stop_id in (tokenizer.eos_token_id, eot_id):
        if stop_id is not None and int(stop_id) not in stop_ids:
            stop_ids.append(int(stop_id))
    if not stop_ids:
        raise RuntimeError("❌ TEST generation 没有可用 eos/eot stop id。")

    output_parent = os.path.dirname(predictions_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    old_padding_side = tokenizer.padding_side
    old_config_use_cache = model.config.use_cache
    old_gen_use_cache = getattr(model.generation_config, "use_cache", None)

    tokenizer.padding_side = "left"
    model.config.use_cache = True
    if old_gen_use_cache is not None:
        model.generation_config.use_cache = True

    clear_cuda_memory("TEST 生成开始前")
    start_time = time.time()
    shown = 0

    try:
        with open(predictions_path, "w", encoding="utf-8") as output_file:
            with torch.inference_mode():
                for start in tqdm(
                    range(0, len(test_records), batch_size),
                    desc="TEST Generate",
                ):
                    batch = test_records[start:start + batch_size]
                    raw_texts = _generate_raw_texts_with_oom_fallback(
                        model=model,
                        tokenizer=tokenizer,
                        batch=batch,
                        device=device,
                        stop_ids=stop_ids,
                        max_new_tokens=max_new_tokens,
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

                        true_ids = set(item["true_label_ids"])
                        pred_ids = set(prediction["pred_ids"])
                        invalid = bool(prediction["invalid_for_micro"])

                        if prediction["parse_ok"]:
                            exact_parse_count += 1
                        if prediction["all_labels_valid"]:
                            valid_output_count += 1
                        if (
                            prediction["all_labels_valid"]
                            and prediction["hierarchy_closed"]
                        ):
                            hierarchy_closed_count += 1
                        if not invalid and pred_ids == true_ids:
                            exact_match_count += 1

                        total_unknown_labels += len(
                            prediction["unknown_labels"]
                        )
                        total_wrong_depth_labels += len(
                            prediction["wrong_depth_labels"]
                        )
                        total_duplicate_labels += int(
                            prediction["duplicate_count"]
                        )
                        total_whitespace_error_labels += len(
                            prediction["whitespace_error_labels"]
                        )
                        total_order_error_depths += len(
                            prediction["order_error_depths"]
                        )
                        empty_prediction_count += int(
                            prediction["empty_prediction"]
                        )
                        missing_top_prediction_count += int(
                            prediction["missing_top_prediction"]
                        )
                        actual_empty_raw_generation_count += int(
                            not _strip_trailing_stop_strings(raw_resp)
                        )
                        parse_reason_counter[parse_reason] += 1

                        true_id_sets.append(true_ids)
                        pred_id_sets.append(pred_ids)
                        invalid_flags.append(invalid)

                        sorted_pred_labels = sorted(
                            prediction["pred_labels"],
                            key=label2id.get,
                        )

                        record = {
                            "source_index": int(item["source_index"]),
                            "body_was_truncated": bool(
                                item["body_was_truncated"]
                            ),
                            "raw_body_token_count": int(
                                item["raw_body_token_count"]
                            ),
                            "truncated_body_token_count": int(
                                item["truncated_body_token_count"]
                            ),
                            "true_labels": item["true_labels"],
                            "true_label_ids": sorted(true_ids),
                            "true_layers": {
                                str(depth): labels
                                for depth, labels in item["true_layers"].items()
                            },
                            "true_leaf_labels": item["true_leaf_labels"],
                            "true_leaf_label_ids": item["true_leaf_label_ids"],
                            "true_leaf_count": int(item["true_leaf_count"]),
                            "is_multi_path": bool(item["is_multi_path"]),
                            "raw_generation": raw_resp,
                            "parse_reason": parse_reason,
                            "parsed_layers": (
                                {
                                    str(depth): labels
                                    for depth, labels in parsed_layers.items()
                                }
                                if parsed_layers is not None
                                else None
                            ),
                            "pred_labels": sorted_pred_labels,
                            "pred_label_ids": sorted(pred_ids),
                            "parse_ok": bool(prediction["parse_ok"]),
                            "all_labels_valid": bool(
                                prediction["all_labels_valid"]
                            ),
                            "hierarchy_closed": bool(
                                prediction["hierarchy_closed"]
                            ),
                            "invalid_for_micro": invalid,
                            "exact_set_match": bool(
                                not invalid and pred_ids == true_ids
                            ),
                            "unknown_labels": prediction["unknown_labels"],
                            "wrong_depth_labels": prediction[
                                "wrong_depth_labels"
                            ],
                            "duplicate_count": int(
                                prediction["duplicate_count"]
                            ),
                            "whitespace_error_labels": prediction[
                                "whitespace_error_labels"
                            ],
                            "order_error_depths": prediction[
                                "order_error_depths"
                            ],
                            "empty_prediction": bool(
                                prediction["empty_prediction"]
                            ),
                            "missing_top_prediction": bool(
                                prediction["missing_top_prediction"]
                            ),
                            "actual_empty_raw_generation": bool(
                                not _strip_trailing_stop_strings(raw_resp)
                            ),
                        }
                        output_file.write(
                            json.dumps(record, ensure_ascii=False) + "\n"
                        )

                        if shown < preview_samples:
                            shown += 1
                            print("\n" + "-" * 88)
                            print(f"样本 {item['source_index']}")
                            print(f"真实标签: {item['true_labels']}")
                            print(f"预测标签: {sorted_pred_labels}")
                            print(f"严格解析状态: {parse_reason}")
                            print(
                                "有效/祖先闭包/非法: "
                                f"{prediction['all_labels_valid']}/"
                                f"{prediction['hierarchy_closed']}/"
                                f"{invalid}"
                            )
                            print("模型原始输出:")
                            print(repr(raw_resp))
                            print("-" * 88)
    finally:
        tokenizer.padding_side = old_padding_side
        model.config.use_cache = old_config_use_cache
        if old_gen_use_cache is not None:
            model.generation_config.use_cache = old_gen_use_cache
        clear_cuda_memory("TEST 生成完成后")

    elapsed_seconds = time.time() - start_time
    sample_count = max(1, len(test_records))

    diagnostics = {
        "num_samples": len(test_records),
        "elapsed_seconds": float(elapsed_seconds),
        "seconds_per_sample": float(elapsed_seconds / sample_count),
        "exact_parse_rate": exact_parse_count / sample_count,
        "valid_output_rate": valid_output_count / sample_count,
        "ancestor_closure_rate": hierarchy_closed_count / sample_count,
        "exact_set_match_rate": exact_match_count / sample_count,
        "invalid_output_rate": sum(invalid_flags) / sample_count,
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
        "actual_empty_raw_generation_count": int(
            actual_empty_raw_generation_count
        ),
        "strict_parse_failure_count": int(
            len(test_records) - exact_parse_count
        ),
        "parse_reason_counts": dict(parse_reason_counter),
    }

    return true_id_sets, pred_id_sets, invalid_flags, diagnostics


# ======================================================================
# 8. Main
# ======================================================================
def main(args) -> None:
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

    model, tokenizer, device, checkpoint_type = load_model_and_tokenizer(args)

    token_ids = validate_atomic_reserved_tokens(
        tokenizer,
        expected_max_depth=args.expected_max_depth,
    )
    check_reserved_token_layout(
        tokenizer=tokenizer,
        system_prompt=args.system_prompt,
        token_ids=token_ids,
        label_space=label_space,
    )

    test_records, sequence_summary = prepare_test_records(
        test_data=test_data,
        tokenizer=tokenizer,
        label_space=label_space,
        system_prompt=args.system_prompt,
        max_seq_length=args.max_seq_length,
        reserve_tokens=args.reserve_tokens,
        max_new_tokens=args.max_new_tokens,
        token_ids=token_ids,
    )

    predictions_path = os.path.join(args.output_dir, "test_predictions.jsonl")
    metrics_path = os.path.join(args.output_dir, "test_metrics.json")

    print("\n" + "=" * 88)
    print("🚀 开始 BGC 学生模型最终测试集详细诊断")
    print(f"模型目录: {args.model_path}")
    print(f"模型类型: {checkpoint_type}")
    print(
        "Tokenizer 来源: "
        f"{args.base_model_path if checkpoint_type == 'peft' else args.model_path}"
    )
    print(f"测试数据: {args.test_data_path}")
    print(f"测试样本: {len(test_records)}")
    print(f"batch_size: {args.batch_size}")
    print(f"max_seq_length: {args.max_seq_length}")
    print(f"reserve_tokens: {args.reserve_tokens}")
    print(f"正文上限: {args.max_seq_length - args.reserve_tokens}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print("生成方式: greedy decoding, do_sample=False")
    print("层内顺序: D1 按固定 label ID；D2-D4 按上一层父节点顺序分组，同父节点内按固定 label ID")
    print("=" * 88 + "\n")

    true_id_sets, pred_id_sets, invalid_flags, diagnostics = (
        run_test_generation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            test_records=test_records,
            label_space=label_space,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            predictions_path=predictions_path,
            preview_samples=args.preview_samples,
        )
    )

    metrics = compute_multilabel_metric_bundle(
        true_id_sets=true_id_sets,
        pred_id_sets=pred_id_sets,
        invalid_flags=invalid_flags,
        label_space=label_space,
    )

    detailed_analysis = compute_detailed_error_analysis(
        true_id_sets=true_id_sets,
        pred_id_sets=pred_id_sets,
        invalid_flags=invalid_flags,
        test_records=test_records,
        label_space=label_space,
        output_dir=args.output_dir,
        top_k=args.analysis_top_k,
    )

    report = {
        "model_path": args.model_path,
        "checkpoint_type": checkpoint_type,
        "base_model_path": args.base_model_path,
        "test_data_path": args.test_data_path,
        "label_space_path": args.label_space_path,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "reserve_tokens": args.reserve_tokens,
        "max_text_tokens": args.max_seq_length - args.reserve_tokens,
        "max_new_tokens": args.max_new_tokens,
        "sequence_summary": sequence_summary,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "detailed_error_analysis": detailed_analysis,
        "evaluation_protocol": {
            "prompt_matches_latest_training": True,
            "uses_final_test_split": True,
            "strict_exact_four_line_parser": True,
            "global_macro_rule": "present ground-truth labels only",
            "global_macro_all146_retained": True,
            "known_wrong-depth_labels_kept_in_flat_global_f1": True,
            "stop_at_first_eos_or_eot": True,
            "greedy_generation": True,
            "peft_tokenizer_loaded_from_base_model": (
                checkpoint_type == "peft"
            ),
            "within_depth_order_rule": (
                "D1 uses canonical label-ID order; D2-D4 group labels "
                "by parent order in the previous depth, then use "
                "canonical label-ID order within each parent group"
            ),
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("🎯 BGC TEST 基础指标（与最新训练验证口径一致）")
    print("正式指标")
    print(
        f"Global Micro-F1          : {metrics['global_micro']:.6f} "
        f"({metrics['global_micro'] * 100:.2f})"
    )
    print(
        f"Global Macro-F1(Present): {metrics['global_macro']:.6f} "
        f"({metrics['global_macro'] * 100:.2f}) | "
        f"Present={metrics['present_label_count']}/"
        f"{label_space['num_labels']}"
    )
    print(
        f"Global Macro-F1-All146  : "
        f"{metrics['global_macro_all_labels']:.6f} "
        f"({metrics['global_macro_all_labels'] * 100:.2f})"
    )
    print(
        f"Samples-F1              : {metrics['global_samples']:.6f} "
        f"({metrics['global_samples'] * 100:.2f})"
    )

    print("\n严格生成诊断指标")
    print(
        f"Strict Micro-F1         : {metrics['strict_micro']:.6f} "
        f"({metrics['strict_micro'] * 100:.2f})"
    )
    print(
        f"Strict Samples-F1       : {metrics['strict_samples']:.6f} "
        f"({metrics['strict_samples'] * 100:.2f})"
    )

    print("\n各层指标")
    for current_depth in range(1, label_space["max_depth"] + 1):
        print(
            f"D{current_depth} Micro/Macro: "
            f"{metrics[f'depth_{current_depth}_micro']:.6f}/"
            f"{metrics[f'depth_{current_depth}_macro']:.6f} | "
            f"PresentLabels="
            f"{metrics[f'depth_{current_depth}_present_labels']}"
        )

    print("\n生成与层级合法性")
    print(f"Exact Parse Rate       : {diagnostics['exact_parse_rate']:.6f}")
    print(f"Valid Output Rate      : {diagnostics['valid_output_rate']:.6f}")
    print(
        f"Ancestor Closure Rate  : "
        f"{diagnostics['ancestor_closure_rate']:.6f}"
    )
    print(
        f"Exact Set Match Rate   : "
        f"{diagnostics['exact_set_match_rate']:.6f}"
    )
    print(
        f"Invalid Output Rate    : "
        f"{diagnostics['invalid_output_rate']:.6f}"
    )
    print(
        "Unknown/WrongDepth/Duplicate/Whitespace/OrderDepth: "
        f"{diagnostics['unknown_label_count']}/"
        f"{diagnostics['wrong_depth_label_count']}/"
        f"{diagnostics['duplicate_label_count']}/"
        f"{diagnostics['whitespace_error_label_count']}/"
        f"{diagnostics['canonical_order_error_depth_count']}"
    )
    print(
        "StrictParseFail/EmptyPrediction/ActualEmptyRaw/MissingTop: "
        f"{diagnostics['strict_parse_failure_count']}/"
        f"{diagnostics['empty_prediction_count']}/"
        f"{diagnostics['actual_empty_raw_generation_count']}/"
        f"{diagnostics['missing_top_prediction_count']}"
    )
    print(f"Parse reason counts    : {diagnostics['parse_reason_counts']}")
    print(
        f"总耗时: {diagnostics['elapsed_seconds'] / 3600:.3f} 小时 | "
        f"平均每条: {diagnostics['seconds_per_sample']:.3f} 秒"
    )
    print(f"逐样本结果: {predictions_path}")
    print(f"汇总指标:   {metrics_path}")
    print("=" * 100 + "\n")


# ======================================================================
# 9. 参数解析
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
            "Diagnose the BGC generative student model on test data with the exact Prompt, "
            "truncation, stopping, parsing, metrics, and OOM fallback "
            "used by the latest parent-ordered training validation, now applied to the final BGC test split."
        )
    )

    # ================= 模型与数据路径 =================
    # 默认测试周期 PEFT checkpoint-7500；tokenizer 从基础模型目录加载。
    parser.add_argument(
        "--model_path",
        type=str,
        default=(
            "/home/lzp/dataspace/checkpoints/"
            "bgc_student_1token/best_sft_model/"
        ),
    )
    parser.add_argument(
        "--checkpoint_type",
        type=str,
        choices=("auto", "merged", "peft"),
        default="auto",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/home/lzp/dataspace/llama-3.2_1b/",
    )
    parser.add_argument(
        "--merge_peft_for_test",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="/home/lzp/dataspace/data/bgc/test_formatted.jsonl",
    )
    parser.add_argument(
        "--label_space_path",
        type=str,
        default="/home/lzp/dataspace/data/bgc/label_space.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "/home/lzp/dataspace/checkpoints/"
            "bgc_student_1token/test_results_best/"
        ),
    )

    # ================= 与训练严格一致的输入参数 =================
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_num_labels", type=int, default=146)
    parser.add_argument("--expected_max_depth", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=3072)
    parser.add_argument("--reserve_tokens", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=520)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are a specialized classifier for the BGC hierarchical "
            "multi-label taxonomy. Follow the required four-line "
            "JSON-array output format exactly. Do not produce explanations "
            "or any extra conversational text."
        ),
    )

    # ================= 推理参数 =================
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--preview_samples", type=int, default=5)
    parser.add_argument("--analysis_top_k", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")

    args = parser.parse_args()
    main(args)
