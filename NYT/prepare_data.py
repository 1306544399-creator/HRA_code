import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict, deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ======================================================================
# 0. 通用工具
# ======================================================================
def str2bool(value):
    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_preview(text: str, max_chars: int = 180) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[:max_chars] + "..."


def load_json_or_jsonl(path: str) -> List[Any]:
    """
    同时支持：
      1. 顶层为 list 的普通 JSON；
      2. 每行一个 JSON object 的 JSONL。

    NYT 的 qd 文件虽然扩展名通常是 .json，实际有可能是 JSONL，
    因此不能只按一种格式读取。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到输入文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        raise ValueError(f"输入文件为空: {path}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        rows: List[Any] = []
        for line_no, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} 第 {line_no} 行不是合法 JSON: {exc}"
                ) from exc
        return rows

    if not isinstance(data, list):
        raise TypeError(
            f"{path} 的 JSON 顶层必须是 list，实际为 {type(data).__name__}。"
        )

    return data


# ======================================================================
# 1. NYT taxonomy / vocab
# ======================================================================
def parse_nyt_taxonomy(taxonomy_path: str):
    """
    解析 HiAGM/HILL 使用的 NYT taxonomy。

    文件每行格式为：
        父节点<TAB>子节点1<TAB>子节点2<...>

    NYT 不是 WOS 的固定两层结构，而是：
      - 多路径；
      - 最大深度 8；
      - 除 Root 外共有 166 个任务标签。
    """
    if not os.path.isfile(taxonomy_path):
        raise FileNotFoundError(f"找不到 taxonomy: {taxonomy_path}")

    children: Dict[str, List[str]] = {}
    parent: Dict[str, str] = {}

    with open(taxonomy_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            raw_line = raw_line.rstrip("\n\r")
            if not raw_line.strip():
                continue

            parts = [part.strip() for part in raw_line.split("\t")]
            parts = [part for part in parts if part]

            if len(parts) < 2:
                raise ValueError(
                    f"taxonomy 第 {line_no} 行至少需要一个父节点和一个子节点: "
                    f"{raw_line!r}"
                )

            node = parts[0]
            node_children = parts[1:]

            if node in children:
                raise ValueError(f"taxonomy 中父节点重复出现: {node!r}")
            if len(node_children) != len(set(node_children)):
                raise ValueError(
                    f"taxonomy 第 {line_no} 行存在重复子节点: {node!r}"
                )

            children[node] = node_children

            for child in node_children:
                if child in parent and parent[child] != node:
                    raise ValueError(
                        f"NYT taxonomy 不是树：{child!r} 同时属于 "
                        f"{parent[child]!r} 和 {node!r}。"
                    )
                parent[child] = node

    if "Root" not in children:
        raise ValueError("taxonomy 中未找到 Root。")
    if "Root" in parent:
        raise ValueError("Root 不应存在父节点。")

    all_nodes: Set[str] = {"Root"}
    all_nodes.update(children.keys())
    all_nodes.update(parent.keys())

    # 从 Root 做 BFS，检查可达性、环和深度。
    depth: Dict[str, int] = {"Root": 0}
    queue = deque(["Root"])

    while queue:
        node = queue.popleft()
        for child in children.get(node, []):
            proposed_depth = depth[node] + 1
            if child in depth:
                if depth[child] != proposed_depth:
                    raise ValueError(
                        f"taxonomy 深度冲突或存在环: {child!r}"
                    )
                continue
            depth[child] = proposed_depth
            queue.append(child)

    unreachable = all_nodes - set(depth)
    if unreachable:
        raise ValueError(
            "taxonomy 中存在无法从 Root 到达的节点: "
            f"{sorted(unreachable)[:20]}"
        )

    # 某些叶节点不会单独作为父节点出现在文件中，补为空列表。
    for node in all_nodes:
        children.setdefault(node, [])

    top_ancestor: Dict[str, str] = {}
    ancestors: Dict[str, List[str]] = {}

    for node in all_nodes:
        if node == "Root":
            ancestors[node] = []
            continue

        chain: List[str] = []
        current = node
        seen: Set[str] = set()

        while current != "Root":
            if current in seen:
                raise ValueError(f"taxonomy 存在环: {node!r}")
            seen.add(current)
            chain.append(current)

            if current not in parent:
                raise ValueError(f"节点缺少父节点: {current!r}")
            current = parent[current]

        chain.reverse()
        ancestors[node] = chain
        top_ancestor[node] = chain[0]

    return {
        "children": children,
        "parent": parent,
        "depth": depth,
        "ancestors": ancestors,
        "top_ancestor": top_ancestor,
        "nodes": all_nodes,
        "top_labels": list(children["Root"]),
        "max_depth": max(depth.values()),
    }


def load_nyt_vocab(vocab_path: str) -> List[str]:
    """
    读取 nyt_label.vocab。

    官方文件第一行是 Root，其后 166 行是任务标签。标签名称中包含空格，
    因此必须逐行读取，不能按普通空白字符 split。
    """
    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(f"找不到 vocab: {vocab_path}")

    with open(vocab_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    if not labels:
        raise ValueError(f"vocab 为空: {vocab_path}")
    if labels[0] != "Root":
        raise ValueError(
            f"nyt_label.vocab 第一行必须是 Root，实际为 {labels[0]!r}。"
        )
    if len(labels) != len(set(labels)):
        duplicates = [
            label for label, count in Counter(labels).items() if count > 1
        ]
        raise ValueError(f"vocab 中存在重复标签: {duplicates[:20]}")

    return labels


def build_label_space(
    taxonomy_path: str,
    vocab_path: str,
    expected_num_labels: int,
    expected_max_depth: int,
):
    hierarchy = parse_nyt_taxonomy(taxonomy_path)
    vocab_with_root = load_nyt_vocab(vocab_path)
    task_labels = vocab_with_root[1:]

    taxonomy_task_nodes = hierarchy["nodes"] - {"Root"}
    vocab_task_nodes = set(task_labels)

    if taxonomy_task_nodes != vocab_task_nodes:
        missing = taxonomy_task_nodes - vocab_task_nodes
        extra = vocab_task_nodes - taxonomy_task_nodes
        raise ValueError(
            "nyt_label.vocab 与 nyt.taxonomy 的节点集合不一致。\n"
            f"vocab 缺失={sorted(missing)[:20]}\n"
            f"vocab 多余={sorted(extra)[:20]}"
        )

    if expected_num_labels > 0 and len(task_labels) != expected_num_labels:
        raise ValueError(
            f"NYT 标签数异常: {len(task_labels)} != {expected_num_labels}"
        )

    if (
        expected_max_depth > 0
        and hierarchy["max_depth"] != expected_max_depth
    ):
        raise ValueError(
            f"NYT 最大深度异常: {hierarchy['max_depth']} "
            f"!= {expected_max_depth}"
        )

    label2id = {label: idx for idx, label in enumerate(task_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    top_labels = hierarchy["top_labels"]
    if any(hierarchy["depth"][label] != 1 for label in top_labels):
        raise ValueError("Root 的直接子节点深度检查失败。")

    return {
        **hierarchy,
        "vocab_with_root": vocab_with_root,
        "labels": task_labels,
        "label2id": label2id,
        "id2label": id2label,
    }



def _torch_load_cpu(path: str):
    """兼容不同 PyTorch 版本，安全地在 CPU 上读取标签字典。"""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "读取 label_dict.pt / label_dict_rev.pt 需要安装 PyTorch。"
        ) from exc

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        # 某些旧文件包含普通 Python 容器，weights_only=True 可能拒绝；
        # 这里退回旧式加载，但文件必须来自用户自己的可信数据目录。
        return torch.load(path, map_location="cpu")


def _int_like(value: Any) -> Optional[int]:
    """把 Python/NumPy/PyTorch 标量转换为 int；失败时返回 None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return int(value)
    try:
        # 只接受没有小数部分的整数字符串/标量。
        parsed = int(value)
        if str(value).strip() == str(parsed):
            return parsed
    except (TypeError, ValueError):
        pass
    return None


def _extract_mapping_candidates(obj: Any, source_name: str) -> List[Tuple[str, Dict]]:
    """
    从 .pt 对象中提取可能的 label->id 或 id->label 映射。

    兼容：
      - 直接保存的 dict；
      - 包在 label_dict / label2id / id2label 等字段中的 dict；
      - 按 ID 顺序保存的字符串 list/tuple。
    """
    candidates: List[Tuple[str, Dict]] = []

    if isinstance(obj, (list, tuple)) and all(isinstance(x, str) for x in obj):
        candidates.append((f"{source_name}:sequence_id2label", {i: x for i, x in enumerate(obj)}))
        return candidates

    if not isinstance(obj, dict):
        return candidates

    # 先检查当前 dict 本身。
    if obj:
        values_as_int = [_int_like(v) for v in obj.values()]
        keys_as_int = [_int_like(k) for k in obj.keys()]

        if all(isinstance(k, str) for k in obj.keys()) and all(v is not None for v in values_as_int):
            candidates.append(
                (
                    f"{source_name}:direct_label2id",
                    {str(k).strip(): int(v) for k, v in zip(obj.keys(), values_as_int)},
                )
            )

        if all(k is not None for k in keys_as_int) and all(isinstance(v, str) for v in obj.values()):
            candidates.append(
                (
                    f"{source_name}:direct_id2label",
                    {int(k): str(v).strip() for k, v in zip(keys_as_int, obj.values())},
                )
            )

    # 再递归检查常见嵌套字段。
    preferred_keys = (
        "label_dict",
        "label2id",
        "label_to_id",
        "label_dict_rev",
        "id2label",
        "id_to_label",
        "mapping",
        "vocab",
    )
    for key in preferred_keys:
        if key in obj:
            candidates.extend(
                _extract_mapping_candidates(obj[key], f"{source_name}.{key}")
            )

    return candidates


def load_source_integer_label_mapping(
    label_dict_path: str,
    label_dict_rev_path: str,
    label_space: dict,
) -> Dict[int, str]:
    """
    读取 qd 文件原本使用的整数标签字典。

    关键原则：
      1. qd 中的整数 ID 先按 label_dict*.pt 恢复为标签字符串；
      2. 再按 nyt_label.vocab 的唯一全局顺序重新编码输出。

    因此不会把“源数据 ID 顺序”和“最终统一 ID 顺序”误认为同一套顺序。
    """
    paths = [label_dict_path, label_dict_rev_path]
    existing_paths = [path for path in paths if path and os.path.isfile(path)]
    if not existing_paths:
        raise FileNotFoundError(
            "没有找到 qd 整数标签映射文件。请确认以下文件至少存在一个：\n"
            f"  {label_dict_path}\n"
            f"  {label_dict_rev_path}"
        )

    canonical_set = set(label_space["labels"])
    normalized_id2label_candidates: List[Tuple[str, Dict[int, str]]] = []

    for path in existing_paths:
        raw = _torch_load_cpu(path)
        for name, mapping in _extract_mapping_candidates(raw, os.path.basename(path)):
            if not mapping:
                continue

            # label -> id
            if all(isinstance(k, str) for k in mapping.keys()):
                id2label: Dict[int, str] = {}
                valid = True
                for label, raw_id in mapping.items():
                    label = str(label).strip()
                    idx = _int_like(raw_id)
                    if idx is None:
                        valid = False
                        break
                    if label in {"Root", "Top"}:
                        continue
                    if idx in id2label and id2label[idx] != label:
                        raise ValueError(
                            f"{name} 中源 ID={idx} 同时映射到 "
                            f"{id2label[idx]!r} 和 {label!r}。"
                        )
                    id2label[idx] = label
                if valid and id2label:
                    normalized_id2label_candidates.append((name, id2label))

            # id -> label
            elif all(isinstance(k, int) for k in mapping.keys()):
                id2label = {}
                for raw_id, raw_label in mapping.items():
                    label = str(raw_label).strip()
                    if label in {"Root", "Top"}:
                        continue
                    id2label[int(raw_id)] = label
                if id2label:
                    normalized_id2label_candidates.append((name, id2label))

    if not normalized_id2label_candidates:
        raise ValueError(
            "label_dict.pt / label_dict_rev.pt 中没有识别出合法的 "
            "label->id 或 id->label 映射。"
        )

    # 只接受标签集合与 166 个任务标签完全相同的候选。
    exact_candidates: List[Tuple[str, Dict[int, str]]] = []
    diagnostics = []
    for name, id2label in normalized_id2label_candidates:
        labels = list(id2label.values())
        label_set = set(labels)
        duplicate_labels = len(labels) - len(label_set)
        missing = canonical_set - label_set
        extra = label_set - canonical_set
        diagnostics.append(
            (
                name,
                len(id2label),
                duplicate_labels,
                sorted(missing)[:10],
                sorted(extra)[:10],
            )
        )
        if not missing and not extra and duplicate_labels == 0:
            exact_candidates.append((name, id2label))

    if not exact_candidates:
        lines = [
            "源标签字典与 nyt_label.vocab 的 166 个任务标签不完全一致。",
            "候选诊断：",
        ]
        for name, count, dup, missing, extra in diagnostics:
            lines.append(
                f"  {name}: entries={count}, duplicate_labels={dup}, "
                f"missing={missing}, extra={extra}"
            )
        raise ValueError("\n".join(lines))

    chosen_name, chosen = exact_candidates[0]

    # 如果多个文件都给出了完整映射，它们必须逐 ID 一致。
    for other_name, other in exact_candidates[1:]:
        if other != chosen:
            differing_ids = sorted(set(chosen) | set(other))
            examples = []
            for idx in differing_ids:
                if chosen.get(idx) != other.get(idx):
                    examples.append((idx, chosen.get(idx), other.get(idx)))
                if len(examples) >= 20:
                    break
            raise ValueError(
                "label_dict.pt 与 label_dict_rev.pt 的源 ID 映射不一致。\n"
                f"基准={chosen_name}, 对比={other_name}\n"
                f"前 20 个差异={examples}"
            )

    source_ids = sorted(chosen)
    contiguous_zero_based = source_ids == list(range(len(chosen)))
    contiguous_one_based = source_ids == list(range(1, len(chosen) + 1))

    print("\n" + "=" * 76)
    print("✅ qd 源整数标签映射校验完成")
    print(f"使用映射: {chosen_name}")
    print(f"源标签数量: {len(chosen)}")
    print(f"源 ID 最小/最大值: {source_ids[0]} / {source_ids[-1]}")
    print(f"源 ID 为连续 0-based: {contiguous_zero_based}")
    print(f"源 ID 为连续 1-based: {contiguous_one_based}")
    print("说明：qd 整数 ID 先恢复为字符串，再按 nyt_label.vocab 统一重编码。")
    print("=" * 76 + "\n")

    return dict(sorted(chosen.items()))


def save_label_space(label_space: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    payload = {
        "num_labels": len(label_space["labels"]),
        "max_depth": int(label_space["max_depth"]),
        "label2id": label_space["label2id"],
        "id2label": {
            str(idx): label for idx, label in label_space["id2label"].items()
        },
        "top_labels": label_space["top_labels"],
        "parent": label_space["parent"],
        "children": label_space["children"],
        "depth": {
            label: int(value) for label, value in label_space["depth"].items()
        },
        "top_ancestor": label_space["top_ancestor"],
        "canonical_order_source": "nyt_label.vocab excluding Root",
        "task_definition": (
            "NYT hierarchical multi-label classification; Root is excluded from "
            "the 166 task labels"
        ),
        "generation_target": (
            "line 1 generates depth-1 coarse labels; line 2 generates "
            "sample-specific deepest endpoint labels"
        ),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ======================================================================
# 2. 从 qd JSON 提取正文和标签
# ======================================================================
TEXT_FIELD_CANDIDATES = (
    "text",
    "doc_text",
    "document",
    "content",
    "raw_text",
    "article",
    "doc_token",
    "token",
    "tokens",
    "input_ids",
)

LABEL_FIELD_CANDIDATES = (
    "labels",
    "label",
    "doc_label",
    "doc_labels",
    "targets",
    "target",
    "y",
    "Y",
)


class LazyBertDecoder:
    """只在确实遇到 BERT token ID 列表时才加载 tokenizer。"""

    def __init__(self, tokenizer_path: str):
        self.tokenizer_path = tokenizer_path
        self._tokenizer = None

    def get(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "检测到正文是 BERT token ID，但当前环境没有 transformers。"
                ) from exc

            print(
                "\n检测到 qd 文件中的正文是整数 token ID，"
                f"正在加载 BERT tokenizer: {self.tokenizer_path}"
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path
            )
        return self._tokenizer


def _find_first_present_field(item: dict, candidates: Sequence[str]):
    for field in candidates:
        if field in item:
            return field, item[field]
    return None, None


def _normalize_text_value(
    value: Any,
    decoder: LazyBertDecoder,
) -> Tuple[str, str]:
    """
    返回：
      text
      source_mode
    """
    if value is None:
        return "", "none"

    if isinstance(value, str):
        return value.strip(), "string"

    if isinstance(value, dict):
        for nested_key in ("text", "content", "input_ids", "ids"):
            if nested_key in value:
                return _normalize_text_value(value[nested_key], decoder)
        raise TypeError(
            f"正文 dict 中未找到可识别字段，keys={sorted(value.keys())}"
        )

    if isinstance(value, (list, tuple)):
        values = list(value)
        if not values:
            return "", "empty_list"

        if all(isinstance(x, str) for x in values):
            return " ".join(values).strip(), "string_tokens"

        if all(
            isinstance(x, int) and not isinstance(x, bool)
            for x in values
        ):
            tokenizer = decoder.get()
            text = tokenizer.decode(
                values,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return text.strip(), "bert_token_ids"

        raise TypeError(
            "正文 list 必须全部为 str 或全部为 int，"
            f"实际前 20 个元素类型={[type(x).__name__ for x in values[:20]]}"
        )

    raise TypeError(f"无法识别正文类型: {type(value).__name__}")


def extract_text(
    item: dict,
    decoder: LazyBertDecoder,
    split_name: str,
    index: int,
) -> Tuple[str, str, str]:
    field, value = _find_first_present_field(item, TEXT_FIELD_CANDIDATES)
    if field is None:
        raise KeyError(
            f"{split_name} 第 {index} 条未找到正文字段，"
            f"候选字段={list(TEXT_FIELD_CANDIDATES)}，"
            f"实际 keys={sorted(item.keys())}"
        )

    text, mode = _normalize_text_value(value, decoder)
    text = text.strip()

    if not text:
        raise ValueError(
            f"{split_name} 第 {index} 条正文为空，来源字段={field!r}。"
        )

    return text, field, mode


def _truthy_multihot(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return bool(value)


def _map_integer_label_ids(
    values: Sequence[int],
    label_space: dict,
    integer_label_mode: str,
    source_id2label: Optional[Dict[int, str]] = None,
) -> List[str]:
    ids = [int(value) for value in values]
    if not ids:
        return []

    # 首选 qd 自带的 label_dict*.pt；这能精确恢复源 ID 含义，
    # 不再根据单个样本是否恰好出现 0/166 来猜测编码方式。
    if source_id2label is not None:
        unknown_ids = sorted(set(ids) - set(source_id2label))
        if unknown_ids:
            raise ValueError(
                "qd 中出现源标签字典未定义的整数 ID: "
                f"{unknown_ids[:20]}"
            )
        return [source_id2label[idx] for idx in ids]

    # 仅当没有提供源标签字典时保留旧兼容逻辑。
    num_labels = len(label_space["labels"])
    vocab_with_root = label_space["vocab_with_root"]
    id2label = label_space["id2label"]

    mode = integer_label_mode
    if mode == "auto":
        raise ValueError(
            "整数标签不能按单个样本可靠自动判断是 0-based 还是 1-based。"
            "请提供 label_dict.pt / label_dict_rev.pt，或显式设置 "
            "--integer_label_mode。"
        )

    if mode == "zero_based_without_root":
        labels: List[str] = []
        for value in ids:
            if value < 0 or value >= num_labels:
                raise ValueError(
                    f"zero_based_without_root 标签 ID 越界: {value}"
                )
            labels.append(id2label[value])
        return labels

    if mode == "vocab_index_with_root":
        labels = []
        for value in ids:
            if value < 0 or value >= len(vocab_with_root):
                raise ValueError(
                    f"vocab_index_with_root 标签 ID 越界: {value}"
                )
            label = vocab_with_root[value]
            if label != "Root":
                labels.append(label)
        return labels

    raise ValueError(f"不支持的 integer_label_mode: {integer_label_mode}")

def _normalize_label_value(
    value: Any,
    label_space: dict,
    integer_label_mode: str,
    source_id2label: Optional[Dict[int, str]] = None,
) -> Tuple[List[str], str]:
    num_labels = len(label_space["labels"])
    vocab_with_root = label_space["vocab_with_root"]

    if value is None:
        return [], "none"

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return [], "empty_string"

        # 只把完整 JSON 字符串数组当作数组解析；不能按逗号拆分，
        # 因为 NYT 标签本身可能包含逗号。
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                labels, mode = _normalize_label_value(
                    parsed,
                    label_space,
                    integer_label_mode,
                    source_id2label,
                )
                return labels, f"json_string_{mode}"

        return [stripped], "single_string"

    if isinstance(value, dict):
        labels = [str(key).strip() for key, flag in value.items() if _truthy_multihot(flag)]
        return labels, "dict_multihot"

    if isinstance(value, (list, tuple)):
        values = list(value)
        if not values:
            return [], "empty_list"

        if all(isinstance(x, str) for x in values):
            return [str(x).strip() for x in values if str(x).strip()], "string_list"

        if all(
            isinstance(x, (int, bool)) and not isinstance(x, float)
            for x in values
        ):
            int_values = [int(x) for x in values]

            # 常见 Y 向量：166 维（不含 Root）或 167 维（含 Root）。
            if (
                len(int_values) in {num_labels, num_labels + 1}
                and set(int_values).issubset({0, 1})
            ):
                if source_id2label is not None:
                    active_positions = [
                        idx for idx, flag in enumerate(int_values) if flag == 1
                    ]
                    unknown_positions = [
                        idx for idx in active_positions if idx not in source_id2label
                    ]
                    if unknown_positions:
                        raise ValueError(
                            "multi-hot 中置 1 的位置不在源标签字典中: "
                            f"{unknown_positions[:20]}"
                        )
                    labels = [source_id2label[idx] for idx in active_positions]
                    return labels, "multihot_source_mapping"

                if len(int_values) == num_labels:
                    labels = [
                        label_space["id2label"][idx]
                        for idx, flag in enumerate(int_values)
                        if flag == 1
                    ]
                    return labels, "multihot_without_root"

                labels = [
                    vocab_with_root[idx]
                    for idx, flag in enumerate(int_values)
                    if flag == 1 and vocab_with_root[idx] != "Root"
                ]
                return labels, "multihot_with_root"

            labels = _map_integer_label_ids(
                int_values,
                label_space,
                integer_label_mode,
                source_id2label,
            )
            mapping_mode = (
                "source_mapping"
                if source_id2label is not None
                else integer_label_mode
            )
            return labels, f"integer_ids_{mapping_mode}"

        raise TypeError(
            "标签 list 必须全部为字符串、整数 ID 或 multi-hot 0/1，"
            f"实际前 20 个元素类型={[type(x).__name__ for x in values[:20]]}"
        )

    raise TypeError(f"无法识别标签类型: {type(value).__name__}")


def extract_labels(
    item: dict,
    label_space: dict,
    integer_label_mode: str,
    source_id2label: Optional[Dict[int, str]],
    split_name: str,
    index: int,
) -> Tuple[List[str], str, str]:
    field, value = _find_first_present_field(item, LABEL_FIELD_CANDIDATES)
    if field is None:
        raise KeyError(
            f"{split_name} 第 {index} 条未找到标签字段，"
            f"候选字段={list(LABEL_FIELD_CANDIDATES)}，"
            f"实际 keys={sorted(item.keys())}"
        )

    labels, mode = _normalize_label_value(
        value,
        label_space,
        integer_label_mode,
        source_id2label,
    )

    labels = [label for label in labels if label and label != "Root"]
    if not labels:
        raise ValueError(
            f"{split_name} 第 {index} 条没有有效任务标签，来源字段={field!r}。"
        )

    return labels, field, mode


# ======================================================================
# 3. 单条样本的层级规范化
# ======================================================================
def canonicalize_and_validate_labels(
    raw_labels: Sequence[str],
    label_space: dict,
    split_name: str,
    index: int,
    ancestor_policy: str,
):
    label2id = label_space["label2id"]
    parent = label_space["parent"]
    depth = label_space["depth"]
    children = label_space["children"]

    cleaned = [str(label).strip() for label in raw_labels]
    cleaned = [label for label in cleaned if label and label != "Root"]

    duplicate_count = len(cleaned) - len(set(cleaned))
    label_set = set(cleaned)

    unknown = label_set - set(label2id)
    if unknown:
        raise ValueError(
            f"{split_name} 第 {index} 条含未知标签: {sorted(unknown)[:20]}"
        )

    missing_ancestors: Set[str] = set()
    for label in list(label_set):
        current = label
        while parent[current] != "Root":
            current = parent[current]
            if current not in label_set:
                missing_ancestors.add(current)

    if missing_ancestors:
        if ancestor_policy == "strict":
            raise ValueError(
                f"{split_name} 第 {index} 条不是祖先闭包标签集，缺失祖先: "
                f"{sorted(missing_ancestors, key=label2id.get)[:20]}"
            )
        if ancestor_policy == "add_missing":
            label_set.update(missing_ancestors)
        else:
            raise ValueError(f"不支持的 ancestor_policy: {ancestor_policy}")

    labels = sorted(label_set, key=label2id.get)
    label_ids = [label2id[label] for label in labels]

    top_labels = [label for label in labels if depth[label] == 1]
    top_label_ids = [label2id[label] for label in top_labels]

    # “样本端点”不是 taxonomy 的全局叶节点，而是：当前样本标签集合中，
    # 没有任何已标注直接子节点的标签。它可以是 taxonomy 内部节点。
    leaf_labels = [
        label
        for label in labels
        if not any(child in label_set for child in children[label])
    ]
    leaf_label_ids = [label2id[label] for label in leaf_labels]

    if not top_labels:
        raise ValueError(
            f"{split_name} 第 {index} 条没有任何深度 1 的粗粒度标签。"
        )
    if not leaf_labels:
        raise ValueError(f"{split_name} 第 {index} 条没有样本端点标签。")

    # HiAGM 官方 NYT split 已去掉无标签和仅一个一层标签的样本。
    # 这里不硬编码“标签数必须 >=2”，但会报告这些异常样本数量。
    return {
        "labels": labels,
        "label_ids": label_ids,
        "top_labels": top_labels,
        "top_label_ids": top_label_ids,
        "leaf_labels": leaf_labels,
        "leaf_label_ids": leaf_label_ids,
        "duplicate_label_count": duplicate_count,
        "added_ancestor_count": len(missing_ancestors)
        if ancestor_policy == "add_missing"
        else 0,
    }


# ======================================================================
# 4. 重复与 split 统计
# ======================================================================
def _decode_label_id_tuple(
    label_ids: Tuple[int, ...],
    label_space: dict,
) -> List[str]:
    """把固定顺序的标签 ID 元组恢复为标签字符串列表。"""
    id2label = label_space["id2label"]
    return [id2label[int(label_id)] for label_id in label_ids]


def _format_label_set_collection(
    label_sets: Set[Tuple[int, ...]],
    label_space: dict,
) -> List[dict]:
    """把同一正文对应的一组或多组标签集合转换成可读结构。"""
    formatted = []
    for label_ids in sorted(label_sets):
        formatted.append(
            {
                "label_ids": list(label_ids),
                "labels": _decode_label_id_tuple(label_ids, label_space),
            }
        )
    return formatted


def print_single_depth_one_examples(
    split_name: str,
    examples: List[dict],
    report_limit: int,
) -> None:
    """
    打印“整个标签集合只有一个深度 1 标签”的样本。

    这些样本不是普通的“只有一个顶层标签”，而是没有任何更深层标签：
        labels == top_labels == leaf_labels == [某个深度 1 标签]
    """
    print(f"\n🔎 {split_name} 仅含一个深度 1 标签的样本检查")
    print(f"   样本数: {len(examples)}")

    if not examples:
        print("   ✅ 未发现此类样本。")
        return

    limit = max(0, int(report_limit))
    if limit == 0:
        print("   已关闭示例打印；可增大 --single_depth_one_report_limit。")
        return

    print(f"   ⚠️ 最多打印前 {limit} 条：")
    for example in examples[:limit]:
        print(
            f"      index={example['index']} | "
            f"label={example['label']!r} | "
            f"label_id={example['label_id']}"
        )
        print(f"      preview={example['preview']!r}")

    remaining = len(examples) - min(len(examples), limit)
    if remaining > 0:
        print(f"   其余 {remaining} 条未在终端打印。")


def print_within_split_duplicate_report(
    split_name: str,
    text_hash_counter: Counter,
    hash_examples: dict,
    hash_label_sets: dict,
    report_limit: int,
) -> None:
    duplicate_hashes = {
        h: count for h, count in text_hash_counter.items() if count > 1
    }
    duplicate_extra = sum(count - 1 for count in duplicate_hashes.values())
    conflicting = [
        h for h in duplicate_hashes if len(hash_label_sets[h]) > 1
    ]

    print(f"\n🔍 {split_name} split 内重复正文检查")
    print(f"   唯一正文数: {len(text_hash_counter)}")
    print(f"   重复正文组数: {len(duplicate_hashes)}")
    print(f"   重复额外样本数: {duplicate_extra}")
    print(f"   同正文但标签集合不同的组数: {len(conflicting)}")

    for h, count in sorted(
        duplicate_hashes.items(), key=lambda pair: (-pair[1], pair[0])
    )[:report_limit]:
        example = hash_examples[h]
        print(
            f"      count={count} | first_idx={example['index']} | "
            f"label_sets={len(hash_label_sets[h])}"
        )
        print(f"      preview={example['preview']!r}")

    if conflicting:
        print("   ❌ 警告：完全相同正文对应了不同标签集合，请人工检查。")


def _find_cross_split_label_conflicts(
    overlap_hashes: Set[str],
    left_stats: dict,
    right_stats: dict,
) -> Set[str]:
    """找出两个 split 中正文相同但标签集合不完全一致的 hash。"""
    return {
        h
        for h in overlap_hashes
        if left_stats["hash_label_sets"][h]
        != right_stats["hash_label_sets"][h]
    }


def print_overlap_examples(
    title: str,
    overlap_hashes: Set[str],
    left_stats: dict,
    right_stats: dict,
    label_space: dict,
    limit: int,
    conflicts_only: bool = False,
) -> None:
    """
    打印跨 split 完全相同正文及其两侧标签集合。

    conflicts_only=True 时只打印“同正文但标签不一致”的高风险样本。
    """
    if not overlap_hashes:
        return

    selected_hashes = set(overlap_hashes)
    if conflicts_only:
        selected_hashes = _find_cross_split_label_conflicts(
            overlap_hashes,
            left_stats,
            right_stats,
        )

    if not selected_hashes:
        return

    prefix = "❌" if conflicts_only else "⚠️"
    suffix = "标签冲突示例" if conflicts_only else "重叠示例"
    print(f"   {prefix} {title} 前 {limit} 个{suffix}:")

    for h in sorted(selected_hashes)[:limit]:
        left = left_stats["hash_examples"][h]
        right = right_stats["hash_examples"][h]
        left_label_sets = left_stats["hash_label_sets"][h]
        right_label_sets = right_stats["hash_label_sets"][h]
        labels_match = left_label_sets == right_label_sets

        print(
            f"      hash={h[:16]}... | "
            f"labels_match={labels_match}"
        )
        print(
            f"      {left_stats['split_name']}_idx={left['index']} | "
            f"{right_stats['split_name']}_idx={right['index']}"
        )
        print(
            f"      {left_stats['split_name']}_label_sets="
            f"{_format_label_set_collection(left_label_sets, label_space)}"
        )
        print(
            f"      {right_stats['split_name']}_label_sets="
            f"{_format_label_set_collection(right_label_sets, label_space)}"
        )
        print(f"      preview={left['preview']!r}")


def print_cross_split_overlap_report(
    train_stats: dict,
    val_stats: dict,
    test_stats: dict,
    label_space: dict,
    report_limit: int,
) -> None:
    """
    同时检查：
      1. 不同 split 是否有完全相同正文；
      2. 相同正文在不同 split 中的完整标签集合是否完全一致。

    只报告，不删除、不移动、不合并官方 split。
    """
    train_hashes = train_stats["text_hashes"]
    val_hashes = val_stats["text_hashes"]
    test_hashes = test_stats["text_hashes"]

    train_val = train_hashes & val_hashes
    train_test = train_hashes & test_hashes
    val_test = val_hashes & test_hashes
    all_three = train_hashes & val_hashes & test_hashes

    train_val_conflicts = _find_cross_split_label_conflicts(
        train_val, train_stats, val_stats
    )
    train_test_conflicts = _find_cross_split_label_conflicts(
        train_test, train_stats, test_stats
    )
    val_test_conflicts = _find_cross_split_label_conflicts(
        val_test, val_stats, test_stats
    )

    all_three_conflicts = {
        h
        for h in all_three
        if not (
            train_stats["hash_label_sets"][h]
            == val_stats["hash_label_sets"][h]
            == test_stats["hash_label_sets"][h]
        )
    }

    print("\n" + "=" * 76)
    print("train / validation / test 正文完全重复与标签一致性检查")
    print("=" * 76)
    print(f"train ∩ validation: {len(train_val)}")
    print(f"train ∩ test:       {len(train_test)}")
    print(f"validation ∩ test:  {len(val_test)}")
    print(f"三者共同重叠:       {len(all_three)}")

    print("\n跨 split 相同正文的标签集合一致性:")
    print(
        f"train ∩ validation: 一致={len(train_val) - len(train_val_conflicts)}, "
        f"冲突={len(train_val_conflicts)}"
    )
    print(
        f"train ∩ test:       一致={len(train_test) - len(train_test_conflicts)}, "
        f"冲突={len(train_test_conflicts)}"
    )
    print(
        f"validation ∩ test:  一致={len(val_test) - len(val_test_conflicts)}, "
        f"冲突={len(val_test_conflicts)}"
    )
    print(
        f"三者共同重叠:       一致={len(all_three) - len(all_three_conflicts)}, "
        f"冲突={len(all_three_conflicts)}"
    )

    print_overlap_examples(
        "train ∩ validation",
        train_val,
        train_stats,
        val_stats,
        label_space,
        report_limit,
    )
    print_overlap_examples(
        "train ∩ test",
        train_test,
        train_stats,
        test_stats,
        label_space,
        report_limit,
    )
    print_overlap_examples(
        "validation ∩ test",
        val_test,
        val_stats,
        test_stats,
        label_space,
        report_limit,
    )

    # 标签冲突属于比“正文重复”更严重的问题，单独再次打印。
    print_overlap_examples(
        "train ∩ validation",
        train_val,
        train_stats,
        val_stats,
        label_space,
        report_limit,
        conflicts_only=True,
    )
    print_overlap_examples(
        "train ∩ test",
        train_test,
        train_stats,
        test_stats,
        label_space,
        report_limit,
        conflicts_only=True,
    )
    print_overlap_examples(
        "validation ∩ test",
        val_test,
        val_stats,
        test_stats,
        label_space,
        report_limit,
        conflicts_only=True,
    )

    any_overlap = bool(train_val or train_test or val_test)
    any_conflict = bool(
        train_val_conflicts
        or train_test_conflicts
        or val_test_conflicts
        or all_three_conflicts
    )

    if any_conflict:
        print(
            "❌ 发现跨 split 完全相同正文对应不同标签集合。"
            "这是高风险标注冲突，请根据上面的样本索引人工核查。"
        )
    elif any_overlap:
        print(
            "✅ 所有跨 split 完全相同正文的标签集合均一致。"
            "仍存在精确正文泄漏风险，但没有发现跨 split 标注冲突。"
        )
    else:
        print("✅ 未发现跨 split 完全相同正文。")

    if any_overlap:
        print("说明：代码只报告，不会擅自删除、移动或混合官方 split。")



def print_cross_split_label_report(
    train_stats: dict,
    val_stats: dict,
    test_stats: dict,
    label_space: dict,
) -> None:
    label2id = label_space["label2id"]
    all_labels = set(label2id)

    train_labels = train_stats["labels_present"]
    val_labels = val_stats["labels_present"]
    test_labels = test_stats["labels_present"]

    val_not_train = val_labels - train_labels
    test_not_train_val = test_labels - train_labels - val_labels
    unseen_all = all_labels - train_labels - val_labels - test_labels

    print("\n" + "=" * 76)
    print("跨 split 标签覆盖报告")
    print("=" * 76)
    print(f"完整任务标签数: {len(all_labels)}")
    print(f"train 出现标签数: {len(train_labels)}")
    print(f"validation 出现标签数: {len(val_labels)}")
    print(f"test 出现标签数: {len(test_labels)}")

    print(f"\nvalidation 有、train 无: {len(val_not_train)}")
    if val_not_train:
        print("  ", sorted(val_not_train, key=label2id.get))

    print(f"test 有、train+validation 无: {len(test_not_train_val)}")
    if test_not_train_val:
        for label in sorted(test_not_train_val, key=label2id.get):
            print(
                f"   {label!r} | "
                f"train={train_stats['label_counter'].get(label, 0)} | "
                f"validation={val_stats['label_counter'].get(label, 0)} | "
                f"test={test_stats['label_counter'].get(label, 0)}"
            )

    print(f"三个 split 都未出现: {len(unseen_all)}")
    if unseen_all:
        print("  ", sorted(unseen_all, key=label2id.get))


# ======================================================================
# 5. 处理一个 split
# ======================================================================
def process_split(
    split_name: str,
    input_path: str,
    output_path: str,
    label_space: dict,
    decoder: LazyBertDecoder,
    integer_label_mode: str,
    source_id2label: Optional[Dict[int, str]],
    ancestor_policy: str,
    duplicate_report_limit: int,
    single_depth_one_report_limit: int,
) -> dict:
    print("\n" + "=" * 76)
    print(f"正在处理 {split_name}: {input_path}")
    print("=" * 76)

    raw_data = load_json_or_jsonl(input_path)
    if not raw_data:
        raise ValueError(f"{split_name} 没有样本。")

    if isinstance(raw_data[0], str):
        raise TypeError(
            f"{input_path} 看起来是 idnewnyt_*.json：其中只有 XML 路径，"
            "没有正文与标签，不能直接用于 Llama 训练。请使用截图中的 "
            "nyt_train_qd.json / nyt_val_qd.json / nyt_test_qd.json，"
            "或重新从原始 XML 抽取正文。"
        )

    processed: List[dict] = []
    label_counter: Counter = Counter()
    depth_counter: Counter = Counter()
    text_source_counter: Counter = Counter()
    label_source_counter: Counter = Counter()

    text_hash_counter: Counter = Counter()
    hash_examples: Dict[str, dict] = {}
    hash_label_sets: Dict[str, Set[Tuple[int, ...]]] = defaultdict(set)

    duplicate_label_total = 0
    added_ancestor_total = 0
    only_one_depth_one_count = 0
    only_one_depth_one_examples: List[dict] = []

    label_counts: List[int] = []
    top_counts: List[int] = []
    leaf_counts: List[int] = []

    for index, item in enumerate(raw_data):
        if not isinstance(item, dict):
            raise TypeError(
                f"{split_name} 第 {index} 条必须是 dict，实际为 "
                f"{type(item).__name__}。"
            )

        text, text_field, text_mode = extract_text(
            item,
            decoder,
            split_name,
            index,
        )
        raw_labels, label_field, label_mode = extract_labels(
            item,
            label_space,
            integer_label_mode,
            source_id2label,
            split_name,
            index,
        )

        normalized = canonicalize_and_validate_labels(
            raw_labels,
            label_space,
            split_name,
            index,
            ancestor_policy,
        )

        labels = normalized["labels"]
        label_ids = normalized["label_ids"]
        top_labels = normalized["top_labels"]
        top_label_ids = normalized["top_label_ids"]
        leaf_labels = normalized["leaf_labels"]
        leaf_label_ids = normalized["leaf_label_ids"]

        duplicate_label_total += normalized["duplicate_label_count"]
        added_ancestor_total += normalized["added_ancestor_count"]

        if len(labels) == 1 and label_space["depth"][labels[0]] == 1:
            only_one_depth_one_count += 1
            only_one_depth_one_examples.append(
                {
                    "index": index,
                    "label": labels[0],
                    "label_id": label_ids[0],
                    "preview": text_preview(text),
                }
            )

        label_counts.append(len(labels))
        top_counts.append(len(top_labels))
        leaf_counts.append(len(leaf_labels))

        for label in labels:
            label_counter[label] += 1
            depth_counter[label_space["depth"][label]] += 1

        text_source_counter[f"{text_field}:{text_mode}"] += 1
        label_source_counter[f"{label_field}:{label_mode}"] += 1

        h = text_sha256(text)
        text_hash_counter[h] += 1
        hash_label_sets[h].add(tuple(label_ids))
        if h not in hash_examples:
            hash_examples[h] = {
                "index": index,
                "preview": text_preview(text),
            }

        processed.append(
            {
                "text": text,
                "labels": labels,
                "label_ids": label_ids,
                "top_labels": top_labels,
                "top_label_ids": top_label_ids,
                "leaf_labels": leaf_labels,
                "leaf_label_ids": leaf_label_ids,
            }
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def average(values: Sequence[int]) -> float:
        return sum(values) / max(1, len(values))

    print(f"✅ {split_name} 处理完成")
    print(f"   样本数: {len(processed)}")
    print(f"   实际出现标签: {len(label_counter)} / {len(label_space['labels'])}")
    print(f"   平均完整标签数: {average(label_counts):.4f}")
    print(f"   平均粗粒度标签数: {average(top_counts):.4f}")
    print(f"   平均样本端点数: {average(leaf_counts):.4f}")
    print(f"   最大完整标签数: {max(label_counts)}")
    print(f"   最大样本端点数: {max(leaf_counts)}")
    print(f"   重复标签元素总数: {duplicate_label_total}")
    print(f"   自动补充祖先总数: {added_ancestor_total}")
    print(f"   仅含一个深度 1 标签的样本数: {only_one_depth_one_count}")
    print(f"   正文来源统计: {dict(text_source_counter)}")
    print(f"   标签来源统计: {dict(label_source_counter)}")
    print(f"   输出文件: {output_path}")

    print_single_depth_one_examples(
        split_name=split_name,
        examples=only_one_depth_one_examples,
        report_limit=single_depth_one_report_limit,
    )

    print_within_split_duplicate_report(
        split_name,
        text_hash_counter,
        hash_examples,
        hash_label_sets,
        duplicate_report_limit,
    )

    return {
        "split_name": split_name,
        "count": len(processed),
        "labels_present": set(label_counter),
        "label_counter": label_counter,
        "depth_counter": depth_counter,
        "text_hashes": set(text_hash_counter),
        "text_hash_counter": text_hash_counter,
        "hash_examples": hash_examples,
        "hash_label_sets": hash_label_sets,
        "average_labels": average(label_counts),
        "average_top_labels": average(top_counts),
        "average_leaf_labels": average(leaf_counts),
        "max_labels": max(label_counts),
        "max_leaf_labels": max(leaf_counts),
        "only_one_depth_one_count": only_one_depth_one_count,
        "only_one_depth_one_examples": only_one_depth_one_examples,
    }


# ======================================================================
# 6. 主流程
# ======================================================================
def main(args) -> None:
    print("=" * 76)
    print("NYT 多路径层次多标签数据：统一干净预处理")
    print("=" * 76)

    label_space = build_label_space(
        taxonomy_path=args.taxonomy_path,
        vocab_path=args.vocab_path,
        expected_num_labels=args.expected_num_labels,
        expected_max_depth=args.expected_max_depth,
    )

    print(
        f"✅ 标签空间校验完成：任务标签 {len(label_space['labels'])} 个，"
        f"最大深度 {label_space['max_depth']}，"
        f"Root 直接子节点 {len(label_space['top_labels'])} 个。"
    )
    print(f"粗粒度标签: {label_space['top_labels']}")

    save_label_space(label_space, args.label_space_output)
    print(f"✅ 全局标签空间已保存: {args.label_space_output}")

    source_id2label = load_source_integer_label_mapping(
        label_dict_path=args.source_label_dict_path,
        label_dict_rev_path=args.source_label_dict_rev_path,
        label_space=label_space,
    )

    decoder = LazyBertDecoder(args.bert_tokenizer_path)

    train_stats = process_split(
        split_name="TRAIN",
        input_path=args.train_input,
        output_path=args.train_output,
        label_space=label_space,
        decoder=decoder,
        integer_label_mode=args.integer_label_mode,
        source_id2label=source_id2label,
        ancestor_policy=args.ancestor_policy,
        duplicate_report_limit=args.duplicate_report_limit,
        single_depth_one_report_limit=args.single_depth_one_report_limit,
    )
    val_stats = process_split(
        split_name="VALIDATION",
        input_path=args.val_input,
        output_path=args.val_output,
        label_space=label_space,
        decoder=decoder,
        integer_label_mode=args.integer_label_mode,
        source_id2label=source_id2label,
        ancestor_policy=args.ancestor_policy,
        duplicate_report_limit=args.duplicate_report_limit,
        single_depth_one_report_limit=args.single_depth_one_report_limit,
    )
    test_stats = process_split(
        split_name="TEST",
        input_path=args.test_input,
        output_path=args.test_output,
        label_space=label_space,
        decoder=decoder,
        integer_label_mode=args.integer_label_mode,
        source_id2label=source_id2label,
        ancestor_policy=args.ancestor_policy,
        duplicate_report_limit=args.duplicate_report_limit,
        single_depth_one_report_limit=args.single_depth_one_report_limit,
    )

    print_cross_split_label_report(
        train_stats,
        val_stats,
        test_stats,
        label_space,
    )
    print_cross_split_overlap_report(
        train_stats,
        val_stats,
        test_stats,
        label_space,
        args.overlap_report_limit,
    )

    print("\n" + "=" * 76)
    print("✅ NYT 预处理全部完成")
    print("=" * 76)
    print(f"标签空间: {args.label_space_output}")
    print(f"训练集:   {args.train_output}")
    print(f"验证集:   {args.val_output}")
    print(f"测试集:   {args.test_output}")
    print("\n每条输出字段:")
    print("   text")
    print("   labels / label_ids              # 完整祖先闭包标签集")
    print("   top_labels / top_label_ids      # Root 的直接子节点")
    print("   leaf_labels / leaf_label_ids    # 当前样本最深端点")
    print("\n说明:")
    print("   - 不使用 idnewnyt_*.json 作为正文数据；它们只是 XML 路径列表。")
    print("   - 如果 qd 的 token 是 BERT ID，会使用指定 BERT tokenizer 解码。")
    print("   - qd 源整数 ID 先由 label_dict*.pt 恢复为标签字符串。")
    print("   - 最终标签 ID 再由 nyt_label.vocab 的稳定顺序统一生成。")
    print("   - Root 不参与 166 类指标。")
    print("   - 单独打印仅含一个深度 1 标签的样本，便于人工核查。")
    print("   - 跨 split 重复正文会继续检查完整标签集合是否一致。")
    print("   - 只报告重复/泄漏，不自动删除、移动或混合官方 split。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess HiAGM/HILL NYT qd data into clean JSONL for "
            "Llama generative hierarchical multi-label training."
        )
    )

    # ================= 路径参数 =================
    parser.add_argument("--train_input", type=str, default="/data1/lzp/code/data/nyt/nyt_train_qd.json")
    parser.add_argument("--val_input", type=str, default="/data1/lzp/code/data/nyt/nyt_val_qd.json")
    parser.add_argument("--test_input", type=str, default="/data1/lzp/code/data/nyt/nyt_test_qd.json")
    parser.add_argument("--taxonomy_path", type=str, default="/data1/lzp/code/data/nyt/nyt.taxonomy")
    parser.add_argument("--vocab_path", type=str, default="/data1/lzp/code/data/nyt/nyt_label.vocab")
    parser.add_argument("--bert_tokenizer_path", type=str, default="bert-base-uncased")
    parser.add_argument("--source_label_dict_path", type=str, default="/data1/lzp/code/data/nyt/label_dict.pt")
    parser.add_argument("--source_label_dict_rev_path", type=str, default="/data1/lzp/code/data/nyt/label_dict_rev.pt")

    parser.add_argument("--train_output", type=str, default="/data1/lzp/code/data/nyt/train_formatted.jsonl")
    parser.add_argument("--val_output", type=str, default="/data1/lzp/code/data/nyt/val_formatted.jsonl")
    parser.add_argument("--test_output", type=str, default="/data1/lzp/code/data/nyt/test_formatted.jsonl")
    parser.add_argument("--label_space_output", type=str, default="/data1/lzp/code/data/nyt/label_space.json")

    # ================= 标签空间严格检查 =================
    parser.add_argument("--expected_num_labels", type=int, default=166)
    parser.add_argument("--expected_max_depth", type=int, default=8)
    parser.add_argument(
        "--ancestor_policy",
        type=str,
        choices=["strict", "add_missing"],
        default="strict",
        help="strict=缺祖先直接报错；add_missing=自动补充 taxonomy 祖先。",
    )
    parser.add_argument(
        "--integer_label_mode",
        type=str,
        choices=[
            "auto",
            "zero_based_without_root",
            "vocab_index_with_root",
        ],
        default="auto",
        help=(
            "仅在没有可用 label_dict.pt / label_dict_rev.pt 时作为回退。"
            "正常情况下整数 ID 会直接按源标签字典恢复，不再自动猜测。"
        ),
    )

    # ================= 报告参数 =================
    parser.add_argument("--duplicate_report_limit", type=int, default=5)
    parser.add_argument("--overlap_report_limit", type=int, default=5)
    parser.add_argument(
        "--single_depth_one_report_limit",
        type=int,
        default=20,
        help=(
            "每个 split 最多打印多少条‘整个标签集合只有一个深度 1 标签’"
            "的样本。默认 20，足以打印当前检测到的 13 条。"
        ),
    )

    main(parser.parse_args())
