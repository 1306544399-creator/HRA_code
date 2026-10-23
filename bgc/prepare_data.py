#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict, deque
from typing import Dict, List, Sequence, Set, Tuple


# =============================================================================
# 0. 基础工具
# =============================================================================
def load_jsonl(path: str) -> List[dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} 第 {line_no} 行不是合法 JSON: {exc}"
                ) from exc

            if not isinstance(item, dict):
                raise TypeError(
                    f"{path} 第 {line_no} 行必须是 JSON object，"
                    f"实际={type(item).__name__}"
                )
            rows.append(item)

    if not rows:
        raise ValueError(f"文件为空或没有有效 JSONL 记录: {path}")

    return rows


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def preview(text: str, max_chars: int = 180) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= max_chars else one_line[:max_chars] + "..."


# =============================================================================
# 1. 解析 HyILR 的 bgc.taxonomy
# =============================================================================
def parse_bgc_taxonomy(
    taxonomy_path: str,
    expected_num_labels: int = 146,
    expected_max_depth: int = 4,
) -> dict:
    """
    HyILR 的 bgc.taxonomy 与你的 NYT taxonomy 形式一致：
        parent<TAB>child1<TAB>child2<...>

    第一行显式包含 Root，所以这里不再构造虚拟 Root。
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

            parts = [part.strip() for part in raw_line.split("\t") if part.strip()]
            if len(parts) < 2:
                raise ValueError(
                    f"taxonomy 第 {line_no} 行至少应包含父节点和一个子节点: "
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
                        f"taxonomy 不是单父节点树：{child!r} 同时属于 "
                        f"{parent[child]!r} 和 {node!r}"
                    )
                parent[child] = node

    if "Root" not in children:
        raise ValueError("bgc.taxonomy 中没有 Root。")
    if "Root" in parent:
        raise ValueError("Root 不应有父节点。")

    all_nodes: Set[str] = {"Root"}
    all_nodes.update(children.keys())
    all_nodes.update(parent.keys())

    # 叶节点可能不会单独作为 parent 出现，补 children=[]
    for node in all_nodes:
        children.setdefault(node, [])

    # BFS 计算深度
    depth: Dict[str, int] = {"Root": 0}
    queue = deque(["Root"])

    while queue:
        node = queue.popleft()
        for child in children[node]:
            proposed = depth[node] + 1
            if child in depth:
                if depth[child] != proposed:
                    raise ValueError(
                        f"taxonomy 深度冲突或存在环: {child!r}"
                    )
                continue
            depth[child] = proposed
            queue.append(child)

    unreachable = all_nodes - set(depth)
    if unreachable:
        raise ValueError(
            "taxonomy 中存在无法从 Root 到达的节点: "
            f"{sorted(unreachable)[:20]}"
        )

    task_nodes = all_nodes - {"Root"}
    max_depth = max(depth.values())

    if expected_num_labels > 0 and len(task_nodes) != expected_num_labels:
        raise ValueError(
            f"BGC 标签数异常: {len(task_nodes)} != {expected_num_labels}"
        )

    if expected_max_depth > 0 and max_depth != expected_max_depth:
        raise ValueError(
            f"BGC 最大深度异常: {max_depth} != {expected_max_depth}"
        )

    # 采用 taxonomy 的稳定遍历顺序：
    # Root 的孩子顺序固定；每个父节点的孩子顺序按文件中顺序固定。
    labels: List[str] = []
    queue = deque(children["Root"])
    while queue:
        node = queue.popleft()
        labels.append(node)
        for child in children[node]:
            queue.append(child)

    if len(labels) != len(task_nodes) or set(labels) != task_nodes:
        raise RuntimeError("taxonomy BFS 标签顺序构造失败。")

    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    top_labels = list(children["Root"])

    ancestors: Dict[str, List[str]] = {}
    for label in labels:
        chain = []
        current = label
        seen = set()

        while current != "Root":
            if current in seen:
                raise ValueError(f"taxonomy 存在环: {label!r}")
            seen.add(current)
            chain.append(current)
            current = parent[current]

        chain.reverse()
        if len(chain) != depth[label]:
            raise ValueError(
                f"{label!r} 的祖先链长度与 depth 不一致。"
            )
        ancestors[label] = chain

    depth_to_labels = {
        d: [label for label in labels if depth[label] == d]
        for d in range(1, max_depth + 1)
    }

    print("\n" + "=" * 78)
    print("✅ HyILR BGC taxonomy 严格校验完成")
    print(f"任务标签数: {len(labels)}")
    print(f"最大深度: {max_depth}")
    print(f"Root 直接子节点: {len(top_labels)} -> {top_labels}")
    for d in range(1, max_depth + 1):
        print(f"Depth {d}: {len(depth_to_labels[d])} 个标签")
    print("=" * 78 + "\n")

    return {
        "num_labels": len(labels),
        "max_depth": max_depth,
        "labels": labels,
        "label2id": label2id,
        "id2label": id2label,
        "top_labels": top_labels,
        "parent": parent,
        "children": children,
        "depth": depth,
        "ancestors": ancestors,
        "depth_to_labels": depth_to_labels,
    }


def save_label_space(label_space: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    payload = {
        "num_labels": int(label_space["num_labels"]),
        "max_depth": int(label_space["max_depth"]),
        "label2id": label_space["label2id"],
        "id2label": {
            str(idx): label for idx, label in label_space["id2label"].items()
        },
        "top_labels": label_space["top_labels"],
        "parent": label_space["parent"],
        "children": label_space["children"],
        "depth": {
            label: int(d) for label, d in label_space["depth"].items()
        },
        "depth_to_labels": {
            str(d): labels
            for d, labels in label_space["depth_to_labels"].items()
        },
        "canonical_order_source": (
            "breadth-first traversal using child order in HyILR bgc.taxonomy"
        ),
        "task_definition": (
            "BGC hierarchical multi-label classification; Root is excluded "
            "from the 146 task labels"
        ),
        "source": "HyILR preprocess/BGC",
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# =============================================================================
# 2. 单条样本规范化
# =============================================================================
def normalize_sample(
    item: dict,
    split_name: str,
    index: int,
    label_space: dict,
    ancestor_policy: str,
) -> Tuple[dict, int]:
    """
    HyILR JSONL:
        {
          "token": "Title: ... Text: ...",
          "label": ["Fiction", ...]
        }

    保留 token 原文，不再 lowercase、不走 RoBERTa tokenizer。
    """
    if "token" not in item:
        raise KeyError(
            f"{split_name} 第 {index} 条缺少 token 字段。"
        )
    if "label" not in item:
        raise KeyError(
            f"{split_name} 第 {index} 条缺少 label 字段。"
        )

    text = item["token"]
    if not isinstance(text, str):
        raise TypeError(
            f"{split_name} 第 {index} 条 token 必须是 str，"
            f"实际={type(text).__name__}"
        )
    text = text.strip()
    if not text:
        raise ValueError(f"{split_name} 第 {index} 条 token 为空。")

    raw_labels = item["label"]
    if not isinstance(raw_labels, list):
        raise TypeError(
            f"{split_name} 第 {index} 条 label 必须是 list，"
            f"实际={type(raw_labels).__name__}"
        )
    if not raw_labels:
        raise ValueError(f"{split_name} 第 {index} 条 label 为空。")

    labels_clean = []
    for pos, raw_label in enumerate(raw_labels):
        if not isinstance(raw_label, str):
            raise TypeError(
                f"{split_name} 第 {index} 条 label[{pos}] 必须是 str。"
            )
        label = raw_label.strip()
        if not label:
            raise ValueError(
                f"{split_name} 第 {index} 条 label[{pos}] 为空。"
            )
        labels_clean.append(label)

    if len(labels_clean) != len(set(labels_clean)):
        raise ValueError(
            f"{split_name} 第 {index} 条存在重复标签: {labels_clean}"
        )

    label2id = label_space["label2id"]
    parent = label_space["parent"]
    depth = label_space["depth"]
    children = label_space["children"]

    unknown = set(labels_clean) - set(label2id)
    if unknown:
        raise ValueError(
            f"{split_name} 第 {index} 条含 taxonomy 未定义标签: "
            f"{sorted(unknown)[:20]}"
        )

    label_set = set(labels_clean)

    # 检查祖先闭包
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
                f"{split_name} 第 {index} 条不是祖先闭包，缺失: "
                f"{sorted(missing_ancestors, key=label2id.get)}"
            )
        elif ancestor_policy == "add_missing":
            label_set.update(missing_ancestors)
        else:
            raise ValueError(
                f"不支持 ancestor_policy={ancestor_policy!r}"
            )

    # 最终统一成唯一全局 label ID 顺序
    labels = sorted(label_set, key=label2id.get)
    label_ids = [label2id[label] for label in labels]

    top_labels = [
        label for label in labels if depth[label] == 1
    ]
    top_label_ids = [label2id[label] for label in top_labels]

    leaf_labels = [
        label
        for label in labels
        if not any(child in label_set for child in children[label])
    ]
    leaf_label_ids = [label2id[label] for label in leaf_labels]

    if not top_labels:
        raise ValueError(
            f"{split_name} 第 {index} 条没有任何 D1 标签。"
        )
    if not leaf_labels:
        raise ValueError(
            f"{split_name} 第 {index} 条没有样本端点标签。"
        )

    output = {
        "text": text,
        "labels": labels,
        "label_ids": label_ids,
        "top_labels": top_labels,
        "top_label_ids": top_label_ids,
        "leaf_labels": leaf_labels,
        "leaf_label_ids": leaf_label_ids,
    }

    return output, len(missing_ancestors)


# =============================================================================
# 3. split 处理 + 统计
# =============================================================================
def process_split(
    split_name: str,
    input_path: str,
    output_path: str,
    label_space: dict,
    ancestor_policy: str,
    duplicate_report_limit: int,
) -> dict:
    print("\n" + "=" * 78)
    print(f"正在处理 {split_name}: {input_path}")
    print("=" * 78)

    raw_data = load_jsonl(input_path)

    label_counter = Counter()
    depth_counter = Counter()

    label_counts = []
    top_counts = []
    leaf_counts = []

    added_ancestor_total = 0

    text_hash_counter = Counter()
    hash_examples = {}
    hash_label_sets = defaultdict(set)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(raw_data):
            normalized, added_count = normalize_sample(
                item=item,
                split_name=split_name,
                index=idx,
                label_space=label_space,
                ancestor_policy=ancestor_policy,
            )

            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")

            labels = normalized["labels"]
            label_ids = normalized["label_ids"]
            top_labels = normalized["top_labels"]
            leaf_labels = normalized["leaf_labels"]

            label_counts.append(len(labels))
            top_counts.append(len(top_labels))
            leaf_counts.append(len(leaf_labels))
            added_ancestor_total += added_count

            for label in labels:
                label_counter[label] += 1
                depth_counter[label_space["depth"][label]] += 1

            h = text_sha256(normalized["text"])
            text_hash_counter[h] += 1
            hash_label_sets[h].add(tuple(label_ids))
            if h not in hash_examples:
                hash_examples[h] = {
                    "index": idx,
                    "preview": preview(normalized["text"]),
                }

    def avg(values: Sequence[int]) -> float:
        return sum(values) / max(1, len(values))

    duplicate_hashes = {
        h: c for h, c in text_hash_counter.items() if c > 1
    }
    conflicting_hashes = [
        h for h in duplicate_hashes if len(hash_label_sets[h]) > 1
    ]

    print(f"✅ {split_name} 处理完成")
    print(f"样本数: {len(raw_data)}")
    print(
        f"实际出现标签: {len(label_counter)} / "
        f"{label_space['num_labels']}"
    )
    print(f"平均完整标签数: {avg(label_counts):.4f}")
    print(f"平均 D1 标签数: {avg(top_counts):.4f}")
    print(f"平均样本端点数: {avg(leaf_counts):.4f}")
    print(f"最大完整标签数: {max(label_counts)}")
    print(f"最大样本端点数: {max(leaf_counts)}")
    print(f"自动补祖先总数: {added_ancestor_total}")
    for d in range(1, label_space["max_depth"] + 1):
        print(f"Depth {d} 标签实例数: {depth_counter[d]}")
    print(f"输出: {output_path}")

    print("\n🔍 split 内完全相同正文检查")
    print(f"唯一正文数: {len(text_hash_counter)}")
    print(f"重复正文组数: {len(duplicate_hashes)}")
    print(
        f"重复额外样本数: "
        f"{sum(c - 1 for c in duplicate_hashes.values())}"
    )
    print(f"同正文但标签集合不同的组数: {len(conflicting_hashes)}")

    for h, count in sorted(
        duplicate_hashes.items(),
        key=lambda x: (-x[1], x[0]),
    )[: max(0, duplicate_report_limit)]:
        ex = hash_examples[h]
        print(
            f"  count={count} | first_idx={ex['index']} | "
            f"label_sets={len(hash_label_sets[h])}"
        )
        print(f"  preview={ex['preview']!r}")

    if conflicting_hashes:
        print("❌ 警告：同正文对应不同标签集合。")

    return {
        "split_name": split_name,
        "count": len(raw_data),
        "labels_present": set(label_counter),
        "label_counter": label_counter,
        "text_hashes": set(text_hash_counter),
        "hash_label_sets": hash_label_sets,
        "hash_examples": hash_examples,
    }


def print_cross_split_label_report(
    train_stats: dict,
    val_stats: dict,
    test_stats: dict,
    label_space: dict,
) -> None:
    all_labels = set(label_space["labels"])
    train_labels = train_stats["labels_present"]
    val_labels = val_stats["labels_present"]
    test_labels = test_stats["labels_present"]

    val_not_train = val_labels - train_labels
    test_not_train_val = test_labels - train_labels - val_labels
    unseen_all = all_labels - train_labels - val_labels - test_labels

    label2id = label_space["label2id"]

    print("\n" + "=" * 78)
    print("跨 split 标签覆盖报告")
    print("=" * 78)
    print(f"完整任务标签数: {len(all_labels)}")
    print(f"train 出现: {len(train_labels)}")
    print(f"validation 出现: {len(val_labels)}")
    print(f"test 出现: {len(test_labels)}")

    print(f"\nvalidation 有、train 无: {len(val_not_train)}")
    if val_not_train:
        print("  ", sorted(val_not_train, key=label2id.get))

    print(f"test 有、train+validation 无: {len(test_not_train_val)}")
    if test_not_train_val:
        for label in sorted(test_not_train_val, key=label2id.get):
            print(
                f"  {label!r} | "
                f"train={train_stats['label_counter'].get(label, 0)} | "
                f"validation={val_stats['label_counter'].get(label, 0)} | "
                f"test={test_stats['label_counter'].get(label, 0)}"
            )

    print(f"三个 split 都未出现: {len(unseen_all)}")
    if unseen_all:
        print("  ", sorted(unseen_all, key=label2id.get))


def print_cross_split_overlap_report(
    train_stats: dict,
    val_stats: dict,
    test_stats: dict,
    limit: int,
) -> None:
    pairs = [
        ("train ∩ validation", train_stats, val_stats),
        ("train ∩ test", train_stats, test_stats),
        ("validation ∩ test", val_stats, test_stats),
    ]

    print("\n" + "=" * 78)
    print("train / validation / test 正文重复与潜在泄漏检查")
    print("=" * 78)

    any_overlap = False
    any_conflict = False

    for title, left, right in pairs:
        overlap = left["text_hashes"] & right["text_hashes"]
        conflicts = {
            h
            for h in overlap
            if left["hash_label_sets"][h] != right["hash_label_sets"][h]
        }

        any_overlap |= bool(overlap)
        any_conflict |= bool(conflicts)

        print(
            f"{title}: overlap={len(overlap)}, "
            f"label_conflict={len(conflicts)}"
        )

        for h in sorted(overlap)[: max(0, limit)]:
            le = left["hash_examples"][h]
            re = right["hash_examples"][h]
            print(
                f"  hash={h[:16]}... | "
                f"{left['split_name']}_idx={le['index']} | "
                f"{right['split_name']}_idx={re['index']}"
            )
            print(f"  preview={le['preview']!r}")

    if any_conflict:
        print("❌ 发现跨 split 同正文但标签集合冲突。")
    elif any_overlap:
        print("⚠️ 存在跨 split 完全相同正文，属于潜在数据泄漏。")
    else:
        print("✅ 未发现跨 split 完全相同正文。")

    print("说明：只报告，不自动删除、移动或混合官方 split。")


# =============================================================================
# 4. 主流程
# =============================================================================
def main(args) -> None:
    print("=" * 78)
    print("HyILR BGC：统一干净预处理")
    print("=" * 78)

    data_dir = args.data_dir

    taxonomy_path = args.taxonomy_path or os.path.join(
        data_dir, "bgc.taxonomy"
    )
    train_input = args.train_input or os.path.join(
        data_dir, "train_data.jsonl"
    )
    val_input = args.val_input or os.path.join(
        data_dir, "dev_data.jsonl"
    )
    test_input = args.test_input or os.path.join(
        data_dir, "test_data.jsonl"
    )

    train_output = args.train_output or os.path.join(
        data_dir, "train_formatted.jsonl"
    )
    val_output = args.val_output or os.path.join(
        data_dir, "val_formatted.jsonl"
    )
    test_output = args.test_output or os.path.join(
        data_dir, "test_formatted.jsonl"
    )
    label_space_output = args.label_space_output or os.path.join(
        data_dir, "label_space.json"
    )

    label_space = parse_bgc_taxonomy(
        taxonomy_path=taxonomy_path,
        expected_num_labels=args.expected_num_labels,
        expected_max_depth=args.expected_max_depth,
    )

    save_label_space(label_space, label_space_output)
    print(f"✅ label_space 已保存: {label_space_output}")

    train_stats = process_split(
        "TRAIN",
        train_input,
        train_output,
        label_space,
        args.ancestor_policy,
        args.duplicate_report_limit,
    )
    val_stats = process_split(
        "VALIDATION",
        val_input,
        val_output,
        label_space,
        args.ancestor_policy,
        args.duplicate_report_limit,
    )
    test_stats = process_split(
        "TEST",
        test_input,
        test_output,
        label_space,
        args.ancestor_policy,
        args.duplicate_report_limit,
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
        args.overlap_report_limit,
    )

    print("\n" + "=" * 78)
    print("✅ HyILR BGC 预处理全部完成")
    print("=" * 78)
    print(f"label_space: {label_space_output}")
    print(f"train:       {train_output}")
    print(f"validation:  {val_output}")
    print(f"test:        {test_output}")
    print("\n输出字段与 NYT 保持一致:")
    print("  text")
    print("  labels / label_ids")
    print("  top_labels / top_label_ids")
    print("  leaf_labels / leaf_label_ids")
    print("\n注意:")
    print("  - bgc.taxonomy 已自带 Root，不再构造虚拟 Root。")
    print("  - Root 不计入 146 个任务标签。")
    print("  - 保留 HyILR token 字段原文，例如 'Title: ... Text: ...'。")
    print("  - 不执行 HyILR data_bgc.py 中的 lowercase/RoBERTa tokenization。")
    print("  - 默认 strict：缺祖先直接报错，不静默修正。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare HyILR BGC JSONL for Llama generative HTC."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data1/lzp/code/data/bgc",
    )

    parser.add_argument("--taxonomy_path", type=str, default="")
    parser.add_argument("--train_input", type=str, default="")
    parser.add_argument("--val_input", type=str, default="")
    parser.add_argument("--test_input", type=str, default="")

    parser.add_argument("--train_output", type=str, default="")
    parser.add_argument("--val_output", type=str, default="")
    parser.add_argument("--test_output", type=str, default="")
    parser.add_argument("--label_space_output", type=str, default="")

    parser.add_argument(
        "--expected_num_labels",
        type=int,
        default=146,
    )
    parser.add_argument(
        "--expected_max_depth",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--ancestor_policy",
        type=str,
        choices=["strict", "add_missing"],
        default="strict",
        help="strict=缺祖先直接报错；add_missing=自动补 taxonomy 祖先。",
    )

    parser.add_argument(
        "--duplicate_report_limit",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--overlap_report_limit",
        type=int,
        default=5,
    )

    main(parser.parse_args())
