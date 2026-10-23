import hashlib
import json
import os
from collections import Counter, defaultdict

# ============================================================
# 路径配置
# ============================================================
TAXNOMY_FILE = "/data1/lzp/code/data/wos/wos.taxnomy"

TRAIN_INPUT_FILE = "/data1/lzp/code/data/wos/wos_train.json"
VAL_INPUT_FILE = "/data1/lzp/code/data/wos/wos_val.json"
TEST_INPUT_FILE = "/data1/lzp/code/data/wos/wos_test.json"

# 注意：文件名继续沿用你原来的命名，
# 但文件内容不再包含 Prompt / Chat Template，改为干净的 text + label + id。
TRAIN_OUTPUT_FILE = "/data1/lzp/code/data/wos/train_formatted.jsonl"
VAL_OUTPUT_FILE = "/data1/lzp/code/data/wos/val_formatted.jsonl"
TEST_OUTPUT_FILE = "/data1/lzp/code/data/wos/test_formatted.jsonl"

# 唯一全局标签字典：只由 wos.taxnomy 生成，train / val / test 共用
LABEL_DICT_FILE = "/data1/lzp/code/data/wos/label2id.json"

# 重复文本报告时，每类最多打印多少个示例
DUPLICATE_REPORT_LIMIT = 5
OVERLAP_REPORT_LIMIT = 5


# ============================================================
# 通用读取函数
# ============================================================
def load_json_or_jsonl(file_path):
    """兼容读取 JSON 列表文件或 JSONL 文件。"""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"找不到数据文件: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                f"{file_path} 的 JSON 顶层必须是 list，"
                f"当前类型为 {type(data).__name__}。"
            )
        return data

    except json.JSONDecodeError:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"{file_path} 第 {line_num} 行不是合法 JSON: {e}"
                    ) from e

                data.append(item)

        return data


def text_sha256(text):
    """
    对已经完成当前脚本清洗后的正文做精确 SHA256。

    注意：这里只检查“当前清洗结果完全相同”的正文：
    - 不转小写
    - 不做模糊匹配
    - 不额外压缩内部空白

    这样避免把本来不同的文本误判成重复。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_preview(text, max_chars=180):
    """生成单行短预览，便于重复/泄漏报告。"""
    one_line = " ".join(text.split())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[:max_chars] + "..."


# ============================================================
# 从 taxnomy 提取完整两级标签空间
# ============================================================
def load_full_label_space_from_taxnomy(taxnomy_file):
    """
    从 WOS taxnomy 中提取完整两级标签空间。

    预期 taxnomy 每行格式：
        父节点<TAB>子节点1<TAB>子节点2<...>

    例如：
        Root<TAB>CS<TAB>Medical<TAB>...
        CS<TAB>Machine learning<TAB>Computer vision<TAB>...

    返回：
        l1_label2id: 完整 L1 标签 -> ID
        l2_label2id: 完整 L2 标签 -> ID
        l2_to_l1:    L2 标签 -> 所属 L1 标签
    """
    if not os.path.isfile(taxnomy_file):
        raise FileNotFoundError(f"找不到 taxnomy 文件: {taxnomy_file}")

    parent_to_children = {}

    with open(taxnomy_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.rstrip("\n\r")
            if not line.strip():
                continue

            # 标签名称可能包含空格，因此必须按 TAB 分列，不能使用 split()
            parts = [part.strip() for part in line.split("\t") if part.strip()]

            if len(parts) < 2:
                raise ValueError(
                    f"taxnomy 第 {line_num} 行格式异常，"
                    f"至少应包含父节点和一个子节点: {line!r}"
                )

            parent = parts[0]
            children = parts[1:]

            if parent in parent_to_children:
                raise ValueError(f"taxnomy 中父节点重复出现: {parent!r}")

            if len(children) != len(set(children)):
                raise ValueError(
                    f"taxnomy 第 {line_num} 行存在重复子节点: {parent!r}"
                )

            parent_to_children[parent] = children

    if "Root" not in parent_to_children:
        raise ValueError("taxnomy 中未找到 Root 行，无法确定完整 L1 标签。")

    # Root 的直接子节点 = 完整 L1 标签；保持 taxnomy 原始顺序
    l1_labels = parent_to_children["Root"]

    if len(l1_labels) != len(set(l1_labels)):
        raise ValueError("taxnomy 的 Root 行中存在重复 L1 标签。")

    # 每个 L1 的直接子节点 = 完整 L2 标签
    l2_labels = []
    l2_to_l1 = {}

    for l1 in l1_labels:
        if l1 not in parent_to_children:
            raise ValueError(
                f"taxnomy 中 L1 标签 {l1!r} 没有对应的子节点行，"
                "无法提取完整 L2 标签。"
            )

        for l2 in parent_to_children[l1]:
            if l2 in l2_to_l1:
                raise ValueError(
                    f"L2 标签 {l2!r} 在 taxnomy 中重复归属于 "
                    f"{l2_to_l1[l2]!r} 和 {l1!r}。"
                )

            l2_to_l1[l2] = l1
            l2_labels.append(l2)

    l1_label2id = {label: idx for idx, label in enumerate(l1_labels)}
    l2_label2id = {label: idx for idx, label in enumerate(l2_labels)}

    return l1_label2id, l2_label2id, l2_to_l1


# ============================================================
# 保存唯一全局标签字典
# ============================================================
def save_label_dict(l1_label2id, l2_label2id, output_file):
    """
    保存由 taxnomy 生成的唯一全局标签映射。

    保留原有字段：
        l1_label2id
        l2_label2id

    同时增加：
        l1_id2label
        l2_id2label

    后续学生训练 / 教师训练需要把 ID 恢复成字符串标签时，
    可以直接读取 id2label，无需每次临时反转字典。
    """
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    l1_id2label = {str(idx): label for label, idx in l1_label2id.items()}
    l2_id2label = {str(idx): label for label, idx in l2_label2id.items()}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "l1_label2id": l1_label2id,
                "l2_label2id": l2_label2id,
                "l1_id2label": l1_id2label,
                "l2_id2label": l2_id2label,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# 单个 split 内重复文本报告
# ============================================================
def print_within_split_duplicate_report(
    split_name,
    text_hash_counter,
    hash_examples,
    hash_label_pairs,
):
    """报告当前 split 内完全相同正文的重复情况，不自动删除任何样本。"""
    duplicate_hashes = {
        h: count
        for h, count in text_hash_counter.items()
        if count > 1
    }

    duplicate_group_count = len(duplicate_hashes)
    duplicate_extra_sample_count = sum(
        count - 1 for count in duplicate_hashes.values()
    )

    conflicting_label_hashes = [
        h
        for h in duplicate_hashes
        if len(hash_label_pairs[h]) > 1
    ]

    print(f"\n🔍 {split_name} split 内重复正文检查")
    print(f"   唯一正文数: {len(text_hash_counter)}")
    print(f"   重复正文组数: {duplicate_group_count}")
    print(f"   重复额外样本数: {duplicate_extra_sample_count}")
    print(f"   同正文但标签不一致的重复组数: {len(conflicting_label_hashes)}")

    if duplicate_hashes:
        print(f"   ⚠️ 前 {DUPLICATE_REPORT_LIMIT} 个重复组示例:")
        sorted_items = sorted(
            duplicate_hashes.items(),
            key=lambda x: (-x[1], x[0]),
        )
        for h, count in sorted_items[:DUPLICATE_REPORT_LIMIT]:
            example = hash_examples[h]
            label_pairs = sorted(hash_label_pairs[h])
            print(
                f"      count={count} | first_idx={example['item_idx']} | "
                f"labels={label_pairs}"
            )
            print(f"      preview={example['text_preview']!r}")

    if conflicting_label_hashes:
        print(
            "   ❌ 警告：发现完全相同正文对应不同标签。"
            "这属于高风险数据问题，请人工核查。"
        )


# ============================================================
# 单独处理一个 split
# ============================================================
def process_split(
    split_name,
    input_file,
    output_file,
    l1_label2id,
    l2_label2id,
    l2_to_l1,
):
    """
    使用 taxnomy 生成的同一套全局标签 ID，处理一个数据集 split。

    本阶段只负责保存：
        text
        l1_label
        l2_label
        l1_id
        l2_id

    明确不负责：
        - 构造 Prompt
        - 写入 Chat Template
        - 添加 <LVL1_Q> / <LVL2_Q>
        - 构造 Completion
        - tokenizer 处理

    严格模式：
        - 空正文直接报错
        - doc_label 缺失/类型错误直接报错
        - 标签层数必须严格等于 2，否则直接报错
        - 不再静默跳过异常样本

    上述模型相关逻辑应在学生训练、教师训练、层级选择、
    特征提取和蒸馏代码运行时统一构造。
    """
    print("\n" + "=" * 70)
    print(f"正在处理 {split_name}: {input_file}")

    raw_data = load_json_or_jsonl(input_file)
    processed_data = []

    l1_counter = Counter()
    l2_counter = Counter()

    # 用于 split 内/跨 split 精确重复正文检查
    text_hash_counter = Counter()
    hash_examples = {}
    hash_label_pairs = defaultdict(set)

    for item_idx, item in enumerate(raw_data):
        if not isinstance(item, dict):
            raise ValueError(
                f"{split_name} 第 {item_idx} 条样本不是 JSON object/dict，"
                f"而是 {type(item).__name__}。"
            )

        # ----------------------------------------------------
        # 1. 获取原始正文，并严格检查空正文
        # ----------------------------------------------------
        if "doc_token" not in item:
            raise ValueError(
                f"{split_name} 第 {item_idx} 条样本缺少 doc_token 字段。"
            )

        text = item["doc_token"]

        if isinstance(text, list):
            text = " ".join(str(token) for token in text)
        elif text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)

        text = text.strip()

        if not text:
            raise ValueError(
                f"{split_name} 第 {item_idx} 条样本正文为空，"
                "严格模式下不允许继续处理。"
            )

        # ----------------------------------------------------
        # 2. 获取层级标签，并严格要求恰好两层
        # ----------------------------------------------------
        if "doc_label" not in item:
            raise ValueError(
                f"{split_name} 第 {item_idx} 条样本缺少 doc_label 字段。"
            )

        labels = item["doc_label"]

        if not isinstance(labels, (list, tuple)):
            raise ValueError(
                f"{split_name} 第 {item_idx} 条样本的 doc_label "
                f"必须是 list/tuple，实际为 {type(labels).__name__}: "
                f"{labels!r}"
            )

        if len(labels) != 2:
            raise ValueError(
                f"{split_name} 第 {item_idx} 条样本必须严格有 2 层标签，"
                f"实际为 {len(labels)} 层: {labels!r}"
            )

        l1_str = str(labels[0]).strip()
        l2_str = str(labels[1]).strip()

        if not l1_str or not l2_str:
            raise ValueError(
                f"{split_name} 第 {item_idx} 条样本存在空标签: "
                f"{labels!r}"
            )

        # ----------------------------------------------------
        # 3. 严格查询 taxnomy 全局标签空间
        # ----------------------------------------------------
        if l1_str not in l1_label2id:
            raise ValueError(
                f"{split_name} 第 {item_idx} 条样本的 L1 标签 "
                f"{l1_str!r} 不在 taxnomy 中。"
            )

        if l2_str not in l2_label2id:
            raise ValueError(
                f"{split_name} 第 {item_idx} 条样本的 L2 标签 "
                f"{l2_str!r} 不在 taxnomy 中。"
            )

        # ----------------------------------------------------
        # 4. 检查 L1-L2 父子关系
        # ----------------------------------------------------
        expected_l1 = l2_to_l1[l2_str]
        if l1_str != expected_l1:
            raise ValueError(
                f"{split_name} 第 {item_idx} 条样本层级关系与 taxnomy 不一致："
                f"样本为 {l1_str!r} > {l2_str!r}，"
                f"taxnomy 规定为 {expected_l1!r} > {l2_str!r}。"
            )

        # ----------------------------------------------------
        # 5. 使用原有 ID 方式（保持不变）
        # ----------------------------------------------------
        l1_id = l1_label2id[l1_str]
        l2_id = l2_label2id[l2_str]

        l1_counter[l1_str] += 1
        l2_counter[l2_str] += 1

        # ----------------------------------------------------
        # 6. 记录正文 hash，用于重复/泄漏检查
        # ----------------------------------------------------
        h = text_sha256(text)
        text_hash_counter[h] += 1
        hash_label_pairs[h].add((l1_str, l2_str))

        if h not in hash_examples:
            hash_examples[h] = {
                "item_idx": item_idx,
                "text_preview": text_preview(text),
                "l1_label": l1_str,
                "l2_label": l2_str,
            }

        # ----------------------------------------------------
        # 7. 只保存干净数据，不生成任何 Prompt
        # ----------------------------------------------------
        processed_data.append(
            {
                "text": text,
                "l1_label": l1_str,
                "l2_label": l2_str,
                "l1_id": l1_id,
                "l2_id": l2_id,
            }
        )

    # --------------------------------------------------------
    # 8. 当前 split 独立保存
    # --------------------------------------------------------
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ {split_name} 处理完成")
    print(f"   原始样本数: {len(raw_data)}")
    print(f"   成功转换数: {len(processed_data)}")
    print("   跳过样本数: 0（严格模式：异常直接报错，不静默跳过）")
    print(
        f"   当前 split 实际出现的 L1: "
        f"{len(l1_counter)} / {len(l1_label2id)}"
    )
    print(
        f"   当前 split 实际出现的 L2: "
        f"{len(l2_counter)} / {len(l2_label2id)}"
    )
    print(f"   输出文件: {output_file}")

    # split 内重复正文检查
    print_within_split_duplicate_report(
        split_name=split_name,
        text_hash_counter=text_hash_counter,
        hash_examples=hash_examples,
        hash_label_pairs=hash_label_pairs,
    )

    return {
        "split_name": split_name,
        "raw_count": len(raw_data),
        "processed_count": len(processed_data),
        "skipped_count": 0,
        "l1_present": set(l1_counter.keys()),
        "l2_present": set(l2_counter.keys()),
        "l1_counter": l1_counter,
        "l2_counter": l2_counter,
        "text_hashes": set(text_hash_counter.keys()),
        "text_hash_counter": text_hash_counter,
        "hash_examples": hash_examples,
        "hash_label_pairs": hash_label_pairs,
    }


# ============================================================
# 三个 split 之间的标签覆盖检查
# ============================================================
def print_cross_split_report(
    train_stats,
    val_stats,
    test_stats,
    l1_label2id,
    l2_label2id,
    l2_to_l1,
):
    """
    只做统计报告，不修改任何数据。

    可以看出：
        - 哪些标签只在 val 出现
        - 哪些标签只在 test 出现
        - 哪些 taxnomy 标签三个 split 都没有出现
        - test-only L2 的父类及 train/val/test 具体频次
    """
    train_l1 = train_stats["l1_present"]
    val_l1 = val_stats["l1_present"]
    test_l1 = test_stats["l1_present"]

    train_l2 = train_stats["l2_present"]
    val_l2 = val_stats["l2_present"]
    test_l2 = test_stats["l2_present"]

    all_tax_l1 = set(l1_label2id.keys())
    all_tax_l2 = set(l2_label2id.keys())

    val_only_l1 = val_l1 - train_l1
    test_only_l1 = test_l1 - train_l1 - val_l1

    val_only_l2 = val_l2 - train_l2
    test_only_l2 = test_l2 - train_l2 - val_l2

    unseen_l1 = all_tax_l1 - train_l1 - val_l1 - test_l1
    unseen_l2 = all_tax_l2 - train_l2 - val_l2 - test_l2

    print("\n" + "=" * 70)
    print("跨 train / val / test 标签覆盖报告")
    print("=" * 70)

    print(f"taxnomy 完整 L1 数量: {len(all_tax_l1)}")
    print(f"taxnomy 完整 L2 数量: {len(all_tax_l2)}")

    print(f"\nval 中出现、但 train 未出现的 L1 数量: {len(val_only_l1)}")
    if val_only_l1:
        print("  ", sorted(val_only_l1))

    print(f"test 中出现、但 train+val 未出现的 L1 数量: {len(test_only_l1)}")
    if test_only_l1:
        print("  ", sorted(test_only_l1))

    print(f"\nval 中出现、但 train 未出现的 L2 数量: {len(val_only_l2)}")
    if val_only_l2:
        print("  ", sorted(val_only_l2))

    print(f"test 中出现、但 train+val 未出现的 L2 数量: {len(test_only_l2)}")
    if test_only_l2:
        print("  ", sorted(test_only_l2))

        # 关键新增：把类似 Electrical generator 的具体频次和父类打印出来
        print("\n   test-only L2 详细频次:")
        for label in sorted(test_only_l2):
            parent = l2_to_l1[label]
            train_count = train_stats["l2_counter"].get(label, 0)
            val_count = val_stats["l2_counter"].get(label, 0)
            test_count = test_stats["l2_counter"].get(label, 0)
            print(
                f"      L2={label!r} | Parent L1={parent!r} | "
                f"train={train_count} | val={val_count} | test={test_count}"
            )

    print(
        f"\ntaxnomy 中存在、但三个 split 都未出现的 L1 数量: "
        f"{len(unseen_l1)}"
    )
    if unseen_l1:
        print("  ", sorted(unseen_l1))

    print(
        f"taxnomy 中存在、但三个 split 都未出现的 L2 数量: "
        f"{len(unseen_l2)}"
    )
    if unseen_l2:
        print("  ", sorted(unseen_l2))

    print(
        "\n说明：以上只是统计，不会把三个 split 混合，"
        "也不会改变模型训练方式。"
    )


# ============================================================
# 跨 split 重复/泄漏检查
# ============================================================
def print_overlap_examples(
    overlap_name,
    overlap_hashes,
    left_stats,
    right_stats,
    limit=OVERLAP_REPORT_LIMIT,
):
    """打印两个 split 之间完全相同正文的示例。"""
    if not overlap_hashes:
        return

    print(f"   ⚠️ {overlap_name} 前 {limit} 个重叠正文示例:")

    for h in sorted(overlap_hashes)[:limit]:
        left_ex = left_stats["hash_examples"][h]
        right_ex = right_stats["hash_examples"][h]

        left_pairs = sorted(left_stats["hash_label_pairs"][h])
        right_pairs = sorted(right_stats["hash_label_pairs"][h])

        print(
            f"      hash={h[:16]}... | "
            f"{left_stats['split_name']}_labels={left_pairs} | "
            f"{right_stats['split_name']}_labels={right_pairs}"
        )
        print(
            f"      {left_stats['split_name']}_first_idx="
            f"{left_ex['item_idx']} | "
            f"{right_stats['split_name']}_first_idx="
            f"{right_ex['item_idx']}"
        )
        print(f"      preview={left_ex['text_preview']!r}")


def print_cross_split_text_overlap_report(
    train_stats,
    val_stats,
    test_stats,
):
    """
    检查 train / val / test 之间是否存在完全相同的清洗后正文。

    只报告，不自动删除、不移动样本，避免脚本擅自改变官方 split。
    """
    train_hashes = train_stats["text_hashes"]
    val_hashes = val_stats["text_hashes"]
    test_hashes = test_stats["text_hashes"]

    train_val_overlap = train_hashes & val_hashes
    train_test_overlap = train_hashes & test_hashes
    val_test_overlap = val_hashes & test_hashes
    all_three_overlap = train_hashes & val_hashes & test_hashes

    print("\n" + "=" * 70)
    print("train / val / test 正文重复与潜在泄漏检查")
    print("=" * 70)
    print("判定方式: 对当前脚本清洗后的正文做精确 SHA256 匹配")
    print(f"train ∩ val  重叠唯一正文数: {len(train_val_overlap)}")
    print(f"train ∩ test 重叠唯一正文数: {len(train_test_overlap)}")
    print(f"val ∩ test   重叠唯一正文数: {len(val_test_overlap)}")
    print(f"三者共同重叠唯一正文数: {len(all_three_overlap)}")

    print_overlap_examples(
        "train ∩ val",
        train_val_overlap,
        train_stats,
        val_stats,
    )
    print_overlap_examples(
        "train ∩ test",
        train_test_overlap,
        train_stats,
        test_stats,
    )
    print_overlap_examples(
        "val ∩ test",
        val_test_overlap,
        val_stats,
        test_stats,
    )

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print(
            "\n⚠️ 检测到跨 split 完全相同正文。"
            "脚本不会自动删除或移动样本，请根据报告人工确认是否属于数据泄漏。"
        )
    else:
        print("\n✅ 未检测到跨 split 完全相同正文。")


# ============================================================
# 主流程
# ============================================================
def process_wos_all_splits():
    print("=" * 70)
    print("WOS 全量标签提取 + train/val/test 干净统一预处理")
    print("=" * 70)

    # 1. 只从 taxnomy 建立唯一完整标签空间
    print(f"正在读取 taxnomy: {TAXNOMY_FILE}")
    l1_label2id, l2_label2id, l2_to_l1 = (
        load_full_label_space_from_taxnomy(TAXNOMY_FILE)
    )

    print(
        f"✅ taxnomy 解析完成：L1 共 {len(l1_label2id)} 类，"
        f"L2 共 {len(l2_label2id)} 类。"
    )

    # 2. 保存唯一全局标签映射
    save_label_dict(
        l1_label2id=l1_label2id,
        l2_label2id=l2_label2id,
        output_file=LABEL_DICT_FILE,
    )
    print(f"✅ 唯一全局标签字典已保存: {LABEL_DICT_FILE}")

    # 3. 分别处理 train / validation / test
    #    三者共用同一个 label2id，但输出到三个独立文件
    #    每条样本只保存 text + label string + id
    train_stats = process_split(
        split_name="TRAIN",
        input_file=TRAIN_INPUT_FILE,
        output_file=TRAIN_OUTPUT_FILE,
        l1_label2id=l1_label2id,
        l2_label2id=l2_label2id,
        l2_to_l1=l2_to_l1,
    )

    val_stats = process_split(
        split_name="VALIDATION",
        input_file=VAL_INPUT_FILE,
        output_file=VAL_OUTPUT_FILE,
        l1_label2id=l1_label2id,
        l2_label2id=l2_label2id,
        l2_to_l1=l2_to_l1,
    )

    test_stats = process_split(
        split_name="TEST",
        input_file=TEST_INPUT_FILE,
        output_file=TEST_OUTPUT_FILE,
        l1_label2id=l1_label2id,
        l2_label2id=l2_label2id,
        l2_to_l1=l2_to_l1,
    )

    # 4. 打印跨 split 标签覆盖情况，并显示 test-only L2 具体频次
    print_cross_split_report(
        train_stats=train_stats,
        val_stats=val_stats,
        test_stats=test_stats,
        l1_label2id=l1_label2id,
        l2_label2id=l2_label2id,
        l2_to_l1=l2_to_l1,
    )

    # 5. 检查 split 内/跨 split 完全相同正文
    #    split 内检查已在 process_split() 中打印；这里检查跨 split。
    print_cross_split_text_overlap_report(
        train_stats=train_stats,
        val_stats=val_stats,
        test_stats=test_stats,
    )

    print("\n" + "=" * 70)
    print("✅ 全部完成")
    print("=" * 70)
    print(f"全局标签字典: {LABEL_DICT_FILE}")
    print(f"训练集输出:     {TRAIN_OUTPUT_FILE}")
    print(f"验证集输出:     {VAL_OUTPUT_FILE}")
    print(f"测试集输出:     {TEST_OUTPUT_FILE}")
    print("\n输出样本字段:")
    print("   text")
    print("   l1_label")
    print("   l2_label")
    print("   l1_id")
    print("   l2_id")
    print("\n严格性检查:")
    print("   - 空正文直接报错")
    print("   - doc_label 缺失或类型错误直接报错")
    print("   - 标签层数必须严格等于 2")
    print("   - 不静默跳过异常样本")
    print("   - 报告 split 内重复正文")
    print("   - 报告跨 split 完全相同正文")
    print("   - 报告 test-only L2 的父类及 train/val/test 频次")
    print("\n注意：本预处理脚本不再生成任何 Prompt 或 Chat Template。")
    print("注意：重复检查只做报告，不自动删除、不移动、不混合任何 split。")


if __name__ == "__main__":
    process_wos_all_splits()
