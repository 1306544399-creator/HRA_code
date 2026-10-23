#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NYT 8B -> 1B 离线单-token-per-depth 蒸馏训练。

支持四种实验：
  traditional : Task CE + logits KD
  align       : Task CE + logits KD + D1-D8 hidden align
  hier        : Task CE + logits KD + D1->D2...D7->D8 relation KD
  full        : Task CE + logits KD + hidden align + relation KD

Teacher 不在线加载。Teacher logits 由缓存的 Completion final hidden 与
common-vocab lm_head 重建。D1-D8 Teacher hidden 由 depth_rel_offsets.npy 定位。
"""

import os
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import hashlib
import json
import random
import re
import shutil
import time
import peft
import transformers
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import swanlab
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from tqdm import tqdm
from swanlab.integration.transformers import SwanLabCallback
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed as hf_set_seed,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model


CTX_START_TOKEN = "<|reserved_special_token_110|>"
CTX_END_TOKEN = "<|reserved_special_token_111|>"
DEPTH_PROBE_TOKENS = {
    d: f"<|reserved_special_token_{111 + d}|>" for d in range(1, 9)
}
MAX_DEPTH = 8
FULL_IO_WEIGHTS_NAME = "full_input_output_embeddings.pt"
PROJECTOR_WEIGHTS_NAME = "hidden_state_projector.pt"
NYT_RESUME_STATE_NAME = "nyt_resume_state.json"
NYT_PENDING_NAME = "nyt_save_in_progress.json"
KD_TRAINER_STATE_PREFIX = "kd_trainer_state_rank_"



def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def install_chat_template(tokenizer) -> None:
    tokenizer.chat_template = (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + message['role'] + "
        "'<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' %}"
        "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
        "{{ content }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
        "{% endif %}"
    )


def load_json_or_jsonl(path: str) -> List[dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        raise ValueError(f"空文件: {path}")
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        return [json.loads(x) for x in raw.splitlines() if x.strip()]


def load_label_space(path: str, expected_num_labels: int = 166) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        x = json.load(f)
    required = {"num_labels", "max_depth", "label2id", "parent", "depth"}
    missing = required - set(x)
    if missing:
        raise KeyError(f"label_space 缺少字段: {sorted(missing)}")
    if int(x["num_labels"]) != expected_num_labels:
        raise ValueError(f"标签数异常: {x['num_labels']} != {expected_num_labels}")
    if int(x["max_depth"]) != MAX_DEPTH:
        raise ValueError(f"最大深度异常: {x['max_depth']} != {MAX_DEPTH}")
    label2id = {str(k): int(v) for k, v in x["label2id"].items()}
    if sorted(label2id.values()) != list(range(expected_num_labels)):
        raise ValueError("label2id 不是连续 0-based")
    x["label2id"] = label2id
    x["depth"] = {str(k): int(v) for k, v in x["depth"].items()}
    x["parent"] = {str(k): str(v) for k, v in x["parent"].items()}
    return x


def validate_clean_data(data: List[dict], label_space: dict, split: str) -> None:
    required = {
        "text", "labels", "label_ids", "top_labels", "top_label_ids",
        "leaf_labels", "leaf_label_ids",
    }
    l2i = label_space["label2id"]
    parent = label_space["parent"]
    for i, row in enumerate(data):
        miss = required - set(row)
        if miss:
            raise KeyError(f"{split} 第{i}条缺字段: {sorted(miss)}")
        text = row["text"]
        labels = row["labels"]
        ids = row["label_ids"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{split} 第{i}条正文为空")
        if len(labels) != len(ids) or len(labels) != len(set(labels)):
            raise ValueError(f"{split} 第{i}条 labels/ids 异常")
        expected = sorted(labels, key=l2i.get)
        if labels != expected or ids != [l2i[x] for x in expected]:
            raise ValueError(f"{split} 第{i}条标签顺序或ID错误")
        s = set(labels)
        for label in labels:
            if label not in l2i:
                raise ValueError(f"{split} 第{i}条未知标签: {label}")
            cur = label
            while parent[cur] != "Root":
                cur = parent[cur]
                if cur not in s:
                    raise ValueError(f"{split} 第{i}条不是祖先闭包")
    print(f"✅ {split} 校验完成: {len(data):,} 条")


def validate_reserved_tokens(tokenizer) -> Dict[str, object]:
    token_map = {"ctx_start": CTX_START_TOKEN, "ctx_end": CTX_END_TOKEN}
    token_map.update({f"depth_{d}": DEPTH_PROBE_TOKENS[d] for d in range(1, 9)})
    out = {}
    for name, tok in token_map.items():
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if tok not in tokenizer.get_vocab() or len(ids) != 1:
            raise RuntimeError(f"特殊 token 非原子或不存在: {tok}, ids={ids}")
        out[name] = int(ids[0])
    if len(set(out.values())) != len(out):
        raise RuntimeError("特殊 token ID 重复")
    return {
        "ctx_start": out["ctx_start"],
        "ctx_end": out["ctx_end"],
        "depth": {d: out[f"depth_{d}"] for d in range(1, 9)},
    }


def sort_depth_labels(labels: Sequence[str], depth: int,
                      prev_labels: Sequence[str], label_space: dict) -> List[str]:
    l2i = label_space["label2id"]
    parent = label_space["parent"]
    if depth == 1:
        return sorted(labels, key=l2i.get)
    rank = {x: i for i, x in enumerate(prev_labels)}
    return sorted(
        labels,
        key=lambda x: (
            0 if parent[x] in rank else 1,
            rank.get(parent[x], l2i.get(parent[x], len(l2i))),
            l2i[x],
        ),
    )


def group_labels_by_depth(labels: Sequence[str], label_space: dict) -> Dict[int, List[str]]:
    grouped = {d: [] for d in range(1, 9)}
    for x in labels:
        grouped[int(label_space["depth"][x])].append(x)
    for d in range(1, 9):
        grouped[d] = sort_depth_labels(
            grouped[d], d, grouped[d - 1] if d > 1 else [], label_space
        )
    return grouped


def build_user_content(text: str) -> str:
    assignments = "\n".join(f"D{d}={DEPTH_PROBE_TOKENS[d]}" for d in range(1, 9))
    return (
        f"{CTX_START_TOKEN}\n{text}\n{CTX_END_TOKEN}\n\n"
        "Classify the document according to the NYT hierarchical multi-label taxonomy "
        "with depths 1-8.\n\n"
        "Hierarchy definitions:\n"
        "- Depth Dk contains every label assigned to the document at exactly taxonomy depth k.\n"
        "- The complete output must be ancestor-closed across all paths.\n"
        "- For each depth, use the document and all previously generated shallower-depth "
        "labels as hierarchical context.\n\n"
        "Hierarchy-token assignments:\n"
        "Each Llama reserved special token below is dedicated to representing the document's "
        "hierarchy information at its assigned taxonomy depth and serves as the depth-specific "
        "prediction-state marker for all labels at that depth.\n"
        f"{assignments}\n\n"
        "Output exactly 8 lines in depth order.\n"
        "On each line, reproduce the corresponding reserved special token shown above, followed "
        "by one space and a valid JSON array containing all labels at that depth.\n"
        "At D1, keep labels in canonical label-ID order. At each deeper depth Dk, first group "
        "labels by the order of their parent labels in D(k-1), then keep children of the same "
        "parent in canonical label-ID order. Use [] when the depth has no label. Use exact "
        "taxonomy label strings.\n"
        "Stop immediately after the 8th line.\n"
        "Do not output numbering, colons, explanations, or any other text."
    )


def build_completion_text(labels: Sequence[str], label_space: dict) -> str:
    grouped = group_labels_by_depth(labels, label_space)
    lines = []
    for d in range(1, 9):
        arr = json.dumps(grouped[d], ensure_ascii=False, separators=(",", ":"))
        lines.append(f"{DEPTH_PROBE_TOKENS[d]} {arr}")
    return "\n".join(lines) + "\n"


def truncate_text(text: str, tokenizer, max_seq_length: int, reserve_tokens: int) -> str:
    budget = max_seq_length - reserve_tokens
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > budget:
        text = tokenizer.decode(ids[:budget], skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
    return text


def sha256_ids(ids: Sequence[int]) -> str:
    return hashlib.sha256(",".join(map(str, map(int, ids))).encode("ascii")).hexdigest()


def _strip_generation_stop_strings(text: str) -> str:
    """与 NYT Student SFT 一致：移除任意数量的尾部 EOT/EOS 字符串。"""
    text = text.replace("\r\n", "\n").strip()
    stop_strings = ("<|eot_id|>", "<|end_of_text|>")
    changed = True
    while changed:
        changed = False
        for stop in stop_strings:
            if text.endswith(stop):
                text = text[:-len(stop)].rstrip()
                changed = True
    return text


def trim_generated_at_first_stop(
    generated_ids: torch.Tensor,
    stop_ids: Sequence[int],
) -> List[List[int]]:
    """
    与最新 NYT Student SFT 验证完全一致：逐样本截到第一个 EOT/EOS。

    HF 批量 generate 中，某条样本提前结束后仍会被 pad 到该 batch 最长
    生成长度。若 pad_token_id==eos_token_id 且 decode 时保留 special token，
    不先在 token ID 层截断会把大量尾部 PAD/EOS 一并解码，导致严格 8 行
    JSON parser 将本来正确的输出判成 parse failure。
    """
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


def _is_ancestor_closed(label_set: Set[str], label_space: dict) -> bool:
    parent = label_space["parent"]
    for label in label_set:
        current = label
        while parent[current] != "Root":
            current = parent[current]
            if current not in label_set:
                return False
    return True


def parse_nyt_generation(raw_text: str, label_space: dict) -> dict:
    """与 NYT 训练/测试定义一致：8 行 depth token + JSON array。"""
    text = _strip_generation_stop_strings(raw_text)
    lines = text.split("\n") if text else []
    label2id = label_space["label2id"]
    depth_map = label_space["depth"]

    parsed_layers = {}
    parse_ok = len(lines) == MAX_DEPTH
    if parse_ok:
        for d, line in enumerate(lines, 1):
            prefix = DEPTH_PROBE_TOKENS[d] + " "
            if not line.startswith(prefix):
                parse_ok = False
                break
            payload = line[len(prefix):]
            try:
                arr = json.loads(payload)
            except json.JSONDecodeError:
                parse_ok = False
                break
            if not isinstance(arr, list) or not all(isinstance(x, str) for x in arr):
                parse_ok = False
                break
            parsed_layers[d] = arr

    if not parse_ok:
        return {
            "pred_ids": set(), "pred_labels": set(), "parse_ok": False,
            "all_labels_valid": False, "hierarchy_closed": False,
            "invalid_for_strict": True, "duplicate_count": 0,
            "unknown_labels": [], "wrong_depth_labels": [],
            "order_error_depths": [], "empty_prediction": True,
            "missing_top_prediction": True,
        }

    pred_labels: Set[str] = set()
    duplicate_count = 0
    unknown_labels = []
    wrong_depth_labels = []
    order_error_depths = []
    whitespace_errors = []

    for d in range(1, MAX_DEPTH + 1):
        layer = parsed_layers[d]
        local_seen = set()
        known_correct_depth = all(
            x in label2id and int(depth_map[x]) == d for x in layer
        )
        if known_correct_depth:
            expected = sort_depth_labels(
                layer, d, parsed_layers[d - 1] if d > 1 else [], label_space
            )
            if layer != expected:
                order_error_depths.append(d)

        for label in layer:
            if label != label.strip():
                whitespace_errors.append(label)
            if label in local_seen or label in pred_labels:
                duplicate_count += 1
            local_seen.add(label)
            if label not in label2id:
                unknown_labels.append(label)
                continue
            pred_labels.add(label)
            if int(depth_map[label]) != d:
                wrong_depth_labels.append(label)

    empty_prediction = not pred_labels
    missing_top_prediction = len(parsed_layers.get(1, [])) == 0
    structural_valid = (
        duplicate_count == 0
        and not unknown_labels
        and not wrong_depth_labels
        and not whitespace_errors
        and not order_error_depths
        and not empty_prediction
        and not missing_top_prediction
    )
    hierarchy_closed = _is_ancestor_closed(pred_labels, label_space) if structural_valid else False
    return {
        "pred_ids": {label2id[x] for x in pred_labels},
        "pred_labels": pred_labels,
        "parse_ok": True,
        "all_labels_valid": structural_valid,
        "hierarchy_closed": hierarchy_closed,
        "invalid_for_strict": not (structural_valid and hierarchy_closed),
        "duplicate_count": duplicate_count,
        "unknown_labels": unknown_labels,
        "wrong_depth_labels": wrong_depth_labels,
        "order_error_depths": order_error_depths,
        "empty_prediction": empty_prediction,
        "missing_top_prediction": missing_top_prediction,
    }


def compute_nyt_metrics(true_sets: Sequence[Set[int]], pred_sets: Sequence[Set[int]], num_labels: int) -> dict:
    y_true = np.zeros((len(true_sets), num_labels), dtype=np.int32)
    y_pred = np.zeros((len(pred_sets), num_labels), dtype=np.int32)
    for i, ids in enumerate(true_sets):
        for x in ids:
            y_true[i, int(x)] = 1
    for i, ids in enumerate(pred_sets):
        for x in ids:
            if 0 <= int(x) < num_labels:
                y_pred[i, int(x)] = 1

    present = np.flatnonzero(y_true.sum(axis=0) > 0)
    if present.size == 0:
        raise RuntimeError("验证集没有任何真实标签，无法计算 Macro-F1")

    return {
        "global_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "global_macro": float(f1_score(y_true[:, present], y_pred[:, present], average="macro", zero_division=0)),
        "global_macro_all166": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "global_samples": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "present_label_count": int(present.size),
    }


class GenerativeNYTMacroCallback(TrainerCallback):
    """定期生成式验证；按 Global Macro-F1 保存 best，Macro 相同再比较 Micro。"""
    def __init__(self, val_data, val_dataset, tokenizer, label_space, system_prompt,
                 max_seq_length, reserve_tokens, eval_steps, eval_batch_size,
                 loss_eval_batch_size, max_new_tokens, save_dir,
                 early_stopping_patience, early_stopping_min_delta,
                 eval_loss_logit_chunk_size=16, resume_checkpoint=None,
                 run_signature=None):
        self.val_data = val_data
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.label_space = label_space
        self.system_prompt = system_prompt
        self.max_seq_length = int(max_seq_length)
        self.reserve_tokens = int(reserve_tokens)
        self.eval_steps = int(eval_steps)
        self.eval_batch_size = int(eval_batch_size)
        self.loss_eval_batch_size = int(loss_eval_batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.save_dir = save_dir
        self.resume_checkpoint = resume_checkpoint
        self.run_signature = run_signature or {}
        self.history_records = []
        self.early_stopping_patience = int(early_stopping_patience)
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.eval_loss_logit_chunk_size = int(eval_loss_logit_chunk_size)
        self.best_saved_ma = -1.0
        self.best_saved_mi = -1.0
        self.early_stop_best_ma = -1.0
        self.early_stop_bad_count = 0
        self.early_stop_triggered = False
        self.last_evaluated_step = -1
        self._force_final_eval = False
        self.history_path = os.path.join(save_dir, "eval_history.jsonl")
        self.best_path = os.path.join(save_dir, "best_macro_checkpoint")
        self.best_meta_path = os.path.join(save_dir, "best_macro_checkpoint.json")
        if self.eval_steps <= 0 or self.early_stopping_patience <= 0 or self.early_stopping_min_delta < 0:
            raise ValueError("eval_steps/patience 必须 > 0，min_delta 必须 >= 0。")
        if not resume_checkpoint and os.path.isfile(self.history_path) and os.path.getsize(self.history_path):
            raise RuntimeError("已有验证历史，请使用 --resume_from_checkpoint auto 或新的 --save_dir。")

        self.val_prompts = []
        for idx, row in enumerate(val_data):
            text = truncate_text(row["text"].strip(), tokenizer, self.max_seq_length, self.reserve_tokens)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_content(text)},
            ]
            prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            re_ids = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
            if list(map(int, prompt_ids)) != list(map(int, re_ids)):
                raise RuntimeError(f"VALIDATION 第{idx}条 prompt 两种编码方式不一致")
            self.val_prompts.append({
                "prompt": prompt,
                "prompt_ids": list(map(int, prompt_ids)),
                "true_ids": set(map(int, row["label_ids"])),
            })

    def _state_dict(self) -> dict:
        return {
            "early_stop_best_ma": self.early_stop_best_ma,
            "early_stop_bad_count": self.early_stop_bad_count,
            "early_stop_triggered": self.early_stop_triggered,
            "best_saved_ma": self.best_saved_ma,
            "best_saved_mi": self.best_saved_mi,
            "last_evaluated_step": self.last_evaluated_step,
        }

    def _restore_callback_state(self, payload: dict) -> None:
        for key in self._state_dict():
            if key not in payload:
                raise RuntimeError(f"❌ 回调断点缺少状态字段: {key}")
            setattr(self, key, payload[key])
        self._force_final_eval = False

    def _replay_legacy_history(self, records: List[dict]) -> None:
        # 完全沿用原早停公式，支持非零 min_delta。
        for record in records:
            ma, mi = float(record["global_macro"]), float(record["global_micro"])
            if ma > self.best_saved_ma or (abs(ma - self.best_saved_ma) < 1e-12 and mi > self.best_saved_mi):
                self.best_saved_ma, self.best_saved_mi = ma, mi
            if self.early_stop_best_ma < 0 or ma > self.early_stop_best_ma + self.early_stopping_min_delta:
                self.early_stop_best_ma = ma
                self.early_stop_bad_count = 0
            else:
                self.early_stop_bad_count += 1
            self.last_evaluated_step = int(record["step"])
        self.early_stop_triggered = self.early_stop_bad_count >= self.early_stopping_patience

    def _root_action(self, state, action):
        """让其他 DDP rank 等待主进程，并同步主进程错误。"""
        result = [None]
        if state.is_world_process_zero:
            try:
                result[0] = {"value": action(), "error": None}
            except Exception as exc:
                result[0] = {"value": None, "error": f"{type(exc).__name__}: {exc}"}
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(result, src=0)
        if result[0] is None:
            raise RuntimeError("❌ 当前非主进程没有初始化分布式通信。")
        if result[0]["error"]:
            raise RuntimeError(result[0]["error"])
        return result[0]["value"]

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.resume_checkpoint:
            return control

        def restore():
            step = int(read_json_file(os.path.join(self.resume_checkpoint, "trainer_state.json"))["global_step"])
            if int(state.global_step) != step:
                raise RuntimeError("❌ Trainer 未正确恢复 global_step。")
            custom_path = os.path.join(self.resume_checkpoint, NYT_RESUME_STATE_NAME)
            if os.path.isfile(custom_path):
                saved = read_json_file(custom_path)
                records = saved["eval_history"]
                self._restore_callback_state(saved["callback_state"])
            else:
                records = read_eval_history_for_resume(self.history_path, step)
                if step >= self.eval_steps and (not records or int(records[-1]["step"]) != step):
                    raise RuntimeError("❌ 旧周期断点缺少对应 step 的验证记录，无法准确恢复回调。")
                self._replay_legacy_history(records)
            previous = -1
            for record in records:
                current = int(record["step"])
                if current <= previous or current > step:
                    raise RuntimeError("❌ 断点内验证历史与训练步数不一致。")
                previous = current
            if self.last_evaluated_step != (int(records[-1]["step"]) if records else -1):
                raise RuntimeError("❌ 回调验证步数与断点内历史不一致。")
            if self.early_stop_triggered:
                raise RuntimeError(
                    "❌ 该断点已触发早停，不再执行同一实验的续训。"
                    "若要更改早停策略继续微调，请另开实验。"
                )
            if step >= int(state.max_steps):
                raise RuntimeError("❌ 该断点已完成原训练计划，无剩余训练步数。")
            ensure_historical_best_available(self.save_dir, records, bool(self.run_signature.get("projector_required", False)))
            write_history_snapshot(self.history_path, records)
            archive_future_checkpoints(self.save_dir, step)
            self.history_records = records
            print(
                f"✅ 已恢复 step={step}, last_eval={self.last_evaluated_step}, "
                f"best_macro={self.best_saved_ma:.6f}, "
                f"early_stop_bad_count={self.early_stop_bad_count}/{self.early_stopping_patience}"
            )
            return {"callback_state": self._state_dict(), "eval_history": records}

        restored = self._root_action(state, restore)
        self._restore_callback_state(restored["callback_state"])
        self.history_records = restored["eval_history"]
        return control

    def on_save(self, args, state, control, **kwargs):
        # 先等所有 rank 写完自己的 RNG，再由主进程写完成标记。
        distributed_barrier()

        def save():
            checkpoint = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            self.tokenizer.save_pretrained(checkpoint)
            payload = {
                "format_version": 1,
                "global_step": int(state.global_step),
                "run_signature": self.run_signature,
                "callback_state": self._state_dict(),
                "eval_history": self.history_records,
                "checkpoint_files": checkpoint_file_sizes(checkpoint, int(args.world_size), projector_required=bool(self.run_signature.get("projector_required", False)), fp16=bool(self.run_signature.get("fp16", False)), require_kd_state=True),
            }
            atomic_write_json(os.path.join(checkpoint, NYT_RESUME_STATE_NAME), payload)
            pending = os.path.join(checkpoint, NYT_PENDING_NAME)
            if os.path.isfile(pending):
                os.remove(pending)
            print(f"✅ 完整续训断点保存完成: {checkpoint}")

        self._root_action(state, save)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        should_eval = (
            step > 0 and step != self.last_evaluated_step
            and (self._force_final_eval or step % self.eval_steps == 0)
        )
        if not should_eval:
            return control

        def evaluate():
            eval_kwargs = dict(kwargs)
            if eval_kwargs.get("model") is None:
                raise RuntimeError("❌ 验证回调未取得模型。")
            eval_kwargs["model"] = unwrap(eval_kwargs["model"])
            self._evaluate_at_step(args, state, control, **eval_kwargs)
            return {
                "callback_state": self._state_dict(),
                "should_training_stop": bool(control.should_training_stop),
            }

        result = self._root_action(state, evaluate)
        self._restore_callback_state(result["callback_state"])
        control.should_training_stop = result["should_training_stop"]
        return control

    def _compute_eval_loss(self, model, device):
        collator = DistillCollator(self.tokenizer.pad_token_id)
        loader = DataLoader(
            self.val_dataset, batch_size=self.loss_eval_batch_size,
            shuffle=False, collate_fn=collator, num_workers=0,
        )
        decoder, lm_head = get_causal_components(model)
        total_sum = 0.0
        total_n = 0
        with torch.inference_mode():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                out = decoder(input_ids=input_ids, attention_mask=attention_mask,
                              use_cache=False, return_dict=True)
                h = out.last_hidden_state[:, :-1, :]
                t = labels[:, 1:]
                mask = t.ne(-100)
                hs = h[mask]
                ts = t[mask]
                for st in range(0, hs.size(0), self.eval_loss_logit_chunk_size):
                    ed = min(st + self.eval_loss_logit_chunk_size, hs.size(0))
                    logits = lm_head(hs[st:ed])
                    loss = F.cross_entropy(logits.float(), ts[st:ed], reduction="sum")
                    total_sum += float(loss.detach().cpu())
                    total_n += ed - st
                del out, h, t, hs, ts
        return total_sum / max(1, total_n)

    def _generate_batch(self, model, batch, device, stop_ids):
        prompts = [x["prompt"] for x in batch]
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=False,
            add_special_tokens=False,
        ).to(device)
        for row_idx, item in enumerate(batch):
            actual = inputs.input_ids[row_idx][inputs.attention_mask[row_idx].bool()].detach().cpu().tolist()
            if actual != item["prompt_ids"]:
                raise RuntimeError("生成验证 prompt IDs 与预构造 prompt 不一致")
        generated = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=list(stop_ids),
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        prompt_width = inputs.input_ids.shape[1]
        generated_ids = generated[:, prompt_width:]

        # 关键：必须像最新 NYT Student SFT 一样，在 decode 前逐样本截到
        # 第一个 EOT/EOS，不能把 batch 里提前结束样本后续补的 PAD/EOS 解码进去。
        trimmed_generated_ids = trim_generated_at_first_stop(
            generated_ids=generated_ids,
            stop_ids=stop_ids,
        )
        result = [
            self.tokenizer.decode(
                row_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for row_ids in trimmed_generated_ids
        ]
        del inputs, generated, generated_ids, trimmed_generated_ids
        return result

    def _append_history(self, payload):
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.history_records.append(dict(payload))

    def _evaluate_at_step(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None or not state.is_world_process_zero or state.global_step == 0:
            return control
        if not self._force_final_eval and state.global_step % self.eval_steps != 0:
            return control
        if int(state.global_step) == self.last_evaluated_step:
            return control

        was_training = model.training
        model.eval()
        model.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        device = next(model.parameters()).device
        print(f"\n[Step {state.global_step}] 开始 NYT 生成式验证（best 按 Global Macro-F1）...")

        try:
            eval_loss = self._compute_eval_loss(model, device)
            eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot is None or eot == self.tokenizer.unk_token_id:
                eot = self.tokenizer.eos_token_id
            stop_ids = []
            for sid in [self.tokenizer.eos_token_id, eot]:
                if sid is not None and int(sid) not in stop_ids:
                    stop_ids.append(int(sid))
            if not stop_ids:
                raise RuntimeError("没有可用 generation stop id")

            old_padding = self.tokenizer.padding_side
            raw_model = unwrap(model)
            old_use_cache = getattr(raw_model.config, "use_cache", False)
            self.tokenizer.padding_side = "left"
            raw_model.config.use_cache = True

            true_sets, pred_sets = [], []
            parse_ok = valid_ok = hier_ok = exact_match = 0
            try:
                with torch.inference_mode():
                    for st in tqdm(range(0, len(self.val_prompts), self.eval_batch_size),
                                   desc=f"Step {state.global_step} NYT Eval"):
                        batch = self.val_prompts[st:st + self.eval_batch_size]
                        raw_texts = self._generate_batch(model, batch, device, stop_ids)
                        for item, raw_text in zip(batch, raw_texts):
                            parsed = parse_nyt_generation(raw_text, self.label_space)
                            true_ids = item["true_ids"]
                            pred_ids = parsed["pred_ids"]
                            true_sets.append(true_ids)
                            pred_sets.append(pred_ids)
                            parse_ok += int(parsed["parse_ok"])
                            valid_ok += int(parsed["all_labels_valid"])
                            hier_ok += int(parsed["hierarchy_closed"])
                            exact_match += int(pred_ids == true_ids)
            finally:
                self.tokenizer.padding_side = old_padding
                raw_model.config.use_cache = old_use_cache

            metrics = compute_nyt_metrics(
                true_sets, pred_sets, int(self.label_space["num_labels"])
            )
            n = max(1, len(true_sets))
            metrics.update({
                "exact_parse_rate": parse_ok / n,
                "valid_output_rate": valid_ok / n,
                "hierarchy_closed_rate": hier_ok / n,
                "exact_match_rate": exact_match / n,
            })
            print(
                f"[Step {state.global_step}] EvalLoss={eval_loss:.6f} | "
                f"Micro={metrics['global_micro']:.6f} | "
                f"Macro={metrics['global_macro']:.6f} | "
                f"Macro-All166={metrics['global_macro_all166']:.6f} | "
                f"Samples={metrics['global_samples']:.6f}"
            )
            print(
                f"[Step {state.global_step}] Parse={metrics['exact_parse_rate']:.4f} | "
                f"Valid={metrics['valid_output_rate']:.4f} | "
                f"HierClosed={metrics['hierarchy_closed_rate']:.4f} | "
                f"ExactMatch={metrics['exact_match_rate']:.4f}"
            )

            swanlab.log(
                {
                    "Eval/Loss": float(eval_loss),
                    "Eval/Global_Micro_F1": float(metrics["global_micro"]),
                    "Eval/Global_Macro_F1": float(metrics["global_macro"]),
                    "Eval/Global_Macro_F1_All166": float(metrics["global_macro_all166"]),
                    "Eval/Samples_F1": float(metrics["global_samples"]),
                    "Eval/Exact_Parse_Rate": float(metrics["exact_parse_rate"]),
                    "Eval/Valid_Output_Rate": float(metrics["valid_output_rate"]),
                    "Eval/Ancestor_Closure_Rate": float(metrics["hierarchy_closed_rate"]),
                    "Eval/Exact_Set_Match_Rate": float(metrics["exact_match_rate"]),
                },
                step=state.global_step,
            )

            record = {"step": int(state.global_step), "eval_loss": float(eval_loss), **metrics}
            self._append_history(record)
            current_ma = metrics["global_macro"]
            current_mi = metrics["global_micro"]
            is_better = current_ma > self.best_saved_ma or (
                abs(current_ma - self.best_saved_ma) < 1e-12 and current_mi > self.best_saved_mi
            )
            if is_better:
                self.best_saved_ma, self.best_saved_mi = current_ma, current_mi
                best_meta = {
                    "step": int(state.global_step),
                    "global_macro": float(current_ma),
                    "global_micro": float(current_mi),
                    "eval_loss": float(eval_loss),
                    "selection_rule": "original KD rule: higher Macro; abs(Macro difference)<1e-12 uses higher Micro",
                }
                save_checkpoint(model, self.tokenizer, self.best_path, checkpoint_metadata=best_meta)
                atomic_write_json(self.best_meta_path, best_meta)
                print(f"🏆 更新 best Macro checkpoint: step={state.global_step}, MA={current_ma:.6f}, MI={current_mi:.6f}")

            if self.early_stop_best_ma < 0 or current_ma > self.early_stop_best_ma + self.early_stopping_min_delta:
                self.early_stop_best_ma = current_ma
                self.early_stop_bad_count = 0
            else:
                self.early_stop_bad_count += 1
            print(f"[Step {state.global_step}] EarlyStop MA best={self.early_stop_best_ma:.6f} bad={self.early_stop_bad_count}/{self.early_stopping_patience}")
            if self.early_stop_bad_count >= self.early_stopping_patience:
                self.early_stop_triggered = True
                control.should_training_stop = True
                print("🛑 Global Macro-F1 连续多次未提升，触发早停。")

            self.last_evaluated_step = int(state.global_step)
        finally:
            if was_training:
                model.train()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if state.global_step == 0:
            return control
        if int(state.global_step) == self.last_evaluated_step:
            return control
        self._force_final_eval = True
        try:
            return self.on_step_end(args, state, control, **kwargs)
        finally:
            self._force_final_eval = False


class NYTDataset(Dataset):
    def __init__(self, data, tokenizer, label_space, token_ids, max_length,
                 reserve_tokens, system_prompt, include_sample_index):
        self.rows = []
        self.include_sample_index = include_sample_index
        for idx, row in enumerate(data):
            text = truncate_text(row["text"].strip(), tokenizer, max_length, reserve_tokens)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_content(text)},
            ]
            prompt_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            prompt_ids = [int(x) for x in prompt_ids]
            completion_ids = tokenizer.encode(
                build_completion_text(row["labels"], label_space),
                add_special_tokens=False,
            )
            eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot is None or eot == tokenizer.unk_token_id:
                eot = tokenizer.eos_token_id
            completion_ids = [int(x) for x in completion_ids] + [int(eot)]
            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids
            if len(input_ids) > max_length:
                raise RuntimeError(f"样本{idx}长度{len(input_ids)}>{max_length}")

            depth_positions = []
            for d in range(1, 9):
                pos = [i for i, x in enumerate(input_ids) if x == token_ids["depth"][d]]
                if len(pos) != 2:
                    raise RuntimeError(f"样本{idx} D{d} occurrence={pos}")
                depth_positions.append(pos[1])

            shift_labels = labels[1:]
            valid_positions = [p for p, y in enumerate(shift_labels) if y != -100]
            pos_to_rel = {p: j for j, p in enumerate(valid_positions)}
            depth_rel = []
            for d, p in enumerate(depth_positions, 1):
                if p not in pos_to_rel:
                    raise RuntimeError(f"样本{idx} D{d} position 不在 causal mask")
                depth_rel.append(pos_to_rel[p])

            item = {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
                "depth_rel_offsets": depth_rel,
                "input_sha256": sha256_ids(input_ids),
            }
            if include_sample_index:
                item["sample_index"] = idx
            self.rows.append(item)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


@dataclass
class DistillCollator:
    pad_token_id: int

    def __call__(self, features: List[dict]) -> dict:
        max_len = max(len(x["input_ids"]) for x in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for x in features:
            n = max_len - len(x["input_ids"])
            batch["input_ids"].append(x["input_ids"] + [self.pad_token_id] * n)
            batch["attention_mask"].append(x["attention_mask"] + [0] * n)
            batch["labels"].append(x["labels"] + [-100] * n)
        out = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
        if "sample_index" in features[0]:
            out["sample_index"] = torch.tensor(
                [x["sample_index"] for x in features], dtype=torch.long
            )
        out["depth_rel_offsets"] = torch.tensor(
            [x["depth_rel_offsets"] for x in features], dtype=torch.long
        )
        return out


class TeacherCache:
    def __init__(self, cache_dir: str, expected_train_size: int,
                 expected_max_length: int, expected_reserve: int,
                 expected_system_prompt: str):
        manifest_path = os.path.join(cache_dir, "manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        m = self.manifest
        if str(m.get("dataset", "")).upper() != "NYT":
            raise ValueError(f"缓存 dataset 不是 NYT: {m.get('dataset')}")
        if int(m["num_samples"]) != expected_train_size:
            raise ValueError(f"缓存样本数 {m['num_samples']} != 训练集 {expected_train_size}")
        if int(m["max_seq_length"]) != expected_max_length:
            raise ValueError("缓存 max_seq_length 与训练不一致")
        if int(m["reserve_tokens"]) != expected_reserve:
            raise ValueError("缓存 reserve_tokens 与训练不一致")
        if m["system_prompt"] != expected_system_prompt:
            raise ValueError("缓存 system_prompt 与训练不一致")

        files = m["files"]
        self.offsets = np.load(os.path.join(cache_dir, files["offsets"]), mmap_mode="r")
        self.valid_counts = np.load(os.path.join(cache_dir, files["valid_counts"]), mmap_mode="r")
        self.depth_rel = np.load(os.path.join(cache_dir, files["depth_rel_offsets"]), mmap_mode="r")
        self.hashes = np.load(os.path.join(cache_dir, files["student_input_sha256"]), mmap_mode="r")
        self.hidden = np.load(os.path.join(cache_dir, files["teacher_completion_hidden"]), mmap_mode="r")
        self.student_common_ids_np = np.load(os.path.join(cache_dir, files["student_common_ids"]), mmap_mode="r")
        self.teacher_head = np.load(os.path.join(cache_dir, files["teacher_common_lm_head"]), mmap_mode="r")
        bias_name = files.get("teacher_common_lm_head_bias")
        self.teacher_bias = (
            np.load(os.path.join(cache_dir, bias_name), mmap_mode="r") if bias_name else None
        )
        if self.depth_rel.shape != (expected_train_size, 8):
            raise ValueError(f"depth_rel_offsets shape 异常: {self.depth_rel.shape}")
        if self.hidden.shape[1] != int(m["teacher_hidden_size"]):
            raise ValueError("Teacher hidden size 与 manifest 不一致")
        if self.teacher_head.shape != (
            int(m["common_vocab_size"]), int(m["teacher_hidden_size"])
        ):
            raise ValueError(f"teacher_common_lm_head shape 异常: {self.teacher_head.shape}")
        print("✅ Teacher cache 校验完成")
        print(f"   samples={expected_train_size:,}, positions={self.hidden.shape[0]:,}")
        print(f"   common_vocab={self.teacher_head.shape[0]:,}, teacher_hidden={self.hidden.shape[1]}")

    @property
    def teacher_hidden_size(self):
        return int(self.hidden.shape[1])

    @property
    def common_vocab_size(self):
        """Teacher/Student 共同词表大小；与 manifest 和 cached lm_head 行数一致。"""
        return int(self.teacher_head.shape[0])

    def verify_dataset(self, dataset: NYTDataset) -> None:
        for i, row in enumerate(dataset.rows):
            expected = self.hashes[i]
            if isinstance(expected, bytes):
                expected = expected.decode("ascii")
            else:
                expected = bytes(expected).decode("ascii")
            if row["input_sha256"] != expected:
                raise RuntimeError(f"训练样本{i} token hash 与 Teacher cache 不一致")
            if row["depth_rel_offsets"] != self.depth_rel[i].astype(int).tolist():
                raise RuntimeError(f"训练样本{i} depth offset 与 Teacher cache 不一致")
        print("✅ 全量训练样本 token hash / D1-D8 offset 与缓存一致")

    def get_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        s, e = int(self.offsets[idx]), int(self.offsets[idx + 1])
        return np.asarray(self.hidden[s:e]), np.asarray(self.depth_rel[idx])


def unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def get_causal_components(model):
    raw = unwrap(model)
    causal = raw.get_base_model() if isinstance(raw, PeftModel) else raw
    decoder = getattr(causal, "model", None)
    lm_head = causal.get_output_embeddings()
    if decoder is None or lm_head is None:
        raise RuntimeError("无法取得 decoder/lm_head")
    return decoder, lm_head


def enable_full_io(model):
    raw = unwrap(model)
    raw.get_input_embeddings().weight.requires_grad_(True)
    raw.get_output_embeddings().weight.requires_grad_(True)


def save_full_io(model, path: str):
    raw = unwrap(model)
    inp = raw.get_input_embeddings().weight
    out = raw.get_output_embeddings().weight
    payload = {"input_embeddings": inp.detach().cpu(),
               "actually_tied": inp.data_ptr() == out.data_ptr()}
    if not payload["actually_tied"]:
        payload["output_embeddings"] = out.detach().cpu()
    torch.save(payload, os.path.join(path, FULL_IO_WEIGHTS_NAME))


def save_checkpoint(model, tokenizer, path: str, checkpoint_metadata=None):
    """模型权重目录使用临时保存，并保留上一份模型。"""
    staging = path + f".saving-{os.getpid()}-{time.time_ns()}"
    os.makedirs(staging, exist_ok=False)
    raw = unwrap(model)
    raw.save_pretrained(staging)
    save_full_io(raw, staging)
    save_projector_weights(raw, staging)
    tokenizer.save_pretrained(staging)
    if checkpoint_metadata is not None:
        atomic_write_json(os.path.join(staging, "best_macro_metadata.json"), checkpoint_metadata)
    previous = path + ".previous"
    if os.path.isdir(path):
        if os.path.isdir(previous):
            shutil.rmtree(previous)
        os.replace(path, previous)
    os.replace(staging, path)


def export_merged(base_path: str, checkpoint: str, output: str, tokenizer, dtype):
    payload = torch.load(os.path.join(checkpoint, FULL_IO_WEIGHTS_NAME), map_location="cpu")
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=dtype, device_map="cpu")
    inp = base.get_input_embeddings().weight
    out = base.get_output_embeddings().weight
    with torch.no_grad():
        inp.copy_(payload["input_embeddings"].to(inp.dtype))
        if not payload["actually_tied"]:
            if inp.data_ptr() == out.data_ptr():
                base.set_output_embeddings(nn.Linear(out.in_features, out.out_features, bias=False,
                                                     dtype=out.dtype, device=out.device))
                out = base.get_output_embeddings().weight
            out.copy_(payload["output_embeddings"].to(out.dtype))
            base.config.tie_word_embeddings = False
    merged = PeftModel.from_pretrained(base, checkpoint).merge_and_unload()
    if os.path.isdir(output):
        shutil.rmtree(output)
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(output)


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
                          projector_required: bool = False, fp16: bool = False,
                          require_kd_state: bool = False) -> dict:
    """当前脚本的普通单卡 / DDP checkpoint；不适用于 DeepSpeed/FSDP。"""
    required = [
        "adapter_config.json", FULL_IO_WEIGHTS_NAME,
        "trainer_state.json", "optimizer.pt", "scheduler.pt", "training_args.bin",
    ]
    if projector_required:
        required.append(PROJECTOR_WEIGHTS_NAME)
    elif os.path.isfile(os.path.join(checkpoint, PROJECTOR_WEIGHTS_NAME)):
        raise RuntimeError("断点含 Projector，但当前模式不创建 Projector，请沿用原蒸馏模式与权重。")
    if fp16:
        required.append("scaler.pt")
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


def validate_checkpoint(checkpoint: str, world_size: int,
                        projector_required: bool = False, fp16: bool = False) -> int:
    if os.path.exists(os.path.join(checkpoint, NYT_PENDING_NAME)):
        raise RuntimeError(f"❌ 该断点保存时被中断，不能用于续训: {checkpoint}")
    custom_path = os.path.join(checkpoint, NYT_RESUME_STATE_NAME)
    sizes = checkpoint_file_sizes(checkpoint, world_size, projector_required, fp16,
                                  require_kd_state=os.path.isfile(custom_path))
    state = read_json_file(os.path.join(checkpoint, "trainer_state.json"))
    step = int(state["global_step"])
    if step <= 0:
        raise RuntimeError(f"❌ 断点 global_step 无效: {checkpoint}")
    folder_match = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(checkpoint))
    if folder_match is None or int(folder_match.group(1)) != step:
        raise RuntimeError("❌ 续训必须指定与 global_step 一致的周期 checkpoint-N 目录。")
    custom_path = os.path.join(checkpoint, NYT_RESUME_STATE_NAME)
    if os.path.isfile(custom_path):
        custom = read_json_file(custom_path)
        if int(custom["global_step"]) != step:
            raise RuntimeError("❌ Trainer 与 NYT 回调保存步数不一致。")
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
                validate_checkpoint(path, world_size, kd_projector_required(args), not args.bf16)
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
    validate_checkpoint(path, world_size, kd_projector_required(args), not args.bf16)
    return path


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kd_projector_required(args) -> bool:
    mode = args.experiment_mode.lower().strip()
    return ((mode in ("align", "full") and args.align_weight > 0)
            or (mode in ("hier", "full") and args.hier_weight > 0))


def cache_file_fingerprint(path: str) -> dict:
    """小文件全量 SHA；大缓存检查大小、修改时间和三段采样 SHA。"""
    stat = os.stat(path)
    result = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if stat.st_size <= 64 * 1024 * 1024:
        result["sha256"] = sha256_file(path)
        result["hash_policy"] = "full"
    else:
        chunk = 1024 * 1024
        offsets = sorted({0, max(0, stat.st_size // 2 - chunk // 2), max(0, stat.st_size - chunk)})
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for offset in offsets:
                f.seek(offset)
                digest.update(str(offset).encode("ascii"))
                digest.update(f.read(chunk))
        result["sha256"] = digest.hexdigest()
        result["hash_policy"] = "head_middle_tail_1MiB"
    return result


def build_run_signature(args) -> dict:
    fields = (
        "experiment_mode", "seed", "expected_num_labels", "max_seq_length",
        "reserve_tokens", "system_prompt", "task_weight", "kd_weight",
        "align_weight", "hier_weight", "kd_temperature", "align_alpha",
        "kd_position_chunk", "num_train_epochs", "train_batch_size", "eval_batch_size",
        "accumulation_steps", "learning_rate", "io_lr", "projector_lr", "weight_decay",
        "lr_scheduler_type", "warmup_ratio", "lora_rank", "lora_alpha", "lora_dropout",
        "eval_steps", "generation_eval_batch_size", "max_new_tokens",
        "early_stopping_patience", "early_stopping_min_delta",
        "eval_loss_logit_chunk_size", "bf16", "gradient_checkpointing",
        "attn_implementation", "num_workers", "dataloader_pin_memory",
    )
    signature = {name: getattr(args, name) for name in fields}
    signature["dataset"] = "NYT"
    signature["max_depth"] = MAX_DEPTH
    signature["hierarchy_policy"] = "hier_all_adjacent_D1_D8;align_all_D1_D8"
    signature["projector_required"] = kd_projector_required(args)
    signature["fp16"] = not args.bf16
    signature["world_size"] = int(os.environ.get("WORLD_SIZE", "1"))
    signature["model_path"] = os.path.realpath(args.model_path)
    signature["teacher_cache_dir"] = os.path.realpath(args.teacher_cache_dir)
    signature["data_sha256"] = {
        name: sha256_file(getattr(args, name))
        for name in ("train_data_path", "val_data_path", "label_space_path")
    }
    signature["base_configuration_sha256"] = {
        name: sha256_file(os.path.join(args.model_path, name))
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        if os.path.isfile(os.path.join(args.model_path, name))
    }
    manifest_path = os.path.join(args.teacher_cache_dir, "manifest.json")
    manifest = read_json_file(manifest_path)
    cache_files = {}
    for name, relative_path in sorted(manifest["files"].items()):
        if relative_path:
            cache_files[name] = cache_file_fingerprint(os.path.join(args.teacher_cache_dir, relative_path))
    signature["teacher_cache_fingerprint"] = {
        "manifest_sha256": sha256_file(manifest_path), "files": cache_files,
    }
    signature["versions"] = {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "peft": peft.__version__,
    }
    return signature


def validate_resume_configuration(checkpoint: str, signature: dict) -> None:
    path = os.path.join(checkpoint, NYT_RESUME_STATE_NAME)
    if os.path.isfile(path):
        saved = read_json_file(path)["run_signature"]
        changed = [key for key in sorted(set(saved) | set(signature))
                   if saved.get(key) != signature.get(key)]
        if changed:
            raise RuntimeError(
                "续训配置与断点不一致: " + ", ".join(changed)
                + "。请沿用原蒸馏模式、损失参数、教师缓存、数据、环境和训练计划。"
            )
        return
    # 旧脚本没有保存完整配置指纹；使用原实验 distill_config 核对。
    old_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint)), "distill_config.json")
    if not os.path.isfile(old_path):
        raise RuntimeError("旧蒸馏断点缺少 distill_config.json，无法核对蒸馏模式和损失参数。")
    saved = read_json_file(old_path)
    if saved.get("dataset", "NYT") != "NYT" or saved.get("max_depth", MAX_DEPTH) != MAX_DEPTH:
        raise RuntimeError("旧蒸馏配置不是 NYT 八层级实验，不能混用断点。")
    if saved.get("hierarchy_policy", signature["hierarchy_policy"]) != signature["hierarchy_policy"]:
        raise RuntimeError("旧蒸馏配置不是全层级 Hier，不能与 only-label 版本混用断点。")
    fingerprint_fields = {
        "data_sha256", "base_configuration_sha256", "teacher_cache_fingerprint", "versions",
        "world_size", "projector_required", "fp16", "hierarchy_policy", "dataset", "max_depth",
    }
    changed = []
    for key in signature:
        if key in fingerprint_fields:
            continue
        if key not in saved:
            changed.append(key + "(旧配置缺失)")
            continue
        old_value = saved[key]
        if key in ("model_path", "teacher_cache_dir"):
            old_value = os.path.realpath(old_value)
        if old_value != signature[key]:
            changed.append(key)
    # 同时核对实际 Trainer 配置与 LoRA 配置。
    try:
        trainer_args = torch.load(os.path.join(checkpoint, "training_args.bin"),
                                  map_location="cpu", weights_only=False)
    except TypeError:
        trainer_args = torch.load(os.path.join(checkpoint, "training_args.bin"), map_location="cpu")
    mapping = {
        "seed": "seed", "num_train_epochs": "num_train_epochs",
        "train_batch_size": "per_device_train_batch_size",
        "accumulation_steps": "gradient_accumulation_steps",
        "learning_rate": "learning_rate", "weight_decay": "weight_decay",
        "warmup_ratio": "warmup_ratio", "bf16": "bf16",
        "gradient_checkpointing": "gradient_checkpointing",
        "eval_steps": "save_steps", "lr_scheduler_type": "lr_scheduler_type",
    }
    for key, trainer_key in mapping.items():
        value = getattr(trainer_args, trainer_key, None)
        if getattr(value, "value", value) != signature[key]:
            changed.append(key + "(Trainer)")
    adapter = read_json_file(os.path.join(checkpoint, "adapter_config.json"))
    for key, adapter_key in (("lora_rank", "r"), ("lora_alpha", "lora_alpha"), ("lora_dropout", "lora_dropout")):
        if adapter.get(adapter_key) != signature[key]:
            changed.append(key + "(LoRA)")
    if changed:
        raise RuntimeError("旧蒸馏断点配置不一致: " + ", ".join(changed))
    print("⚠️ 旧断点未记录层级策略与缓存/数据指纹；请确认它来自 NYT 全层级 Hier 版本，且使用原数据和原教师缓存。")


def load_full_input_output_weights_for_resume(model, checkpoint: str) -> None:
    raw = unwrap(model)
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
        raise RuntimeError("❌ 旧断点缺少 eval_history.jsonl，无法恢复早停与历史最佳状态。")
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
        if ma > best_ma or (abs(ma - best_ma) < 1e-12 and mi > best_mi):
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


def ensure_historical_best_available(save_dir: str, records: List[dict],
                                     projector_required: bool = False) -> None:
    """恢复较早断点时，不能将未来的 best 混入恢复后的实验。"""
    best = select_best_record(records)
    if best is None:
        return
    online = os.path.join(save_dir, "best_macro_checkpoint")
    meta_path = os.path.join(save_dir, "best_macro_checkpoint.json")
    source = None
    for candidate in (online, online + ".previous"):
        embedded = os.path.join(candidate, "best_macro_metadata.json")
        candidate_meta = embedded if os.path.isfile(embedded) else (
            meta_path if candidate == online else None
        )
        if candidate_meta and os.path.isfile(candidate_meta) and os.path.isdir(candidate):
            meta = read_json_file(candidate_meta)
            matches = all(meta.get(key) == best[key] for key in ("step", "global_macro", "global_micro"))
            files_exist = all(os.path.isfile(os.path.join(candidate, name)) for name in (
                FULL_IO_WEIGHTS_NAME, "adapter_config.json",
            )) and any(os.path.isfile(os.path.join(candidate, name)) for name in (
                "adapter_model.safetensors", "adapter_model.bin",
            ))
            if projector_required:
                files_exist = files_exist and os.path.isfile(os.path.join(candidate, PROJECTOR_WEIGHTS_NAME))
            if matches and files_exist:
                if candidate == online:
                    atomic_write_json(meta_path, meta)
                    return
                source = candidate
                break
    if source is None:
        source = os.path.join(save_dir, "checkpoints", f"checkpoint-{int(best['step'])}")
    required = [FULL_IO_WEIGHTS_NAME, "adapter_config.json"]
    if projector_required:
        required.append(PROJECTOR_WEIGHTS_NAME)
    if not all(os.path.isfile(os.path.join(source, name)) for name in required) or not any(
        os.path.isfile(os.path.join(source, name))
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ) or os.path.exists(os.path.join(source, NYT_PENDING_NAME)):
        raise RuntimeError(
            f"❌ 恢复历史最佳 step={best['step']} 的模型文件已不存在。"
            "这通常发生于选择较早断点且此前 best 已被覆盖。"
            "请优先从最新完整断点恢复，或从备份还原历史最佳模型。"
        )
    # 先完整拷贝到临时目录，再备份并替换旧在线 best，不删除旧模型。
    stamp = str(time.time_ns())
    temporary = online + ".restore-" + stamp
    shutil.copytree(source, temporary)
    if os.path.isdir(online):
        os.replace(online, online + ".before-resume-" + stamp)
    os.replace(temporary, online)
    atomic_write_json(meta_path, {
        "step": int(best["step"]), "global_macro": float(best["global_macro"]),
        "global_micro": float(best["global_micro"]),
        "global_samples": float(best.get("global_samples", 0)),
        "eval_loss": float(best.get("eval_loss", 0)),
        "selection_rule": "highest Macro; exact Macro ties use higher Micro",
    })



def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_projector_weights(model, checkpoint: str) -> None:
    projector = getattr(unwrap(model), "hidden_projector", None)
    if projector is not None:
        torch.save({"state_dict": {key: value.detach().cpu()
                                   for key, value in projector.state_dict().items()}},
                   os.path.join(checkpoint, PROJECTOR_WEIGHTS_NAME))


def load_projector_weights_for_resume(model, checkpoint: str) -> None:
    projector = getattr(unwrap(model), "hidden_projector", None)
    path = os.path.join(checkpoint, PROJECTOR_WEIGHTS_NAME)
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
    print(f"✅ 已恢复全层级共享 Projector: {checkpoint}")


KD_METER_FIELDS = (
    "_meter", "_meter_align_depth_mse", "_meter_align_depth_cos",
    "_meter_align_depth_total", "_meter_hier_relation_mse", "_meter_n",
)


class NYTDistillTrainer(Trainer):
    def __init__(self, *args, teacher_cache: TeacherCache, task_weight: float,
                 kd_weight: float, align_weight: float, hier_weight: float,
                 temperature: float, align_alpha: float,
                 kd_position_chunk: int, io_learning_rate: float,
                 projector_learning_rate: float,
                 student_depth_token_ids: Dict[int, int], **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = teacher_cache
        self.task_weight = float(task_weight)
        self.kd_weight = float(kd_weight)
        self.align_weight = float(align_weight)
        self.hier_weight = float(hier_weight)
        self.temperature = float(temperature)
        self.align_alpha = float(align_alpha)
        self.kd_position_chunk = int(kd_position_chunk)
        self.io_learning_rate = float(io_learning_rate)
        self.projector_learning_rate = float(projector_learning_rate)
        self.student_depth_token_ids = {
            int(d): int(token_id)
            for d, token_id in student_depth_token_ids.items()
        }

        if set(self.student_depth_token_ids) != set(range(1, 9)):
            raise ValueError(
                "student_depth_token_ids 必须恰好包含 D1-D8。"
            )

        # 与 WOS 离线 hidden 蒸馏实现保持同一损失定义与参数合法性检查。
        if self.temperature <= 0:
            raise ValueError(f"kd_temperature 必须 > 0，实际={self.temperature}")
        if min(self.task_weight, self.kd_weight, self.align_weight, self.hier_weight) < 0:
            raise ValueError("task/kd/align/hier weight 不能为负数")
        if not (0.0 <= self.align_alpha <= 1.0):
            raise ValueError(f"align_alpha 必须位于 [0,1]，实际={self.align_alpha}")
        if self.kd_position_chunk <= 0:
            raise ValueError(f"kd_position_chunk 必须 > 0，实际={self.kd_position_chunk}")

        self._teacher_head_gpu = None
        self._teacher_bias_gpu = None
        # 与 WOS offline KD 一致：bf16 训练时 Teacher cache 的重建矩阵
        # 固定使用 bf16；否则使用 float32。KL 前再统一转 float32。
        # 这样避免 student_outputs.logits 的 dtype 在不同 transformers 版本下
        # 意外把 128K x hidden 的 Teacher head 提升为 float32，造成额外显存占用。
        self.teacher_compute_dtype = (
            torch.bfloat16 if self.args.bf16 else torch.float32
        )
        self.student_common_ids = torch.tensor(
            np.asarray(self.cache.student_common_ids_np),
            dtype=torch.long,
            device=self.args.device,
        )
        student_vocab_rows = int(
            unwrap(self.model).get_output_embeddings().weight.shape[0]
        )
        if self.student_common_ids.numel() != int(self.cache.common_vocab_size):
            raise RuntimeError(
                "Student common-vocab 数与 Teacher cache 不一致: "
                f"student={self.student_common_ids.numel()}, "
                f"cache={self.cache.common_vocab_size}"
            )
        if int(self.student_common_ids.max().item()) >= student_vocab_rows:
            raise RuntimeError("student_common_ids 超出 Student lm_head 行数。")
        self._printed_forward_sanity = False

        # 与 WOS SwanLab 日志风格保持一致：除了总损失，
        # NYT 额外逐深度记录 D1-D8 Align，并逐相邻关系记录 7 条 Hier MSE。
        self._meter = {k: 0.0 for k in ["task", "kd", "align", "hier", "total"]}
        self._meter_align_depth_mse = [0.0 for _ in range(8)]
        self._meter_align_depth_cos = [0.0 for _ in range(8)]
        self._meter_align_depth_total = [0.0 for _ in range(8)]
        self._meter_hier_relation_mse = [0.0 for _ in range(7)]
        self._meter_n = 0

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        target = self.model if model is None else model
        load_full_input_output_weights_for_resume(target, resume_from_checkpoint)
        load_projector_weights_for_resume(target, resume_from_checkpoint)
        path = os.path.join(
            resume_from_checkpoint, f"{KD_TRAINER_STATE_PREFIX}{self.args.process_index}.json"
        )
        if os.path.isfile(path):
            payload = read_json_file(path)
            expected_step = int(read_json_file(os.path.join(resume_from_checkpoint, "trainer_state.json"))["global_step"])
            if int(payload["global_step"]) != expected_step:
                raise RuntimeError("蒸馏损失日志状态与断点步数不一致。")
            for field in KD_METER_FIELDS:
                if field not in payload["meters"]:
                    raise RuntimeError(f"蒸馏损失日志状态缺少字段: {field}")
                setattr(self, field, payload["meters"][field])

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        if output_dir is None:
            output_dir = self.args.output_dir
        if self.is_world_process_zero():
            save_full_io(self.model, output_dir)
            save_projector_weights(self.model, output_dir)

    def _save_checkpoint(self, model, trial, *args, **kwargs):
        output_dir = os.path.join(self._get_output_dir(trial), f"checkpoint-{self.state.global_step}")
        if self.is_world_process_zero():
            os.makedirs(output_dir, exist_ok=True)
            atomic_write_json(os.path.join(output_dir, NYT_PENDING_NAME),
                              {"global_step": int(self.state.global_step)})
        distributed_barrier()
        result = super()._save_checkpoint(model, trial, *args, **kwargs)
        # 每个 rank 各自保存日志累积器；GPU Teacher tensors 由原缓存重新构建。
        atomic_write_json(
            os.path.join(output_dir, f"{KD_TRAINER_STATE_PREFIX}{self.args.process_index}.json"),
            {"global_step": int(self.state.global_step),
             "meters": {field: getattr(self, field) for field in KD_METER_FIELDS}},
        )
        return result

    def _teacher_tensors(self, device):
        # common-vocab Teacher head 只在第一次真正需要 KD 的 batch 搬到 GPU，后续复用。
        # 计算 dtype 严格固定为 WOS 同款 teacher_compute_dtype。
        if self._teacher_head_gpu is None:
            self._teacher_head_gpu = torch.from_numpy(
                np.array(self.cache.teacher_head, copy=True)
            ).to(
                device=device, dtype=self.teacher_compute_dtype
            ).contiguous()
            self._teacher_head_gpu.requires_grad_(False)
            if self.cache.teacher_bias is not None:
                self._teacher_bias_gpu = torch.from_numpy(
                    np.array(self.cache.teacher_bias, copy=True)
                ).to(
                    device=device, dtype=self.teacher_compute_dtype
                ).contiguous()
                self._teacher_bias_gpu.requires_grad_(False)
        return self._teacher_head_gpu, self._teacher_bias_gpu

    def _get_student_final_norm_module(self, model):
        """严格复用 WOS 的 final-norm hook 定位逻辑。"""
        raw = unwrap(model)
        base_causal_lm = raw.get_base_model() if hasattr(raw, "get_base_model") else raw
        decoder = getattr(base_causal_lm, "model", None)
        if decoder is None:
            raise RuntimeError("Student 找不到 Llama decoder (.model)。")
        final_norm = getattr(decoder, "norm", None)
        if final_norm is None:
            raise RuntimeError("Student decoder 找不到 final norm。")
        return final_norm

    def _resolve_depth_positions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        valid_prediction_mask: torch.Tensor,
        depth_rel_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """
        将每条样本 cache 中的 D1-D8 relative offset 恢复为完整序列中的
        第二次 depth-token absolute position，并严格核验 token ID 与顺序。

        valid_prediction_mask[p] 对应 hidden/logits 位置 p，用来预测 token p+1；
        cache 提取阶段 depth_rel_offsets 也是在这同一 mask 内定义的。
        """
        if depth_rel_offsets.ndim != 2 or depth_rel_offsets.shape[1] != 8:
            raise RuntimeError(
                f"depth_rel_offsets batch shape 异常: {tuple(depth_rel_offsets.shape)}"
            )

        depth_positions = []
        for row in range(input_ids.size(0)):
            actual_len = int(attention_mask[row].sum().item())
            if actual_len < 2:
                raise RuntimeError(f"batch row={row} 实际序列长度不足 2。")

            valid_positions = torch.nonzero(
                valid_prediction_mask[row, : actual_len - 1],
                as_tuple=False,
            ).flatten()
            rel = depth_rel_offsets[row].to(
                device=valid_positions.device, dtype=torch.long
            )
            if (rel < 0).any() or (rel >= valid_positions.numel()).any():
                raise RuntimeError(
                    f"batch row={row} D1-D8 relative offset 越界: "
                    f"rel={rel.detach().cpu().tolist()}, valid_count={valid_positions.numel()}"
                )

            absolute = valid_positions.index_select(0, rel)
            if not torch.all(absolute[1:] > absolute[:-1]):
                raise RuntimeError(
                    f"batch row={row} D1-D8 absolute position 顺序异常: "
                    f"{absolute.detach().cpu().tolist()}"
                )

            # 进一步核验这些 absolute position 真的是 Completion 中的 D1-D8 token。
            for d in range(1, 9):
                pos = int(absolute[d - 1].item())
                actual_token_id = int(input_ids[row, pos].item())
                expected_token_id = self.student_depth_token_ids[d]
                if actual_token_id != expected_token_id:
                    raise RuntimeError(
                        f"batch row={row} D{d} absolute position={pos} token ID 错误: "
                        f"actual={actual_token_id}, expected={expected_token_id}"
                    )

            depth_positions.append(absolute)

        return torch.stack(depth_positions, dim=0).to(
            device=input_ids.device, dtype=torch.long
        )

    def _verify_batch_cache_layout(
        self,
        sample_index: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        valid_prediction_mask: torch.Tensor,
        depth_rel_offsets: torch.Tensor,
    ) -> None:
        """每个 batch 再做一次 WOS 风格的 sample/hash/count/offset 严格审计。"""
        indices = sample_index.detach().cpu().tolist()
        for row, idx in enumerate(indices):
            idx = int(idx)
            if idx < 0 or idx >= len(self.cache.valid_counts):
                raise IndexError(f"sample_index 越界: {idx}")

            actual_len = int(attention_mask[row].sum().item())
            current_ids = input_ids[row, :actual_len].detach().cpu().tolist()
            current_hash = sha256_ids(current_ids)
            cached_hash = self.cache.hashes[idx]
            if isinstance(cached_hash, bytes):
                cached_hash = cached_hash.decode("ascii")
            else:
                cached_hash = bytes(cached_hash).decode("ascii")
            if current_hash != cached_hash:
                raise RuntimeError(f"样本{idx} 当前 batch token hash 与 Teacher cache 不一致")

            current_valid_count = int(
                valid_prediction_mask[row, : max(0, actual_len - 1)].sum().item()
            )
            cached_valid_count = int(self.cache.valid_counts[idx])
            if current_valid_count != cached_valid_count:
                raise RuntimeError(
                    f"样本{idx} Completion KD position 数不一致: "
                    f"batch={current_valid_count}, cache={cached_valid_count}"
                )

            current_rel = depth_rel_offsets[row].detach().cpu().tolist()
            cached_rel = self.cache.depth_rel[idx].astype(int).tolist()
            if current_rel != cached_rel:
                raise RuntimeError(
                    f"样本{idx} D1-D8 relative offset 与 cache 不一致: "
                    f"batch={current_rel}, cache={cached_rel}"
                )

    def create_optimizer(self):
        """严格按 WOS/NYT SFT 的参数分组逻辑：LoRA / 完整 IO / Projector。"""
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        raw = unwrap(opt_model)
        input_module = raw.get_input_embeddings()
        output_module = raw.get_output_embeddings()
        if input_module is None or output_module is None:
            raise RuntimeError("创建优化器时找不到 input embedding 或 lm_head。")

        io_param_ids = {
            id(input_module.weight),
            id(output_module.weight),
        }

        projector = getattr(raw, "hidden_projector", None)
        projector_param_ids = set()
        if projector is not None:
            projector_param_ids = {
                id(p) for p in projector.parameters() if p.requires_grad
            }
            if not projector_param_ids:
                raise RuntimeError("hidden_projector 存在但没有可训练参数。")

        needs_projector = self.align_weight > 0 or self.hier_weight > 0
        if needs_projector and projector is None:
            raise RuntimeError("Align/Hier 已开启，但 Student 没有 hidden_projector。")
        if (not needs_projector) and projector is not None:
            raise RuntimeError("Traditional-only 模式不应创建 hidden_projector。")
        if projector is not None:
            if int(projector.out_features) != int(self.cache.teacher_hidden_size):
                raise RuntimeError(
                    "Projector 输出维度与 Teacher hidden size 不一致: "
                    f"projector={projector.out_features}, "
                    f"teacher={self.cache.teacher_hidden_size}"
                )

        # 与 Trainer 默认规则保持一致：Norm / bias 不 weight decay。
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
        seen = set()

        for name, p in opt_model.named_parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)

            is_io = pid in io_param_ids
            is_projector = pid in projector_param_ids
            if is_io and is_projector:
                raise RuntimeError("参数同时被识别为 IO 与 Projector。")

            use_decay = name in decay_parameter_names
            if is_io:
                key = "io_decay" if use_decay else "io_no_decay"
            elif is_projector:
                key = "projector_decay" if use_decay else "projector_no_decay"
            else:
                key = "lora_decay" if use_decay else "lora_no_decay"
            grouped[key].append(p)

        lora_count = sum(
            p.numel()
            for key in ("lora_decay", "lora_no_decay")
            for p in grouped[key]
        )
        io_count = sum(
            p.numel()
            for key in ("io_decay", "io_no_decay")
            for p in grouped[key]
        )
        projector_count = sum(
            p.numel()
            for key in ("projector_decay", "projector_no_decay")
            for p in grouped[key]
        )
        total_grouped = lora_count + io_count + projector_count
        total_trainable = sum(
            p.numel() for p in opt_model.parameters() if p.requires_grad
        )

        if total_grouped != total_trainable:
            raise RuntimeError(
                "优化器参数审计失败: "
                f"grouped={total_grouped:,}, trainable={total_trainable:,}"
            )
        if io_count <= 0:
            raise RuntimeError("完整 input embedding/lm_head 没有进入优化器。")
        if needs_projector and projector_count <= 0:
            raise RuntimeError("Align/Hier 模式下 Projector 没有进入优化器。")

        groups = []
        spec = [
            ("lora_decay", self.args.learning_rate, self.args.weight_decay),
            ("lora_no_decay", self.args.learning_rate, 0.0),
            ("io_decay", self.io_learning_rate, self.args.weight_decay),
            ("io_no_decay", self.io_learning_rate, 0.0),
            ("projector_decay", self.projector_learning_rate, self.args.weight_decay),
            ("projector_no_decay", self.projector_learning_rate, 0.0),
        ]
        for key, lr, wd in spec:
            if grouped[key]:
                groups.append({
                    "params": grouped[key],
                    "lr": float(lr),
                    "weight_decay": float(wd),
                })

        self.optimizer = torch.optim.AdamW(
            groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )

        print("\n" + "=" * 100)
        print("✅ Optimizer 参数审计通过（WOS/NYT-SFT 同款分组原则）")
        print(f"LoRA/other trainable: {lora_count:,} | lr={self.args.learning_rate}")
        print(f"Full IO             : {io_count:,} | lr={self.io_learning_rate}")
        print(f"Projector           : {projector_count:,} | lr={self.projector_learning_rate}")
        print(f"Total trainable     : {total_trainable:,}")
        print("=" * 100 + "\n")
        return self.optimizer

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        **kwargs,
    ):
        """
        严格按 WOS 离线蒸馏的 Student forward 方式实现：

        1) Student 走完整 CausalLM/PeftModel forward：
             student_outputs = model(..., labels=labels)
           Task CE 直接使用 student_outputs.loss，和 Student SFT baseline 一致。

        2) Traditional KD 直接使用同一次 forward 的 student_outputs.logits。

        3) 只有启用 Align/Hier 时，才在同一次 forward 上给 final norm 挂 hook，
           提取 D1-D8 第二次 occurrence 的 Student final hidden。

        Teacher 仍完全离线，不加载 8B Transformer。
        """
        del num_items_in_batch, kwargs

        sample_index = inputs.pop("sample_index", None)
        depth_rel_offsets = inputs.pop("depth_rel_offsets", None)
        if "labels" not in inputs:
            raise RuntimeError("KD batch 缺少 labels。")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        # Causal shift / Completion-only mask 在 Student forward 前即可确定。
        shift_labels = labels[:, 1:]
        valid_prediction_mask = shift_labels.ne(-100)
        valid_count = int(valid_prediction_mask.sum().item())
        if valid_count <= 0:
            raise RuntimeError("当前 batch 没有 Completion supervision token。")

        is_distill_train_batch = sample_index is not None and model.training
        if is_distill_train_batch:
            if depth_rel_offsets is None:
                raise RuntimeError("Offline KD train batch 缺少 depth_rel_offsets。")
            depth_positions = self._resolve_depth_positions(
                input_ids=input_ids,
                attention_mask=attention_mask,
                valid_prediction_mask=valid_prediction_mask,
                depth_rel_offsets=depth_rel_offsets,
            )
            self._verify_batch_cache_layout(
                sample_index=sample_index,
                input_ids=input_ids,
                attention_mask=attention_mask,
                valid_prediction_mask=valid_prediction_mask,
                depth_rel_offsets=depth_rel_offsets,
            )
        else:
            depth_positions = None

        # --------------------------------------------------------------
        # 1. Student 正常完整 CausalLM forward：与 WOS / SFT 一致。
        # --------------------------------------------------------------
        need_student_hidden = (
            is_distill_train_batch
            and (self.align_weight > 0 or self.hier_weight > 0)
        )
        captured = {}
        handle = None
        if need_student_hidden:
            final_norm = self._get_student_final_norm_module(model)

            def _capture_final_hidden(_module, _inputs, output):
                captured["last_hidden"] = output

            handle = final_norm.register_forward_hook(_capture_final_hidden)

        try:
            student_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
        finally:
            if handle is not None:
                handle.remove()

        task_loss = student_outputs.loss
        if task_loss is None:
            raise RuntimeError("Student 完整 CausalLM forward 没有返回 task loss。")
        if student_outputs.logits is None:
            raise RuntimeError("Student 完整 CausalLM forward 没有返回 logits。")

        student_shift_logits = student_outputs.logits[:, :-1, :]
        student_selected_full_logits = student_shift_logits[
            valid_prediction_mask
        ]
        selected_targets = shift_labels[valid_prediction_mask]
        if student_selected_full_logits.size(0) != valid_count:
            raise RuntimeError("Student Completion logits 展平数量异常。")

        # 首个训练 batch 做严格数值自检：官方 CausalLM loss 必须和直接 CE 一致。
        if model.training and not self._printed_forward_sanity:
            manual_task_loss = F.cross_entropy(
                student_selected_full_logits.float(),
                selected_targets,
                reduction="mean",
            )
            diff = abs(
                float(task_loss.detach().float().item())
                - float(manual_task_loss.detach().float().item())
            )
            tolerance = max(
                1e-4,
                5e-4 * abs(float(task_loss.detach().float().item())),
            )
            if diff > tolerance:
                raise RuntimeError(
                    "Student official task loss 与 Completion-only direct CE 不一致: "
                    f"official={task_loss.detach().float().item():.8f}, "
                    f"manual={manual_task_loss.detach().float().item():.8f}, "
                    f"diff={diff:.8g}, tol={tolerance:.8g}"
                )
            print("\n" + "=" * 100)
            print("✅ Student forward sanity check 通过")
            print("✅ Task loss 来自完整 model(..., labels=labels) 官方 CausalLM forward")
            print(
                f"official Task CE={task_loss.detach().float().item():.8f} | "
                f"direct Completion CE={manual_task_loss.detach().float().item():.8f} | "
                f"diff={diff:.3e}"
            )
            print("=" * 100 + "\n")
            self._printed_forward_sanity = True

        # Validation / 普通 evaluate 没有 sample_index，只返回和 SFT 一致的 Task CE。
        if not is_distill_train_batch:
            return (task_loss, student_outputs) if return_outputs else task_loss

        indices = sample_index.detach().cpu().tolist()
        expected_counts = valid_prediction_mask.sum(dim=1).detach().cpu().tolist()

        # --------------------------------------------------------------
        # 2. 按 sample_index 读取离线 Teacher Completion hidden。
        # --------------------------------------------------------------
        need_teacher_hidden = (
            self.kd_weight > 0
            or self.align_weight > 0
            or self.hier_weight > 0
        )
        teacher_hidden = None
        t_depth = None
        if need_teacher_hidden:
            teacher_parts = []
            teacher_depth_batch = []
            for row, idx in enumerate(indices):
                arr, rel_np = self.cache.get_sample(int(idx))
                if arr.shape[0] != int(expected_counts[row]):
                    raise RuntimeError(
                        f"样本{idx} cache valid_count={arr.shape[0]} "
                        f"!= batch={expected_counts[row]}"
                    )

                batch_rel = depth_rel_offsets[row].detach().cpu().tolist()
                cache_rel = rel_np.astype(int).tolist()
                if cache_rel != batch_rel:
                    raise RuntimeError(
                        f"样本{idx} depth_rel_offsets 与 batch 不一致"
                    )

                t_piece = torch.from_numpy(
                    np.array(arr, copy=True)
                ).to(
                    device=student_outputs.logits.device,
                    dtype=self.teacher_compute_dtype,
                    non_blocking=True,
                )
                teacher_parts.append(t_piece)

                if self.align_weight > 0 or self.hier_weight > 0:
                    rel_t = depth_rel_offsets[row].to(
                        device=t_piece.device, dtype=torch.long
                    )
                    teacher_depth_batch.append(
                        t_piece.index_select(0, rel_t)
                    )

            teacher_hidden = torch.cat(teacher_parts, dim=0)
            if teacher_hidden.shape[0] != valid_count:
                raise RuntimeError(
                    f"当前 batch Teacher hidden 数={teacher_hidden.shape[0]} "
                    f"!= Student Completion positions={valid_count}"
                )
            if teacher_depth_batch:
                t_depth = torch.stack(teacher_depth_batch, dim=0)

        # --------------------------------------------------------------
        # 3. Traditional Full-Completion common-vocab logits KD。
        #    Student logits 来自上面的同一次完整 CausalLM forward。
        # --------------------------------------------------------------
        kd_loss = task_loss.new_zeros(())
        if self.kd_weight > 0:
            common_ids = self.student_common_ids
            if common_ids.device != student_selected_full_logits.device:
                common_ids = common_ids.to(student_selected_full_logits.device)
            teacher_head, teacher_bias = self._teacher_tensors(
                student_selected_full_logits.device
            )

            kd_sum = task_loss.new_zeros(())
            kd_count = 0
            T = self.temperature
            for st in range(0, valid_count, self.kd_position_chunk):
                ed = min(st + self.kd_position_chunk, valid_count)
                student_common = student_selected_full_logits[
                    st:ed
                ].index_select(-1, common_ids).float()
                teacher_common = F.linear(
                    teacher_hidden[st:ed],
                    teacher_head,
                    teacher_bias,
                ).float()
                chunk_kd = F.kl_div(
                    F.log_softmax(student_common / T, dim=-1),
                    F.softmax(teacher_common / T, dim=-1),
                    reduction="batchmean",
                ) * (T * T)
                kd_sum = kd_sum + chunk_kd * (ed - st)
                kd_count += ed - st
            kd_loss = kd_sum / max(1, kd_count)

        # --------------------------------------------------------------
        # 4. D1-D8 hidden Align / D1->D2...D7->D8 relation MSE。
        #    Student hidden 来自同一次完整 forward 的 final-norm hook。
        # --------------------------------------------------------------
        align_loss = task_loss.new_zeros(())
        hier_loss = task_loss.new_zeros(())
        align_depth_mse = [task_loss.new_zeros(()) for _ in range(8)]
        align_depth_cos = [task_loss.new_zeros(()) for _ in range(8)]
        align_depth_total = [task_loss.new_zeros(()) for _ in range(8)]
        hier_relation_mse = [task_loss.new_zeros(()) for _ in range(7)]

        if self.align_weight > 0 or self.hier_weight > 0:
            if "last_hidden" not in captured:
                raise RuntimeError("Student final-norm hook 没有捕获到 hidden。")
            student_last_hidden = captured["last_hidden"]
            if not torch.is_tensor(student_last_hidden):
                raise TypeError("Student final-norm hook 输出不是 Tensor。")
            if student_last_hidden.shape[:2] != input_ids.shape:
                raise RuntimeError(
                    "Student final hidden shape 与 input_ids 不一致: "
                    f"hidden={tuple(student_last_hidden.shape)}, "
                    f"input={tuple(input_ids.shape)}"
                )
            if t_depth is None:
                raise RuntimeError("启用了 hidden loss 但没有取得 Teacher D1-D8 hidden。")

            raw = unwrap(model)
            projector = getattr(raw, "hidden_projector", None)
            if projector is None:
                raise RuntimeError("启用了 hidden loss 但 model.hidden_projector 不存在")

            row_ids = torch.arange(
                input_ids.size(0),
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(1).expand(-1, 8)
            s_depth = student_last_hidden[
                row_ids,
                depth_positions,
                :,
            ]
            if s_depth.shape[1] != 8:
                raise RuntimeError(f"Student D1-D8 hidden shape 异常: {tuple(s_depth.shape)}")

            p_depth = projector(s_depth.float())
            t_depth_f = t_depth.float()
            if p_depth.shape != t_depth_f.shape:
                raise RuntimeError(
                    f"Projected Student / Teacher depth hidden shape 不一致: "
                    f"student={tuple(p_depth.shape)}, teacher={tuple(t_depth_f.shape)}"
                )

            if self.align_weight > 0:
                for depth_idx in range(8):
                    d_mse = F.mse_loss(
                        p_depth[:, depth_idx, :],
                        t_depth_f[:, depth_idx, :],
                        reduction="mean",
                    )
                    d_cos = 1.0 - F.cosine_similarity(
                        p_depth[:, depth_idx, :],
                        t_depth_f[:, depth_idx, :],
                        dim=-1,
                        eps=1e-8,
                    ).mean()
                    d_total = (
                        self.align_alpha * d_mse
                        + (1.0 - self.align_alpha) * d_cos
                    )
                    align_depth_mse[depth_idx] = d_mse
                    align_depth_cos[depth_idx] = d_cos
                    align_depth_total[depth_idx] = d_total
                align_loss = torch.stack(align_depth_total).mean()

            if self.hier_weight > 0:
                for rel_idx in range(7):
                    s_rel = (
                        p_depth[:, rel_idx + 1, :]
                        - p_depth[:, rel_idx, :]
                    )
                    t_rel = (
                        t_depth_f[:, rel_idx + 1, :]
                        - t_depth_f[:, rel_idx, :]
                    )
                    rel_mse = F.mse_loss(
                        s_rel,
                        t_rel,
                        reduction="mean",
                    )
                    hier_relation_mse[rel_idx] = rel_mse
                hier_loss = torch.stack(hier_relation_mse).mean()

        total = (
            self.task_weight * task_loss
            + self.kd_weight * kd_loss
            + self.align_weight * align_loss
            + self.hier_weight * hier_loss
        )
        if not torch.isfinite(total):
            raise FloatingPointError(
                "total loss 出现 NaN/Inf: "
                f"task={task_loss.detach().float().item():.6f}, "
                f"kd={kd_loss.detach().float().item():.6f}, "
                f"align={align_loss.detach().float().item():.6f}, "
                f"hier={hier_loss.detach().float().item():.6f}"
            )

        for k, v in [
            ("task", task_loss),
            ("kd", kd_loss),
            ("align", align_loss),
            ("hier", hier_loss),
            ("total", total),
        ]:
            self._meter[k] += float(v.detach().cpu())

        for depth_idx in range(8):
            self._meter_align_depth_mse[depth_idx] += float(
                align_depth_mse[depth_idx].detach().cpu()
            )
            self._meter_align_depth_cos[depth_idx] += float(
                align_depth_cos[depth_idx].detach().cpu()
            )
            self._meter_align_depth_total[depth_idx] += float(
                align_depth_total[depth_idx].detach().cpu()
            )

        for rel_idx in range(7):
            self._meter_hier_relation_mse[rel_idx] += float(
                hier_relation_mse[rel_idx].detach().cpu()
            )
        self._meter_n += 1

        return (total, student_outputs) if return_outputs else total

    def log(self, logs, *args, **kwargs):
        logs = dict(logs)
        if self._meter_n > 0 and "loss" in logs:
            n = float(self._meter_n)

            # 与 WOS 命名保持一致。
            logs["Train/Task_Loss"] = self._meter["task"] / n
            logs["Train/KD_Loss"] = self._meter["kd"] / n

            # NYT D1-D8：每层分别保存 MSE / CosLoss / Total。
            for depth_idx in range(8):
                depth = depth_idx + 1
                logs[f"Train/Align_D{depth}_MSE"] = (
                    self._meter_align_depth_mse[depth_idx] / n
                )
                logs[f"Train/Align_D{depth}_CosLoss"] = (
                    self._meter_align_depth_cos[depth_idx] / n
                )
                logs[f"Train/Align_D{depth}_Total"] = (
                    self._meter_align_depth_total[depth_idx] / n
                )

            logs["Train/Align_Loss"] = self._meter["align"] / n

            # NYT 七条相邻层级关系：每条 relation MSE 单独保存。
            for rel_idx in range(7):
                d1 = rel_idx + 1
                d2 = rel_idx + 2
                logs[f"Train/Hier_D{d1}_to_D{d2}_MSE"] = (
                    self._meter_hier_relation_mse[rel_idx] / n
                )

            # 与 WOS 的 Train/Hier_Hidden_MSE 对应。
            logs["Train/Hier_Hidden_MSE"] = self._meter["hier"] / n
            logs["Train/Total_Loss"] = self._meter["total"] / n

            logs["Train/Task_Weight"] = self.task_weight
            logs["Train/KD_Weight"] = self.kd_weight
            logs["Train/Align_Weight"] = self.align_weight
            logs["Train/Hier_Weight"] = self.hier_weight
            logs["Train/Align_Alpha"] = self.align_alpha
            logs["Train/KD_Temperature"] = self.temperature
            logs["Train/Common_Vocab_Size"] = float(
                self.student_common_ids.numel()
            )

            self._meter = {k: 0.0 for k in self._meter}
            self._meter_align_depth_mse = [0.0 for _ in range(8)]
            self._meter_align_depth_cos = [0.0 for _ in range(8)]
            self._meter_align_depth_total = [0.0 for _ in range(8)]
            self._meter_hier_relation_mse = [0.0 for _ in range(7)]
            self._meter_n = 0

        return super().log(logs, *args, **kwargs)


def main(args):
    set_all_seeds(args.seed)
    mode = args.experiment_mode.lower().strip()
    if mode not in {"traditional", "align", "hier", "full"}:
        raise ValueError("experiment_mode 必须是 traditional/align/hier/full")
    align_weight = args.align_weight if mode in {"align", "full"} else 0.0
    hier_weight = args.hier_weight if mode in {"hier", "full"} else 0.0

    if not args.save_dir:
        raise ValueError("--save_dir 不能为空")
    args.save_dir = os.path.abspath(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    resume_checkpoint = resolve_resume_checkpoint(args)
    run_signature = build_run_signature(args)
    if resume_checkpoint:
        validate_resume_configuration(resume_checkpoint, run_signature)
        print(f"🔄 将从完整蒸馏断点恢复: {resume_checkpoint}")
    else:
        print("🆕 开始新的 NYT 蒸馏实验。")

    label_space = load_label_space(args.label_space_path, args.expected_num_labels)
    train_data = load_json_or_jsonl(args.train_data_path)
    val_data = load_json_or_jsonl(args.val_data_path)
    validate_clean_data(train_data, label_space, "TRAIN")
    validate_clean_data(val_data, label_space, "VALIDATION")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    install_chat_template(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    token_ids = validate_reserved_tokens(tokenizer)

    train_dataset = NYTDataset(
        train_data, tokenizer, label_space, token_ids,
        args.max_seq_length, args.reserve_tokens,
        args.system_prompt, True,
    )
    val_dataset = NYTDataset(
        val_data, tokenizer, label_space, token_ids,
        args.max_seq_length, args.reserve_tokens,
        args.system_prompt, False,
    )

    cache = TeacherCache(
        args.teacher_cache_dir, len(train_dataset), args.max_seq_length,
        args.reserve_tokens, args.system_prompt,
    )
    cache.verify_dataset(train_dataset)
    cached_common = np.asarray(cache.student_common_ids_np)
    if cached_common.min() < 0 or cached_common.max() >= len(tokenizer):
        raise RuntimeError("缓存 student_common_ids 超出当前 tokenizer 范围")

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora)
    enable_full_io(model)

    if align_weight > 0 or hier_weight > 0:
        student_hidden = int(model.config.hidden_size)
        # 与 WOS 原 hidden 蒸馏一致：共享 Linear Projector，bias=True，float32。
        projector = nn.Linear(
            student_hidden,
            cache.teacher_hidden_size,
            bias=True,
            dtype=torch.float32,
        )
        projector.requires_grad_(True)
        model.add_module("hidden_projector", projector)
        print(
            f"✅ D1-D8 共享 float32 Projector: "
            f"{student_hidden} -> {cache.teacher_hidden_size} | bias=True"
        )

    train_args = TrainingArguments(
        output_dir=os.path.join(args.save_dir, "checkpoints"),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=False,
        ignore_data_skip=False,
        load_best_model_at_end=False,
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        optim="adamw_torch",
    )

    trainer = NYTDistillTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DistillCollator(tokenizer.pad_token_id),
        teacher_cache=cache,
        task_weight=args.task_weight,
        kd_weight=args.kd_weight,
        align_weight=align_weight,
        hier_weight=hier_weight,
        temperature=args.kd_temperature,
        align_alpha=args.align_alpha,
        kd_position_chunk=args.kd_position_chunk,
        io_learning_rate=args.io_lr,
        projector_learning_rate=args.projector_lr,
        student_depth_token_ids=token_ids["depth"],
    )

    swanlab_callback = SwanLabCallback(
        project=args.project_name,
        experiment_name=args.run_name,
    )
    trainer.add_callback(swanlab_callback)

    macro_callback = GenerativeNYTMacroCallback(
        val_data=val_data,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        label_space=label_space,
        system_prompt=args.system_prompt,
        max_seq_length=args.max_seq_length,
        reserve_tokens=args.reserve_tokens,
        eval_steps=args.eval_steps,
        eval_batch_size=args.generation_eval_batch_size,
        loss_eval_batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
        save_dir=args.save_dir,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        eval_loss_logit_chunk_size=args.eval_loss_logit_chunk_size,
        resume_checkpoint=resume_checkpoint,
        run_signature=run_signature,
    )
    trainer.add_callback(macro_callback)

    config = vars(args).copy()
    config["effective_align_weight"] = align_weight
    config["effective_hier_weight"] = hier_weight
    config["hierarchy_policy"] = run_signature["hierarchy_policy"]
    config["dataset"] = "NYT"
    config["max_depth"] = MAX_DEPTH
    if trainer.is_world_process_zero():
        config["resolved_resume_checkpoint"] = resume_checkpoint
        atomic_write_json(os.path.join(args.save_dir, "distill_config.json"), config)

    print("\n" + "=" * 100)
    print(f"NYT 离线蒸馏 | mode={mode}")
    print(f"Task={args.task_weight}, KD={args.kd_weight}, Align={align_weight}, Hier={hier_weight}")
    print(f"T={args.kd_temperature}, alpha={args.align_alpha}")
    print(f"save_dir={args.save_dir}")
    print("=" * 100 + "\n")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    if trainer.is_world_process_zero():
        final_ckpt = os.path.join(args.save_dir, "final_kd_checkpoint")
        save_checkpoint(trainer.model, tokenizer, final_ckpt)

        online_best = os.path.join(args.save_dir, "best_macro_checkpoint")
        if not os.path.isdir(online_best):
            raise RuntimeError("训练结束后没有生成 best_macro_checkpoint；请检查生成式验证是否运行。")

        best_ckpt = os.path.join(args.save_dir, "best_kd_checkpoint")
        if os.path.isdir(best_ckpt):
            shutil.rmtree(best_ckpt)
        shutil.copytree(online_best, best_ckpt)
        tokenizer.save_pretrained(best_ckpt)

        if args.export_merged_model:
            final_model = os.path.join(args.save_dir, "final_model")
            best_model = os.path.join(args.save_dir, "best_kd_model")
            export_merged(args.model_path, final_ckpt, final_model, tokenizer, dtype)
            export_merged(args.model_path, best_ckpt, best_model, tokenizer, dtype)
            print(f"🎉 final merged model: {final_model}")
            print(f"🎉 best merged model (按 MA): {best_model}")
        print(f"✅ final checkpoint: {final_ckpt}")
        print(f"✅ best checkpoint (按 MA): {best_ckpt}")


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in {"1", "true", "yes", "y", "t"}:
        return True
    if s in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError("Boolean expected")


if __name__ == "__main__":
    p = argparse.ArgumentParser("NYT offline logits/hidden distillation")
    p.add_argument("--model_path", default="/data1/lzp/code/llama-3.2_1b/")
    p.add_argument("--train_data_path", default="/data1/lzp/code/data/nyt/train_formatted.jsonl")
    p.add_argument("--val_data_path", default="/data1/lzp/code/data/nyt/val_formatted.jsonl")
    p.add_argument("--label_space_path", default="/data1/lzp/code/data/nyt/label_space.json")
    p.add_argument("--teacher_cache_dir", default="/data1/lzp/code/hierarchical_distill/teacher_nyt_hiddenstates/")
    p.add_argument("--save_dir", default="/data1/lzp/code/hierarchical_distill/checkpoints_nyt_logits_kd")
    p.add_argument(
        "--resume_from_checkpoint", default="",
        help="不传则新实验；auto/true/latest 恢复最近完整断点；也可指定原 save_dir/checkpoints/checkpoint-N。",
    )

    p.add_argument("--experiment_mode", choices=["traditional", "align", "hier", "full"], default="traditional")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--expected_num_labels", type=int, default=166)
    p.add_argument("--max_seq_length", type=int, default=3072)
    p.add_argument("--reserve_tokens", type=int, default=1024)
    p.add_argument("--system_prompt", default=(
        "You are a specialized classifier for the NYT hierarchical multi-label taxonomy. "
        "Follow the required eight-line JSON-array output format exactly. "
        "Do not produce explanations or any extra conversational text."
    ))

    p.add_argument("--task_weight", type=float, default=1.0)
    p.add_argument("--kd_weight", type=float, default=1.0)
    p.add_argument("--align_weight", type=float, default=1.0)
    p.add_argument("--hier_weight", type=float, default=1.0)
    p.add_argument("--kd_temperature", type=float, default=2.0)
    p.add_argument("--align_alpha", type=float, default=0.5)
    p.add_argument("--kd_position_chunk", type=int, default=16,
                   help="每次计算多少个 Completion position 的全词表/common词表 logits")

    p.add_argument("--num_train_epochs", type=float, default=8.0)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=1e-4)
    p.add_argument("--io_lr", type=float, default=5e-6)
    p.add_argument("--projector_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lr_scheduler_type", default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.0)

    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.06)

    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--generation_eval_batch_size", type=int, default=20)
    p.add_argument("--max_new_tokens", type=int, default=520)
    p.add_argument("--early_stopping_patience", type=int, default=5)
    p.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    p.add_argument("--eval_loss_logit_chunk_size", type=int, default=128)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_total_limit", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--dataloader_pin_memory", type=str2bool, default=True)
    p.add_argument("--bf16", type=str2bool, default=True)
    p.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    p.add_argument("--attn_implementation", default="sdpa")
    p.add_argument("--export_merged_model", type=str2bool, default=True)

    # SwanLab：与原 NYT 教师/学生训练保持同样的 Callback + 显式验证指标记录方式。
    p.add_argument("--project_name", type=str, default="NYT-Distillation")
    p.add_argument(
        "--run_name",
        type=str,
        default="Llama3.2-1B-NYT-Traditional-LogitsKD-Seed42-Lora16",
    )

    main(p.parse_args())
