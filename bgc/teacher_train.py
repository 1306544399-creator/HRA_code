import os

# 在导入 torch 前设置，尽量保证可复现性。
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# 减少长序列验证时的大块显存碎片；必须在 import torch 前设置。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import hashlib
import json
import random
import re
import shutil
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import peft
import swanlab
import transformers
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from swanlab.integration.transformers import SwanLabCallback
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed as hf_set_seed,
)


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
# 1. Llama 预留特殊 Token + 完整输入/输出词表训练与保存
# ======================================================================
# 正文边界：只用于精确定位正文 token span。
CTX_START_TOKEN = "<|reserved_special_token_110|>"
CTX_END_TOKEN = "<|reserved_special_token_111|>"

# BGC 最大深度为 4：一个预留特殊 token 负责预测一整层标签。
# Prompt 中第一次出现用于绑定层级角色；Completion 中第二次出现并位于
# 对应深度 JSON 标签数组之前。
DEPTH_PROBE_TOKENS: Dict[int, str] = {
    depth: f"<|reserved_special_token_{111 + depth}|>"
    for depth in range(1, 5)
}

FULL_IO_WEIGHTS_NAME = "full_input_output_embeddings.pt"
FULL_IO_META_NAME = "full_input_output_embeddings_meta.json"
BGC_RESUME_STATE_NAME = "bgc_resume_state.json"
BGC_PENDING_NAME = "bgc_save_in_progress.json"


def unwrap_training_model(model):
    """兼容 DDP 等外层包装，返回实际模型对象。"""
    while hasattr(model, "module"):
        model = model.module
    return model


def clear_cuda_memory(stage: str = "") -> None:
    """
    清理已经失去 Python 引用的临时张量与 CUDA 缓存。

    该函数不会释放仍被模型、优化器或 scheduler 持有的显存；它主要用于
    训练、Eval Loss 和生成式验证三个阶段切换时减少缓存与碎片。
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if stage:
        print(f"🧹 CUDA 临时显存清理完成: {stage}")


def is_cuda_oom_error(exc: BaseException) -> bool:
    """兼容不同 PyTorch 版本的 CUDA OOM 判断。"""
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    type_match = oom_type is not None and isinstance(exc, oom_type)
    message = str(exc).lower()
    return bool(type_match or "cuda out of memory" in message)


def get_causal_lm_for_memory_efficient_eval(model):
    """
    从 DDP/PEFT 外层包装中取得实际 CausalLM。

    Eval Loss 会直接调用其中的 decoder 和 lm_head，只对真正受监督的
    Completion 位置分块计算词表 logits，避免构造完整 [B, L, V] logits。
    LoRA 已注入基础模型模块，因此这种调用仍使用当前训练后的 LoRA 权重。
    """
    raw_model = unwrap_training_model(model)

    if isinstance(raw_model, PeftModel):
        causal_lm = raw_model.get_base_model()
    else:
        causal_lm = raw_model

    decoder = getattr(causal_lm, "model", None)
    lm_head = causal_lm.get_output_embeddings()

    if decoder is None or lm_head is None:
        raise RuntimeError(
            "❌ 无法取得 CausalLM decoder 或 lm_head，不能执行分块 Eval Loss。"
        )

    return causal_lm, decoder, lm_head


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


def find_all_token_positions(input_ids: List[int], token_id: int) -> List[int]:
    return [
        idx
        for idx, current_id in enumerate(input_ids)
        if int(current_id) == int(token_id)
    ]


def is_input_output_actually_tied(model) -> bool:
    model = unwrap_training_model(model)
    input_weight = model.get_input_embeddings().weight
    output_module = model.get_output_embeddings()

    if output_module is None:
        raise RuntimeError("❌ 模型缺少 lm_head/output embeddings。")

    return input_weight.data_ptr() == output_module.weight.data_ptr()


def enable_full_input_output_training(model) -> bool:
    """
    不主动解绑，也不新增 token。

    完整训练：
      - 整张 input embedding；
      - 整张 lm_head/output embedding；
      - LoRA 参数。

    如果基础模型原本权重共享，则继续共享；如果原本未共享，则保持未共享，
    但两张矩阵都会完整训练。
    """
    model = unwrap_training_model(model)
    input_weight = model.get_input_embeddings().weight
    output_module = model.get_output_embeddings()

    if output_module is None:
        raise RuntimeError("❌ 模型缺少 lm_head/output embeddings。")

    input_weight.requires_grad_(True)
    output_module.weight.requires_grad_(True)

    actually_tied = input_weight.data_ptr() == output_module.weight.data_ptr()

    print("\n" + "=" * 78)
    print("🎯 完整输入/输出词表训练检查")
    print(f"实际共享 input embedding / lm_head: {actually_tied}")
    print(f"config.tie_word_embeddings: {model.config.tie_word_embeddings}")
    print(f"input embedding requires_grad: {input_weight.requires_grad}")
    print(f"lm_head requires_grad: {output_module.weight.requires_grad}")
    print(f"input shape: {tuple(input_weight.shape)}")
    print(f"output shape: {tuple(output_module.weight.shape)}")
    print("说明：本代码不执行解绑；两张矩阵按模型原始共享状态训练。")
    print("=" * 78 + "\n")

    return actually_tied


def save_reserved_token_info(
    save_dir: str,
    tokenizer,
    token_ids: Dict[str, object],
    max_depth: int,
) -> None:
    depth_payload = {}
    depth_ids = token_ids["depth"]

    for depth in range(1, max_depth + 1):
        depth_payload[str(depth)] = {
            "token": DEPTH_PROBE_TOKENS[depth],
            "token_id": int(depth_ids[depth]),
            "semantic_role": (
                f"is dedicated to representing BGC hierarchy information at taxonomy depth {depth}"
            ),
        }

    payload = {
        "ctx_start_token": CTX_START_TOKEN,
        "ctx_start_token_id": int(token_ids["ctx_start"]),
        "ctx_end_token": CTX_END_TOKEN,
        "ctx_end_token_id": int(token_ids["ctx_end"]),
        "max_depth": int(max_depth),
        "depth_tokens": depth_payload,
        "tokenizer_length": len(tokenizer),
        "tokens_added": False,
        "embedding_resized": False,
        "prompt_occurrence": (
            "first occurrence binds each reserved token to one taxonomy depth"
        ),
        "completion_occurrence": (
            "second occurrence precedes the JSON array containing all labels "
            "at that depth"
        ),
        "representation_for_future_probing_and_distillation": (
            "final-layer contextual hidden state at the second occurrence "
            "of each depth token"
        ),
        "body_span_rule": (
            "tokens strictly between ctx_start_token_id and ctx_end_token_id"
        ),
        "target_format": (
            "exactly 4 lines; each line is: depth_reserved_token + space + "
            "JSON string array; [] means no label at that depth"
        ),
    }

    path = os.path.join(save_dir, "reserved_token_info.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ 预留特殊 Token 元信息已保存到: {path}")


def _reset_directory(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def save_full_input_output_weights(model, output_dir: str) -> None:
    """
    PEFT 默认只保存 adapter。由于本脚本手动将完整 input embedding/lm_head
    设为可训练，因此额外保存完整输入/输出词表权重。
    """
    model = unwrap_training_model(model)
    os.makedirs(output_dir, exist_ok=True)

    input_weight = model.get_input_embeddings().weight
    output_module = model.get_output_embeddings()

    if output_module is None:
        raise RuntimeError("❌ 保存时模型缺少 lm_head/output embeddings。")

    actually_tied = input_weight.data_ptr() == output_module.weight.data_ptr()

    payload = {
        "input_embeddings": input_weight.detach().cpu(),
        "actually_tied": actually_tied,
    }

    if not actually_tied:
        payload["output_embeddings"] = output_module.weight.detach().cpu()

    weight_path = os.path.join(output_dir, FULL_IO_WEIGHTS_NAME)
    torch.save(payload, weight_path)

    meta = {
        "actually_tied": actually_tied,
        "config_tie_word_embeddings": bool(
            getattr(model.config, "tie_word_embeddings", False)
        ),
        "input_shape": list(input_weight.shape),
        "output_shape": list(output_module.weight.shape),
        "contains_output_tensor_separately": not actually_tied,
    }

    meta_path = os.path.join(output_dir, FULL_IO_META_NAME)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def save_peft_checkpoint_with_full_io(
    model, tokenizer, output_dir: str, overwrite: bool = True,
    checkpoint_metadata: Optional[dict] = None,
) -> None:
    """先完整写临时目录；保留上一份模型，防止最佳模型保存中断后丢失。"""
    model = unwrap_training_model(model)
    if not overwrite:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        save_full_input_output_weights(model, output_dir)
        tokenizer.save_pretrained(output_dir)
        if checkpoint_metadata is not None:
            atomic_write_json(os.path.join(output_dir, "best_macro_metadata.json"), checkpoint_metadata)
    else:
        staging = output_dir + f".saving-{os.getpid()}-{time.time_ns()}"
        os.makedirs(staging, exist_ok=False)
        model.save_pretrained(staging)
        save_full_input_output_weights(model, staging)
        tokenizer.save_pretrained(staging)
        if checkpoint_metadata is not None:
            atomic_write_json(os.path.join(staging, "best_macro_metadata.json"), checkpoint_metadata)
        previous = output_dir + ".previous"
        if os.path.isdir(output_dir):
            if os.path.isdir(previous):
                shutil.rmtree(previous)
            os.replace(output_dir, previous)
        os.replace(staging, output_dir)
    print(f"✅ 已保存 PEFT + 完整输入/输出词表 checkpoint: {output_dir}")


def select_and_materialize_best_checkpoint(
    save_dir: str,
    final_step: int,
) -> dict:
    """
    训练结束后根据完整 eval_history.jsonl 离线确认 best：
      1. Global Macro-F1 最高者优先；
      2. Macro 完全相同时，Global Micro-F1 更高者优先；
      3. Macro 与 Micro 都完全相同时，保留更早达到该结果的 step。

    为防止最高 Macro 对应的周期 checkpoint 被 save_total_limit 删除，
    验证回调会在 Macro 刷新（或 Macro 相同但 Micro 更高）时，在线覆盖保存
    best_macro_checkpoint。训练结束后，本函数核对历史记录并将该模型物化为
    best_sft_checkpoint。
    """
    history_path = os.path.join(save_dir, "eval_history.jsonl")
    if not os.path.isfile(history_path):
        raise FileNotFoundError(
            f"❌ 缺少验证历史，无法离线选择 best: {history_path}"
        )

    records = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"❌ eval_history.jsonl 第 {line_no} 行解析失败: {exc}"
                ) from exc
            records.append(record)

    if not records:
        raise RuntimeError("❌ eval_history.jsonl 中没有任何验证记录。")

    selected = records[0]
    for record in records[1:]:
        record_ma = float(record["global_macro"])
        record_mi = float(record["global_micro"])
        selected_ma = float(selected["global_macro"])
        selected_mi = float(selected["global_micro"])

        if record_ma > selected_ma or (
            record_ma == selected_ma and record_mi > selected_mi
        ):
            selected = record

    selected_step = int(selected["step"])

    online_best_path = os.path.join(save_dir, "best_macro_checkpoint")
    online_meta_path = os.path.join(save_dir, "best_macro_checkpoint.json")
    final_checkpoint_path = os.path.join(save_dir, "final_sft_checkpoint")
    periodic_checkpoint_path = os.path.join(
        save_dir,
        "checkpoints",
        f"checkpoint-{selected_step}",
    )

    selected_source = None
    if os.path.isfile(online_meta_path) and os.path.isdir(online_best_path):
        with open(online_meta_path, "r", encoding="utf-8") as f:
            online_meta = json.load(f)

        online_matches = (
            int(online_meta.get("step", -1)) == selected_step
            and float(online_meta.get("global_macro", -1.0))
            == float(selected["global_macro"])
            and float(online_meta.get("global_micro", -1.0))
            == float(selected["global_micro"])
        )
        if online_matches:
            selected_source = online_best_path

    if selected_source is None:
        if selected_step == int(final_step) and os.path.isdir(final_checkpoint_path):
            selected_source = final_checkpoint_path
        elif os.path.isdir(periodic_checkpoint_path):
            selected_source = periodic_checkpoint_path

    if selected_source is None:
        available = []
        checkpoints_root = os.path.join(save_dir, "checkpoints")
        if os.path.isdir(checkpoints_root):
            available = sorted(os.listdir(checkpoints_root))
        raise RuntimeError(
            "❌ 最高 Global Macro-F1 对应的 checkpoint 不存在。\n"
            f"最高记录 step={selected_step}, "
            f"Macro={float(selected['global_macro']):.6f}, "
            f"Micro={float(selected['global_micro']):.6f}\n"
            f"在线 best 路径={online_best_path}\n"
            f"现存周期 checkpoint={available}"
        )

    best_checkpoint_path = os.path.join(save_dir, "best_sft_checkpoint")
    _reset_directory(best_checkpoint_path)
    shutil.copytree(
        selected_source,
        best_checkpoint_path,
        dirs_exist_ok=True,
    )

    max_macro = float(selected["global_macro"])
    tied_max_macro_steps = [
        int(r["step"])
        for r in records
        if float(r["global_macro"]) == max_macro
    ]

    summary = {
        "selection_rule": (
            "highest Global Macro-F1; exact Macro ties are broken by higher "
            "Global Micro-F1; exact Macro and Micro ties keep the earlier step"
        ),
        "selected_step": selected_step,
        "selected_global_macro": max_macro,
        "selected_global_micro": float(selected["global_micro"]),
        "selected_global_samples": float(
            selected.get("global_samples", 0.0)
        ),
        "selected_eval_loss": float(selected.get("eval_loss", 0.0)),
        "steps_with_exact_same_highest_macro": tied_max_macro_steps,
        "source_checkpoint": selected_source,
        "materialized_best_checkpoint": best_checkpoint_path,
    }

    summary_path = os.path.join(save_dir, "best_checkpoint_selection.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print("🏆 训练结束后最高 Macro checkpoint 选择完成")
    print(
        f"最终选择 Step {summary['selected_step']} | "
        f"Macro={summary['selected_global_macro']:.6f} | "
        f"Micro={summary['selected_global_micro']:.6f} | "
        f"Samples={summary['selected_global_samples']:.6f}"
    )
    if len(tied_max_macro_steps) > 1:
        print(
            "最高 Macro 出现于多个 step，已按 Micro-F1 决胜: "
            f"{tied_max_macro_steps}"
        )
    print(f"来源: {selected_source}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"选择记录: {summary_path}")
    print("=" * 78 + "\n")

    return summary


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def checkpoint_file_sizes(checkpoint: str, world_size: int) -> dict:
    """当前脚本的普通单卡 / DDP checkpoint；不适用于 DeepSpeed/FSDP。"""
    required = [
        "adapter_config.json", FULL_IO_WEIGHTS_NAME, FULL_IO_META_NAME,
        "trainer_state.json", "optimizer.pt", "scheduler.pt", "training_args.bin",
    ]
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


def validate_checkpoint(checkpoint: str, world_size: int) -> int:
    if os.path.exists(os.path.join(checkpoint, BGC_PENDING_NAME)):
        raise RuntimeError(f"❌ 该断点保存时被中断，不能用于续训: {checkpoint}")
    sizes = checkpoint_file_sizes(checkpoint, world_size)
    state = read_json_file(os.path.join(checkpoint, "trainer_state.json"))
    step = int(state["global_step"])
    if step <= 0:
        raise RuntimeError(f"❌ 断点 global_step 无效: {checkpoint}")
    folder_match = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(checkpoint))
    if folder_match is None or int(folder_match.group(1)) != step:
        raise RuntimeError("❌ 续训必须指定与 global_step 一致的周期 checkpoint-N 目录。")
    custom_path = os.path.join(checkpoint, BGC_RESUME_STATE_NAME)
    if os.path.isfile(custom_path):
        custom = read_json_file(custom_path)
        if int(custom["global_step"]) != step:
            raise RuntimeError("❌ Trainer 与 BGC 回调保存步数不一致。")
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
                validate_checkpoint(path, world_size)
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
    validate_checkpoint(path, world_size)
    return path


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_signature(args) -> dict:
    # 保持总 epoch 不变：linear scheduler 的总计划也必须保持不变。
    fields = (
        "seed", "num_epochs", "batch_size", "accumulation_steps", "lr", "io_lr",
        "lr_scheduler_type", "warmup_ratio", "lora_rank", "lora_alpha", "lora_dropout",
        "max_seq_length", "reserve_tokens", "max_new_tokens", "system_prompt",
        "expected_num_labels", "expected_max_depth", "eval_steps",
        "eval_batch_size", "loss_eval_batch_size", "eval_loss_logit_chunk_size",
        "early_stopping_patience", "early_stopping_min_delta", "bf16",
        "gradient_checkpointing", "attn_implementation", "num_workers",
    )
    signature = {name: getattr(args, name) for name in fields}
    # 教师版另有验证 DataLoader 参数；学生版没有此参数。
    if hasattr(args, "eval_num_workers"):
        signature["eval_num_workers"] = args.eval_num_workers
        signature["dataloader_pin_memory"] = args.dataloader_pin_memory
    signature["model_path"] = os.path.realpath(args.model_path)
    signature["base_configuration_sha256"] = {
        name: sha256_file(os.path.join(args.model_path, name))
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        if os.path.isfile(os.path.join(args.model_path, name))
    }
    signature["world_size"] = int(os.environ.get("WORLD_SIZE", "1"))
    signature["data_sha256"] = {
        name: sha256_file(getattr(args, name))
        for name in ("train_data_path", "val_data_path", "label_space_path")
    }
    signature["versions"] = {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "peft": peft.__version__,
    }
    return signature


def validate_resume_configuration(checkpoint: str, signature: dict) -> None:
    path = os.path.join(checkpoint, BGC_RESUME_STATE_NAME)
    if os.path.isfile(path):
        saved = read_json_file(path)["run_signature"]
        changed = [
            key for key in sorted(set(saved) | set(signature))
            if saved.get(key) != signature.get(key)
        ]
        if changed:
            raise RuntimeError(
                "❌ 续训配置与断点不一致: " + ", ".join(changed)
                + "。请沿用原数据、环境、总 epoch、batch、学习率等配置。"
            )
        return
    # 旧脚本没有本脚本的配置快照；从本人的旧断点读取 TrainingArguments。
    # training_args.bin 含 Python 对象，不能使用仅 tensor 的加载方式。
    args_path = os.path.join(checkpoint, "training_args.bin")
    try:
        saved = torch.load(args_path, map_location="cpu", weights_only=False)
    except TypeError:
        saved = torch.load(args_path, map_location="cpu")
    mapping = {
        "seed": "seed", "num_epochs": "num_train_epochs",
        "batch_size": "per_device_train_batch_size",
        "accumulation_steps": "gradient_accumulation_steps",
        "lr": "learning_rate", "warmup_ratio": "warmup_ratio",
        "bf16": "bf16", "gradient_checkpointing": "gradient_checkpointing",
        "eval_steps": "save_steps", "lr_scheduler_type": "lr_scheduler_type",
    }
    changed = []
    for current_name, saved_name in mapping.items():
        value = getattr(saved, saved_name, None)
        value = getattr(value, "value", value)
        if value != signature[current_name]:
            changed.append(current_name)
    adapter = read_json_file(os.path.join(checkpoint, "adapter_config.json"))
    for current_name, saved_name in (
        ("lora_rank", "r"), ("lora_alpha", "lora_alpha"),
        ("lora_dropout", "lora_dropout"),
    ):
        if adapter.get(saved_name) != signature[current_name]:
            changed.append(current_name)
    if changed:
        raise RuntimeError("❌ 旧断点配置不一致: " + ", ".join(changed))
    print(
        "⚠️ 正在兼容旧脚本断点：已核对 TrainingArguments 和 LoRA 配置；"
        "旧断点未记录数据指纹、io_lr、Prompt、早停参数，请沿用原实验设置。"
    )


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
    # 已有历史按 step 升序，相同 Macro/Micro 保留更早的 step。
    return max(records, key=lambda r: (float(r["global_macro"]), float(r["global_micro"])))


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


def ensure_historical_best_available(save_dir: str, records: List[dict]) -> None:
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
                FULL_IO_WEIGHTS_NAME, FULL_IO_META_NAME, "adapter_config.json",
            )) and any(os.path.isfile(os.path.join(candidate, name)) for name in (
                "adapter_model.safetensors", "adapter_model.bin",
            ))
            if matches and files_exist:
                if candidate == online:
                    atomic_write_json(meta_path, meta)
                    return
                source = candidate
                break
    if source is None:
        source = os.path.join(save_dir, "checkpoints", f"checkpoint-{int(best['step'])}")
    required = [FULL_IO_WEIGHTS_NAME, "adapter_config.json", FULL_IO_META_NAME]
    if not all(os.path.isfile(os.path.join(source, name)) for name in required) or not any(
        os.path.isfile(os.path.join(source, name))
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ) or os.path.exists(os.path.join(source, BGC_PENDING_NAME)):
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



def export_merged_full_model(
    base_model_path: str,
    peft_checkpoint_dir: str,
    output_dir: str,
    tokenizer,
    torch_dtype,
) -> None:
    """
    将“基础模型 + LoRA adapter + 完整输入/输出词表权重”导出为可直接通过
    AutoModelForCausalLM.from_pretrained() 加载的完整合并模型。
    """
    weight_path = os.path.join(
        peft_checkpoint_dir,
        FULL_IO_WEIGHTS_NAME,
    )

    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            f"❌ 缺少完整输入/输出词表权重: {weight_path}"
        )

    print("\n" + "=" * 78)
    print(f"📦 导出完整合并模型: {output_dir}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map="cpu",
    )

    payload = _torch_load_cpu(weight_path)
    saved_input = payload["input_embeddings"]
    saved_tied = bool(payload["actually_tied"])

    input_weight = base_model.get_input_embeddings().weight
    output_module = base_model.get_output_embeddings()

    if output_module is None:
        raise RuntimeError("❌ 导出时基础模型缺少 lm_head。")

    if tuple(saved_input.shape) != tuple(input_weight.shape):
        raise RuntimeError(
            "❌ 保存的 input embedding 形状与基础模型不一致："
            f"saved={tuple(saved_input.shape)}, "
            f"base={tuple(input_weight.shape)}"
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
                raise RuntimeError("❌ 无法恢复训练时的共享输入/输出权重。")
        else:
            saved_output = payload.get("output_embeddings")
            if saved_output is None:
                raise RuntimeError("❌ 未共享模型缺少 output_embeddings 权重。")

            if tuple(saved_output.shape) != tuple(output_module.weight.shape):
                raise RuntimeError(
                    "❌ 保存的 lm_head 形状与基础模型不一致。"
                )

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

            with torch.no_grad():
                output_module.weight.copy_(
                    saved_output.to(dtype=output_module.weight.dtype)
                )

            base_model.config.tie_word_embeddings = False

    peft_model = PeftModel.from_pretrained(
        base_model,
        peft_checkpoint_dir,
        is_trainable=False,
    )
    merged_model = peft_model.merge_and_unload()
    merged_model.config.use_cache = True

    _reset_directory(output_dir)
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(output_dir)

    print(f"✅ 完整合并模型已保存: {output_dir}")
    print("=" * 78 + "\n")

# ======================================================================
# 2. Prompt：4 个层级 Token 首次绑定 + Completion 第二次生成整层标签
# ======================================================================
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


# ======================================================================
# 3. 严格读取 BGC clean formatted 数据与 label_space.json
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

# ======================================================================
# 4. BGC label_space.json 与 clean split 严格一致性检查
# ======================================================================
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
      - Root 深度为 0，任务标签深度为 1...8；
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


def save_resolved_label_space(save_dir: str, label_space: dict) -> None:
    payload = {
        "num_labels": int(label_space["num_labels"]),
        "max_depth": int(label_space["max_depth"]),
        "label2id": label_space["label2id"],
        "id2label": {
            str(label_id): label
            for label_id, label in label_space["id2label"].items()
        },
        "top_labels": label_space["top_labels"],
        "parent": label_space["parent"],
        "children": label_space["children"],
        "depth": label_space["depth"],
        "depth_to_labels": {
            str(current_depth): labels
            for current_depth, labels in label_space["depth_to_labels"].items()
        },
        "generation_rule": (
            "one reserved token predicts the complete JSON label array for "
            "one taxonomy depth; all 4 lines are always generated; D1 uses "
            "canonical label-ID order; each deeper depth groups labels by "
            "the previous depth's parent order, with siblings kept in "
            "canonical label-ID order"
        ),
        "metric_rule": (
            "Global Micro-F1 uses the complete 146-label space and includes "
            "every exactly matched known label even when generated at the "
            "wrong depth; official Global Macro-F1 averages only labels "
            "present in the current ground-truth split; Global "
            "Macro-F1-All146 is retained as a diagnostic; Strict "
            "Micro/Samples-F1 additionally use one invalid-output sentinel "
            "dimension for format, depth, order, whitespace, empty-output, "
            "and hierarchy errors"
        ),
    }

    path = os.path.join(save_dir, "resolved_label_space.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ 训练实际使用的标签空间已保存: {path}")


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

# ======================================================================
# 5. 截断、正文边界审计、序列预算、生成解析与多标签指标
# ======================================================================
def truncate_text_for_prompt(
    text: str,
    tokenizer,
    max_seq_length: int = 3072,
    reserve_tokens: int = 1024,
) -> str:
    """
    train / validation generation / validation loss 共用同一截断逻辑。

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


def _token_ids_sha256(token_ids: List[int]) -> str:
    payload = ",".join(str(int(token_id)) for token_id in token_ids)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _decode_ids(tokenizer, token_ids: List[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _find_first_non_whitespace_token_offset(tokenizer, token_ids: List[int]):
    for end in range(1, len(token_ids) + 1):
        if _decode_ids(tokenizer, token_ids[:end]).strip():
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
    token_ids: Dict[str, object],
    split_name: str,
    source_index: int,
    print_detail: bool,
    preview_chars: int,
) -> dict:
    """
    按 token ID 提取 CTX_START 与 CTX_END 之间的权威正文 span。

    严格要求边界内部 token IDs 与编码 "\n" + text + "\n" 完全一致。
    """
    start_id = int(token_ids["ctx_start"])
    end_id = int(token_ids["ctx_end"])

    start_positions = find_all_token_positions(prompt_ids, start_id)
    end_positions = find_all_token_positions(prompt_ids, end_id)

    if len(start_positions) != 1 or len(end_positions) != 1:
        raise RuntimeError(
            f"❌ {split_name} 第 {source_index} 条正文边界次数异常："
            f"CTX_START={start_positions}, CTX_END={end_positions}"
        )

    start_pos = start_positions[0]
    end_pos = end_positions[0]
    if end_pos <= start_pos + 1:
        raise RuntimeError(
            f"❌ {split_name} 第 {source_index} 条正文边界顺序错误或正文为空："
            f"start={start_pos}, end={end_pos}"
        )

    between_ids = [int(x) for x in prompt_ids[start_pos + 1 : end_pos]]
    expected_between_text = f"\n{truncated_text}\n"
    expected_between_ids = tokenizer.encode(
        expected_between_text,
        add_special_tokens=False,
    )
    expected_between_ids = [int(x) for x in expected_between_ids]

    token_exact_match = between_ids == expected_between_ids
    decoded_between_text = _decode_ids(tokenizer, between_ids)
    decoded_segment_match = decoded_between_text == expected_between_text

    if (
        decoded_between_text.startswith("\n")
        and decoded_between_text.endswith("\n")
    ):
        extracted_text = decoded_between_text[1:-1]
    else:
        extracted_text = decoded_between_text

    extracted_text_match = extracted_text == truncated_text

    if not token_exact_match or not decoded_segment_match or not extracted_text_match:
        raise RuntimeError(
            f"❌ {split_name} 第 {source_index} 条正文 span 与构造正文不一致。\n"
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

    record = {
        "split": split_name,
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
        "prompt_token_sha256": _token_ids_sha256(prompt_ids),
        "first_token_after_ctx_start_id": int(between_ids[0]),
        "first_token_after_ctx_start": tokenizer.convert_ids_to_tokens(
            between_ids[0]
        ),
        "first_token_after_ctx_start_decoded": _decode_ids(
            tokenizer,
            [between_ids[0]],
        ),
        "last_token_before_ctx_end_id": int(between_ids[-1]),
        "last_token_before_ctx_end": tokenizer.convert_ids_to_tokens(
            between_ids[-1]
        ),
        "last_token_before_ctx_end_decoded": _decode_ids(
            tokenizer,
            [between_ids[-1]],
        ),
        "first_non_whitespace_token_absolute_position": (
            int(first_non_ws_abs_pos)
            if first_non_ws_abs_pos is not None
            else None
        ),
    }

    if print_detail:
        preview_chars = max(50, int(preview_chars))
        print("\n" + "=" * 88)
        print(
            f"🔬 {split_name} 正文 token span 审计："
            f"source_index={source_index}"
        )
        print(f"CTX_START 位置/ID: {start_pos}/{start_id}")
        print(f"CTX_END   位置/ID: {end_pos}/{end_id}")
        print(f"边界内部 token 数: {len(between_ids)}")
        print(f"token ID 与预期完全一致: {token_exact_match}")
        print(f"解码正文与截断正文逐字符一致: {extracted_text_match}")
        print("出现额外 '>'、边界碎片或正文首 token 黏连: False")
        print(f"正文文本 SHA-256: {record['body_text_sha256']}")
        print(f"正文 token SHA-256: {record['body_token_sha256']}")
        print(f"完整 Prompt token SHA-256: {record['prompt_token_sha256']}")

        print("\n【截断后正文开头】")
        print(repr(extracted_text[:preview_chars]))
        print("【截断后正文结尾】")
        print(repr(extracted_text[-preview_chars:]))
        print("【边界内部解码开头（包含正文前换行）】")
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
            print(
                "tokenizer token: "
                f"{tokenizer.convert_ids_to_tokens(first_id)!r}"
            )
            print(
                "single-token decode: "
                f"{_decode_ids(tokenizer, [first_id])!r}"
            )

        print("=" * 88 + "\n")

    return record


def audit_split_body_spans(
    data: List[dict],
    split_name: str,
    tokenizer,
    token_ids: Dict[str, object],
    max_seq_length: int,
    reserve_tokens: int,
    system_prompt: str,
    label_space: dict,
    save_dir: str,
    print_samples: int,
    preview_chars: int,
    scan_limit: int = -1,
) -> str:
    """训练启动前审计一个 split 的正文边界。"""
    total = len(data)
    scan_count = total if scan_limit < 0 else min(total, scan_limit)
    if scan_count <= 0:
        raise ValueError(
            f"❌ {split_name} body audit scan_count 必须大于 0。"
        )

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", split_name.lower())
    output_path = os.path.join(
        save_dir,
        f"{safe_name}_body_span_audit.jsonl",
    )

    records = []
    truncated_count = 0
    max_depth = int(label_space["max_depth"])
    depth_ids = token_ids["depth"]

    print("\n" + "=" * 88)
    print(
        f"🔎 开始 {split_name} 正文边界审计："
        f"扫描 {scan_count}/{total} 条"
    )
    print("权威正文 span = prompt_ids[CTX_START位置 + 1 : CTX_END位置]")
    print("=" * 88)

    for idx in tqdm(
        range(scan_count),
        desc=f"审计{split_name}正文边界",
        leave=False,
    ):
        raw_text = get_clean_sample_fields(
            data[idx], split_name, idx
        )[0]
        raw_token_count = len(
            tokenizer.encode(raw_text, add_special_tokens=False)
        )
        truncated_text = truncate_text_for_prompt(
            raw_text,
            tokenizer,
            max_seq_length=max_seq_length,
            reserve_tokens=reserve_tokens,
        )
        truncated_token_count = len(
            tokenizer.encode(truncated_text, add_special_tokens=False)
        )
        if truncated_token_count < raw_token_count:
            truncated_count += 1

        messages = build_prompt_messages(
            text=truncated_text,
            system_prompt=system_prompt,
            max_depth=max_depth,
        )
        _, prompt_ids = encode_chat_prompt_consistently(
            tokenizer=tokenizer,
            messages=messages,
            split_name=split_name,
            source_index=idx,
        )

        if len(prompt_ids) > max_seq_length:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条 Prompt 已超过 max_seq_length："
                f"{len(prompt_ids)} > {max_seq_length}。\n"
                f"当前 reserve_tokens={reserve_tokens}，"
                "请增大 --reserve_tokens 或减小正文预算。"
            )

        if prompt_ids.count(int(token_ids["ctx_start"])) != 1:
            raise RuntimeError(
                f"❌ {split_name} 第 {idx} 条 CTX_START 次数异常。"
            )
        if prompt_ids.count(int(token_ids["ctx_end"])) != 1:
            raise RuntimeError(
                f"❌ {split_name} 第 {idx} 条 CTX_END 次数异常。"
            )

        for current_depth in range(1, max_depth + 1):
            current_id = int(depth_ids[current_depth])
            if prompt_ids.count(current_id) != 1:
                raise RuntimeError(
                    f"❌ {split_name} 第 {idx} 条 Prompt 中 D{current_depth} "
                    f"token_id={current_id} 应出现 1 次。"
                )

        record = audit_prompt_body_span(
            prompt_ids=prompt_ids,
            truncated_text=truncated_text,
            tokenizer=tokenizer,
            token_ids=token_ids,
            split_name=split_name,
            source_index=idx,
            print_detail=(idx < max(0, print_samples)),
            preview_chars=preview_chars,
        )
        record["raw_body_token_count"] = int(raw_token_count)
        record["truncated_body_token_count"] = int(truncated_token_count)
        record["body_was_truncated"] = bool(
            truncated_token_count < raw_token_count
        )
        record["chat_template_direct_tokenization_match"] = True
        records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n" + "=" * 88)
    print(f"✅ {split_name} 正文边界审计通过：{scan_count}/{total} 条")
    print("✅ CTX_START / CTX_END 均按原子 token ID 定位")
    print("✅ 边界内部 token IDs 与截断正文编码逐 token 完全一致")
    print("✅ 4 个深度 token 在 Prompt 中均恰好出现 1 次")
    print("✅ chat template 直接编码与字符串重编码逐 token 完全一致")
    print(
        f"正文发生截断：{truncated_count}/{scan_count} "
        f"({truncated_count / max(1, scan_count):.2%})"
    )
    print(f"💾 审计日志: {output_path}")
    print("=" * 88 + "\n")

    return output_path


def _get_eot_id(tokenizer) -> int:
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is None or eot_id == tokenizer.unk_token_id:
        eot_id = tokenizer.eos_token_id
    if eot_id is None:
        raise RuntimeError(
            "❌ tokenizer 同时缺少 <|eot_id|> 和 eos_token_id。"
        )
    return int(eot_id)


def validate_sequence_budgets(
    data: List[dict],
    split_name: str,
    tokenizer,
    label_space: dict,
    system_prompt: str,
    max_seq_length: int,
    reserve_tokens: int,
    max_new_tokens: int,
) -> dict:
    """
    全量检查 Prompt + 正确 Completion + EOT 是否能装入 max_seq_length。

    该检查会根据当前 max_seq_length / reserve_tokens / max_new_tokens
    扫描全部样本；只要预算不够，就在训练前给出首个超长样本、
    最大长度和建议 reserve，不会等到训练中途才报错。
    """
    max_depth = int(label_space["max_depth"])
    eot_id = _get_eot_id(tokenizer)

    max_prompt_len = 0
    max_completion_len = 0
    max_full_len = 0
    max_required_reserve = 0
    max_raw_body_len = 0
    max_truncated_body_len = 0
    truncated_count = 0
    first_overflow = None
    first_generation_overflow = None

    for idx, item in enumerate(
        tqdm(data, desc=f"检查{split_name}序列预算", leave=False)
    ):
        text, labels = get_clean_sample_fields(
            item, split_name, idx
        )[:2]

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
        _, prompt_ids = encode_chat_prompt_consistently(
            tokenizer=tokenizer,
            messages=messages,
            split_name=f"{split_name}_BUDGET",
            source_index=idx,
        )

        completion_ids = tokenizer.encode(
            build_completion_text(labels, label_space),
            add_special_tokens=False,
        )
        completion_with_eot_len = len(completion_ids) + 1
        full_len = len(prompt_ids) + completion_with_eot_len

        # 近似但保守地统计“非正文预算”。正文边界内还包含前后换行，
        # 因此再加 4 token 安全余量给出建议值。
        required_reserve = (
            len(prompt_ids)
            - len(truncated_body_ids)
            + completion_with_eot_len
            + 4
        )

        max_prompt_len = max(max_prompt_len, len(prompt_ids))
        max_completion_len = max(
            max_completion_len, completion_with_eot_len
        )
        max_full_len = max(max_full_len, full_len)
        max_required_reserve = max(
            max_required_reserve, required_reserve
        )
        max_raw_body_len = max(max_raw_body_len, len(raw_body_ids))
        max_truncated_body_len = max(
            max_truncated_body_len, len(truncated_body_ids)
        )

        if full_len > max_seq_length and first_overflow is None:
            first_overflow = {
                "index": idx,
                "prompt_len": len(prompt_ids),
                "completion_with_eot_len": completion_with_eot_len,
                "full_len": full_len,
                "raw_body_len": len(raw_body_ids),
                "truncated_body_len": len(truncated_body_ids),
                "required_reserve_estimate": required_reserve,
            }

        # generate 的 max_new_tokens 要能容纳完整正确输出；EOT 也算一个 token。
        if (
            completion_with_eot_len > max_new_tokens
            and first_generation_overflow is None
        ):
            first_generation_overflow = {
                "index": idx,
                "completion_with_eot_len": completion_with_eot_len,
            }

    print("\n" + "=" * 88)
    print(f"📏 {split_name} 序列预算检查")
    print(f"reserve_tokens: {reserve_tokens}")
    print(f"max_seq_length: {max_seq_length}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"最大原始正文 token 数: {max_raw_body_len}")
    print(f"最大截断后正文 token 数: {max_truncated_body_len}")
    print(f"最大 Prompt token 数: {max_prompt_len}")
    print(f"最大正确 Completion+EOT token 数: {max_completion_len}")
    print(f"最大完整训练序列 token 数: {max_full_len}")
    print(
        f"正文发生截断: {truncated_count}/{len(data)} "
        f"({truncated_count / max(1, len(data)):.2%})"
    )
    print(f"建议 reserve_tokens 至少约为: {max_required_reserve}")
    print("=" * 88 + "\n")

    if first_generation_overflow is not None:
        raise ValueError(
            "❌ 正确 BGC 4 行 Completion 超过 max_new_tokens。\n"
            f"首个样本={first_generation_overflow}\n"
            f"请把 --max_new_tokens 至少调到 {max_completion_len}。"
        )

    if first_overflow is not None:
        raise ValueError(
            f"❌ 当前 reserve_tokens={reserve_tokens} 无法容纳某些 "
            "BGC 样本的 Prompt + 4 行 Completion。\n"
            f"首个超长样本={first_overflow}\n"
            f"全量最大完整长度={max_full_len}，max_seq_length={max_seq_length}\n"
            f"建议把 --reserve_tokens 从 {reserve_tokens} 调到至少 "
            f"{max_required_reserve}，然后重新运行。"
        )

    print(
        f"✅ {split_name} 所有样本均可在当前预算下训练："
        f"max_seq_length={max_seq_length}, "
        f"reserve_tokens={reserve_tokens}, "
        f"正文上限={max_seq_length - reserve_tokens}, "
        f"max_new_tokens={max_new_tokens}"
    )

    return {
        "max_prompt_len": max_prompt_len,
        "max_completion_len": max_completion_len,
        "max_full_len": max_full_len,
        "max_required_reserve": max_required_reserve,
        "truncated_count": truncated_count,
    }


def _strip_trailing_stop_strings(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()

    stop_strings = (
        "<|eot_id|>",
        "<|end_of_text|>",
    )

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
# 6. 普通 Completion CE：显存友好的验证辅助函数
# ======================================================================
def compute_completion_ce_sum_from_hidden(
    shifted_hidden: torch.Tensor,
    shifted_labels: torch.Tensor,
    lm_head,
    logit_chunk_size: int,
) -> Tuple[torch.Tensor, int]:
    """
    对所有有效 Completion token 等权计算交叉熵总和。

    位置 t 的 hidden state 预测位置 t+1 的 label，因此调用方应传入：
      shifted_hidden = hidden[:, :-1, :]
      shifted_labels = labels[:, 1:]

    labels == -100 的 Prompt/padding 位置不参与监督。其余 Completion token
    （包括深度特殊 token、JSON 结构、标签文本、换行和 EOT）全部等权。
    返回 loss_sum 与有效 token 数，调用方在完整验证集上执行全局平均。
    """
    if logit_chunk_size <= 0:
        raise ValueError(
            f"❌ logit_chunk_size 必须大于 0，实际={logit_chunk_size}"
        )
    if shifted_hidden.shape[:2] != shifted_labels.shape:
        raise RuntimeError(
            "❌ shifted_hidden 与 shifted_labels 前两维不一致。"
        )

    valid_mask = shifted_labels.ne(-100)
    valid_hidden = shifted_hidden[valid_mask]
    valid_targets = shifted_labels[valid_mask]
    valid_count = int(valid_targets.numel())

    if valid_count <= 0:
        raise RuntimeError("❌ 当前 batch 没有有效 Completion 监督 token。")

    total_loss = shifted_hidden.new_zeros((), dtype=torch.float32)

    for chunk_start in range(0, valid_count, logit_chunk_size):
        chunk_end = min(
            chunk_start + logit_chunk_size,
            valid_count,
        )
        hidden_chunk = valid_hidden[chunk_start:chunk_end]
        target_chunk = valid_targets[chunk_start:chunk_end]

        logits_chunk = lm_head(hidden_chunk)
        total_loss = total_loss + F.cross_entropy(
            logits_chunk.float(),
            target_chunk,
            reduction="sum",
        )

    return total_loss, valid_count


# ======================================================================
# 6. Dataset
# ======================================================================
class GenerativeBGCDataset(Dataset):
    def __init__(
        self,
        data: List[dict],
        tokenizer,
        label_space: dict,
        max_length: int,
        reserve_tokens: int,
        system_prompt: str,
        split_name: str = "训练",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.label_space = label_space
        self.max_length = max_length
        self.reserve_tokens = reserve_tokens
        self.system_prompt = system_prompt
        self.split_name = split_name
        self.max_depth = int(label_space["max_depth"])

        self.ctx_start_id = tokenizer.convert_tokens_to_ids(CTX_START_TOKEN)
        self.ctx_end_id = tokenizer.convert_tokens_to_ids(CTX_END_TOKEN)
        self.depth_token_ids = {
            current_depth: tokenizer.convert_tokens_to_ids(
                DEPTH_PROBE_TOKENS[current_depth]
            )
            for current_depth in range(1, self.max_depth + 1)
        }

        print(
            f"✅ {self.split_name} Dataset 就绪：{len(self.data)} 条 | "
            "普通 Completion-only CE | 所有 Completion token 等权"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text, labels = get_clean_sample_fields(
            item,
            self.split_name,
            idx,
        )[:2]

        text = truncate_text_for_prompt(
            text,
            self.tokenizer,
            self.max_length,
            self.reserve_tokens,
        )

        messages = build_prompt_messages(
            text=text,
            system_prompt=self.system_prompt,
            max_depth=self.max_depth,
        )

        # 训练反向传播入口直接使用 tokenize=True，不会重复添加 BOS。
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_ids = [int(x) for x in prompt_ids]

        # Completion-only supervision：
        # D1-D4 token 在这里第二次出现；每个 token 预测对应深度的一整层。
        completion_text = build_completion_text(
            labels,
            self.label_space,
        )
        completion_ids = self.tokenizer.encode(
            completion_text,
            add_special_tokens=False,
        )
        completion_ids = [int(x) for x in completion_ids]
        completion_ids.append(_get_eot_id(self.tokenizer))

        input_ids = prompt_ids + completion_ids
        labels_ids = [-100] * len(prompt_ids) + completion_ids

        if input_ids.count(int(self.ctx_start_id)) != 1:
            raise RuntimeError(
                f"❌ {self.split_name} 第 {idx} 条 CTX_START 出现次数异常。"
            )
        if input_ids.count(int(self.ctx_end_id)) != 1:
            raise RuntimeError(
                f"❌ {self.split_name} 第 {idx} 条 CTX_END 出现次数异常。"
            )

        second_positions = []
        for current_depth in range(1, self.max_depth + 1):
            current_id = int(self.depth_token_ids[current_depth])
            positions = find_all_token_positions(input_ids, current_id)
            if len(positions) != 2:
                raise RuntimeError(
                    f"❌ {self.split_name} 第 {idx} 条 D{current_depth} "
                    f"Token 应出现 2 次，实际={positions}"
                )
            second_positions.append(positions[1])

        if second_positions != sorted(second_positions):
            raise RuntimeError(
                f"❌ {self.split_name} 第 {idx} 条 D1-D4 第二次 "
                f"occurrence 顺序异常: {second_positions}"
            )

        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            raise ValueError(
                f"❌ {self.split_name} 第 {idx} 条样本仍超长 "
                f"{overflow} tokens。\n"
                "请增大 --reserve_tokens、增大 --max_seq_length，或减小正文预算。"
            )

        if len(input_ids) != len(labels_ids):
            raise RuntimeError(
                f"❌ {self.split_name} 第 {idx} 条 input/label 长度不一致。"
            )

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels_ids,
        }


class CachedTokenizedBGCDataset(Dataset):
    """
    缓存已经完成 Prompt/Completion 构造与 tokenization 的验证 Loss 样本。

    缓存只改变数据准备方式，不改变 input_ids、labels、padding 规则或
    Completion-only CE 的定义。样本按完整序列长度排序，以减少 batch 内
    right-padding。
    """

    def __init__(self, items: List[dict], split_name: str = "验证loss缓存"):
        self.items = items
        self.split_name = split_name
        print(
            f"✅ {self.split_name}就绪：{len(self.items)} 条；"
            "已预先 tokenize，并按完整序列长度排序"
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        # DataCollatorForSeq2Seq 会在组 batch 时补齐 labels。返回副本，
        # 避免 collator 修改缓存中的原始数组，保证多轮验证完全复用。
        return {
            "input_ids": item["input_ids"],
            "labels": item["labels"].copy(),
        }


# ======================================================================
# 7. 生成式验证 Callback
# ======================================================================
class GenerativeBGCF1EvalCallback(TrainerCallback):
    def __init__(
        self,
        val_data: List[dict],
        label_space: dict,
        tokenizer,
        max_seq_length: int = 3072,
        reserve_tokens: int = 1024,
        system_prompt: str = (
            "You are a specialized BGC hierarchical multi-label classifier. "
            "Follow the required output format exactly."
        ),
        eval_steps: int = 1000,
        eval_batch_size: int = 16,
        loss_eval_batch_size: int = 8,
        eval_loss_logit_chunk_size: int = 512,
        loss_eval_num_workers: int = 4,
        loss_eval_pin_memory: bool = True,
        early_stopping_patience: int = 5,
        early_stopping_min_delta: float = 0.0,
        max_new_tokens: int = 520,
        save_dir: str = "",
        resume_checkpoint: Optional[str] = None,
        run_signature: Optional[dict] = None,
    ):
        self.eval_steps = eval_steps
        self.eval_batch_size = int(eval_batch_size)
        self.loss_eval_batch_size = int(loss_eval_batch_size)
        self.eval_loss_logit_chunk_size = int(eval_loss_logit_chunk_size)
        self.loss_eval_num_workers = int(loss_eval_num_workers)
        self.loss_eval_pin_memory = bool(loss_eval_pin_memory)

        if self.eval_batch_size <= 0:
            raise ValueError(
                f"❌ eval_batch_size 必须大于 0，实际={self.eval_batch_size}"
            )
        if self.loss_eval_batch_size <= 0:
            raise ValueError(
                "❌ loss_eval_batch_size 必须大于 0，"
                f"实际={self.loss_eval_batch_size}"
            )
        if self.loss_eval_num_workers < 0:
            raise ValueError(
                "❌ loss_eval_num_workers 不能为负，"
                f"实际={self.loss_eval_num_workers}"
            )
        if self.eval_loss_logit_chunk_size <= 0:
            raise ValueError(
                "❌ eval_loss_logit_chunk_size 必须大于 0，"
                f"实际={self.eval_loss_logit_chunk_size}"
            )
        self.early_stopping_patience = int(early_stopping_patience)
        if self.early_stopping_patience <= 0:
            raise ValueError(
                "❌ early_stopping_patience 必须大于 0，"
                f"实际={self.early_stopping_patience}"
            )

        self.early_stopping_min_delta = float(early_stopping_min_delta)
        if self.early_stopping_min_delta < 0:
            raise ValueError(
                "❌ early_stopping_min_delta 不能为负，"
                f"实际={self.early_stopping_min_delta}"
            )

        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_length
        self.reserve_tokens = reserve_tokens
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        self.resume_checkpoint = resume_checkpoint
        self.run_signature = run_signature or {}
        self.history_records = []
        self.label_space = label_space
        self.max_depth = int(label_space["max_depth"])
        self.num_labels = int(label_space["num_labels"])

        # 早停只监控 Global Macro-F1。只有提高超过 min_delta，
        # 才视为一次有效提升并重置耐心计数。
        self.early_stop_best_ma = -1.0
        self.early_stop_bad_count = 0
        self.early_stop_triggered = False

        # 独立保存真正的最高 Macro checkpoint，避免它被
        # save_total_limit=10 删除。Macro 相同才比较 Micro。
        self.best_saved_ma = -1.0
        self.best_saved_mi = -1.0
        self.best_macro_checkpoint_path = os.path.join(
            save_dir,
            "best_macro_checkpoint",
        )
        self.best_macro_meta_path = os.path.join(
            save_dir,
            "best_macro_checkpoint.json",
        )

        # 记录最后一次已经完成验证的 step，避免最终步与常规验证重复。
        self.last_evaluated_step = -1
        self._force_final_eval = False

        self.history_path = os.path.join(
            save_dir,
            "eval_history.jsonl",
        )
        if not resume_checkpoint:
            if os.path.isfile(self.history_path) and os.path.getsize(self.history_path):
                raise RuntimeError(
                    "❌ save_dir 已有验证历史。续训请使用 --resume_from_checkpoint auto；"
                    "新实验请更换 --save_dir。"
                )

        # --------------------------------------------------------------
        # Eval Loss 数据只在 Callback 初始化时 tokenize 一次。
        # 后续每轮验证复用完全相同的 input_ids/labels，并按完整序列长度
        # 排序以减少 right-padding。该优化不改变任何监督 token。
        # --------------------------------------------------------------
        raw_loss_eval_dataset = GenerativeBGCDataset(
            data=val_data,
            tokenizer=tokenizer,
            label_space=label_space,
            max_length=max_seq_length,
            reserve_tokens=reserve_tokens,
            system_prompt=system_prompt,
            split_name="验证loss预处理",
        )

        cached_loss_eval_items: List[dict] = []
        for cache_idx in tqdm(
            range(len(raw_loss_eval_dataset)),
            desc="缓存验证loss token",
            leave=False,
        ):
            tokenized_item = raw_loss_eval_dataset[cache_idx]
            cached_loss_eval_items.append(
                {
                    # int32 仅用于 CPU 缓存；collator 输出仍会转换为
                    # 模型要求的 torch.long，不改变 token ID。
                    "input_ids": np.asarray(
                        tokenized_item["input_ids"],
                        dtype=np.int32,
                    ),
                    "labels": np.asarray(
                        tokenized_item["labels"],
                        dtype=np.int32,
                    ),
                }
            )

        cached_loss_eval_items.sort(
            key=lambda item: len(item["input_ids"])
        )
        self.loss_eval_dataset = CachedTokenizedBGCDataset(
            cached_loss_eval_items,
            split_name="验证loss缓存",
        )
        del raw_loss_eval_dataset
        del cached_loss_eval_items

        # CausalLM 的 Eval Loss 不需要 decoder_input_ids，因此 collator
        # 不持有大模型对象，允许安全使用多进程 worker。
        self.loss_eval_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=None,
            label_pad_token_id=-100,
            padding=True,
            pad_to_multiple_of=8,
        )

        loss_loader_kwargs = {
            "dataset": self.loss_eval_dataset,
            "batch_size": self.loss_eval_batch_size,
            "shuffle": False,
            "collate_fn": self.loss_eval_collator,
            "num_workers": self.loss_eval_num_workers,
            "pin_memory": self.loss_eval_pin_memory,
            "persistent_workers": self.loss_eval_num_workers > 0,
        }
        if self.loss_eval_num_workers > 0:
            loss_loader_kwargs["prefetch_factor"] = 2

        # DataLoader 只创建一次；persistent workers 可跨多轮验证复用。
        self.loss_eval_dataloader = DataLoader(**loss_loader_kwargs)

        # 生成式验证 Prompt：初始化时构造并保存权威 prompt_ids。
        self.val_prompts = []

        for idx, item in enumerate(val_data):
            text, labels, label_ids = get_clean_sample_fields(
                item,
                "VALIDATION",
                idx,
            )[:3]

            text = truncate_text_for_prompt(
                text,
                tokenizer,
                self.max_seq_length,
                self.reserve_tokens,
            )

            messages = build_prompt_messages(
                text=text,
                system_prompt=self.system_prompt,
                max_depth=self.max_depth,
            )

            full_prompt_str, full_prompt_ids = (
                encode_chat_prompt_consistently(
                    tokenizer=tokenizer,
                    messages=messages,
                    split_name="VALIDATION_CALLBACK",
                    source_index=idx,
                )
            )

            if len(full_prompt_ids) > self.max_seq_length:
                raise ValueError(
                    f"❌ VALIDATION 第 {idx} 条 Prompt 超过 "
                    f"max_seq_length：{len(full_prompt_ids)} > "
                    f"{self.max_seq_length}"
                )

            reference_completion_length = len(
                tokenizer.encode(
                    build_completion_text(labels, self.label_space),
                    add_special_tokens=False,
                )
            ) + 1  # 正确 Completion 后还会追加 EOT。

            self.val_prompts.append(
                {
                    "source_index": int(idx),
                    "prompt": full_prompt_str,
                    "prompt_ids": full_prompt_ids,
                    "true_label_ids": set(int(x) for x in label_ids),
                    "true_labels": labels,
                    # 只用于把相近预计生成长度的样本放入同一 batch；
                    # 不会送入模型，也不会参与指标计算。
                    "reference_completion_length": int(
                        reference_completion_length
                    ),
                }
            )

        # 完整验证集不减少任何样本。先按 128-token Prompt 桶分组，
        # 再按正确 Completion 长度排序，兼顾 Prompt padding 与 batch 内
        # 生成尾部等待。排序只改变执行顺序，不改变模型输入或评测规则。
        self.val_prompts.sort(
            key=lambda item: (
                len(item["prompt_ids"]) // 128,
                item["reference_completion_length"],
                len(item["prompt_ids"]),
                item["source_index"],
            )
        )

        print(
            f"✅ BGC 验证生成 Prompt 准备完成："
            f"{len(self.val_prompts)} 条；已按 Prompt 长度桶和参考输出长度排序"
        )
        print(
            "✅ 验证效率配置："
            f"generate_batch={self.eval_batch_size}, "
            f"loss_batch={self.loss_eval_batch_size}, "
            f"loss_chunk={self.eval_loss_logit_chunk_size}, "
            f"loss_workers={self.loss_eval_num_workers}, "
            f"pin_memory={self.loss_eval_pin_memory}"
        )

    def _compute_eval_loss_batch_with_oom_fallback(
        self,
        cpu_batch: dict,
        decoder,
        lm_head,
        device,
        logit_chunk_size: int,
    ) -> Tuple[float, int]:
        """
        计算一个 Eval Loss batch 的交叉熵总和。

        默认使用配置的 loss batch 与 logits chunk；若极端长 batch OOM，
        只对该 batch 自动二分。若 batch=1 仍在 lm_head 分块处 OOM，则
        自动减半 chunk。拆分只改变执行粒度，不改变有效 token 或 CE 定义。
        """
        input_ids_cpu = cpu_batch["input_ids"]
        attention_mask_cpu = cpu_batch["attention_mask"]
        labels_cpu = cpu_batch["labels"]

        batch_size = int(input_ids_cpu.shape[0])
        if batch_size <= 0:
            raise RuntimeError("❌ Eval Loss 收到空 batch。")

        # 当前阶段使用 right padding。递归拆分后重新裁掉子 batch 尾部
        # 无效 padding，降低峰值显存，但不删除任何真实 token。
        valid_lengths = attention_mask_cpu.sum(dim=1)
        max_valid_length = int(valid_lengths.max().item())
        input_ids_cpu = input_ids_cpu[:, :max_valid_length]
        attention_mask_cpu = attention_mask_cpu[:, :max_valid_length]
        labels_cpu = labels_cpu[:, :max_valid_length]

        input_ids = None
        attention_mask = None
        labels = None
        decoder_outputs = None
        shifted_hidden = None
        shifted_labels = None
        batch_loss_sum = None

        try:
            input_ids = input_ids_cpu.to(
                device,
                dtype=torch.long,
                non_blocking=self.loss_eval_pin_memory,
            )
            attention_mask = attention_mask_cpu.to(
                device,
                dtype=torch.long,
                non_blocking=self.loss_eval_pin_memory,
            )
            labels = labels_cpu.to(
                device,
                dtype=torch.long,
                non_blocking=self.loss_eval_pin_memory,
            )

            decoder_outputs = decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

            shifted_hidden = decoder_outputs.last_hidden_state[:, :-1, :]
            shifted_labels = labels[:, 1:]

            batch_loss_sum, batch_valid_tokens = (
                compute_completion_ce_sum_from_hidden(
                    shifted_hidden=shifted_hidden,
                    shifted_labels=shifted_labels,
                    lm_head=lm_head,
                    logit_chunk_size=logit_chunk_size,
                )
            )

            return (
                float(batch_loss_sum.detach().cpu().item()),
                int(batch_valid_tokens),
            )

        except RuntimeError as exc:
            if not is_cuda_oom_error(exc):
                raise

            # 先释放本次失败调用创建的 GPU 临时张量。
            del input_ids
            del attention_mask
            del labels
            del decoder_outputs
            del shifted_hidden
            del shifted_labels
            del batch_loss_sum
            clear_cuda_memory(
                f"Eval Loss batch={batch_size}, chunk={logit_chunk_size} OOM"
            )

            if batch_size > 1:
                split_at = batch_size // 2
                left_batch = {
                    key: value[:split_at]
                    for key, value in cpu_batch.items()
                }
                right_batch = {
                    key: value[split_at:]
                    for key, value in cpu_batch.items()
                }
                print(
                    f"⚠️ Eval Loss batch={batch_size} 显存不足，"
                    f"自动按 {split_at} + {batch_size - split_at} 重试。"
                )
                left_sum, left_count = (
                    self._compute_eval_loss_batch_with_oom_fallback(
                        cpu_batch=left_batch,
                        decoder=decoder,
                        lm_head=lm_head,
                        device=device,
                        logit_chunk_size=logit_chunk_size,
                    )
                )
                right_sum, right_count = (
                    self._compute_eval_loss_batch_with_oom_fallback(
                        cpu_batch=right_batch,
                        decoder=decoder,
                        lm_head=lm_head,
                        device=device,
                        logit_chunk_size=logit_chunk_size,
                    )
                )
                return left_sum + right_sum, left_count + right_count

            if logit_chunk_size > 1:
                smaller_chunk = max(1, logit_chunk_size // 2)
                print(
                    f"⚠️ Eval Loss batch=1 在 chunk={logit_chunk_size} OOM，"
                    f"自动改用 chunk={smaller_chunk} 重试当前样本。"
                )
                return self._compute_eval_loss_batch_with_oom_fallback(
                    cpu_batch=cpu_batch,
                    decoder=decoder,
                    lm_head=lm_head,
                    device=device,
                    logit_chunk_size=smaller_chunk,
                )

            raise RuntimeError(
                "❌ Eval Loss 在 batch=1 且 logits chunk=1 时仍然 OOM。"
            ) from exc

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
            if ma > self.best_saved_ma or (ma == self.best_saved_ma and mi > self.best_saved_mi):
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
            custom_path = os.path.join(self.resume_checkpoint, BGC_RESUME_STATE_NAME)
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
            ensure_historical_best_available(self.save_dir, records)
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
                "checkpoint_files": checkpoint_file_sizes(checkpoint, int(args.world_size)),
            }
            atomic_write_json(os.path.join(checkpoint, BGC_RESUME_STATE_NAME), payload)
            pending = os.path.join(checkpoint, BGC_PENDING_NAME)
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
            eval_kwargs["model"] = unwrap_training_model(eval_kwargs["model"])
            self._evaluate_at_step(args, state, control, **eval_kwargs)
            return {
                "callback_state": self._state_dict(),
                "should_training_stop": bool(control.should_training_stop),
            }

        result = self._root_action(state, evaluate)
        self._restore_callback_state(result["callback_state"])
        control.should_training_stop = result["should_training_stop"]
        return control

    def _compute_eval_loss(self, model, device) -> float:
        """
        teacher-forcing 验证 loss：
          - 普通 Completion-only cross entropy；
          - 所有有效 Completion token 等权；
          - 在完整验证集的全部有效 token 上全局平均；
          - right padding；
          - 不生成；
          - 不构造完整 [batch, sequence, vocabulary] logits。

        先只运行 decoder 得到 hidden states，再选出 labels[:, 1:] 中
        不为 -100 的有效 Completion 位置，最后按小块经过 lm_head 计算
        交叉熵总和。峰值显存由 [B,L,V] 降为 [chunk,V]，但损失定义与
        Hugging Face CausalLM 的普通 Completion CE 保持一致。
        """
        clear_cuda_memory("进入 Eval Loss 前")

        old_padding_side = self.tokenizer.padding_side
        old_config_use_cache = model.config.use_cache
        old_gen_use_cache = getattr(
            model.generation_config,
            "use_cache",
            None,
        )

        self.tokenizer.padding_side = "right"
        model.config.use_cache = False

        if old_gen_use_cache is not None:
            model.generation_config.use_cache = False

        # 复用初始化阶段已经建立的缓存数据与 persistent DataLoader。
        dataloader = self.loss_eval_dataloader

        _, decoder, lm_head = get_causal_lm_for_memory_efficient_eval(model)

        total_loss_sum = 0.0
        total_valid_tokens = 0

        try:
            with torch.inference_mode():
                for batch in tqdm(
                    dataloader,
                    desc="Eval Loss",
                    leave=False,
                ):
                    batch_loss_sum, batch_valid_tokens = (
                        self._compute_eval_loss_batch_with_oom_fallback(
                            cpu_batch=batch,
                            decoder=decoder,
                            lm_head=lm_head,
                            device=device,
                            logit_chunk_size=(
                                self.eval_loss_logit_chunk_size
                            ),
                        )
                    )

                    total_loss_sum += float(batch_loss_sum)
                    total_valid_tokens += int(batch_valid_tokens)
                    del batch
        finally:
            self.tokenizer.padding_side = old_padding_side
            model.config.use_cache = old_config_use_cache

            if old_gen_use_cache is not None:
                model.generation_config.use_cache = old_gen_use_cache

            # dataloader 与 worker 跨验证轮次复用，不在这里销毁。
            clear_cuda_memory("Eval Loss 完成后")

        if total_valid_tokens <= 0:
            raise RuntimeError(
                "❌ 完整验证集中没有有效 Completion 监督 token。"
            )

        return total_loss_sum / total_valid_tokens

    def _generate_raw_texts_once(
        self,
        model,
        batch: List[dict],
        device,
        stop_ids: Sequence[int],
    ) -> List[str]:
        """用当前 batch 大小执行一次贪心生成。

        Prompt 已在 Callback 初始化时完成权威 tokenization。这里直接在 CPU
        上做等价的 left padding，不再调用 tokenizer.pad，也不会改变任何
        prompt token。无论生成成功还是 OOM，都会在函数退出前释放本次尝试
        创建的 CPU/GPU 张量，防止失败尝试占住显存。
        """
        cpu_input_ids = None
        cpu_attention_mask = None
        input_ids = None
        attention_mask = None
        output_ids = None
        generated_ids = None
        trimmed_generated_ids = None

        try:
            batch_prompt_ids = [item["prompt_ids"] for item in batch]
            max_prompt_len = max(len(ids) for ids in batch_prompt_ids)
            batch_size = len(batch_prompt_ids)

            # 与 tokenizer.padding_side="left" + tokenizer.pad 完全等价。
            cpu_input_ids = torch.full(
                (batch_size, max_prompt_len),
                fill_value=int(self.tokenizer.pad_token_id),
                dtype=torch.long,
            )
            cpu_attention_mask = torch.zeros(
                (batch_size, max_prompt_len),
                dtype=torch.long,
            )

            for row_idx, prompt_ids in enumerate(batch_prompt_ids):
                prompt_tensor = torch.as_tensor(
                    prompt_ids,
                    dtype=torch.long,
                )
                prompt_len = int(prompt_tensor.numel())
                cpu_input_ids[row_idx, -prompt_len:] = prompt_tensor
                cpu_attention_mask[row_idx, -prompt_len:] = 1

            if self.loss_eval_pin_memory and torch.cuda.is_available():
                cpu_input_ids = cpu_input_ids.pin_memory()
                cpu_attention_mask = cpu_attention_mask.pin_memory()

            input_ids = cpu_input_ids.to(
                device,
                non_blocking=self.loss_eval_pin_memory,
            )
            attention_mask = cpu_attention_mask.to(
                device,
                non_blocking=self.loss_eval_pin_memory,
            )

            # 严格确认 padding 前后的真实 prompt token 完全一致。
            for row_idx, item in enumerate(batch):
                mask = attention_mask[row_idx].bool()
                actual_prompt_ids = (
                    input_ids[row_idx][mask]
                    .detach()
                    .cpu()
                    .tolist()
                )
                expected_prompt_ids = item["prompt_ids"]
                if actual_prompt_ids != expected_prompt_ids:
                    raise RuntimeError(
                        "❌ 生成式验证 padding 后的 Prompt 与权威 "
                        "prompt_ids 不一致："
                        f"batch_row={row_idx}, "
                        f"actual_len={len(actual_prompt_ids)}, "
                        f"expected_len={len(expected_prompt_ids)}"
                    )

            input_len = input_ids.shape[1]
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=list(stop_ids),
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                use_cache=True,
            )

            generated_ids = output_ids[:, input_len:]
            trimmed_generated_ids = trim_generated_at_first_stop(
                generated_ids=generated_ids,
                stop_ids=stop_ids,
            )

            # 深度 token 本身属于 special token，不能跳过。
            return [
                self.tokenizer.decode(
                    row_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                for row_ids in trimmed_generated_ids
            ]
        finally:
            # 关键修复：即使 model.generate() 中途 OOM，也先释放本轮尝试
            # 的输入、输出和缓存引用，再由上层决定是否拆分重试。
            del cpu_input_ids
            del cpu_attention_mask
            del input_ids
            del attention_mask
            del output_ids
            del generated_ids
            del trimmed_generated_ids

    def _generate_raw_texts_with_oom_fallback(
        self,
        model,
        batch: List[dict],
        device,
        stop_ids: Sequence[int],
    ) -> List[str]:
        """
        先按配置的 eval_batch_size 生成，OOM 时按顺序二分重试。

        这里使用迭代队列而不是在 except 代码块内递归。旧写法会让异常
        traceback 暂时保留失败 generate 调用中的 GPU 张量和 KV cache，导致
        16 -> 8 -> 4 -> 2 -> 1 的重试过程中显存不断叠加，最终出现 batch=1
        仍 OOM。当前写法会先退出 except、释放 traceback 和本轮张量、清空
        CUDA cache，再执行下一个子 batch。预测顺序和指标定义均不改变。
        """
        pending_batches: List[List[dict]] = [batch]
        all_raw_texts: List[str] = []

        while pending_batches:
            current_batch = pending_batches.pop(0)
            failed_size = len(current_batch)
            oom_happened = False
            oom_message = ""

            try:
                current_texts = self._generate_raw_texts_once(
                    model=model,
                    batch=current_batch,
                    device=device,
                    stop_ids=stop_ids,
                )
            except RuntimeError as exc:
                if not is_cuda_oom_error(exc):
                    raise

                oom_happened = True
                oom_message = str(exc)
                # 显式切断 traceback 对失败 generate frame 的引用。
                exc.__traceback__ = None
            else:
                all_raw_texts.extend(current_texts)
                del current_texts

            # 必须离开 except 作用域以后再清理和重试，否则异常对象会继续
            # 持有失败调用的局部变量，empty_cache 也无法真正回收。
            if not oom_happened:
                continue

            clear_cuda_memory(
                f"生成 batch={failed_size} OOM 后已释放失败尝试并准备二分"
            )

            if failed_size <= 1:
                prompt_len = len(current_batch[0]["prompt_ids"])
                raise RuntimeError(
                    "❌ 已彻底释放前序失败尝试后，生成式验证在 batch=1 "
                    "时仍然 CUDA OOM。\n"
                    f"当前样本 source_index={current_batch[0]['source_index']}, "
                    f"prompt_len={prompt_len}, "
                    f"max_new_tokens={self.max_new_tokens}。\n"
                    "这时才说明单样本本身超过当前剩余显存，需要减少模型/"
                    "优化器常驻显存，或降低序列预算。\n"
                    f"原始 OOM 信息: {oom_message}"
                ) from None

            split_at = failed_size // 2
            left_batch = current_batch[:split_at]
            right_batch = current_batch[split_at:]

            print(
                f"⚠️ 生成 batch={failed_size} 显存不足，"
                f"已释放失败尝试，自动按 {len(left_batch)} + "
                f"{len(right_batch)} 重试；eval_batch_size 配置仍保持不变。"
            )

            # 放到队列最前面，保证输出顺序与原 batch 完全一致。
            pending_batches = [left_batch, right_batch] + pending_batches

        return all_raw_texts

    def _append_history(self, payload: dict) -> None:
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.history_records.append(dict(payload))

    def _evaluate_at_step(self, args, state, control, **kwargs):
        model = kwargs.get("model")

        if model is None:
            return control
        if not state.is_world_process_zero:
            return control
        if state.global_step == 0:
            return control

        force_final_eval = bool(self._force_final_eval)

        if (
            not force_final_eval
            and state.global_step % self.eval_steps != 0
        ):
            return control

        if int(state.global_step) == self.last_evaluated_step:
            return control

        eval_trigger = "最终步" if force_final_eval else "定期"
        print(
            f"\n[Step {state.global_step}] "
            f"开始{eval_trigger} BGC 严格生成式验证..."
        )

        was_training = model.training
        model.eval()
        model.zero_grad(set_to_none=True)
        clear_cuda_memory("训练切换到验证")
        device = next(model.parameters()).device

        try:
            eval_loss = self._compute_eval_loss(model, device)

            print(
                f"[Step {state.global_step}] "
                f"Eval Loss: {eval_loss:.6f}"
            )
            clear_cuda_memory("Eval Loss 与生成式验证阶段切换")

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

            eot_id = _get_eot_id(self.tokenizer)
            stop_ids = []
            for stop_id in [self.tokenizer.eos_token_id, eot_id]:
                if stop_id is not None and stop_id not in stop_ids:
                    stop_ids.append(int(stop_id))

            if not stop_ids:
                raise RuntimeError(
                    "❌ generation 没有可用 eos/eot stop id。"
                )

            old_padding_side = self.tokenizer.padding_side
            old_config_use_cache = model.config.use_cache
            old_gen_use_cache = getattr(
                model.generation_config,
                "use_cache",
                None,
            )

            self.tokenizer.padding_side = "left"
            model.config.use_cache = True

            if old_gen_use_cache is not None:
                model.generation_config.use_cache = True

            try:
                with torch.inference_mode():
                    for start in tqdm(
                        range(
                            0,
                            len(self.val_prompts),
                            self.eval_batch_size,
                        ),
                        desc=f"Step {state.global_step} Eval",
                    ):
                        batch = self.val_prompts[
                            start : start + self.eval_batch_size
                        ]
                        raw_texts = self._generate_raw_texts_with_oom_fallback(
                            model=model,
                            batch=batch,
                            device=device,
                            stop_ids=stop_ids,
                        )

                        for raw_resp, item in zip(raw_texts, batch):
                            parsed_layers, _ = parse_generation_exact(
                                raw_resp,
                                max_depth=self.max_depth,
                            )
                            prediction = validate_parsed_prediction(
                                parsed_layers,
                                self.label_space,
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

                            true_id_sets.append(true_ids)
                            pred_id_sets.append(pred_ids)
                            invalid_flags.append(invalid)
            finally:
                self.tokenizer.padding_side = old_padding_side
                model.config.use_cache = old_config_use_cache

                if old_gen_use_cache is not None:
                    model.generation_config.use_cache = old_gen_use_cache

            metrics = compute_multilabel_metric_bundle(
                true_id_sets=true_id_sets,
                pred_id_sets=pred_id_sets,
                invalid_flags=invalid_flags,
                label_space=self.label_space,
            )

            sample_count = max(1, len(true_id_sets))
            exact_parse_rate = exact_parse_count / sample_count
            valid_output_rate = valid_output_count / sample_count
            hierarchy_closed_rate = hierarchy_closed_count / sample_count
            exact_match_rate = exact_match_count / sample_count
            invalid_output_rate = sum(invalid_flags) / sample_count

            print(
                f"[Step {state.global_step}] 正式指标 | "
                f"Global Micro-F1: {metrics['global_micro']:.4f} | "
                f"Global Macro-F1(Present): {metrics['global_macro']:.4f} | "
                f"Macro-All146: {metrics['global_macro_all_labels']:.4f} | "
                f"Samples-F1: {metrics['global_samples']:.4f} | "
                f"PresentLabels: {metrics['present_label_count']}/{self.num_labels}"
            )
            print(
                f"[Step {state.global_step}] 严格生成指标 | "
                f"Strict Micro-F1: {metrics['strict_micro']:.4f} | "
                f"Strict Samples-F1: {metrics['strict_samples']:.4f}"
            )

            for current_depth in range(1, self.max_depth + 1):
                print(
                    f"[Step {state.global_step}] D{current_depth} "
                    f"Micro/Macro: "
                    f"{metrics[f'depth_{current_depth}_micro']:.4f}/"
                    f"{metrics[f'depth_{current_depth}_macro']:.4f} | "
                    f"PresentLabels="
                    f"{metrics[f'depth_{current_depth}_present_labels']}"
                )

            print(
                f"[Step {state.global_step}] "
                f"ExactParse={exact_parse_rate:.4f} | "
                f"ValidOutput={valid_output_rate:.4f} | "
                f"AncestorClosed={hierarchy_closed_rate:.4f} | "
                f"ExactSetMatch={exact_match_rate:.4f} | "
                f"InvalidOutput={invalid_output_rate:.4f}"
            )
            print(
                f"[Step {state.global_step}] "
                f"UnknownLabels={total_unknown_labels} | "
                f"WrongDepthLabels={total_wrong_depth_labels} | "
                f"DuplicateLabels={total_duplicate_labels} | "
                f"WhitespaceErrors={total_whitespace_error_labels} | "
                f"OrderErrorDepths={total_order_error_depths}"
            )
            print(
                f"[Step {state.global_step}] "
                f"EmptyPredictions={empty_prediction_count} | "
                f"MissingTopPredictions={missing_top_prediction_count}"
            )

            log_payload = {
                "Eval/Loss": eval_loss,
                # 正式综合指标：不再重复记录 Eval/Micro_F1 与 Eval/Macro_F1。
                "Eval/Global_Micro_F1": metrics["global_micro"],
                "Eval/Global_Macro_F1": metrics["global_macro"],
                "Eval/Global_Macro_F1_All146": metrics[
                    "global_macro_all_labels"
                ],
                "Eval/Present_Label_Count": metrics["present_label_count"],
                "Eval/Samples_F1": metrics["global_samples"],
                "Eval/Strict_Micro_F1": metrics["strict_micro"],
                "Eval/Strict_Samples_F1": metrics["strict_samples"],
                "Eval/Exact_Parse_Rate": exact_parse_rate,
                "Eval/Valid_Output_Rate": valid_output_rate,
                "Eval/Ancestor_Closure_Rate": hierarchy_closed_rate,
                "Eval/Exact_Set_Match_Rate": exact_match_rate,
                "Eval/Invalid_Output_Rate": invalid_output_rate,
                "Eval/Unknown_Label_Count": total_unknown_labels,
                "Eval/Wrong_Depth_Label_Count": total_wrong_depth_labels,
                "Eval/Duplicate_Label_Count": total_duplicate_labels,
                "Eval/Whitespace_Error_Label_Count": (
                    total_whitespace_error_labels
                ),
                "Eval/Canonical_Order_Error_Depth_Count": (
                    total_order_error_depths
                ),
                "Eval/Empty_Prediction_Count": empty_prediction_count,
                "Eval/Missing_Top_Prediction_Count": (
                    missing_top_prediction_count
                ),
            }

            for current_depth in range(1, self.max_depth + 1):
                log_payload[
                    f"Eval/D{current_depth}_Micro_F1"
                ] = metrics[f"depth_{current_depth}_micro"]
                log_payload[
                    f"Eval/D{current_depth}_Macro_F1"
                ] = metrics[f"depth_{current_depth}_macro"]
                log_payload[
                    f"Eval/D{current_depth}_Present_Label_Count"
                ] = metrics[f"depth_{current_depth}_present_labels"]

            swanlab.log(log_payload, step=state.global_step)

            history_payload = {
                "step": int(state.global_step),
                "eval_loss": float(eval_loss),
                **metrics,
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
            }
            self._append_history(history_payload)

            current_ma = float(metrics["global_macro"])
            current_mi = float(metrics["global_micro"])

            # ----------------------------------------------------------
            # 在线保留最高 Macro checkpoint
            # ----------------------------------------------------------
            # 严格以 Macro 为第一优先级；仅当 Macro 完全相同时，
            # 才用 Micro 决胜。这样即使周期 checkpoint 只保留最近 10 个，
            # 真正最高 Macro 的模型也不会丢失。
            improved_for_best_save = (
                current_ma > self.best_saved_ma
                or (
                    current_ma == self.best_saved_ma
                    and current_mi > self.best_saved_mi
                )
            )

            if improved_for_best_save:
                self.best_saved_ma = current_ma
                self.best_saved_mi = current_mi

                best_meta = {
                    "step": int(state.global_step),
                    "global_macro": current_ma,
                    "global_micro": current_mi,
                    "global_samples": float(metrics["global_samples"]),
                    "eval_loss": float(eval_loss),
                    "selection_rule": (
                        "highest Global Macro-F1; exact Macro ties use "
                        "higher Global Micro-F1"
                    ),
                }
                save_peft_checkpoint_with_full_io(
                    model=model,
                    tokenizer=self.tokenizer,
                    output_dir=self.best_macro_checkpoint_path,
                    overwrite=True,
                    checkpoint_metadata=best_meta,
                )

                with open(
                    self.best_macro_meta_path,
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(best_meta, f, ensure_ascii=False, indent=2)

                print(
                    f"🏆 已更新最高 Macro checkpoint: "
                    f"Step {state.global_step} | "
                    f"Macro={current_ma:.6f} | Micro={current_mi:.6f}"
                )

            # ----------------------------------------------------------
            # Macro 早停
            # ----------------------------------------------------------
            # 第一次验证直接建立基准。默认 min_delta=0，因此之后只要
            # Macro 严格高于历史最佳值，就视为上升并把 bad_count 清零；
            # 连续 patience 次没有刷新历史最佳值即停止训练。
            if self.early_stop_best_ma < 0:
                self.early_stop_best_ma = current_ma
                self.early_stop_bad_count = 0
                improved_for_early_stop = True
            elif (
                current_ma
                > self.early_stop_best_ma + self.early_stopping_min_delta
            ):
                self.early_stop_best_ma = current_ma
                self.early_stop_bad_count = 0
                improved_for_early_stop = True
            else:
                self.early_stop_bad_count += 1
                improved_for_early_stop = False

            print(
                f"[Step {state.global_step}] EarlyStop Macro | "
                f"current={current_ma:.6f} | "
                f"best_effective={self.early_stop_best_ma:.6f} | "
                f"min_delta={self.early_stopping_min_delta:.6f} | "
                f"bad_count={self.early_stop_bad_count}/"
                f"{self.early_stopping_patience} | "
                f"effective_improvement={improved_for_early_stop}"
            )

            # 每次验证指标已经写入 eval_history.jsonl；最高 Macro 模型也已
            # 在线独立保存。训练结束后再根据完整历史核对并物化正式 best。
            if self.early_stop_bad_count >= self.early_stopping_patience:
                self.early_stop_triggered = True
                control.should_training_stop = True
                print(
                    f"\n🛑 触发早停：Global Macro-F1 连续 "
                    f"{self.early_stopping_patience} 次验证没有刷新 "
                    "历史最佳值。\n"
                    f"当前 step={state.global_step}，训练将停止；"
                    "正式 best 将在训练结束后从完整验证历史中离线选择。"
                )

            self.last_evaluated_step = int(state.global_step)

        finally:
            clear_cuda_memory("完整验证结束")
            if was_training:
                model.train()

        return control

    def on_train_end(self, args, state, control, **kwargs):
        """所有 rank 共同参与最终步验证的同步。"""
        if state.global_step == 0 or int(state.global_step) == self.last_evaluated_step:
            return control
        self._force_final_eval = True
        try:
            return self.on_step_end(args, state, control, **kwargs)
        finally:
            self._force_final_eval = False

# ======================================================================
# 8. 运行前环境检查 + 自定义 Trainer 保存完整输入/输出词表
# ======================================================================
def check_runtime_compatibility() -> None:
    print("\n" + "=" * 70)
    print("🧪 运行环境检查")
    print(f"torch: {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"peft: {peft.__version__}")
    print("训练策略: LoRA + 完整 input embedding + 完整 lm_head")
    print("不使用 AddedToken / resize / trainable_token_indices / 主动解绑")
    print("=" * 70 + "\n")


class FullIOPeftTrainer(Trainer):
    """
    1. 每个 Hugging Face checkpoint 额外保存完整 input embedding/lm_head；
    2. 使用参数组学习率：
       - LoRA 参数使用 args.learning_rate；
       - 完整 input embedding/lm_head 使用 io_learning_rate。

    若 input embedding 与 lm_head 实际共享，则它们是同一个 Parameter，
    只会被加入优化器一次，不会重复更新。
    """

    def __init__(
        self,
        *trainer_args,
        io_learning_rate: float,
        **trainer_kwargs,
    ):
        if io_learning_rate <= 0:
            raise ValueError(
                f"❌ io_learning_rate 必须大于 0，实际={io_learning_rate}"
            )

        self.io_learning_rate = float(io_learning_rate)
        super().__init__(*trainer_args, **trainer_kwargs)

    def create_optimizer(self):
        """为 LoRA 和完整输入/输出词表建立不同学习率的参数组。"""
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

        # 与 Trainer 默认逻辑一致：Norm/bias 不做 weight decay。
        decay_parameter_names = set(
            self.get_decay_parameter_names(opt_model)
        )

        grouped = {
            "lora_decay": [],
            "lora_no_decay": [],
            "io_decay": [],
            "io_no_decay": [],
        }
        grouped_names = {key: [] for key in grouped}
        seen_param_ids = set()

        for name, parameter in opt_model.named_parameters():
            if not parameter.requires_grad:
                continue

            parameter_id = id(parameter)
            if parameter_id in seen_param_ids:
                continue
            seen_param_ids.add(parameter_id)

            is_io = parameter_id in io_param_ids
            use_decay = name in decay_parameter_names

            if is_io and use_decay:
                key = "io_decay"
            elif is_io:
                key = "io_no_decay"
            elif use_decay:
                key = "lora_decay"
            else:
                key = "lora_no_decay"

            grouped[key].append(parameter)
            grouped_names[key].append(name)

        io_parameter_count = sum(
            parameter.numel()
            for key in ("io_decay", "io_no_decay")
            for parameter in grouped[key]
        )
        lora_parameter_count = sum(
            parameter.numel()
            for key in ("lora_decay", "lora_no_decay")
            for parameter in grouped[key]
        )

        if io_parameter_count == 0:
            raise RuntimeError(
                "❌ 参数分组失败：没有找到可训练的 input embedding/lm_head。"
            )
        if lora_parameter_count == 0:
            raise RuntimeError(
                "❌ 参数分组失败：没有找到可训练的 LoRA 参数。"
            )

        optimizer_grouped_parameters = []

        def add_group(key: str, learning_rate: float, weight_decay: float):
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

        # 兼容不同 transformers 版本的函数签名。
        try:
            optimizer_cls, optimizer_kwargs = (
                Trainer.get_optimizer_cls_and_kwargs(
                    self.args,
                    opt_model,
                )
            )
        except TypeError:
            optimizer_cls, optimizer_kwargs = (
                Trainer.get_optimizer_cls_and_kwargs(self.args)
            )

        optimizer_kwargs = dict(optimizer_kwargs)
        optimizer_kwargs.pop("params", None)

        # PyTorch 优化器仍需要一个默认 lr；每个参数组自己的 lr 会覆盖它。
        optimizer_kwargs["lr"] = self.args.learning_rate

        self.optimizer = optimizer_cls(
            optimizer_grouped_parameters,
            **optimizer_kwargs,
        )

        input_output_tied = (
            input_module.weight.data_ptr()
            == output_module.weight.data_ptr()
        )

        print("\n" + "=" * 70)
        print("🎛️ 分组学习率优化器创建完成")
        print(f"LoRA 初始学习率: {self.args.learning_rate:.8g}")
        print(f"Embedding/lm_head 初始学习率: {self.io_learning_rate:.8g}")
        print(f"LoRA 可训练参数量: {lora_parameter_count:,}")
        print(f"Embedding/lm_head 可训练参数量: {io_parameter_count:,}")
        print(f"input embedding / lm_head 实际共享: {input_output_tied}")
        print(
            "说明：linear scheduler 会按相同比例衰减每个参数组，"
            "因此两组学习率比例在训练过程中保持不变。"
        )
        print("=" * 70 + "\n")

        return self.optimizer

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # 先由 Trainer 恢复 LoRA；再覆盖手动训练的完整 IO 权重。
        # copy_ 不替换 Parameter，因此保留参数共享关系和优化器参数引用。
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        load_full_input_output_weights_for_resume(
            self.model if model is None else model, resume_from_checkpoint
        )

    def _save_checkpoint(self, model, trial, *args, **kwargs):
        # 标记写入中的目录，防止中断后 auto 选择半个 checkpoint。
        output_dir = os.path.join(
            self._get_output_dir(trial), f"checkpoint-{self.state.global_step}"
        )
        if self.is_world_process_zero():
            os.makedirs(output_dir, exist_ok=True)
            atomic_write_json(
                os.path.join(output_dir, BGC_PENDING_NAME),
                {"global_step": int(self.state.global_step)},
            )
        distributed_barrier()
        return super()._save_checkpoint(model, trial, *args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)

        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_world_process_zero():
            save_full_input_output_weights(self.model, output_dir)


# ======================================================================
# 9. Main
# ======================================================================
def main(args) -> None:
    set_all_seeds(args.seed)
    check_runtime_compatibility()

    os.makedirs(args.save_dir, exist_ok=True)
    args.save_dir = os.path.abspath(args.save_dir)
    resume_checkpoint = resolve_resume_checkpoint(args)
    run_signature = build_run_signature(args)
    if resume_checkpoint:
        validate_resume_configuration(resume_checkpoint, run_signature)
        print(f"🔄 将从完整训练断点恢复: {resume_checkpoint}")
    else:
        print("🆕 本次从基础模型开始新实验。")

    # ------------------------------------------------------------------
    # A. label_space.json + clean split 严格校验
    # ------------------------------------------------------------------
    label_space = load_and_validate_label_space(
        label_space_path=args.label_space_path,
        expected_num_labels=args.expected_num_labels,
        expected_max_depth=args.expected_max_depth,
    )

    save_resolved_label_space(
        save_dir=args.save_dir,
        label_space=label_space,
    )

    train_data = validate_clean_split(
        data_path=args.train_data_path,
        split_name="TRAIN",
        label_space=label_space,
    )

    val_data = validate_clean_split(
        data_path=args.val_data_path,
        split_name="VALIDATION",
        label_space=label_space,
    )

    # ------------------------------------------------------------------
    # B. Tokenizer + 6 个 Llama 原生预留特殊 Token
    #    110/111 为正文边界，112-115 分别负责深度 1-4。
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    # 与 BGC 学生脚本保持逐字一致的 Llama-3 chat template。
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

    tokenizer.padding_side = "right"

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
    save_reserved_token_info(
        save_dir=args.save_dir,
        tokenizer=tokenizer,
        token_ids=token_ids,
        max_depth=label_space["max_depth"],
    )

    # ------------------------------------------------------------------
    # C. 先做全量序列预算检查。
    #    当前默认 3072/1024，使正文最多保留 2048 tokens；若 BGC 的
    #    8 层 Prompt/Completion 仍超出，会在训练前直接报告建议值。
    # ------------------------------------------------------------------
    train_budget = validate_sequence_budgets(
        data=train_data,
        split_name="TRAIN",
        tokenizer=tokenizer,
        label_space=label_space,
        system_prompt=args.system_prompt,
        max_seq_length=args.max_seq_length,
        reserve_tokens=args.reserve_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    val_budget = validate_sequence_budgets(
        data=val_data,
        split_name="VALIDATION",
        tokenizer=tokenizer,
        label_space=label_space,
        system_prompt=args.system_prompt,
        max_seq_length=args.max_seq_length,
        reserve_tokens=args.reserve_tokens,
        max_new_tokens=args.max_new_tokens,
    )

    budget_path = os.path.join(args.save_dir, "sequence_budget_report.json")
    with open(budget_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "max_seq_length": args.max_seq_length,
                "reserve_tokens": args.reserve_tokens,
                "max_new_tokens": args.max_new_tokens,
                "train": train_budget,
                "validation": val_budget,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"✅ 序列预算报告已保存: {budget_path}")

    # ------------------------------------------------------------------
    # D. 训练前严格审计 TRAIN / VALIDATION 正文边界
    # ------------------------------------------------------------------
    train_body_audit_path = audit_split_body_spans(
        data=train_data,
        split_name="TRAIN",
        tokenizer=tokenizer,
        token_ids=token_ids,
        max_seq_length=args.max_seq_length,
        reserve_tokens=args.reserve_tokens,
        system_prompt=args.system_prompt,
        label_space=label_space,
        save_dir=args.save_dir,
        print_samples=args.body_audit_print_samples,
        preview_chars=args.body_audit_preview_chars,
        scan_limit=args.body_audit_scan_limit,
    )
    val_body_audit_path = audit_split_body_spans(
        data=val_data,
        split_name="VALIDATION",
        tokenizer=tokenizer,
        token_ids=token_ids,
        max_seq_length=args.max_seq_length,
        reserve_tokens=args.reserve_tokens,
        system_prompt=args.system_prompt,
        label_space=label_space,
        save_dir=args.save_dir,
        print_samples=args.body_audit_print_samples,
        preview_chars=args.body_audit_preview_chars,
        scan_limit=args.body_audit_scan_limit,
    )
    print(f"✅ TRAIN 正文审计日志: {train_body_audit_path}")
    print(f"✅ VALIDATION 正文审计日志: {val_body_audit_path}")

    # ------------------------------------------------------------------
    # E. Dataset：Prompt 首次出现 + Completion 第二次出现
    # ------------------------------------------------------------------
    train_dataset = GenerativeBGCDataset(
        data=train_data,
        tokenizer=tokenizer,
        label_space=label_space,
        max_length=args.max_seq_length,
        reserve_tokens=args.reserve_tokens,
        system_prompt=args.system_prompt,
        split_name="训练",
    )

    # ------------------------------------------------------------------
    # F. 加载教师模型：不新增 token、不 resize、不主动解绑
    # ------------------------------------------------------------------
    model_dtype = torch.bfloat16 if args.bf16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        attn_implementation=args.attn_implementation,
        device_map=None,
    )

    embedding_rows = model.get_input_embeddings().weight.shape[0]
    if embedding_rows != len(tokenizer):
        raise RuntimeError(
            "❌ 原生 tokenizer 长度与模型 input embedding 行数不一致："
            f"tokenizer={len(tokenizer)}, model={embedding_rows}。"
            "本脚本不会自动 resize。"
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None

    model.config.use_cache = False

    # ------------------------------------------------------------------
    # G. LoRA + 完整 input embedding/lm_head 训练
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # get_peft_model 会冻结基础参数；这里显式重新开放完整输入/输出词表。
    actually_tied = enable_full_input_output_training(model)

    model.enable_input_require_grads()

    input_trainable = model.get_input_embeddings().weight.requires_grad
    output_trainable = model.get_output_embeddings().weight.requires_grad

    if not input_trainable or not output_trainable:
        raise RuntimeError("❌ 完整 input embedding/lm_head 未正确进入训练。")

    print("\n" + "=" * 78)
    print("🎯 最终 BGC 教师训练范围")
    print(f"CTX_START ID: {token_ids['ctx_start']}")
    print(f"CTX_END ID:   {token_ids['ctx_end']}")
    for current_depth in range(1, label_space["max_depth"] + 1):
        print(
            f"D{current_depth} Token: "
            f"{DEPTH_PROBE_TOKENS[current_depth]} -> "
            f"ID={token_ids['depth'][current_depth]}"
        )
    print(f"input/lm_head 实际共享: {actually_tied}")
    print("LoRA 参数: 训练")
    print("完整 input embedding: 训练")
    print("完整 lm_head: 训练")
    print("其余 Transformer 原始主干参数: 冻结")
    print(f"LoRA 初始学习率: {args.lr}")
    print(f"完整 embedding/lm_head 初始学习率: {args.io_lr}")
    print(
        "LoRA rank/alpha/dropout: "
        f"{args.lora_rank}/{args.lora_alpha}/{args.lora_dropout}"
    )
    print(
        "Prompt: 4 个深度 token 第一次出现，分别绑定 taxonomy depth 1-4，"
        "并明确作为对应深度的层级信息表示标记"
    )
    print(
        "Completion: 4 个深度 token 第二次出现；一个 token 预测一整层 JSON 标签数组"
    )
    print("空层输出 []；D1 按固定 label ID，D2-D4 按父节点顺序分组，同父节点内按固定 label ID")
    print("正文边界: 两个不可分割的 Llama 预留特殊 Token")
    print("后续探测/蒸馏: 提取每个深度 token 第二次 occurrence 的最后一层 hidden state")
    print("=" * 78 + "\n")

    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # H. TrainingArguments
    # ------------------------------------------------------------------
    if args.save_total_limit != 10:
        raise ValueError(
            "❌ 本实验要求只保留最近 10 个周期 checkpoint，"
            f"当前 --save_total_limit={args.save_total_limit}。"
        )

    training_args = TrainingArguments(
        output_dir=os.path.join(args.save_dir, "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        learning_rate=args.lr,
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
        dataloader_pin_memory=args.dataloader_pin_memory,
        seed=args.seed,
        data_seed=args.seed,
    )

    swanlab_callback = SwanLabCallback(
        project=args.project_name,
        experiment_name=args.run_name,
    )

    f1_eval_callback = GenerativeBGCF1EvalCallback(
        val_data=val_data,
        label_space=label_space,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        reserve_tokens=args.reserve_tokens,
        system_prompt=args.system_prompt,
        eval_steps=args.eval_steps,
        eval_batch_size=args.eval_batch_size,
        loss_eval_batch_size=args.loss_eval_batch_size,
        eval_loss_logit_chunk_size=args.eval_loss_logit_chunk_size,
        loss_eval_num_workers=args.eval_num_workers,
        loss_eval_pin_memory=args.dataloader_pin_memory,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        max_new_tokens=args.max_new_tokens,
        save_dir=args.save_dir,
        resume_checkpoint=resume_checkpoint,
        run_signature=run_signature,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=True,
    )

    trainer = FullIOPeftTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[swanlab_callback, f1_eval_callback],
        io_learning_rate=args.io_lr,
    )

    print("\n" + "=" * 78)
    print("🚀 开始 BGC 教师模型纯生成式层次多标签 SFT（等价加速验证版）")
    print(f"Train:       {args.train_data_path}")
    print(f"Validation:  {args.val_data_path}")
    print(f"Label space: {args.label_space_path}")
    print(
        f"LoRA LR: {args.lr} | "
        f"Embedding/lm_head LR: {args.io_lr} | "
        f"scheduler: {args.lr_scheduler_type} | "
        f"warmup_ratio: {args.warmup_ratio}"
    )
    print(
        f"max_seq_length={args.max_seq_length} | "
        f"reserve_tokens={args.reserve_tokens} | "
        f"max_new_tokens={args.max_new_tokens}"
    )
    print(
        f"生成验证 batch={args.eval_batch_size} | "
        f"Eval Loss batch={args.loss_eval_batch_size} | "
        f"Eval Loss logits chunk={args.eval_loss_logit_chunk_size} | "
        f"Eval workers={args.eval_num_workers}"
    )
    print(
        "验证等价性：完整验证集、Prompt、正文截断、贪心生成、"
        "max_new_tokens、解析、F1、best 与早停规则均保持不变；"
        "仅优化缓存、排序、padding、并行 batch 与数据传输。"
    )
    print(
        "训练Loss：普通 Completion-only CE | "
        "所有有效 Completion token 等权 | "
        "在 batch 的全部有效 token 上全局平均"
    )
    print(
        f"早停：监控 Global Macro-F1 | 每 {args.eval_steps} step 验证 | "
        f"patience={args.early_stopping_patience} | "
        f"min_delta={args.early_stopping_min_delta}"
    )
    print(
        "Best选择：严格选择 Global Macro-F1 最高的 checkpoint；"
        "只有 Macro 完全相同时才比较 Micro-F1。"
    )
    print(
        f"周期 checkpoint 最多保留最近 {args.save_total_limit} 个；"
        "训练结束后仍会另外保存 final/best。"
    )
    print(
        "验证规则：每 eval_steps 验证一次并执行 Macro 早停；"
        "若最后一步不是整除节点，训练结束时自动补做最终步验证。"
    )
    print(
        f"Epoch: {args.num_epochs} | batch: {args.batch_size} | "
        f"accum: {args.accumulation_steps}"
    )
    print("=" * 78 + "\n")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # ------------------------------------------------------------------
    # I. 保存 final/best checkpoint，并可导出完整合并模型
    # ------------------------------------------------------------------
    if trainer.is_world_process_zero():
        final_checkpoint_path = os.path.join(
            args.save_dir,
            "final_sft_checkpoint",
        )

        save_peft_checkpoint_with_full_io(
            model=model,
            tokenizer=tokenizer,
            output_dir=final_checkpoint_path,
            overwrite=True,
        )

        # 基于完整验证历史核对最高 Macro，并物化正式 best。
        best_selection = select_and_materialize_best_checkpoint(
            save_dir=args.save_dir,
            final_step=int(trainer.state.global_step),
        )
        best_checkpoint_path = best_selection[
            "materialized_best_checkpoint"
        ]
        # Hugging Face 的周期 checkpoint 不一定包含 tokenizer；补充保存，
        # 保证 best_sft_checkpoint 可以独立用于后续加载与导出。
        tokenizer.save_pretrained(best_checkpoint_path)

        if args.export_merged_model:
            final_model_path = os.path.join(args.save_dir, "final_model")
            best_model_path = os.path.join(args.save_dir, "best_sft_model")

            export_merged_full_model(
                base_model_path=args.model_path,
                peft_checkpoint_dir=final_checkpoint_path,
                output_dir=final_model_path,
                tokenizer=tokenizer,
                torch_dtype=model_dtype,
            )

            export_merged_full_model(
                base_model_path=args.model_path,
                peft_checkpoint_dir=best_checkpoint_path,
                output_dir=best_model_path,
                tokenizer=tokenizer,
                torch_dtype=model_dtype,
            )

            print(f"🎉 final 完整模型: {final_model_path}")
            print(f"🎉 best 完整模型:  {best_model_path}")
        else:
            print(f"✅ final checkpoint: {final_checkpoint_path}")
            print(f"✅ best checkpoint:  {best_checkpoint_path}")


# ======================================================================
# 10. 参数解析
#     所有经常修改的训练参数统一集中在最底部。
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
            "BGC Teacher Pure Generative SFT: clean ancestor-closed "
            "multi-path labels + atomic context boundaries + one reserved "
            "token per taxonomy depth + full input/output tuning"
        )
    )

    # ================= 路径参数 =================
    parser.add_argument("--model_path", type=str, default="/home/lzp/dataspace/model/llama3.1/")
    parser.add_argument("--train_data_path", type=str, default="/home/lzp/dataspace/data/bgc/train_formatted.jsonl")
    parser.add_argument("--val_data_path", type=str, default="/home/lzp/dataspace/data/bgc/val_formatted.jsonl")
    parser.add_argument("--label_space_path", type=str, default="/home/lzp/dataspace/data/bgc/label_space.json")
    parser.add_argument("--save_dir", type=str, default="/home/lzp/dataspace/checkpoints/bgc_teacher_1token")

    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="不传则开始新实验；auto/true/latest 自动恢复最近完整断点；也可指定 checkpoint-N 路径。",
    )

    # ================= 数据与随机种子 =================
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_num_labels", type=int, default=146)
    parser.add_argument("--expected_max_depth", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=3072)
    # 3072 - 1024 = 2048，因此正文最多保留 2048 tokens。
    parser.add_argument("--reserve_tokens", type=int, default=1024)

    # ================= 正文边界审计 =================
    # 默认扫描 TRAIN 和 VALIDATION 的全部样本；只打印每个 split 的前 5 条。
    parser.add_argument("--body_audit_print_samples", type=int, default=5)
    parser.add_argument("--body_audit_preview_chars", type=int, default=300)
    parser.add_argument("--body_audit_scan_limit", type=int, default=-1)

    # ================= 核心训练参数 =================
    parser.add_argument("--num_epochs", type=float, default=8.0)
    # 保持当前教师实验使用的 4 × 4，有效 batch size=16。
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    # 本实验保留 LoRA rank 16 与完整 IO 小学习率，只改 dropout 与早停。
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--io_lr", type=float, default=5.0e-6)

    # 不做 warmup，不用 cosine；linear 从初始学习率逐步衰减到 0。
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)

    # ================= LoRA 参数 =================
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.06)

    # ================= 验证与保存参数 =================
    parser.add_argument("--eval_steps", type=int, default=500)
    # 连续 5 次验证 Macro 没有严格高于历史最佳值则早停。
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    # 完整验证集不抽样；只提高并行度和数据准备效率，不改变 Prompt、
    # 生成策略、样本、max_new_tokens、解析规则或 F1 口径。生成 OOM 时
    # 仍会仅对当前 batch 自动二分重试，直到 batch=1。
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--loss_eval_batch_size", type=int, default=8)
    # 分块大小只决定每次通过 lm_head 的监督位置数，不改变 CE 定义。
    parser.add_argument("--eval_loss_logit_chunk_size", type=int, default=512)
    # BGC 保守使用 520；训练前会全量检查正确 Completion+EOT 是否超过该预算。
    parser.add_argument("--max_new_tokens", type=int, default=520)
    parser.add_argument("--save_total_limit", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--export_merged_model", type=str2bool, default=True)

    # ================= 性能参数 =================
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_num_workers", type=int, default=4)
    parser.add_argument("--dataloader_pin_memory", type=str2bool, default=True)
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")

    # ================= Prompt / SwanLab =================
    parser.add_argument("--system_prompt", type=str, default="You are a specialized classifier for the BGC hierarchical multi-label taxonomy. Follow the required four-line JSON-array output format exactly. Do not produce explanations or any extra conversational text.")
    parser.add_argument("--project_name", type=str, default="BGC-Generative-Teacher-FastVal-V2")
    parser.add_argument("--run_name", type=str, default="Llama3.1-8B-BGC-1Token-FastValV2-EB16-LB8-Chunk512-4x4-Seed42")

    args = parser.parse_args()
    main(args)
