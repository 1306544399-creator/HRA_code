import os

# 在导入 torch 前设置，尽量保证可复现性。
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import hashlib
import json
import random
import re
import shutil
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
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

# 每个层级使用一对层级 Token：
#   START：位于当前层级答案之前；
#   END：位于当前层级答案之后。
# Prompt 中四个层级 Token 各出现一次用于角色绑定；
# Completion 中再次出现，形成一条连续层级生成链：
#   L1_START + L1 label + L1_END + L2_START + L2 label + L2_END
#
# 特别注意：层级特殊 Token 与相邻内容之间不插入空格或换行：
#   - L1_START 直接预测 L1 标签的第一个 token；
#   - L1_END 直接预测 L2_START；
#   - L2_START 直接预测 L2 标签的第一个 token；
#   - L2_END 直接预测 EOT。
#
# 后续蒸馏时，四个层级 Token 都使用 Completion 中的第二次 occurrence。
HIER_START_TOKEN_L1 = "<|reserved_special_token_112|>"
HIER_END_TOKEN_L1 = "<|reserved_special_token_113|>"
HIER_START_TOKEN_L2 = "<|reserved_special_token_114|>"
HIER_END_TOKEN_L2 = "<|reserved_special_token_115|>"

FULL_IO_WEIGHTS_NAME = "full_input_output_embeddings.pt"
FULL_IO_META_NAME = "full_input_output_embeddings_meta.json"
WOS_RESUME_STATE_NAME = "wos_paired_student_resume_state.json"
WOS_PENDING_NAME = "wos_paired_student_save_in_progress.json"


def unwrap_training_model(model):
    """兼容 DDP 等外层包装，返回实际模型对象。"""
    while hasattr(model, "module"):
        model = model.module
    return model


def validate_atomic_reserved_tokens(tokenizer) -> Dict[str, int]:
    """
    严格验证六个 Llama 预留特殊 Token：
      - 正文边界 2 个；
      - L1 层级 START/END 2 个；
      - L2 层级 START/END 2 个；
      - 已存在于 tokenizer 词表；
      - add_special_tokens=False 时均严格编码为单 token；
      - 不等于 unk；
      - 六个 token ID 两两不同。

    本脚本不新增 token，也不 resize embedding。
    """
    token_map = {
        "ctx_start": CTX_START_TOKEN,
        "ctx_end": CTX_END_TOKEN,
        "l1_start": HIER_START_TOKEN_L1,
        "l1_end": HIER_END_TOKEN_L1,
        "l2_start": HIER_START_TOKEN_L2,
        "l2_end": HIER_END_TOKEN_L2,
    }

    vocab = tokenizer.get_vocab()
    token_ids: Dict[str, int] = {}

    print("\n" + "=" * 70)
    print("🔎 验证 Llama 预留特殊 Token...")

    for role, token in token_map.items():
        if token not in vocab:
            raise RuntimeError(
                f"❌ {token} 不在当前 tokenizer 词表中。"
                "请检查本地模型/tokenizer 版本。"
            )

        ids = tokenizer.encode(token, add_special_tokens=False)
        converted_id = tokenizer.convert_tokens_to_ids(token)

        if len(ids) != 1:
            raise RuntimeError(
                f"❌ {token} 不是严格单 token，实际 ids={ids}"
            )

        token_id = ids[0]

        if converted_id != token_id:
            raise RuntimeError(
                f"❌ {token} 的 encode ID 与 convert ID 不一致："
                f"encode={token_id}, convert={converted_id}"
            )

        if token_id == tokenizer.unk_token_id:
            raise RuntimeError(f"❌ {token} 被编码成 unk_token。")

        token_ids[role] = token_id
        print(f"✅ {role}: {token} -> ID={token_id}")

    if len(set(token_ids.values())) != len(token_ids):
        raise RuntimeError(
            f"❌ 六个预留特殊 Token 的 ID 未两两区分：{token_ids}"
        )

    print("✅ 六个预留特殊 Token 均为不可分割的原子 token")
    print("=" * 70 + "\n")
    return token_ids


def find_all_token_positions(input_ids: List[int], token_id: int) -> List[int]:
    return [idx for idx, current_id in enumerate(input_ids) if current_id == token_id]


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

    print("\n" + "=" * 70)
    print("🎯 完整输入/输出词表训练检查")
    print(f"实际共享 input embedding / lm_head: {actually_tied}")
    print(f"config.tie_word_embeddings: {model.config.tie_word_embeddings}")
    print(f"input embedding requires_grad: {input_weight.requires_grad}")
    print(f"lm_head requires_grad: {output_module.weight.requires_grad}")
    print(f"input shape: {tuple(input_weight.shape)}")
    print(f"output shape: {tuple(output_module.weight.shape)}")
    print("说明：本代码不执行解绑；两张矩阵按模型原始共享状态训练。")
    print("=" * 70 + "\n")

    return actually_tied


def save_reserved_token_info(
    save_dir: str,
    tokenizer,
    token_ids: Dict[str, int],
) -> None:
    payload = {
        "ctx_start_token": CTX_START_TOKEN,
        "ctx_start_token_id": token_ids["ctx_start"],
        "ctx_end_token": CTX_END_TOKEN,
        "ctx_end_token_id": token_ids["ctx_end"],
        "l1_start_token": HIER_START_TOKEN_L1,
        "l1_start_token_id": token_ids["l1_start"],
        "l1_end_token": HIER_END_TOKEN_L1,
        "l1_end_token_id": token_ids["l1_end"],
        "l2_start_token": HIER_START_TOKEN_L2,
        "l2_start_token_id": token_ids["l2_start"],
        "l2_end_token": HIER_END_TOKEN_L2,
        "l2_end_token_id": token_ids["l2_end"],
        "tokenizer_length": len(tokenizer),
        "tokens_added": False,
        "embedding_resized": False,
        "prompt_occurrence": (
            "first occurrence of each hierarchy token binds its semantic role"
        ),
        "completion_occurrence": (
            "second occurrences form one contiguous causal hierarchy chain: "
            "L1_START + L1_LABEL + L1_END + L2_START + L2_LABEL + L2_END"
        ),
        "completion_has_inter_level_separator": False,
        "completion_has_spaces_around_hierarchy_tokens": False,
        "hierarchy_token_semantic_roles": {
            "l1_start": (
                "represents document information used to initiate L1 prediction; "
                "its hidden state directly predicts the first token of the L1 label"
            ),
            "l1_end": (
                "summarizes completed L1 information and performs the L1-to-L2 "
                "hierarchical transition; its hidden state directly predicts L2_START"
            ),
            "l2_start": (
                "initiates L2 prediction conditioned on the document and preceding "
                "L1 information; its hidden state directly predicts the first token "
                "of the L2 label"
            ),
            "l2_end": (
                "summarizes L2 and the complete two-level hierarchy; its hidden "
                "state directly predicts EOT"
            ),
        },
        "future_distillation_targets": [
            "l1_start_second_occurrence",
            "l1_end_second_occurrence",
            "l2_start_second_occurrence",
            "l2_end_second_occurrence",
        ],
        "representation_for_future_probing_and_distillation": (
            "use the final-layer contextual hidden state at the second "
            "occurrence of every hierarchy START/END token"
        ),
        "body_span_rule": (
            "tokens strictly between ctx_start_token_id and ctx_end_token_id"
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
            atomic_write_json(os.path.join(output_dir, "best_sft_metadata.json"), checkpoint_metadata)
    else:
        staging = output_dir + f".saving-{os.getpid()}-{time.time_ns()}"
        os.makedirs(staging, exist_ok=False)
        model.save_pretrained(staging)
        save_full_input_output_weights(model, staging)
        tokenizer.save_pretrained(staging)
        if checkpoint_metadata is not None:
            atomic_write_json(os.path.join(staging, "best_sft_metadata.json"), checkpoint_metadata)
        previous = output_dir + ".previous"
        if os.path.isdir(output_dir):
            if os.path.isdir(previous):
                shutil.rmtree(previous)
            os.replace(output_dir, previous)
        os.replace(staging, output_dir)
    print(f"✅ 已保存 PEFT + 完整输入/输出词表 checkpoint: {output_dir}")


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
    if os.path.exists(os.path.join(checkpoint, WOS_PENDING_NAME)):
        raise RuntimeError(f"❌ 该断点保存时被中断，不能用于续训: {checkpoint}")
    sizes = checkpoint_file_sizes(checkpoint, world_size)
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
            raise RuntimeError("❌ Trainer 与 WOS 学生回调保存步数不一致。")
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
    fields = (
        "seed", "num_epochs", "batch_size", "accumulation_steps", "lr", "io_lr",
        "lr_scheduler_type", "warmup_ratio", "lora_rank", "lora_alpha", "lora_dropout",
        "max_seq_length", "reserve_tokens", "max_new_tokens", "system_prompt",
        "expected_num_l1", "expected_num_l2", "eval_steps", "eval_batch_size",
        "loss_eval_batch_size", "bf16", "gradient_checkpointing", "attn_implementation",
        "num_workers", "dataloader_pin_memory", "logging_steps",
    )
    signature = {name: getattr(args, name) for name in fields}
    signature.update(dataset="WOS", training_stage="paired_token_student_sft", max_depth=2,
                     hierarchy_policy="paired_START_END_per_level;second_occurrences_contiguous_completion")
    signature["model_path"] = os.path.realpath(args.model_path)
    signature["base_configuration_sha256"] = {
        name: sha256_file(os.path.join(args.model_path, name))
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        if os.path.isfile(os.path.join(args.model_path, name))
    }
    signature["world_size"] = int(os.environ.get("WORLD_SIZE", "1"))
    signature["data_sha256"] = {
        name: sha256_file(getattr(args, name))
        for name in ("train_data_path", "val_data_path", "taxonomy_path", "dict_path")
    }
    signature["token_semantics"] = {
        "ctx_start": CTX_START_TOKEN, "ctx_end": CTX_END_TOKEN,
        "l1_start": HIER_START_TOKEN_L1, "l1_end": HIER_END_TOKEN_L1,
        "l2_start": HIER_START_TOKEN_L2, "l2_end": HIER_END_TOKEN_L2,
    }
    signature["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                             "peft": peft.__version__}
    return signature


def validate_resume_configuration(checkpoint: str, signature: dict) -> None:
    path = os.path.join(checkpoint, WOS_RESUME_STATE_NAME)
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
    # 在主流程覆写元信息之前，核对旧实验记录的标签数和 Token 语义。
    experiment = os.path.dirname(os.path.dirname(checkpoint))
    meta_path = os.path.join(experiment, "resolved_label_space.json")
    if os.path.isfile(meta_path):
        meta = read_json_file(meta_path)
        if (len(meta["l1_label2id"]) != int(signature["expected_num_l1"])
                or len(meta["l2_label2id"]) != int(signature["expected_num_l2"])):
            raise RuntimeError("旧 WOS 实验的 L1/L2 标签数与当前设置不一致。")
    token_path = os.path.join(experiment, "reserved_token_info.json")
    if not os.path.isfile(token_path):
        raise RuntimeError("旧配对学生实验缺少 reserved_token_info.json，无法核对 token 语义。")
    if os.path.isfile(token_path):
        meta = read_json_file(token_path)
        expected = {"ctx_start_token": CTX_START_TOKEN, "ctx_end_token": CTX_END_TOKEN,
                    "l1_start_token": HIER_START_TOKEN_L1, "l1_end_token": HIER_END_TOKEN_L1,
                    "l2_start_token": HIER_START_TOKEN_L2, "l2_end_token": HIER_END_TOKEN_L2,
                    "completion_has_inter_level_separator": False,
                    "completion_has_spaces_around_hierarchy_tokens": False}
        if any(meta.get(key) != value for key, value in expected.items()):
            raise RuntimeError("旧 WOS 实验的 Token 语义与当前 START/END 配对学生脚本不一致。")
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
        "num_workers": "dataloader_num_workers", "dataloader_pin_memory": "dataloader_pin_memory",
        "logging_steps": "logging_steps",
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
        "旧断点未记录数据指纹、io_lr、Prompt 和依赖版本，请沿用原实验设置。"
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


def ensure_wos_historical_best_available(save_dir: str, records: List[dict],
                                         allow_legacy_best: bool) -> None:
    best = select_best_record(records)
    if best is None:
        return
    online = os.path.join(save_dir, "best_sft_checkpoint")
    required = [FULL_IO_WEIGHTS_NAME, FULL_IO_META_NAME, "adapter_config.json"]

    def complete_model(path):
        return (all(os.path.isfile(os.path.join(path, name)) and os.path.getsize(os.path.join(path, name)) > 0
                    for name in required)
                and any(os.path.isfile(os.path.join(path, name)) for name in ("adapter_model.safetensors", "adapter_model.bin"))
                and not os.path.exists(os.path.join(path, WOS_PENDING_NAME)))

    source = None
    for candidate in (online, online + ".previous"):
        metadata_path = os.path.join(candidate, "best_sft_metadata.json")
        if complete_model(candidate) and os.path.isfile(metadata_path):
            meta = read_json_file(metadata_path)
            if all(meta.get(key) == best[key] for key in ("step", "global_macro", "global_micro")):
                if candidate == online:
                    return
                source = candidate
                break
    if source is None:
        periodic = os.path.join(save_dir, "checkpoints", f"checkpoint-{int(best['step'])}")
        if complete_model(periodic):
            source = periodic
        elif (allow_legacy_best and complete_model(online)
              and not os.path.isfile(os.path.join(online, "best_sft_metadata.json"))):
            # 原 producer 在完整验证后保存 best；没有超过恢复步数的历史时，
            # 原在线 best 属于所恢复历史。若有未来记录则不能做此推断。
            atomic_write_json(os.path.join(online, "best_sft_metadata.json"), best)
            print("✅ 已按原 WOS 完整验证历史确认旧 best_sft_checkpoint。")
            return
        else:
            raise RuntimeError(
                f"无法取得恢复历史对应的最佳模型 step={best['step']}。"
                "请恢复最近完整断点，或还原对应历史最佳模型备份。"
            )
    temporary = online + f".restore-{time.time_ns()}"
    shutil.copytree(source, temporary)
    atomic_write_json(os.path.join(temporary, "best_sft_metadata.json"), best)
    if os.path.isdir(online):
        os.replace(online, online + f".before-resume-{time.time_ns()}")
    os.replace(temporary, online)


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

    print("\n" + "=" * 70)
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

            # 如果基础模型意外共享，先复制一个独立 lm_head。
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
    print("=" * 70 + "\n")


# ======================================================================
# 2. Prompt：四个层级 Token 角色绑定 + 连续因果层级链 Completion
# ======================================================================
def build_user_content(text: str) -> str:
    """
    学生训练、教师训练、测试、层级探测、特征提取和后续蒸馏
    必须复用同一套 Prompt 逻辑。

    关键约束：
      - Prompt 中四个层级 Token 各出现且只出现一次，用于角色绑定；
      - Completion 中四个层级 Token 各出现第二次；
      - 后续层级蒸馏严格提取 Completion 中的第二次 occurrence。

    Completion 的连续因果链：

      L1_START + L1_LABEL + L1_END + L2_START + L2_LABEL + L2_END

    即：
      L1_START -> L1 label first token
      L1_END   -> L2_START
      L2_START -> L2 label first token
      L2_END   -> EOT
    """
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
    """
    Completion 中四个层级 Token 都是第二次 occurrence，并且形成连续链：

      L1_START + L1答案 + L1_END + L2_START + L2答案 + L2_END

    不在 START/label、label/END、L1_END/L2_START 之间插入额外空格或换行。
    因此关键 next-token 关系严格为：
      L1_START -> L1 label first token
      L1_END   -> L2_START
      L2_START -> L2 label first token
      L2_END   -> EOT   （EOT 在 Dataset 中紧接着 append）

    后续层级蒸馏定位并蒸馏这四个第二次 occurrence。
    """
    return (
        f"{HIER_START_TOKEN_L1}{l1_label}{HIER_END_TOKEN_L1}"
        f"{HIER_START_TOKEN_L2}{l2_label}{HIER_END_TOKEN_L2}"
    )

def build_prompt_messages(text: str, system_prompt: str) -> List[dict]:
    """统一构造训练、验证、测试、探测、提取和蒸馏所用 messages。"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_content(text)},
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
    token_ids: Dict[str, int],
) -> None:
    """
    严格检查 Prompt / Completion 的特殊 Token 布局。

    Prompt：
      - CTX_START / CTX_END 各出现 1 次；
      - L1_START / L1_END / L2_START / L2_END 各出现 1 次用于角色绑定。

    完整样本：
      - 四个层级 Token 均各出现 2 次；
      - 第二次 occurrence 全部位于 Completion；
      - Completion 是一条连续链：
          L1_START + L1_LABEL + L1_END + L2_START + L2_LABEL + L2_END
      - L1_END 与 L2_START 必须在 token ID 序列中严格相邻；
      - L2_END 后由 Dataset 直接 append EOT，因此训练时 L2_END 直接预测 EOT。
    """
    messages = build_prompt_messages(
        text="probe check text",
        system_prompt=system_prompt,
    )

    _, prompt_ids = encode_chat_prompt_consistently(
        tokenizer=tokenizer,
        messages=messages,
        split_name="LAYOUT_CHECK",
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
                f"❌ Prompt 中 token_id={token_id} 应出现 "
                f"{expected_count} 次，实际 {actual_count} 次。"
            )

    l1_label = "CS"
    l2_label = "Machine learning"
    expected_completion = build_completion_text(l1_label, l2_label)
    completion_ids = tokenizer.encode(
        expected_completion,
        add_special_tokens=False,
    )
    completion_ids = [int(x) for x in completion_ids]
    full_ids = prompt_ids + completion_ids

    position_map = {
        "l1_start": find_all_token_positions(full_ids, token_ids["l1_start"]),
        "l1_end": find_all_token_positions(full_ids, token_ids["l1_end"]),
        "l2_start": find_all_token_positions(full_ids, token_ids["l2_start"]),
        "l2_end": find_all_token_positions(full_ids, token_ids["l2_end"]),
    }

    for role, positions in position_map.items():
        if len(positions) != 2:
            raise RuntimeError(
                f"❌ 完整样本中 {role} 应出现 2 次，实际 {positions}"
            )

    second = {role: positions[1] for role, positions in position_map.items()}

    if not (
        second["l1_start"]
        < second["l1_end"]
        < second["l2_start"]
        < second["l2_end"]
    ):
        raise RuntimeError(
            "❌ Completion 中层级 Token 顺序异常："
            f"{position_map}"
        )

    completion_start = len(prompt_ids)
    if any(pos < completion_start for pos in second.values()):
        raise RuntimeError(
            "❌ 某个层级 Token 的第二次 occurrence 没有落在 Completion 中。"
        )

    # 最重要的层间因果约束：L1_END 的下一个 token 必须就是 L2_START。
    if second["l2_start"] != second["l1_end"] + 1:
        between = full_ids[second["l1_end"] + 1 : second["l2_start"]]
        raise RuntimeError(
            "❌ L1_END 与 L2_START 之间存在额外 token，"
            "无法保证 L1_END 直接预测 L2_START。"
            f" between_ids={between}"
        )

    # START 后必须至少存在一个标签 token，END 前也必须有标签内容。
    if second["l1_end"] <= second["l1_start"] + 1:
        raise RuntimeError("❌ L1_START 与 L1_END 之间没有有效 L1 标签 token。")
    if second["l2_end"] <= second["l2_start"] + 1:
        raise RuntimeError("❌ L2_START 与 L2_END 之间没有有效 L2 标签 token。")

    # 检查 START 的 next token 确实属于标签编码，而不是空格/换行分隔符。
    expected_l1_label_ids = tokenizer.encode(l1_label, add_special_tokens=False)
    expected_l2_label_ids = tokenizer.encode(l2_label, add_special_tokens=False)
    if not expected_l1_label_ids or not expected_l2_label_ids:
        raise RuntimeError("❌ 测试标签编码为空，无法完成层级布局检查。")

    if full_ids[second["l1_start"] + 1] != expected_l1_label_ids[0]:
        raise RuntimeError(
            "❌ L1_START 后的下一个 token 不是 L1 标签首 token。"
        )
    if full_ids[second["l2_start"] + 1] != expected_l2_label_ids[0]:
        raise RuntimeError(
            "❌ L2_START 后的下一个 token 不是 L2 标签首 token。"
        )

    decoded_completion = tokenizer.decode(
        completion_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded_completion != expected_completion:
        raise RuntimeError(
            "❌ Completion 编码/解码后文本与预期不完全一致。\n"
            f"decoded={decoded_completion!r}\n"
            f"expected={expected_completion!r}"
        )

    # 再用严格答案解析器回读一次，确保无需换行也能无歧义提取。
    parsed_l1, parsed_l2 = parse_generation_exact(decoded_completion)
    if parsed_l1 != l1_label or parsed_l2 != l2_label:
        raise RuntimeError(
            "❌ 连续 Completion 的严格答案解析失败："
            f" parsed=({parsed_l1!r}, {parsed_l2!r})"
        )

    ctx_start_pos = full_ids.index(token_ids["ctx_start"])
    ctx_end_pos = full_ids.index(token_ids["ctx_end"])
    if ctx_end_pos <= ctx_start_pos + 1:
        raise RuntimeError("❌ 正文起止 Token 之间没有有效正文 token。")

    print("\n" + "=" * 70)
    print("✅ Prompt / Completion 特殊 Token 连续链布局检查通过")
    print(f"正文开始 Token 位置: {ctx_start_pos}")
    print(f"正文结束 Token 位置: {ctx_end_pos}")
    print(f"L1 START 两次位置: {position_map['l1_start']}")
    print(f"L1 END   两次位置: {position_map['l1_end']}")
    print(f"L2 START 两次位置: {position_map['l2_start']}")
    print(f"L2 END   两次位置: {position_map['l2_end']}")
    print("✅ L1_END 与 L2_START 在 Completion token 序列中严格相邻")
    print("✅ L1_START / L2_START 后直接进入对应标签 token")
    print("✅ 连续无换行 Completion 可由特殊 Token 边界严格解析")
    print(
        "后续层级探测/蒸馏：提取以上四个层级 Token "
        "在 Completion 中的第二次 occurrence。"
    )
    print("=" * 70 + "\n")

# ======================================================================
# 3. 严格读取最终 clean formatted 数据
#    只接受：
#      text
#      l1_label
#      l2_label
#      l1_id
#      l2_id
# ======================================================================
REQUIRED_CLEAN_FIELDS = (
    "text",
    "l1_label",
    "l2_label",
    "l1_id",
    "l2_id",
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
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"❌ JSONL 解析失败: {path}, line={line_no}: {e}"
                ) from e

            if not isinstance(item, dict):
                raise TypeError(
                    f"❌ JSONL 每行必须是 dict: {path}, line={line_no}, "
                    f"实际={type(item).__name__}"
                )
            rows.append(item)
        return rows

    if not isinstance(data, list):
        raise TypeError(
            f"❌ JSON 顶层必须是 list: {path}, "
            f"实际={type(data).__name__}"
        )

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(
                f"❌ JSON 第 {idx} 条必须是 dict: "
                f"实际={type(item).__name__}"
            )

    return data


def get_clean_sample_fields(
    item: dict,
    split_name: str,
    idx: int,
) -> Tuple[str, str, str, int, int]:
    """
    严格读取最终预处理输出，不再兼容旧字段，不再猜字段名。
    """
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

    if not isinstance(text, str):
        raise TypeError(
            f"{split_name} 第 {idx} 条 text 必须是 str，"
            f"实际={type(text).__name__}"
        )
    text = text.strip()
    if not text:
        raise ValueError(f"{split_name} 第 {idx} 条 text 为空。")

    if not isinstance(l1_label, str) or not l1_label.strip():
        raise TypeError(
            f"{split_name} 第 {idx} 条 l1_label 必须是非空 str。"
        )
    if not isinstance(l2_label, str) or not l2_label.strip():
        raise TypeError(
            f"{split_name} 第 {idx} 条 l2_label 必须是非空 str。"
        )

    l1_label = l1_label.strip()
    l2_label = l2_label.strip()

    # bool 是 int 的子类，因此显式排除。
    if isinstance(l1_id, bool) or not isinstance(l1_id, int):
        raise TypeError(
            f"{split_name} 第 {idx} 条 l1_id 必须是 int，"
            f"实际={l1_id!r}"
        )
    if isinstance(l2_id, bool) or not isinstance(l2_id, int):
        raise TypeError(
            f"{split_name} 第 {idx} 条 l2_id 必须是 int，"
            f"实际={l2_id!r}"
        )

    return text, l1_label, l2_label, l1_id, l2_id


# ======================================================================
# 4. taxonomy + label2id 严格一致性检查
# ======================================================================
def parse_wos_taxonomy(taxonomy_path: str):
    """
    与最终预处理脚本保持同一逻辑：
      Root<TAB>L1_1<TAB>L1_2...
      L1_1<TAB>L2_1<TAB>L2_2...
      ...

    返回：
      l1_labels：taxonomy 原始顺序
      l2_labels：按 L1 顺序拼接后的 taxonomy 原始顺序
      l2_to_l1：L2 -> L1
    """
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
                raise ValueError(
                    f"❌ taxonomy 中父节点重复出现: {parent!r}"
                )

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
            raise ValueError(
                f"❌ taxonomy 中 L1 {l1!r} 没有对应子节点行。"
            )

        for l2 in parent_to_children[l1]:
            if l2 in l2_to_l1:
                raise ValueError(
                    f"❌ L2 {l2!r} 重复归属于 "
                    f"{l2_to_l1[l2]!r} 和 {l1!r}"
                )
            l2_to_l1[l2] = l1
            l2_labels.append(l2)

    # 严格检查是否存在预期两级结构之外的额外 parent 行。
    expected_parents = {"Root"} | set(l1_labels)
    unexpected_parents = set(parent_to_children) - expected_parents
    if unexpected_parents:
        raise ValueError(
            "❌ taxonomy 存在两级 WOS 结构之外的额外父节点行: "
            f"{sorted(unexpected_parents)[:20]}"
        )

    overlap = set(l1_labels) & set(l2_labels)
    if overlap:
        raise ValueError(
            "❌ L1/L2 标签名称发生重叠: "
            f"{sorted(overlap)[:20]}"
        )

    return l1_labels, l2_labels, l2_to_l1


def _normalize_label2id_mapping(
    raw_mapping,
    name: str,
) -> Dict[str, int]:
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
    """
    最终训练直接使用 label2id.json 中的 ID，
    同时严格验证它与 taxonomy 顺序完全一致。

    这样 formatted JSONL 中的 l1_id/l2_id、
    label2id.json、taxonomy、训练 metric 四者不会发生错位。
    """
    if not os.path.isfile(dict_path):
        raise FileNotFoundError(f"❌ label2id 文件不存在: {dict_path}")

    with open(dict_path, "r", encoding="utf-8") as f:
        raw_dict = json.load(f)

    if not isinstance(raw_dict, dict):
        raise TypeError("❌ label2id.json 顶层必须是 dict。")

    for key in ("l1_label2id", "l2_label2id"):
        if key not in raw_dict:
            raise KeyError(
                f"❌ label2id.json 缺少必需字段 {key!r}。"
            )

    l1_label2id = _normalize_label2id_mapping(
        raw_dict["l1_label2id"],
        "l1_label2id",
    )
    l2_label2id = _normalize_label2id_mapping(
        raw_dict["l2_label2id"],
        "l2_label2id",
    )

    tax_l1, tax_l2, l2_to_l1 = parse_wos_taxonomy(taxonomy_path)

    if expected_num_l1 > 0 and len(tax_l1) != expected_num_l1:
        raise ValueError(
            f"❌ taxonomy L1 数量异常: "
            f"{len(tax_l1)} != {expected_num_l1}"
        )
    if expected_num_l2 > 0 and len(tax_l2) != expected_num_l2:
        raise ValueError(
            f"❌ taxonomy L2 数量异常: "
            f"{len(tax_l2)} != {expected_num_l2}"
        )

    expected_l1_label2id = {
        label: idx for idx, label in enumerate(tax_l1)
    }
    expected_l2_label2id = {
        label: idx for idx, label in enumerate(tax_l2)
    }

    if l1_label2id != expected_l1_label2id:
        missing = set(expected_l1_label2id) - set(l1_label2id)
        extra = set(l1_label2id) - set(expected_l1_label2id)
        wrong_ids = {
            label: (l1_label2id.get(label), expected_l1_label2id[label])
            for label in expected_l1_label2id
            if label in l1_label2id
            and l1_label2id[label] != expected_l1_label2id[label]
        }
        raise ValueError(
            "❌ label2id.json 的 L1 映射与 taxonomy 顺序不一致。\n"
            f"missing={sorted(missing)[:20]}\n"
            f"extra={sorted(extra)[:20]}\n"
            f"wrong_ids={dict(list(wrong_ids.items())[:20])}"
        )

    if l2_label2id != expected_l2_label2id:
        missing = set(expected_l2_label2id) - set(l2_label2id)
        extra = set(l2_label2id) - set(expected_l2_label2id)
        wrong_ids = {
            label: (l2_label2id.get(label), expected_l2_label2id[label])
            for label in expected_l2_label2id
            if label in l2_label2id
            and l2_label2id[label] != expected_l2_label2id[label]
        }
        raise ValueError(
            "❌ label2id.json 的 L2 映射与 taxonomy 顺序不一致。\n"
            f"missing={sorted(missing)[:20]}\n"
            f"extra={sorted(extra)[:20]}\n"
            f"wrong_ids={dict(list(wrong_ids.items())[:20])}"
        )

    # 如果 id2label 存在，则额外验证它与 label2id 完全互逆。
    if "l1_id2label" in raw_dict:
        expected = {str(v): k for k, v in l1_label2id.items()}
        actual = {
            str(k): str(v)
            for k, v in raw_dict["l1_id2label"].items()
        }
        if actual != expected:
            raise ValueError(
                "❌ label2id.json 的 l1_id2label "
                "与 l1_label2id 不互逆。"
            )

    if "l2_id2label" in raw_dict:
        expected = {str(v): k for k, v in l2_label2id.items()}
        actual = {
            str(k): str(v)
            for k, v in raw_dict["l2_id2label"].items()
        }
        if actual != expected:
            raise ValueError(
                "❌ label2id.json 的 l2_id2label "
                "与 l2_label2id 不互逆。"
            )

    print("\n" + "=" * 70)
    print("🏷️ 标签空间严格校验完成")
    print(f"L1: {len(l1_label2id)} 类")
    print(f"L2: {len(l2_label2id)} 类")
    print(f"总标签: {len(l1_label2id) + len(l2_label2id)} 类")
    print("✅ label2id.json 与 taxonomy 标签及 ID 顺序完全一致")
    print("metric 规则: L1 使用局部 ID；L2 计算全局指标时 + num_l1")
    print("=" * 70 + "\n")

    return l1_label2id, l2_label2id, l2_to_l1


def save_resolved_label_space(
    save_dir: str,
    l1_label2id: Dict[str, int],
    l2_label2id: Dict[str, int],
    l2_to_l1: Dict[str, str],
) -> None:
    payload = {
        "l1_label2id": l1_label2id,
        "l2_label2id": l2_label2id,
        "l1_id2label": {str(v): k for k, v in l1_label2id.items()},
        "l2_id2label": {str(v): k for k, v in l2_label2id.items()},
        "l2_parent": l2_to_l1,
        "metric_id_rule": (
            "L1 local ids [0,N1); "
            "L2 global metric ids = local_l2_id + N1"
        ),
    }

    path = os.path.join(save_dir, "resolved_label_space.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ 训练实际使用的标签空间已保存: {path}")


def validate_clean_split(
    data_path: str,
    split_name: str,
    l1_label2id: Dict[str, int],
    l2_label2id: Dict[str, int],
    l2_to_l1: Dict[str, str],
) -> List[dict]:
    """
    训练开始前一次性严格校验最终 clean JSONL：
      - 五个字段完整
      - text 非空
      - L1/L2 字符串合法
      - l1_id 与 l1_label2id 完全一致
      - l2_id 与 l2_label2id 完全一致
      - L1-L2 父子关系与 taxonomy 一致

    任何错误直接停止，不静默跳过。
    """
    data = load_json_or_jsonl(data_path)

    seen_l1 = set()
    seen_l2 = set()

    for idx, item in enumerate(
        tqdm(data, desc=f"校验{split_name}", leave=False)
    ):
        try:
            _, l1_label, l2_label, l1_id, l2_id = (
                get_clean_sample_fields(item, split_name, idx)
            )
        except Exception as e:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条样本格式错误: {e}\n"
                f"样本={item}"
            ) from e

        if l1_label not in l1_label2id:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条出现未知 L1: "
                f"{l1_label!r}"
            )
        if l2_label not in l2_label2id:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条出现未知 L2: "
                f"{l2_label!r}"
            )

        expected_l1_id = l1_label2id[l1_label]
        expected_l2_id = l2_label2id[l2_label]

        if l1_id != expected_l1_id:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条 L1 ID 错位: "
                f"label={l1_label!r}, sample_id={l1_id}, "
                f"label2id_id={expected_l1_id}"
            )

        if l2_id != expected_l2_id:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条 L2 ID 错位: "
                f"label={l2_label!r}, sample_id={l2_id}, "
                f"label2id_id={expected_l2_id}"
            )

        expected_parent = l2_to_l1.get(l2_label)
        if expected_parent is None:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条 L2 "
                f"不在 taxonomy 父子映射中: {l2_label!r}"
            )

        if expected_parent != l1_label:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条父子关系错误: "
                f"L1={l1_label!r}, L2={l2_label!r}, "
                f"taxonomy parent={expected_parent!r}"
            )

        seen_l1.add(l1_label)
        seen_l2.add(l2_label)

    print("\n" + "=" * 70)
    print(f"✅ {split_name} 严格校验通过")
    print(f"文件: {data_path}")
    print(f"样本数: {len(data)}")
    print(f"实际出现 L1: {len(seen_l1)} / {len(l1_label2id)}")
    print(f"实际出现 L2: {len(seen_l2)} / {len(l2_label2id)}")
    print("五字段缺失: 0")
    print("空正文: 0")
    print("未知标签: 0")
    print("标签-ID 错位: 0")
    print("父子关系错误: 0")
    print("静默跳过: 0")
    print("=" * 70 + "\n")

    return data


# ======================================================================
# 5. 截断、Completion 与 exact 解析
# ======================================================================
def truncate_text_for_prompt(
    text: str,
    tokenizer,
    max_seq_length: int = 2048,
    reserve_tokens: int = 300,
) -> str:
    """
    train / validation generation / validation loss
    共用同一截断逻辑。
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
    """稳定记录 token ID 序列，便于学生/教师/蒸馏逐样本核对。"""
    payload = ",".join(str(int(token_id)) for token_id in token_ids)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _text_sha256(text: str) -> str:
    """记录截断后正文文本；不同 tokenizer 之间也可先核对文本是否一致。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _decode_ids(tokenizer, token_ids: List[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _find_first_non_whitespace_token_offset(tokenizer, token_ids: List[int]):
    """仅用于打印正文首个非空白 token，不改变权威正文 span。"""
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
    token_ids: Dict[str, int],
    split_name: str,
    source_index: int,
    print_detail: bool,
    preview_chars: int,
) -> dict:
    """
    按 token ID 提取 ``CTX_START`` 与 ``CTX_END`` 之间的权威正文 span。

    严格要求边界内部 token IDs 与编码 ``"\n" + text + "\n"`` 完全一致。
    只要该检查通过，就不可能存在旧 ``<CTX>`` 拆分后遗留的 ``>``、
    额外边界碎片或正文首 token 黏连。
    """
    start_positions = find_all_token_positions(
        prompt_ids,
        token_ids["ctx_start"],
    )
    end_positions = find_all_token_positions(
        prompt_ids,
        token_ids["ctx_end"],
    )

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
        print(f"CTX_START 位置/ID: {start_pos}/{token_ids['ctx_start']}")
        print(f"CTX_END   位置/ID: {end_pos}/{token_ids['ctx_end']}")
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
            print(
                "从 CTX_START 到正文首个非空白 token 的联合解码: "
                f"{_decode_ids(tokenizer, prompt_ids[start_pos:first_non_ws_abs_pos + 1])!r}"
            )

        print("=" * 88 + "\n")

    return record


def audit_split_body_spans(
    data: List[dict],
    split_name: str,
    tokenizer,
    token_ids: Dict[str, int],
    max_seq_length: int,
    reserve_tokens: int,
    system_prompt: str,
    save_dir: str,
    print_samples: int,
    preview_chars: int,
    scan_limit: int = -1,
) -> str:
    """
    在训练启动前审计一个 split 的正文边界。默认 ``scan_limit=-1``
    表示扫描全部样本；只打印前 ``print_samples`` 条，全部结果写入 JSONL。
    """
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

    print("\n" + "=" * 88)
    print(
        f"🔎 开始 {split_name} 正文边界全量审计："
        f"扫描 {scan_count}/{total} 条"
    )
    print(
        "权威正文 span = prompt_ids[CTX_START位置 + 1 : CTX_END位置]"
    )
    print("=" * 88)

    for idx in tqdm(
        range(scan_count),
        desc=f"审计{split_name}正文边界",
        leave=False,
    ):
        raw_text, _, _, _, _ = get_clean_sample_fields(
            data[idx],
            split_name,
            idx,
        )
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
        )
        _, prompt_ids = encode_chat_prompt_consistently(
            tokenizer=tokenizer,
            messages=messages,
            split_name=split_name,
            source_index=idx,
        )

        if len(prompt_ids) > max_seq_length:
            raise ValueError(
                f"❌ {split_name} 第 {idx} 条 Prompt 超过 max_seq_length："
                f"{len(prompt_ids)} > {max_seq_length}。"
            )

        expected_counts = {
            token_ids["ctx_start"]: 1,
            token_ids["ctx_end"]: 1,
            token_ids["l1_start"]: 1,
            token_ids["l1_end"]: 1,
            token_ids["l2_start"]: 1,
            token_ids["l2_end"]: 1,
        }
        for token_id, expected_count in expected_counts.items():
            actual_count = prompt_ids.count(token_id)
            if actual_count != expected_count:
                raise RuntimeError(
                    f"❌ {split_name} 第 {idx} 条 Prompt 中 token_id={token_id} "
                    f"应出现 {expected_count} 次，实际 {actual_count} 次。"
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
    print("✅ chat template 直接编码与字符串重编码逐 token 完全一致")
    print("✅ 未发现额外 '>'、边界碎片或正文首 token 黏连")
    print(
        f"正文发生截断：{truncated_count}/{scan_count} "
        f"({truncated_count / max(1, scan_count):.2%})"
    )
    print(f"💾 审计日志: {output_path}")
    print("=" * 88 + "\n")

    return output_path


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


def parse_generation_exact(raw_resp: str) -> Tuple[str, str]:
    """
    严格解析连续层级链 Completion：

      <L1_START>L1_label<L1_END><L2_START>L2_label<L2_END>

    解析完全依赖四个不可分割的 Llama reserved special token 作为边界，
    不依赖换行，也不依赖层间空格。

    要求：
      - 四个层级 Token 各出现且只出现 1 次；
      - 顺序严格为 L1_START -> L1_END -> L2_START -> L2_END；
      - L1_END 与 L2_START 在文本中严格相邻；
      - L1/L2 标签均非空；
      - L2_END 后除停止 Token 外不能有任何额外正文；
      - 不 normalize、不模糊匹配、不候选纠错。
    """
    text = _strip_trailing_stop_strings(raw_resp)

    # 先做 occurrence 数量检查，避免模型重复生成特殊 token 时被误解析。
    required_tokens = (
        HIER_START_TOKEN_L1,
        HIER_END_TOKEN_L1,
        HIER_START_TOKEN_L2,
        HIER_END_TOKEN_L2,
    )
    if any(text.count(token) != 1 for token in required_tokens):
        return "", ""

    prefix = HIER_START_TOKEN_L1
    middle = HIER_END_TOKEN_L1 + HIER_START_TOKEN_L2
    suffix = HIER_END_TOKEN_L2

    if not text.startswith(prefix) or not text.endswith(suffix):
        return "", ""

    body = text[len(prefix) : -len(suffix)]
    if body.count(middle) != 1:
        return "", ""

    l1_label, l2_label = body.split(middle, 1)

    # 不允许靠普通分隔空白“修复”格式；标签本身可以包含内部空格。
    if not l1_label or not l2_label:
        return "", ""
    if l1_label != l1_label.strip() or l2_label != l2_label.strip():
        return "", ""
    if "\n" in l1_label or "\n" in l2_label or "\r" in l1_label or "\r" in l2_label:
        return "", ""

    return l1_label, l2_label

def compute_metric_bundle(
    true_l1: List[int],
    true_l2: List[int],
    pred_l1: List[int],
    pred_l2: List[int],
    num_l1: int,
    num_l2: int,
):
    """
    计算与 WOS 固定任务标签空间一致的指标。

    ID 规则：
      - L1 局部 ID: [0, num_l1)
      - L2 局部 ID: [0, num_l2)
      - Global 中 L2 ID = local_l2_id + num_l1

    非法预测仍使用 sentinel：
      - 非法 L1 = 9999
      - 非法 L2 在 Global 中 = 99999

    关键修正：
      Macro-F1 必须显式限定在真实任务标签空间内。
      否则 sklearn 会把预测中的 9999 / 99999 当成额外类别，
      从而错误拉低 Macro-F1。

    注意：
      - Micro-F1 保持原逻辑，不指定 labels；
        非法预测仍完整计为错误。
      - Macro-F1 固定只平均真实 WOS 类别；
        sentinel 不作为“额外类别”参与平均。
    """
    if not (
        len(true_l1)
        == len(true_l2)
        == len(pred_l1)
        == len(pred_l2)
    ):
        raise ValueError("❌ metric 输入长度不一致。")

    if num_l1 <= 0 or num_l2 <= 0:
        raise ValueError(
            f"❌ num_l1/num_l2 必须为正数，"
            f"实际 num_l1={num_l1}, num_l2={num_l2}"
        )

    all_true: List[int] = []
    all_pred: List[int] = []

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
                p2 + num_l1 if p2 != 9999 else 99999,
            ]
        )

    # --------------------------------------------------------------
    # Micro：保持旧逻辑。
    # sentinel 仍在 y_pred 中，因此非法预测被完整计错。
    # --------------------------------------------------------------
    global_micro = f1_score(
        all_true,
        all_pred,
        average="micro",
        zero_division=0,
    )

    # --------------------------------------------------------------
    # Macro：严格固定到真实 WOS 全部任务标签空间。
    # L1 = 0 ... num_l1-1
    # L2 = num_l1 ... num_l1+num_l2-1
    # 不让 9999 / 99999 变成额外“伪类别”参与平均。
    # --------------------------------------------------------------
    global_valid_labels = list(
        range(num_l1 + num_l2)
    )

    global_macro = f1_score(
        all_true,
        all_pred,
        labels=global_valid_labels,
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------------------
    # L1 / L2 诊断指标。
    # Micro 保持旧逻辑；Macro 固定到各层真实类别空间。
    # --------------------------------------------------------------
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
# 6. Dataset
# ======================================================================
class GenerativeWOSDataset(Dataset):
    def __init__(
        self,
        data: List[dict],
        tokenizer,
        max_length: int,
        reserve_tokens: int,
        system_prompt: str,
        split_name: str = "训练",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reserve_tokens = reserve_tokens
        self.system_prompt = system_prompt
        self.split_name = split_name

        self.ctx_start_id = tokenizer.convert_tokens_to_ids(CTX_START_TOKEN)
        self.ctx_end_id = tokenizer.convert_tokens_to_ids(CTX_END_TOKEN)
        self.l1_start_token_id = tokenizer.convert_tokens_to_ids(
            HIER_START_TOKEN_L1
        )
        self.l1_end_token_id = tokenizer.convert_tokens_to_ids(
            HIER_END_TOKEN_L1
        )
        self.l2_start_token_id = tokenizer.convert_tokens_to_ids(
            HIER_START_TOKEN_L2
        )
        self.l2_end_token_id = tokenizer.convert_tokens_to_ids(
            HIER_END_TOKEN_L2
        )

        print(f"✅ {self.split_name} Dataset 就绪：{len(self.data)} 条")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text, l1_str, l2_str, _, _ = get_clean_sample_fields(
            item,
            self.split_name,
            idx,
        )

        text = truncate_text_for_prompt(
            text,
            self.tokenizer,
            self.max_length,
            self.reserve_tokens,
        )

        messages = build_prompt_messages(
            text=text,
            system_prompt=self.system_prompt,
        )

        # 训练反向传播入口直接使用 tokenize=True，不会重复添加 BOS。
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_ids = [int(x) for x in prompt_ids]

        # Completion-only supervision：
        # 四个层级 Token 在这里第二次出现，并组成连续因果链：
        # L1_START + L1_LABEL + L1_END + L2_START + L2_LABEL + L2_END。
        # 这里不插入普通空格或换行分隔符。
        completion_text = build_completion_text(l1_str, l2_str)

        completion_ids = self.tokenizer.encode(
            completion_text,
            add_special_tokens=False,
        )

        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is None or eot_id == self.tokenizer.unk_token_id:
            eot_id = self.tokenizer.eos_token_id

        if eot_id is None:
            raise RuntimeError(
                "❌ tokenizer 同时缺少 <|eot_id|> 和 eos_token_id。"
            )

        completion_ids.append(eot_id)

        input_ids = prompt_ids + completion_ids
        labels_ids = [-100] * len(prompt_ids) + completion_ids

        # 严格保证正文边界各 1 次；
        # 四个层级 START/END Token 各出现 2 次（Prompt 1 次 + Completion 1 次）。
        if input_ids.count(self.ctx_start_id) != 1:
            raise RuntimeError(
                f"❌ {self.split_name} 第 {idx} 条 CTX_START 出现次数异常。"
            )
        if input_ids.count(self.ctx_end_id) != 1:
            raise RuntimeError(
                f"❌ {self.split_name} 第 {idx} 条 CTX_END 出现次数异常。"
            )

        hierarchy_token_checks = {
            "L1_START": self.l1_start_token_id,
            "L1_END": self.l1_end_token_id,
            "L2_START": self.l2_start_token_id,
            "L2_END": self.l2_end_token_id,
        }
        for role, token_id in hierarchy_token_checks.items():
            positions = find_all_token_positions(input_ids, token_id)
            if len(positions) != 2:
                raise RuntimeError(
                    f"❌ {self.split_name} 第 {idx} 条 {role} "
                    f"应出现 2 次，实际 positions={positions}"
                )

        # 第二次 occurrence 必须全部位于 Completion，并满足边界顺序。
        second_positions = {
            role: find_all_token_positions(input_ids, token_id)[1]
            for role, token_id in hierarchy_token_checks.items()
        }
        if not (
            second_positions["L1_START"]
            < second_positions["L1_END"]
            < second_positions["L2_START"]
            < second_positions["L2_END"]
        ):
            raise RuntimeError(
                f"❌ {self.split_name} 第 {idx} 条 Completion "
                f"层级 Token 顺序异常: {second_positions}"
            )
        if any(
            pos < len(prompt_ids)
            for pos in second_positions.values()
        ):
            raise RuntimeError(
                f"❌ {self.split_name} 第 {idx} 条某个层级 Token "
                "第二次 occurrence 未落在 Completion 中。"
            )

        # L1_END 必须直接预测 L2_START：两者在 token 序列中严格相邻。
        if second_positions["L2_START"] != second_positions["L1_END"] + 1:
            between_ids = input_ids[
                second_positions["L1_END"] + 1 : second_positions["L2_START"]
            ]
            raise RuntimeError(
                f"❌ {self.split_name} 第 {idx} 条 L1_END 与 L2_START "
                f"之间出现额外 token: {between_ids}"
            )

        # START 后必须直接进入当前层级标签的第一个 token。
        l1_label_ids = self.tokenizer.encode(l1_str, add_special_tokens=False)
        l2_label_ids = self.tokenizer.encode(l2_str, add_special_tokens=False)
        if not l1_label_ids or not l2_label_ids:
            raise RuntimeError(
                f"❌ {self.split_name} 第 {idx} 条标签编码为空。"
            )
        if input_ids[second_positions["L1_START"] + 1] != l1_label_ids[0]:
            raise RuntimeError(
                f"❌ {self.split_name} 第 {idx} 条 L1_START 后不是 L1 标签首 token。"
            )
        if input_ids[second_positions["L2_START"] + 1] != l2_label_ids[0]:
            raise RuntimeError(
                f"❌ {self.split_name} 第 {idx} 条 L2_START 后不是 L2 标签首 token。"
            )

        # L2_END 后必须直接是训练停止 token，使其 hidden state 直接预测 EOT/EOS。
        if second_positions["L2_END"] + 1 >= len(input_ids):
            raise RuntimeError(
                f"❌ {self.split_name} 第 {idx} 条 L2_END 后缺少 EOT/EOS。"
            )
        if input_ids[second_positions["L2_END"] + 1] != eot_id:
            raise RuntimeError(
                f"❌ {self.split_name} 第 {idx} 条 L2_END 后不是 EOT/EOS。"
            )

        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            raise ValueError(
                f"❌ {self.split_name} 第 {idx} 条样本仍超长 "
                f"{overflow} tokens。\n"
                "请增大 --reserve_tokens，或减小正文预算。"
            )

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels_ids,
        }


# ======================================================================
# 7. 生成式验证 Callback
# ======================================================================
class GenerativeF1EvalCallback(TrainerCallback):
    def __init__(
        self,
        val_data: List[dict],
        l1_label2id: Dict[str, int],
        l2_label2id: Dict[str, int],
        l2_to_l1: Dict[str, str],
        tokenizer,
        max_seq_length: int = 2048,
        reserve_tokens: int = 300,
        system_prompt: str = (
            "You are a specialized academic classifier. "
            "Do NOT produce extra conversational text."
        ),
        eval_steps: int = 500,
        eval_batch_size: int = 16,
        loss_eval_batch_size: int = 4,
        max_new_tokens: int = 32,
        save_dir: str = "",
        resume_checkpoint: Optional[str] = None,
        run_signature: Optional[dict] = None,
    ):
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size
        self.loss_eval_batch_size = loss_eval_batch_size
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_length
        self.reserve_tokens = reserve_tokens
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        self.resume_checkpoint = resume_checkpoint
        self.run_signature = run_signature or {}
        self.history_records = []
        if self.eval_steps <= 0:
            raise ValueError("eval_steps 必须大于 0。")

        self.l1_label2id = l1_label2id
        self.l2_label2id = l2_label2id
        self.l2_to_l1 = l2_to_l1
        self.num_l1 = len(l1_label2id)
        self.num_l2 = len(l2_label2id)

        self.best_ma = -1.0
        self.best_mi = -1.0

        # 记录最后一次已经完成验证的 step，避免最终步与常规验证重复。
        self.last_evaluated_step = -1
        self._force_final_eval = False

        self.history_path = os.path.join(
            save_dir,
            "eval_history.jsonl",
        )

        if not resume_checkpoint and os.path.isfile(self.history_path) and os.path.getsize(self.history_path):
            raise RuntimeError("save_dir 已有验证历史，请使用 --resume_from_checkpoint auto 或新目录。")

        # 与训练相同的 completion-only loss 数据集。
        self.loss_eval_dataset = GenerativeWOSDataset(
            data=val_data,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            reserve_tokens=reserve_tokens,
            system_prompt=system_prompt,
            split_name="验证loss",
        )

        # 生成式验证 Prompt：运行时由 clean text 构造。
        self.val_prompts = []

        for idx, item in enumerate(val_data):
            text, l1_str, l2_str, l1_id, l2_id = (
                get_clean_sample_fields(
                    item,
                    "VALIDATION",
                    idx,
                )
            )

            text = truncate_text_for_prompt(
                text,
                tokenizer,
                self.max_seq_length,
                self.reserve_tokens,
            )

            messages = build_prompt_messages(
                text=text,
                system_prompt=self.system_prompt,
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

            self.val_prompts.append(
                {
                    "prompt": full_prompt_str,
                    "prompt_ids": full_prompt_ids,
                    # 直接使用已严格校验过的 clean local ID。
                    "true_l1": l1_id,
                    "true_l2": l2_id,
                    "true_l1_label": l1_str,
                    "true_l2_label": l2_str,
                }
            )

        print(
            f"✅ 验证生成 Prompt 准备完成："
            f"{len(self.val_prompts)} 条"
        )

    def _state_dict(self):
        return {"best_ma": self.best_ma, "best_mi": self.best_mi,
                "last_evaluated_step": self.last_evaluated_step}

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
                self.best_ma = float(best["global_macro"]) if best else -1.0
                self.best_mi = float(best["global_micro"]) if best else -1.0
                self.last_evaluated_step = int(records[-1]["step"]) if records else -1
            previous = -1
            for record in records:
                current = int(record["step"])
                if current <= previous or current > step:
                    raise RuntimeError("WOS 验证历史与断点步数不一致。")
                previous = current
            if self.last_evaluated_step != (int(records[-1]["step"]) if records else -1):
                raise RuntimeError("WOS 回调验证步数与历史不一致。")
            allow_legacy_best = legacy and (not full_legacy_history or int(full_legacy_history[-1]["step"]) <= step)
            ensure_wos_historical_best_available(
                self.save_dir, records, allow_legacy_best
            )
            write_history_snapshot(self.history_path, records)
            archive_future_checkpoints(self.save_dir, step)
            self.history_records = records
            print(f"✅ 已恢复 WOS step={step}, last_eval={self.last_evaluated_step}, best_Macro={self.best_ma:.6f}")
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
                    checkpoint, int(args.world_size),
                ),
            })
            pending = os.path.join(checkpoint, WOS_PENDING_NAME)
            if os.path.isfile(pending):
                os.remove(pending)
            print(f"✅ 完整 WOS 续训断点保存完成: {checkpoint}")

        self._root_action(state, save)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step <= 0 or step == self.last_evaluated_step or (not self._force_final_eval and step % self.eval_steps != 0):
            return control

        def evaluate():
            eval_kwargs = dict(kwargs)
            if eval_kwargs.get("model") is None:
                raise RuntimeError("WOS 验证回调未取得模型。")
            eval_kwargs["model"] = unwrap_training_model(eval_kwargs["model"])
            self._evaluate_at_step(args, state, control, **eval_kwargs)
            return {"callback_state": self._state_dict(), "should_training_stop": bool(control.should_training_stop)}

        payload = self._root_action(state, evaluate)
        self._restore_state(payload["callback_state"])
        control.should_training_stop = payload["should_training_stop"]
        return control


    def _compute_eval_loss(self, model, device) -> float:
        """
        teacher-forcing 验证 loss：
          - completion-only labels
          - right padding
          - 不生成
        """
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

        collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=model,
            label_pad_token_id=-100,
            padding=True,
        )

        dataloader = DataLoader(
            self.loss_eval_dataset,
            batch_size=self.loss_eval_batch_size,
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
                    dataloader,
                    desc="Eval Loss",
                    leave=False,
                ):
                    batch = {
                        k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in batch.items()
                    }

                    outputs = model(**batch)
                    bs = batch["input_ids"].shape[0]

                    total_loss += (
                        outputs.loss.detach().float().item() * bs
                    )
                    total_count += bs
        finally:
            self.tokenizer.padding_side = old_padding_side
            model.config.use_cache = old_config_use_cache

            if old_gen_use_cache is not None:
                model.generation_config.use_cache = old_gen_use_cache

        return total_loss / max(1, total_count)

    def _append_history(self, payload: dict) -> None:
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(payload, ensure_ascii=False) + "\n"
            )
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

        # 当总步数恰好是 eval_steps 的整数倍时，on_step_end 已验证过，
        # on_train_end 不再重复验证同一个 step。
        if int(state.global_step) == self.last_evaluated_step:
            return control

        eval_trigger = "最终步" if force_final_eval else "定期"
        print(
            f"\n[Step {state.global_step}] "
            f"开始{eval_trigger}严格生成式验证..."
        )

        was_training = model.training
        model.eval()
        device = next(model.parameters()).device

        try:
            eval_loss = self._compute_eval_loss(model, device)

            print(
                f"[Step {state.global_step}] "
                f"Eval Loss: {eval_loss:.6f}"
            )

            true_l1: List[int] = []
            true_l2: List[int] = []
            pred_l1: List[int] = []
            pred_l2: List[int] = []

            exact_parse_count = 0
            valid_label_pair_count = 0
            hierarchy_consistent_count = 0

            eot_id = self.tokenizer.convert_tokens_to_ids(
                "<|eot_id|>"
            )
            if (
                eot_id is None
                or eot_id == self.tokenizer.unk_token_id
            ):
                eot_id = self.tokenizer.eos_token_id

            stop_ids = []
            for sid in [
                self.tokenizer.eos_token_id,
                eot_id,
            ]:
                if sid is not None and sid not in stop_ids:
                    stop_ids.append(sid)

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
                with torch.no_grad():
                    for i in tqdm(
                        range(
                            0,
                            len(self.val_prompts),
                            self.eval_batch_size,
                        ),
                        desc=f"Step {state.global_step} Eval",
                    ):
                        batch = self.val_prompts[
                            i : i + self.eval_batch_size
                        ]
                        prompts = [x["prompt"] for x in batch]

                        # prompts 已由 chat template 写入 BOS/EOT/header。
                        # 这里必须关闭 add_special_tokens，避免再次添加 BOS。
                        inputs = self.tokenizer(
                            prompts,
                            return_tensors="pt",
                            padding=True,
                            truncation=False,
                            add_special_tokens=False,
                        ).to(device)

                        # 去掉 left padding 后，实际送入 generate 的每条 Prompt
                        # 必须与 apply_chat_template(tokenize=True) 完全一致。
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
                                    "❌ 生成式验证批量编码后的 Prompt 与训练入口不一致："
                                    f"batch_row={row_idx}, "
                                    f"actual_len={len(actual_prompt_ids)}, "
                                    f"expected_len={len(expected_prompt_ids)}"
                                )

                        input_len = inputs.input_ids.shape[1]

                        output_ids = model.generate(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_new_tokens=self.max_new_tokens,
                            eos_token_id=stop_ids,
                            pad_token_id=self.tokenizer.pad_token_id,
                            do_sample=False,
                            use_cache=True,
                        )

                        generated_ids = output_ids[:, input_len:]

                        # 层级 Token 本身属于 special token，不能跳过。
                        raw_texts = self.tokenizer.batch_decode(
                            generated_ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )

                        for raw_resp, item in zip(
                            raw_texts,
                            batch,
                        ):
                            p1_str, p2_str = parse_generation_exact(
                                raw_resp
                            )

                            if p1_str and p2_str:
                                exact_parse_count += 1

                            p1 = self.l1_label2id.get(
                                p1_str,
                                9999,
                            )
                            p2 = self.l2_label2id.get(
                                p2_str,
                                9999,
                            )

                            if p1 != 9999 and p2 != 9999:
                                valid_label_pair_count += 1

                                if (
                                    self.l2_to_l1.get(p2_str)
                                    == p1_str
                                ):
                                    hierarchy_consistent_count += 1

                            pred_l1.append(p1)
                            pred_l2.append(p2)
                            true_l1.append(item["true_l1"])
                            true_l2.append(item["true_l2"])
            finally:
                self.tokenizer.padding_side = old_padding_side
                model.config.use_cache = old_config_use_cache

                if old_gen_use_cache is not None:
                    model.generation_config.use_cache = (
                        old_gen_use_cache
                    )

            metrics = compute_metric_bundle(
                true_l1=true_l1,
                true_l2=true_l2,
                pred_l1=pred_l1,
                pred_l2=pred_l2,
                num_l1=self.num_l1,
                num_l2=self.num_l2,
            )

            n = max(1, len(true_l1))
            exact_parse_rate = exact_parse_count / n
            valid_label_pair_rate = valid_label_pair_count / n
            hierarchy_consistency_rate = (
                hierarchy_consistent_count / n
            )

            print(
                f"[Step {state.global_step}] "
                f"Global Micro-F1: "
                f"{metrics['global_micro']:.4f} | "
                f"Global Macro-F1: "
                f"{metrics['global_macro']:.4f}"
            )

            print(
                f"[Step {state.global_step}] "
                f"L1 Micro/Macro: "
                f"{metrics['l1_micro']:.4f}/"
                f"{metrics['l1_macro']:.4f} | "
                f"L2 Micro/Macro: "
                f"{metrics['l2_micro']:.4f}/"
                f"{metrics['l2_macro']:.4f}"
            )

            print(
                f"[Step {state.global_step}] "
                f"ExactParse={exact_parse_rate:.4f} | "
                f"ValidLabelPair={valid_label_pair_rate:.4f} | "
                f"HierConsistent="
                f"{hierarchy_consistency_rate:.4f}"
            )

            log_payload = {
                "Eval/Loss": eval_loss,
                "Eval/Micro_F1": metrics["global_micro"],
                "Eval/Macro_F1": metrics["global_macro"],
                "Eval/Global_Micro_F1": metrics["global_micro"],
                "Eval/Global_Macro_F1": metrics["global_macro"],
                "Eval/L1_Micro_F1": metrics["l1_micro"],
                "Eval/L1_Macro_F1": metrics["l1_macro"],
                "Eval/L2_Micro_F1": metrics["l2_micro"],
                "Eval/L2_Macro_F1": metrics["l2_macro"],
                "Eval/Exact_Parse_Rate": exact_parse_rate,
                "Eval/Valid_Label_Pair_Rate": valid_label_pair_rate,
                "Eval/Hierarchy_Consistency_Rate": (
                    hierarchy_consistency_rate
                ),
            }

            swanlab.log(
                log_payload,
                step=state.global_step,
            )

            history_payload = {
                "step": int(state.global_step),
                "eval_loss": float(eval_loss),
                **metrics,
                "exact_parse_rate": float(exact_parse_rate),
                "valid_label_pair_rate": float(
                    valid_label_pair_rate
                ),
                "hierarchy_consistency_rate": float(
                    hierarchy_consistency_rate
                ),
            }

            self._append_history(history_payload)

            current_ma = metrics["global_macro"]
            current_mi = metrics["global_micro"]

            is_better = (
                current_ma > self.best_ma
                or (
                    abs(current_ma - self.best_ma) < 1e-12
                    and current_mi > self.best_mi
                )
            )

            if is_better:
                self.best_ma = current_ma
                self.best_mi = current_mi

                best_checkpoint_path = os.path.join(
                    self.save_dir,
                    "best_sft_checkpoint",
                )

                print(
                    f"🏆 新 best："
                    f"Global Macro-F1={current_ma:.4f}, "
                    f"Micro-F1={current_mi:.4f}\n"
                    f"保存到 {best_checkpoint_path}"
                )

                save_peft_checkpoint_with_full_io(
                    model=model,
                    tokenizer=self.tokenizer,
                    output_dir=best_checkpoint_path,
                    overwrite=True,
                    checkpoint_metadata=history_payload,
                )

            # 只有整轮验证和 best 判断均成功完成后才记录。
            self.last_evaluated_step = int(state.global_step)

        finally:
            if was_training:
                model.train()

        return control

    def on_train_end(self, args, state, control, **kwargs):
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
                os.path.join(output_dir, WOS_PENDING_NAME),
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

    args.save_dir = os.path.abspath(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    resume_checkpoint = resolve_resume_checkpoint(args)
    run_signature = build_run_signature(args)
    if resume_checkpoint:
        validate_resume_configuration(resume_checkpoint, run_signature)
        print(f"🔄 将从完整 WOS 学生训练断点恢复: {resume_checkpoint}")
    else:
        print("🆕 开始新的 WOS 学生 SFT 实验。")

    # ------------------------------------------------------------------
    # A. taxonomy + label2id + clean split 严格校验
    # ------------------------------------------------------------------
    l1_label2id, l2_label2id, l2_to_l1 = (
        load_and_validate_label_space(
            dict_path=args.dict_path,
            taxonomy_path=args.taxonomy_path,
            expected_num_l1=args.expected_num_l1,
            expected_num_l2=args.expected_num_l2,
        )
    )

    save_resolved_label_space(
        save_dir=args.save_dir,
        l1_label2id=l1_label2id,
        l2_label2id=l2_label2id,
        l2_to_l1=l2_to_l1,
    )

    train_data = validate_clean_split(
        data_path=args.train_data_path,
        split_name="TRAIN",
        l1_label2id=l1_label2id,
        l2_label2id=l2_label2id,
        l2_to_l1=l2_to_l1,
    )

    val_data = validate_clean_split(
        data_path=args.val_data_path,
        split_name="VALIDATION",
        l1_label2id=l1_label2id,
        l2_label2id=l2_label2id,
        l2_to_l1=l2_to_l1,
    )

    # ------------------------------------------------------------------
    # B. Tokenizer + 六个 Llama 原生预留特殊 Token
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 与旧学生脚本保持一致的 Llama-3 chat template。
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

    token_ids = validate_atomic_reserved_tokens(tokenizer)
    check_reserved_token_layout(
        tokenizer=tokenizer,
        system_prompt=args.system_prompt,
        token_ids=token_ids,
    )
    save_reserved_token_info(
        save_dir=args.save_dir,
        tokenizer=tokenizer,
        token_ids=token_ids,
    )

    # ------------------------------------------------------------------
    # C. 训练前严格审计 TRAIN / VALIDATION 正文边界
    # ------------------------------------------------------------------
    train_body_audit_path = audit_split_body_spans(
        data=train_data,
        split_name="TRAIN",
        tokenizer=tokenizer,
        token_ids=token_ids,
        max_seq_length=args.max_seq_length,
        reserve_tokens=args.reserve_tokens,
        system_prompt=args.system_prompt,
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
        save_dir=args.save_dir,
        print_samples=args.body_audit_print_samples,
        preview_chars=args.body_audit_preview_chars,
        scan_limit=args.body_audit_scan_limit,
    )
    print(f"✅ TRAIN 正文审计日志: {train_body_audit_path}")
    print(f"✅ VALIDATION 正文审计日志: {val_body_audit_path}")

    # ------------------------------------------------------------------
    # D. Dataset：Prompt 首次绑定 + Completion 连续层级因果链
    # ------------------------------------------------------------------
    train_dataset = GenerativeWOSDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        reserve_tokens=args.reserve_tokens,
        system_prompt=args.system_prompt,
        split_name="训练",
    )

    # ------------------------------------------------------------------
    # D. 加载学生模型：不新增 token、不 resize、不主动解绑
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

    model.config.use_cache = False

    # ------------------------------------------------------------------
    # E. LoRA + 完整 input embedding/lm_head 训练
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

    print("\n" + "=" * 70)
    print("🎯 最终学生训练范围")
    print(f"CTX_START ID: {token_ids['ctx_start']}")
    print(f"CTX_END ID:   {token_ids['ctx_end']}")
    print(f"L1 START ID:  {token_ids['l1_start']}")
    print(f"L1 END ID:    {token_ids['l1_end']}")
    print(f"L2 START ID:  {token_ids['l2_start']}")
    print(f"L2 END ID:    {token_ids['l2_end']}")
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
    print("Prompt: L1/L2 各一对 START/END Token，第一次出现用于角色绑定")
    print("Completion: L1_START+L1+L1_END+L2_START+L2+L2_END 连续无分隔链")
    print("因果约束: L1_END 直接预测 L2_START；L2_END 直接预测 EOT")
    print("正文边界: 两个不可分割的 Llama 预留特殊 Token")
    print(
        "后续探测/蒸馏: 四个层级 START/END Token 都提取 Completion 中 "
        "第二次 occurrence 的最后一层 hidden state"
    )
    print("=" * 70 + "\n")

    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # F. TrainingArguments
    # ------------------------------------------------------------------
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

    f1_eval_callback = GenerativeF1EvalCallback(
        val_data=val_data,
        l1_label2id=l1_label2id,
        l2_label2id=l2_label2id,
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

    print("\n" + "=" * 70)
    print("🚀 开始 WOS 学生模型纯生成式 SFT")
    print(f"Train: {args.train_data_path}")
    print(f"Val:   {args.val_data_path}")
    print(f"Tax:   {args.taxonomy_path}")
    print(f"Dict:  {args.dict_path}")
    print(
        f"LoRA LR: {args.lr} | "
        f"Embedding/lm_head LR: {args.io_lr} | "
        f"scheduler: {args.lr_scheduler_type} | "
        f"warmup_ratio: {args.warmup_ratio}"
    )
    print(
        f"周期 checkpoint 最多保留最近 {args.save_total_limit} 个；"
        "训练结束后仍会另外保存 final/best。"
    )
    print(
        "验证规则：每 eval_steps 验证一次；"
        "若最后一步不是整除节点，训练结束时自动补做最终步验证。"
    )
    print(
        f"Epoch: {args.num_epochs} | batch: {args.batch_size} | "
        f"accum: {args.accumulation_steps}"
    )
    print("=" * 70 + "\n")

    if trainer.is_world_process_zero():
        config_path = os.path.join(args.save_dir, "wos_paired_student_training_config.json")
        if os.path.isfile(config_path):
            shutil.copy2(config_path, config_path + f".before-resume-{time.time_ns()}.bak")
        atomic_write_json(config_path, {
            "run_signature": run_signature, "arguments": vars(args),
            "resolved_resume_checkpoint": resume_checkpoint,
        })
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # ------------------------------------------------------------------
    # G. 保存 final/best checkpoint，并可导出完整合并模型
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

        best_checkpoint_path = os.path.join(
            args.save_dir,
            "best_sft_checkpoint",
        )

        if not os.path.isdir(best_checkpoint_path):
            print(
                "⚠️ 训练期间没有触发 best 保存，"
                "将 final checkpoint 同步保存为 best。"
            )
            save_peft_checkpoint_with_full_io(
                model=model,
                tokenizer=tokenizer,
                output_dir=best_checkpoint_path,
                overwrite=True,
            )

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

    raise argparse.ArgumentTypeError(
        "Boolean value expected."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "WOS Student Pure Generative SFT: "
            "clean formatted data + atomic reserved context boundaries "
            "+ contiguous hierarchy START/END causal chain + full input/output tuning"
        )
    )

    # ================= 路径参数 =================
    parser.add_argument("--model_path", type=str, default="/data1/lzp/code/llama-3.2_1b/")
    parser.add_argument("--train_data_path", type=str, default="/data1/lzp/code/data/wos/train_formatted.jsonl")
    parser.add_argument("--val_data_path", type=str, default="/data1/lzp/code/data/wos/val_formatted.jsonl")
    parser.add_argument("--taxonomy_path", type=str, default="/data1/lzp/code/data/wos/wos.taxnomy")
    parser.add_argument("--dict_path", type=str, default="/data1/lzp/code/data/wos/label2id.json")
    parser.add_argument("--save_dir", type=str, default="/data1/lzp/code/hierarchical_distill/checkpoints_student_wos_2")
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="不传则新实验；auto/true/latest 恢复最近完整周期断点；也可指定原 save_dir/checkpoints/checkpoint-N。",
    )

    # ================= 数据与随机种子 =================
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_num_l1", type=int, default=7)
    parser.add_argument("--expected_num_l2", type=int, default=134)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--reserve_tokens", type=int, default=300)

    # ================= 正文边界审计 =================
    # 默认扫描 TRAIN 和 VALIDATION 的全部样本；只打印每个 split 的前 5 条。
    parser.add_argument("--body_audit_print_samples", type=int, default=5)
    parser.add_argument("--body_audit_preview_chars", type=int, default=300)
    parser.add_argument("--body_audit_scan_limit", type=int, default=-1)

    # ================= 核心训练参数 =================
    parser.add_argument("--num_epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    # LoRA 保持原学习率；完整 input embedding/lm_head 使用更小学习率。
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--io_lr", type=float, default=1.0e-5)

    # 不做 warmup，不用 cosine。
    # linear = 从初始学习率逐步线性衰减到 0。
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)

    # ================= LoRA 参数 =================
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # ================= 验证与保存参数 =================
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--loss_eval_batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--save_total_limit", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--export_merged_model", type=str2bool, default=True)

    # ================= 性能参数 =================
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dataloader_pin_memory", type=str2bool, default=True)
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")

    # ================= Prompt / SwanLab =================
    parser.add_argument("--system_prompt", type=str, default="You are a specialized academic classifier for the WOS two-level taxonomy. Follow the required output format exactly. Do not produce explanations or any extra conversational text.")
    parser.add_argument("--project_name", type=str, default="WOS-Generative-Student")
    parser.add_argument("--run_name", type=str, default="Llama3.2-1B-WOS-ContiguousPairedHierarchyTokens-FullIO-GroupLR-IO1e-5-Seed42-Lora16")

    args = parser.parse_args()
    main(args)

