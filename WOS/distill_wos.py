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
import torch.nn as nn
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

# 层级 Token：Prompt 中第一次出现用于角色绑定；
# Completion 中第二次出现，并直接位于对应标签之前。
PROBE_TOKEN_L1 = "<|reserved_special_token_112|>"
PROBE_TOKEN_L2 = "<|reserved_special_token_113|>"

FULL_IO_WEIGHTS_NAME = "full_input_output_embeddings.pt"
FULL_IO_META_NAME = "full_input_output_embeddings_meta.json"
HIDDEN_PROJECTOR_WEIGHTS_NAME = "hidden_state_projector.pt"
HIDDEN_PROJECTOR_META_NAME = "hidden_state_projector_meta.json"
WOS_RESUME_STATE_NAME = "wos_resume_state.json"
WOS_PENDING_NAME = "wos_save_in_progress.json"
KD_TRAINER_STATE_PREFIX = "wos_kd_trainer_state_rank_"
WOS_METER_FIELDS = (
    "_loss_meter_task", "_loss_meter_kd", "_loss_meter_total", "_loss_meter_count",
    "_meter_align_l1_mse", "_meter_align_l1_cos", "_meter_align_l1",
    "_meter_align_l2_mse", "_meter_align_l2_cos", "_meter_align_l2",
    "_meter_align", "_meter_hier",
)


def unwrap_training_model(model):
    """兼容 DDP 等外层包装，返回实际模型对象。"""
    while hasattr(model, "module"):
        model = model.module
    return model


def validate_atomic_reserved_tokens(tokenizer) -> Dict[str, int]:
    """
    严格验证四个 Llama 预留特殊 Token：
      - 已存在于 tokenizer 词表；
      - add_special_tokens=False 时均严格编码为单 token；
      - 不等于 unk；
      - 四个 token ID 两两不同。

    本脚本不新增 token，也不 resize embedding。
    """
    token_map = {
        "ctx_start": CTX_START_TOKEN,
        "ctx_end": CTX_END_TOKEN,
        "l1": PROBE_TOKEN_L1,
        "l2": PROBE_TOKEN_L2,
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
            f"❌ 四个预留特殊 Token 的 ID 未两两区分：{token_ids}"
        )

    print("✅ 四个预留特殊 Token 均为不可分割的原子 token")
    print("=" * 70 + "\n")
    return token_ids



def build_teacher_student_vocab_alignment(
    student_tokenizer,
    teacher_tokenizer,
) -> dict:
    """
    为 Llama-3.1 Teacher / Llama-3.2 Student 建立按“token 字符串语义”对齐的词表映射。

    重要：
      - 不再要求 Teacher / Student 的 token ID 数字完全相同；
      - 只要 token 字符串相同，就认为是同一个词表语义；
      - 传统 KD 只在两边共同 token 集合上计算 KL；
      - Student-only / Teacher-only token 不参与 KD；
      - WOS 必需的 4 个预留 token 被设为 protected tokens，
        只要任意一个不同时存在于两边，程序立即停止。

    返回：
      student_common_ids: 共同 token 按 Student ID 排序后的 Student IDs
      teacher_common_ids: 与上面逐项语义对应的 Teacher IDs
      student_to_teacher: 长度为 len(student_tokenizer) 的 ID 翻译表，
                          common token: student_id -> teacher_id；
                          student-only token: -1
    """
    print("\n" + "=" * 88)
    print("🔎 建立 Teacher / Student 按 token 字符串对齐的共同词表...")

    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()

    student_tokens = set(student_vocab)
    teacher_tokens = set(teacher_vocab)
    common_tokens = student_tokens & teacher_tokens
    student_only = student_tokens - teacher_tokens
    teacher_only = teacher_tokens - student_tokens

    if not common_tokens:
        raise RuntimeError("❌ Teacher / Student 没有任何共同 token，无法做 logits KD。")

    protected_tokens = [
        CTX_START_TOKEN,
        CTX_END_TOKEN,
        PROBE_TOKEN_L1,
        PROBE_TOKEN_L2,
    ]

    # Chat template / completion 中同样必需。它们不是你的层级创新 token，
    # 但如果缺失，Teacher 输入语义会错，因此也进行强制检查。
    required_chat_tokens = [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ]

    print("\n🔐 WOS 四个 protected token 跨模型检查")
    protected_info = {}
    for token in protected_tokens:
        s_ids = student_tokenizer.encode(token, add_special_tokens=False)
        t_ids = teacher_tokenizer.encode(token, add_special_tokens=False)

        if token not in student_vocab:
            raise RuntimeError(f"❌ Student 缺少 protected token: {token}")
        if token not in teacher_vocab:
            raise RuntimeError(f"❌ Teacher 缺少 protected token: {token}")
        if len(s_ids) != 1 or len(t_ids) != 1:
            raise RuntimeError(
                f"❌ protected token 必须在两边都严格编码为单 token: {token}; "
                f"student_ids={s_ids}, teacher_ids={t_ids}"
            )

        sid = int(student_vocab[token])
        tid = int(teacher_vocab[token])

        if int(s_ids[0]) != sid or int(t_ids[0]) != tid:
            raise RuntimeError(
                f"❌ protected token encode/get_vocab ID 不一致: {token}; "
                f"student encode/vocab={s_ids[0]}/{sid}, "
                f"teacher encode/vocab={t_ids[0]}/{tid}"
            )

        if token not in common_tokens:
            raise RuntimeError(
                f"❌ protected token 没进入共同词表，禁止继续训练: {token}"
            )

        protected_info[token] = {
            "student_id": sid,
            "teacher_id": tid,
        }
        print(
            f"✅ {token} | Student ID={sid} | Teacher ID={tid} | "
            "Common vocab=YES"
        )

    if len({v["student_id"] for v in protected_info.values()}) != 4:
        raise RuntimeError("❌ Student 的四个 protected token ID 发生重复。")
    if len({v["teacher_id"] for v in protected_info.values()}) != 4:
        raise RuntimeError("❌ Teacher 的四个 protected token ID 发生重复。")

    print("✅ 4/4 WOS protected tokens 全部保留，绝不会被共同词表过滤掉")

    print("\n🔐 Chat template 必需 token 检查")
    for token in required_chat_tokens:
        if token not in student_vocab or token not in teacher_vocab:
            raise RuntimeError(
                f"❌ Teacher / Student 至少一边缺少 Chat 必需 token: {token}"
            )
        s_ids = student_tokenizer.encode(token, add_special_tokens=False)
        t_ids = teacher_tokenizer.encode(token, add_special_tokens=False)
        if len(s_ids) != 1 or len(t_ids) != 1:
            raise RuntimeError(
                f"❌ Chat 必需 token 不是两边的单 token: {token}; "
                f"student={s_ids}, teacher={t_ids}"
            )
        print(
            f"✅ {token} | Student ID={student_vocab[token]} | "
            f"Teacher ID={teacher_vocab[token]}"
        )

    # 共同 token 按 Student ID 排序。
    # teacher_common_ids[i] 与 student_common_ids[i] 始终代表完全相同的 token 字符串。
    common_sorted = sorted(common_tokens, key=lambda tok: student_vocab[tok])
    student_common_ids = [int(student_vocab[tok]) for tok in common_sorted]
    teacher_common_ids = [int(teacher_vocab[tok]) for tok in common_sorted]

    # Student input IDs -> Teacher input IDs 的语义翻译表。
    # 这使同一份 Student batch 可以安全转换成 Teacher 自己的 ID 空间。
    student_to_teacher = [-1] * len(student_tokenizer)
    for token in common_sorted:
        sid = int(student_vocab[token])
        tid = int(teacher_vocab[token])
        if sid < 0 or sid >= len(student_to_teacher):
            raise RuntimeError(
                f"❌ Student token ID 超出 tokenizer 长度: {token!r} -> {sid}"
            )
        student_to_teacher[sid] = tid

    # 统计“字符串相同但 ID 不同”的 token。
    same_token_wrong_id = [
        (tok, int(student_vocab[tok]), int(teacher_vocab[tok]))
        for tok in common_sorted
        if int(student_vocab[tok]) != int(teacher_vocab[tok])
    ]

    common_ratio_student = len(common_tokens) / max(1, len(student_vocab))
    common_ratio_teacher = len(common_tokens) / max(1, len(teacher_vocab))

    print("\n📚 共同词表统计")
    print(f"Student vocab: {len(student_vocab):,}")
    print(f"Teacher vocab: {len(teacher_vocab):,}")
    print(f"Common token strings: {len(common_tokens):,}")
    print(f"Common / Student: {common_ratio_student:.6%}")
    print(f"Common / Teacher: {common_ratio_teacher:.6%}")
    print(f"Student-only: {len(student_only):,}")
    print(f"Teacher-only: {len(teacher_only):,}")
    print(f"共同 token 中 ID 不同: {len(same_token_wrong_id):,}")

    if student_only:
        print("Student-only tokens:", sorted(student_only)[:30])
    if teacher_only:
        print("Teacher-only tokens:", sorted(teacher_only)[:30])
    if same_token_wrong_id:
        print("同 token 不同 ID（前 20 个）:", same_token_wrong_id[:20])

    print(
        "✅ KD 将按 token 字符串语义对齐："
        "Teacher logits[teacher_common_ids] ↔ "
        "Student logits[student_common_ids]"
    )
    print(
        "✅ 两边各自独有 token 只从 KD soft target 中排除；"
        "不会删除或修改任何模型词表/权重"
    )
    print("=" * 88 + "\n")

    return {
        "student_common_ids": student_common_ids,
        "teacher_common_ids": teacher_common_ids,
        "student_to_teacher": student_to_teacher,
        "common_token_count": len(common_tokens),
        "student_only_tokens": sorted(student_only),
        "teacher_only_tokens": sorted(teacher_only),
        "same_token_wrong_id_count": len(same_token_wrong_id),
        "protected_info": protected_info,
    }


def save_vocab_alignment_info(save_dir: str, alignment: dict) -> None:
    """保存可审计的跨模型词表对齐摘要，不保存 12 万长度的完整映射。"""
    payload = {
        "alignment_rule": "same token string, IDs may differ",
        "kd_vocab_rule": "intersection of teacher/student token strings",
        "common_token_count": int(alignment["common_token_count"]),
        "student_only_tokens": alignment["student_only_tokens"],
        "teacher_only_tokens": alignment["teacher_only_tokens"],
        "same_token_wrong_id_count": int(
            alignment["same_token_wrong_id_count"]
        ),
        "protected_tokens": alignment["protected_info"],
        "protected_tokens_are_never_dropped": True,
    }
    path = os.path.join(save_dir, "teacher_student_vocab_alignment.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"✅ Teacher/Student 词表对齐摘要已保存: {path}")


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
        "lvl1_token": PROBE_TOKEN_L1,
        "lvl1_token_id": token_ids["l1"],
        "lvl2_token": PROBE_TOKEN_L2,
        "lvl2_token_id": token_ids["l2"],
        "tokenizer_length": len(tokenizer),
        "tokens_added": False,
        "embedding_resized": False,
        "prompt_occurrence": "first occurrence binds hierarchy role",
        "completion_occurrence": (
            "second occurrence precedes the corresponding label"
        ),
        "representation_for_hierarchical_align_and_relation": (
            "final-layer contextual hidden state at the second occurrence"
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



def get_hidden_projector(model):
    """如果当前 Student 挂载了共享 hidden-state projector，则返回它。"""
    model = unwrap_training_model(model)
    projector = getattr(model, "hidden_projector", None)
    if projector is not None and not isinstance(projector, nn.Module):
        raise TypeError("❌ model.hidden_projector 存在但不是 nn.Module。")
    return projector


def save_hidden_projector_weights(model, output_dir: str) -> None:
    """
    共享 Projector 只用于训练期 hidden-state Align/Hier，不参与最终生成推理。
    PEFT 的 save_pretrained 不保证保存额外挂载模块，因此单独保存用于审计/恢复。
    """
    projector = get_hidden_projector(model)
    if projector is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "state_dict": {
            k: v.detach().cpu() for k, v in projector.state_dict().items()
        }
    }
    torch.save(payload, os.path.join(output_dir, HIDDEN_PROJECTOR_WEIGHTS_NAME))

    meta = {
        "type": projector.__class__.__name__,
        "in_features": int(projector.in_features),
        "out_features": int(projector.out_features),
        "bias": projector.bias is not None,
        "shared_between_l1_l2": True,
        "used_only_for_training_losses": True,
        "needed_for_inference": False,
    }
    with open(
        os.path.join(output_dir, HIDDEN_PROJECTOR_META_NAME),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

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
        "experiment_mode", "seed", "expected_num_l1", "expected_num_l2",
        "max_seq_length", "reserve_tokens", "system_prompt", "num_epochs",
        "batch_size", "accumulation_steps", "lr", "io_lr", "projector_lr",
        "kd_temperature", "kd_weight", "align_weight", "hier_weight", "align_alpha",
        "l1_special_weight", "l2_special_weight", "lr_scheduler_type", "warmup_ratio",
        "lora_rank", "lora_alpha", "lora_dropout", "eval_steps", "eval_batch_size",
        "loss_eval_batch_size", "max_new_tokens", "bf16", "gradient_checkpointing",
        "num_workers", "dataloader_pin_memory", "attn_implementation",
    )
    signature = {name: getattr(args, name) for name in fields}
    signature.update(dataset="WOS", training_stage="offline_distillation", max_depth=2,
                     hierarchy_policy="shared_L1_L2_projector;L2_minus_L1_relation_mse",
                     projector_required=wos_projector_required(args),
                     world_size=int(os.environ.get("WORLD_SIZE", "1")))
    signature["model_path"] = os.path.realpath(args.model_path)
    signature["teacher_cache_dir"] = os.path.realpath(args.teacher_cache_dir)
    signature["data_sha256"] = {name: sha256_file(getattr(args, name))
                               for name in ("train_data_path", "val_data_path", "taxonomy_path", "dict_path")}
    signature["base_configuration_sha256"] = {
        name: sha256_file(os.path.join(args.model_path, name))
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        if os.path.isfile(os.path.join(args.model_path, name))
    }
    manifest_path = os.path.join(args.teacher_cache_dir, "manifest.json")
    manifest = read_json_file(manifest_path)
    files = manifest.get("files", {})
    fallbacks = {
        "offsets": "offsets.npy", "valid_counts": "valid_counts.npy",
        "l1_rel_offsets": "l1_rel_offsets.npy", "l2_rel_offsets": "l2_rel_offsets.npy",
        "student_input_sha256": "student_input_sha256.npy",
        "teacher_completion_hidden": "teacher_completion_hidden.npy",
        "student_common_ids": "student_common_ids.npy", "teacher_common_ids": "teacher_common_ids.npy",
        "teacher_common_lm_head": "teacher_common_lm_head.npy",
    }
    names = {key: files.get(key) or fallback for key, fallback in fallbacks.items()}
    if files.get("teacher_common_lm_head_bias"):
        names["teacher_common_lm_head_bias"] = files["teacher_common_lm_head_bias"]
    signature["teacher_cache_fingerprint"] = {
        "manifest_sha256": sha256_file(manifest_path),
        "files": {key: cache_file_fingerprint(os.path.join(args.teacher_cache_dir, name))
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
            raise RuntimeError("WOS 续训配置与断点不一致: " + ", ".join(changed))
        return
    experiment = os.path.dirname(os.path.dirname(checkpoint))
    info_path = os.path.join(experiment, "offline_teacher_cache_info.json")
    if not os.path.isfile(info_path):
        raise RuntimeError("旧 WOS 蒸馏实验缺少 offline_teacher_cache_info.json，无法核对离线教师缓存。")
    info = read_json_file(info_path)
    current_manifest = read_json_file(os.path.join(signature["teacher_cache_dir"], "manifest.json"))
    if (info["teacher_cache_manifest"] != current_manifest
            or os.path.realpath(info["teacher_cache_dir"]) != signature["teacher_cache_dir"]):
        raise RuntimeError("旧 WOS 实验的教师缓存路径或 manifest 与当前设置不一致。")
    try:
        saved_args = torch.load(os.path.join(checkpoint, "training_args.bin"), map_location="cpu", weights_only=False)
    except TypeError:
        saved_args = torch.load(os.path.join(checkpoint, "training_args.bin"), map_location="cpu")
    mapping = {"seed": "seed", "num_epochs": "num_train_epochs", "batch_size": "per_device_train_batch_size",
               "accumulation_steps": "gradient_accumulation_steps", "lr": "learning_rate",
               "lr_scheduler_type": "lr_scheduler_type", "warmup_ratio": "warmup_ratio",
               "bf16": "bf16", "gradient_checkpointing": "gradient_checkpointing", "eval_steps": "save_steps"}
    changed = []
    for key, saved_key in mapping.items():
        value = getattr(saved_args, saved_key, None)
        if getattr(value, "value", value) != signature[key]:
            changed.append(key)
    adapter = read_json_file(os.path.join(checkpoint, "adapter_config.json"))
    for key, saved_key in (("lora_rank", "r"), ("lora_alpha", "lora_alpha"), ("lora_dropout", "lora_dropout")):
        if adapter.get(saved_key) != signature[key]:
            changed.append(key)
    if changed:
        raise RuntimeError("旧 WOS 断点配置不一致: " + ", ".join(changed))
    print("⚠️ 旧 WOS 断点没有蒸馏参数/数据/缓存指纹快照；已核对原生参数、LoRA、缓存 manifest。请沿用原模式、KD/Align/Hier 权重、温度、学习率和数据。")


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

    source = None
    for candidate in (online, online + ".previous"):
        metadata_path = os.path.join(candidate, "best_kd_metadata.json")
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
              and not os.path.isfile(os.path.join(online, "best_kd_metadata.json"))):
            # 原 producer 在完整验证后保存 best；没有超过恢复步数的历史时，
            # 原在线 best 属于所恢复历史。若有未来记录则不能做此推断。
            atomic_write_json(os.path.join(online, "best_kd_metadata.json"), best)
            print("✅ 已按原 WOS 完整验证历史确认旧 best_kd_checkpoint。")
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
    print(f"✅ 已恢复WOS L1/L2 共享 Projector: {checkpoint}")



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
# 2. Prompt：预留 Token 首次角色绑定 + Completion 第二次直接生成标签
# ======================================================================
def build_user_content(text: str) -> str:
    """
    学生训练、教师训练、测试、层级探测、特征提取和后续蒸馏
    必须复用同一套 Prompt 逻辑。
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
    """Completion 中是两个层级 Token 的第二次出现。"""
    return (
        f"{PROBE_TOKEN_L1} {l1_label}\n"
        f"{PROBE_TOKEN_L2} {l2_label}\n"
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
    检查：
      - Prompt 中正文起止 Token 各 1 次；
      - Prompt 中 L1/L2 Token 各 1 次；
      - 拼接 Completion 后 L1/L2 Token 各 2 次；
      - 第二次 occurrence 位于对应标签之前。
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
        token_ids["l1"]: 1,
        token_ids["l2"]: 1,
    }

    for token_id, expected_count in expected_prompt_counts.items():
        actual_count = prompt_ids.count(token_id)
        if actual_count != expected_count:
            raise RuntimeError(
                f"❌ Prompt 中 token_id={token_id} 应出现 "
                f"{expected_count} 次，实际 {actual_count} 次。"
            )

    completion_ids = tokenizer.encode(
        build_completion_text("CS", "Machine learning"),
        add_special_tokens=False,
    )
    full_ids = prompt_ids + completion_ids

    l1_positions = find_all_token_positions(full_ids, token_ids["l1"])
    l2_positions = find_all_token_positions(full_ids, token_ids["l2"])

    if len(l1_positions) != 2:
        raise RuntimeError(
            f"❌ 完整样本中 L1 Token 应出现 2 次，实际 {l1_positions}"
        )
    if len(l2_positions) != 2:
        raise RuntimeError(
            f"❌ 完整样本中 L2 Token 应出现 2 次，实际 {l2_positions}"
        )

    if not (
        l1_positions[0] < l2_positions[0]
        < l1_positions[1] < l2_positions[1]
    ):
        raise RuntimeError(
            "❌ 层级 Token occurrence 顺序异常："
            f"L1={l1_positions}, L2={l2_positions}"
        )

    ctx_start_pos = full_ids.index(token_ids["ctx_start"])
    ctx_end_pos = full_ids.index(token_ids["ctx_end"])

    if ctx_end_pos <= ctx_start_pos + 1:
        raise RuntimeError("❌ 正文起止 Token 之间没有有效正文 token。")

    print("\n" + "=" * 70)
    print("✅ Prompt / Completion 特殊 Token 布局检查通过")
    print(f"正文开始 Token 位置: {ctx_start_pos}")
    print(f"正文结束 Token 位置: {ctx_end_pos}")
    print(f"L1 Token 两次位置: {l1_positions}")
    print(f"L2 Token 两次位置: {l2_positions}")
    print("后续层级探测/蒸馏应提取各层级 Token 的第二次 occurrence。")
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
            token_ids["l1"]: 1,
            token_ids["l2"]: 1,
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
    严格解析新 Completion：
      <L1 reserved token> L1_label
      <L2 reserved token> L2_label

    要求：
      - 正好两行；
      - 两个层级 Token 顺序正确；
      - 不 normalize、不模糊匹配、不候选纠错；
      - 第二行之后除停止 Token 外不能有额外正文。
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
        self.l1_token_id = tokenizer.convert_tokens_to_ids(PROBE_TOKEN_L1)
        self.l2_token_id = tokenizer.convert_tokens_to_ids(PROBE_TOKEN_L2)

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
        # 两个层级 Token 在这里第二次出现，并直接位于对应标签之前。
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

        # 严格保证正文边界各 1 次、层级 Token 各 2 次。
        if input_ids.count(self.ctx_start_id) != 1:
            raise RuntimeError(
                f"❌ {self.split_name} 第 {idx} 条 CTX_START 出现次数异常。"
            )
        if input_ids.count(self.ctx_end_id) != 1:
            raise RuntimeError(
                f"❌ {self.split_name} 第 {idx} 条 CTX_END 出现次数异常。"
            )
        if input_ids.count(self.l1_token_id) != 2:
            raise RuntimeError(
                f"❌ {self.split_name} 第 {idx} 条 L1 Token 应出现 2 次。"
            )
        if input_ids.count(self.l2_token_id) != 2:
            raise RuntimeError(
                f"❌ {self.split_name} 第 {idx} 条 L2 Token 应出现 2 次。"
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

        if self.eval_steps <= 0:
            raise ValueError("eval_steps 必须大于 0。")
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
                self.save_dir, records, bool(self.run_signature.get("projector_required", False)), allow_legacy_best
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
                    "best_kd_checkpoint",
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
                    checkpoint_metadata={
                        "step": int(state.global_step),
                        "global_macro": float(current_ma),
                        "global_micro": float(current_mi),
                        "eval_loss": float(eval_loss),
                    },
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
    print("训练策略: Traditional full-completion logits KD + hidden-state Align/Hier")
    print("Student: LoRA + 完整 input embedding/lm_head + 共享线性 Projector")
    print("层级损失只取 Completion 中第二次 L1/L2 特殊 token 的 final hidden state")
    print("不使用 AddedToken / resize / trainable_token_indices / 主动解绑")
    print("=" * 70 + "\n")


class FullIOPeftTrainer(Trainer):
    """
    1. 每个 Hugging Face checkpoint 额外保存完整 input embedding/lm_head；
    2. 若模型挂载 hidden_projector，也额外保存 projector；
    3. 参数组学习率：
       - LoRA 参数使用 args.learning_rate；
       - 完整 input embedding/lm_head 使用 io_learning_rate；
       - 共享 Projector 使用 projector_learning_rate（若未给则等于 LoRA LR）。
    """

    def __init__(
        self,
        *trainer_args,
        io_learning_rate: float,
        projector_learning_rate: float = None,
        **trainer_kwargs,
    ):
        if io_learning_rate <= 0:
            raise ValueError(
                f"❌ io_learning_rate 必须大于 0，实际={io_learning_rate}"
            )
        if projector_learning_rate is not None and projector_learning_rate <= 0:
            raise ValueError(
                "❌ projector_learning_rate 必须大于 0，"
                f"实际={projector_learning_rate}"
            )

        self.io_learning_rate = float(io_learning_rate)
        self.projector_learning_rate = (
            float(projector_learning_rate)
            if projector_learning_rate is not None
            else None
        )
        super().__init__(*trainer_args, **trainer_kwargs)

    def create_optimizer(self):
        """为 LoRA、完整 IO、共享 Projector 建立独立参数组。"""
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

        io_param_ids = {id(input_module.weight), id(output_module.weight)}

        projector = get_hidden_projector(raw_model)
        projector_param_ids = set()
        if projector is not None:
            projector_param_ids = {
                id(parameter)
                for parameter in projector.parameters()
                if parameter.requires_grad
            }
            if not projector_param_ids:
                raise RuntimeError("❌ hidden_projector 存在但没有可训练参数。")

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

            parameter_id = id(parameter)
            if parameter_id in seen_param_ids:
                continue
            seen_param_ids.add(parameter_id)

            is_io = parameter_id in io_param_ids
            is_projector = parameter_id in projector_param_ids
            if is_io and is_projector:
                raise RuntimeError("❌ 参数同时被识别为 IO 与 Projector。")

            use_decay = name in decay_parameter_names

            if is_io:
                key = "io_decay" if use_decay else "io_no_decay"
            elif is_projector:
                key = (
                    "projector_decay" if use_decay else "projector_no_decay"
                )
            else:
                key = "lora_decay" if use_decay else "lora_no_decay"

            grouped[key].append(parameter)

        io_parameter_count = sum(
            p.numel()
            for key in ("io_decay", "io_no_decay")
            for p in grouped[key]
        )
        projector_parameter_count = sum(
            p.numel()
            for key in ("projector_decay", "projector_no_decay")
            for p in grouped[key]
        )
        lora_parameter_count = sum(
            p.numel()
            for key in ("lora_decay", "lora_no_decay")
            for p in grouped[key]
        )

        if io_parameter_count == 0:
            raise RuntimeError(
                "❌ 参数分组失败：没有找到可训练的 input embedding/lm_head。"
            )
        if lora_parameter_count == 0:
            raise RuntimeError("❌ 参数分组失败：没有找到可训练的 LoRA 参数。")
        if projector is not None and projector_parameter_count == 0:
            raise RuntimeError("❌ 参数分组失败：没有找到 Projector 参数。")

        projector_lr = (
            self.projector_learning_rate
            if self.projector_learning_rate is not None
            else self.args.learning_rate
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

        add_group("lora_decay", self.args.learning_rate, self.args.weight_decay)
        add_group("lora_no_decay", self.args.learning_rate, 0.0)
        add_group("io_decay", self.io_learning_rate, self.args.weight_decay)
        add_group("io_no_decay", self.io_learning_rate, 0.0)
        add_group("projector_decay", projector_lr, self.args.weight_decay)
        add_group("projector_no_decay", projector_lr, 0.0)

        try:
            optimizer_cls, optimizer_kwargs = (
                Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)
            )
        except TypeError:
            optimizer_cls, optimizer_kwargs = (
                Trainer.get_optimizer_cls_and_kwargs(self.args)
            )

        optimizer_kwargs = dict(optimizer_kwargs)
        optimizer_kwargs.pop("params", None)
        optimizer_kwargs["lr"] = self.args.learning_rate

        self.optimizer = optimizer_cls(
            optimizer_grouped_parameters,
            **optimizer_kwargs,
        )

        input_output_tied = (
            input_module.weight.data_ptr() == output_module.weight.data_ptr()
        )

        print("\n" + "=" * 76)
        print("🎛️ 分组学习率优化器创建完成")
        print(f"LoRA 初始学习率: {self.args.learning_rate:.8g}")
        print(f"Embedding/lm_head 初始学习率: {self.io_learning_rate:.8g}")
        if projector is not None:
            print(f"Shared Projector 初始学习率: {projector_lr:.8g}")
        print(f"LoRA 可训练参数量: {lora_parameter_count:,}")
        print(f"Embedding/lm_head 可训练参数量: {io_parameter_count:,}")
        if projector is not None:
            print(f"Shared Projector 可训练参数量: {projector_parameter_count:,}")
        print(f"input embedding / lm_head 实际共享: {input_output_tied}")
        print(
            "说明：linear scheduler 会按相同比例衰减每个参数组；"
            "每组初始 LR 比例保持不变。"
        )
        print("=" * 76 + "\n")
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
            for key, value in payload["meters"].items():
                if key not in WOS_METER_FIELDS or not hasattr(self, key):
                    raise RuntimeError(f"当前 WOS Trainer 不支持断点日志字段: {key}")
                setattr(self, key, value)

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
             "meters": {key: getattr(self, key) for key in WOS_METER_FIELDS if hasattr(self, key)}},
        )
        return result


    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        if output_dir is None:
            output_dir = self.args.output_dir
        if self.is_world_process_zero():
            save_full_input_output_weights(self.model, output_dir)
            save_hidden_projector_weights(self.model, output_dir)



# ======================================================================
# 9. Traditional Full-Completion Logits KD Trainer
# ======================================================================
class TraditionalKDTrainer(FullIOPeftTrainer):
    """
    最基础的传统生成式 Logits Knowledge Distillation：

        L_total = L_task + kd_weight * L_KD

    其中：
      - L_task：与学生 SFT baseline 完全一致的 completion-only causal LM loss；
      - L_KD：只在 labels[:, 1:] != -100 的有效预测位置上计算；
      - Teacher / Student logits 不再按“相同数字 ID”硬对齐；
      - 而是在两个 tokenizer 的共同 token 字符串集合上做语义对齐；
      - 两边各自独有的极少量 token 不参与 KL，并在共同词表内重新归一化；
      - WOS 的 110/111/112/113 四个 protected token 已在启动时强制保证保留。

    本 baseline 不包含：
      - Align loss；
      - Hier loss；
      - hidden-state distillation；
      - projector；
      - 单独特殊 token KD。

    为降低 8B Teacher 的显存峰值：
      Teacher 不构造 [B, L, V] 的整段 logits。
      先前向得到最后层 hidden states，然后只对 Completion 有效预测位置调用 lm_head。
    """

    def __init__(
        self,
        *trainer_args,
        teacher_model,
        student_common_ids: List[int],
        teacher_common_ids: List[int],
        student_to_teacher_id_map: List[int],
        kd_temperature: float,
        kd_weight: float,
        teacher_forward_batch_size: int,
        **trainer_kwargs,
    ):
        if kd_temperature <= 0:
            raise ValueError(
                f"❌ kd_temperature 必须 > 0，实际={kd_temperature}"
            )
        if kd_weight < 0:
            raise ValueError(
                f"❌ kd_weight 必须 >= 0，实际={kd_weight}"
            )
        if teacher_forward_batch_size <= 0:
            raise ValueError(
                "❌ teacher_forward_batch_size 必须 > 0，"
                f"实际={teacher_forward_batch_size}"
            )
        if not student_common_ids or not teacher_common_ids:
            raise ValueError("❌ 共同 KD 词表不能为空。")
        if len(student_common_ids) != len(teacher_common_ids):
            raise ValueError(
                "❌ Student / Teacher common ID 列表长度不一致："
                f"student={len(student_common_ids)}, "
                f"teacher={len(teacher_common_ids)}"
            )

        self.teacher_model = teacher_model
        self._student_common_ids_cpu = [int(x) for x in student_common_ids]
        self._teacher_common_ids_cpu = [int(x) for x in teacher_common_ids]
        self._student_to_teacher_cpu = [
            int(x) for x in student_to_teacher_id_map
        ]
        self.kd_temperature = float(kd_temperature)
        self.kd_weight = float(kd_weight)
        self.teacher_forward_batch_size = int(teacher_forward_batch_size)

        self._loss_meter_task = 0.0
        self._loss_meter_kd = 0.0
        self._loss_meter_total = 0.0
        self._loss_meter_count = 0
        self._printed_kd_sanity = False

        super().__init__(*trainer_args, **trainer_kwargs)

        # Student 由 Trainer 管理设备；Teacher 是额外冻结模型。
        self.teacher_model.to(self.args.device)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        self.teacher_model.config.use_cache = False

        # 映射张量全部驻留在当前 rank 对应 GPU；它们很小。
        self.student_common_ids = torch.tensor(
            self._student_common_ids_cpu,
            dtype=torch.long,
            device=self.args.device,
        )
        self.teacher_common_ids = torch.tensor(
            self._teacher_common_ids_cpu,
            dtype=torch.long,
            device=self.args.device,
        )
        self.student_to_teacher_id_map = torch.tensor(
            self._student_to_teacher_cpu,
            dtype=torch.long,
            device=self.args.device,
        )

        teacher_trainable = sum(
            p.numel()
            for p in self.teacher_model.parameters()
            if p.requires_grad
        )
        if teacher_trainable != 0:
            raise RuntimeError(
                f"❌ Teacher 仍有 {teacher_trainable:,} 个可训练参数。"
            )

        student_vocab_rows = (
            unwrap_training_model(self.model)
            .get_output_embeddings()
            .weight.shape[0]
        )
        teacher_vocab_rows = (
            unwrap_training_model(self.teacher_model)
            .get_output_embeddings()
            .weight.shape[0]
        )

        if int(self.student_common_ids.max().item()) >= student_vocab_rows:
            raise RuntimeError("❌ student_common_ids 超出 Student lm_head 行数。")
        if int(self.teacher_common_ids.max().item()) >= teacher_vocab_rows:
            raise RuntimeError("❌ teacher_common_ids 超出 Teacher lm_head 行数。")

        print("\n" + "=" * 88)
        print("🧊 Teacher 冻结 + common-vocab KD 检查通过")
        print(
            f"Teacher device: "
            f"{next(self.teacher_model.parameters()).device}"
        )
        print(f"Student vocab rows: {student_vocab_rows:,}")
        print(f"Teacher vocab rows: {teacher_vocab_rows:,}")
        print(f"Common KD vocab: {len(self._student_common_ids_cpu):,}")
        print(f"KD temperature: {self.kd_temperature}")
        print(f"KD weight: {self.kd_weight}")
        print(
            "Teacher forward micro-batch: "
            f"{self.teacher_forward_batch_size}"
        )
        print("Teacher trainable params: 0")
        print("=" * 88 + "\n")

    def _translate_student_ids_to_teacher(
        self,
        student_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 Student token ID 序列按 token 字符串语义翻译到 Teacher ID 空间。

        例如同一个：
          <|reserved_special_token_112|>
        可以是 Student ID=128117、Teacher ID=128120。

        任何实际输入位置若出现 student-only token，直接报错，绝不静默替换。
        """
        if student_input_ids.min().item() < 0:
            raise RuntimeError("❌ Student input_ids 中出现负数 ID。")
        if student_input_ids.max().item() >= self.student_to_teacher_id_map.numel():
            raise RuntimeError(
                "❌ Student input_ids 超出跨 tokenizer 映射表范围。"
            )

        teacher_input_ids = self.student_to_teacher_id_map[
            student_input_ids
        ]

        # 只检查 attention_mask=1 的实际输入。padding 位正常情况下也应可映射，
        # 但不让 padding 干扰真正语义输入的诊断。
        unmapped_mask = teacher_input_ids.lt(0) & attention_mask.bool()
        if unmapped_mask.any():
            bad_student_ids = torch.unique(
                student_input_ids[unmapped_mask]
            ).detach().cpu().tolist()
            raise RuntimeError(
                "❌ 当前训练 batch 使用了 Teacher 中不存在的 Student token。"
                "这些 token 无法按语义转换到 Teacher 输入空间。"
                f" student_ids={bad_student_ids[:30]}"
            )

        # padding 位如果刚好是 student-only，也不能把 -1 喂给 embedding。
        # 用 Teacher 的第 0 行只是占位，因为 attention_mask=0 完全屏蔽；
        # 正常 Llama pad=eos 是共同 token，所以通常不会触发。
        if teacher_input_ids.lt(0).any():
            teacher_input_ids = teacher_input_ids.masked_fill(
                teacher_input_ids.lt(0),
                0,
            )

        return teacher_input_ids

    def _teacher_common_logits_at_supervised_positions(
        self,
        teacher_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        valid_prediction_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        valid_prediction_mask: [B, L-1]，对应：

            logits[:, :-1, :]  <->  labels[:, 1:]

        Teacher 只在这些位置计算词表 logits，然后只保留共同 KD 词表。
        返回 [N_valid, V_common]。
        """
        teacher = unwrap_training_model(self.teacher_model)

        decoder = getattr(teacher, "model", None)
        if decoder is None:
            raise RuntimeError(
                "❌ 当前 Teacher 不是预期的 LlamaForCausalLM 结构："
                "找不到 teacher.model。"
            )

        lm_head = teacher.get_output_embeddings()
        if lm_head is None:
            raise RuntimeError("❌ Teacher 找不到 lm_head。")

        batch_size = teacher_input_ids.size(0)
        all_teacher_common_logits = []

        with torch.inference_mode():
            for start in range(
                0,
                batch_size,
                self.teacher_forward_batch_size,
            ):
                end = min(
                    start + self.teacher_forward_batch_size,
                    batch_size,
                )

                chunk_input_ids = teacher_input_ids[start:end]
                chunk_attention_mask = attention_mask[start:end]
                chunk_valid_mask = valid_prediction_mask[start:end]

                # right padding：裁掉当前 micro-batch 纯 padding 的尾部。
                actual_lengths = chunk_attention_mask.sum(dim=1)
                max_actual_len = int(actual_lengths.max().item())

                if max_actual_len < 2:
                    raise RuntimeError(
                        "❌ Teacher batch 中出现长度不足 2 的样本。"
                    )

                chunk_input_ids = chunk_input_ids[:, :max_actual_len]
                chunk_attention_mask = chunk_attention_mask[:, :max_actual_len]
                chunk_valid_mask = chunk_valid_mask[:, : max_actual_len - 1]

                outputs = decoder(
                    input_ids=chunk_input_ids,
                    attention_mask=chunk_attention_mask,
                    use_cache=False,
                    return_dict=True,
                )

                hidden_states = outputs.last_hidden_state
                prediction_hidden = hidden_states[:, :-1, :]
                selected_hidden = prediction_hidden[chunk_valid_mask]

                if selected_hidden.numel() == 0:
                    continue

                # 只需要 Completion 有效位置的 Teacher logits。
                full_logits = lm_head(selected_hidden)
                common_logits = full_logits.index_select(
                    dim=-1,
                    index=self.teacher_common_ids,
                )
                all_teacher_common_logits.append(common_logits)

        if not all_teacher_common_logits:
            raise RuntimeError(
                "❌ 当前 batch 没有任何 Completion 有效 KD 位置。"
            )

        return torch.cat(all_teacher_common_logits, dim=0)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """
        Traditional full-completion common-vocabulary KD。

        Student task loss 保持原 SFT 定义；KD 则在共同 token 字符串集合上：

            Student logits[:, student_common_ids]
            Teacher logits[:, teacher_common_ids]

        两组列逐项代表完全相同的 token 字符串。
        """
        del num_items_in_batch

        if "labels" not in inputs:
            raise RuntimeError("❌ KD 训练 batch 缺少 labels。")

        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # --------------------------------------------------------------
        # 1. Student 正常前向：与 Student SFT baseline 完全一致。
        # --------------------------------------------------------------
        student_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

        task_loss = student_outputs.loss
        if task_loss is None:
            raise RuntimeError("❌ Student forward 没有返回 task loss。")

        # --------------------------------------------------------------
        # 2. Causal shift + Completion-only mask。
        # --------------------------------------------------------------
        shift_labels = labels[:, 1:]
        valid_prediction_mask = shift_labels.ne(-100)
        valid_count = int(valid_prediction_mask.sum().item())

        if valid_count <= 0:
            raise RuntimeError(
                "❌ 当前 batch 没有任何有效 Completion supervision token。"
            )

        student_shift_logits = student_outputs.logits[:, :-1, :]
        student_selected_full_logits = student_shift_logits[
            valid_prediction_mask
        ]
        student_common_logits = student_selected_full_logits.index_select(
            dim=-1,
            index=self.student_common_ids,
        )

        # --------------------------------------------------------------
        # 3. 将 Student 输入 IDs 转成 Teacher 自己的 ID 空间。
        #    注意：位置与 token 字符串序列不变，只改变 ID 数字。
        # --------------------------------------------------------------
        teacher_input_ids = self._translate_student_ids_to_teacher(
            student_input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # --------------------------------------------------------------
        # 4. Teacher 冻结前向，同一 causal positions + common vocabulary。
        # --------------------------------------------------------------
        teacher_common_logits = (
            self._teacher_common_logits_at_supervised_positions(
                teacher_input_ids=teacher_input_ids,
                attention_mask=attention_mask,
                valid_prediction_mask=valid_prediction_mask,
            )
        )

        if student_common_logits.shape != teacher_common_logits.shape:
            raise RuntimeError(
                "❌ Student / Teacher common KD logits 形状不一致："
                f"student={tuple(student_common_logits.shape)}, "
                f"teacher={tuple(teacher_common_logits.shape)}"
            )

        # --------------------------------------------------------------
        # 5. 标准 temperature-scaled KL。
        #    两边均只在 common vocabulary 内 softmax，因此自动重新归一化。
        # --------------------------------------------------------------
        temperature = self.kd_temperature

        student_log_probs = F.log_softmax(
            student_common_logits.float() / temperature,
            dim=-1,
        )
        teacher_probs = F.softmax(
            teacher_common_logits.float() / temperature,
            dim=-1,
        )

        kd_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        ) * (temperature ** 2)

        total_loss = task_loss + self.kd_weight * kd_loss

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "❌ KD total loss 出现 NaN/Inf："
                f"task={task_loss.detach().float().item()}, "
                f"kd={kd_loss.detach().float().item()}"
            )

        # --------------------------------------------------------------
        # 6. 首 batch 打印严格对齐信息。
        # --------------------------------------------------------------
        if not self._printed_kd_sanity:
            print("\n" + "=" * 88)
            print("✅ Traditional common-vocab KD 首批次对齐检查")
            print(f"Student input_ids shape: {tuple(input_ids.shape)}")
            print(
                f"Teacher translated input_ids shape: "
                f"{tuple(teacher_input_ids.shape)}"
            )
            print(f"labels shape: {tuple(labels.shape)}")
            print(f"有效 Completion 预测位置数: {valid_count}")
            print(
                "Student common logits shape: "
                f"{tuple(student_common_logits.shape)}"
            )
            print(
                "Teacher common logits shape: "
                f"{tuple(teacher_common_logits.shape)}"
            )
            print(
                "KD mask: labels[:, 1:] != -100；"
                "与 logits[:, :-1, :] 严格 causal 对齐"
            )
            print(
                "输入对齐: Student ID -> 相同 token 字符串的 Teacher ID"
            )
            print(
                "输出对齐: Teacher common IDs ↔ Student common IDs；"
                "逐列 token 字符串完全相同"
            )
            print(
                "WOS protected 110/111/112/113 已在启动时强制保留"
            )
            print(
                "本 baseline 仍蒸馏整个 gold Completion，"
                "没有单独挑 L1/L2 特殊 token。"
            )
            print(f"Task loss: {task_loss.detach().float().item():.6f}")
            print(f"KD loss: {kd_loss.detach().float().item():.6f}")
            print(f"Total loss: {total_loss.detach().float().item():.6f}")
            print("=" * 88 + "\n")
            self._printed_kd_sanity = True

        # --------------------------------------------------------------
        # 7. 累积训练 loss 分解。
        # --------------------------------------------------------------
        if model.training:
            self._loss_meter_task += task_loss.detach().float().item()
            self._loss_meter_kd += kd_loss.detach().float().item()
            self._loss_meter_total += total_loss.detach().float().item()
            self._loss_meter_count += 1

        if return_outputs:
            return total_loss, student_outputs
        return total_loss

    def log(self, logs, *args, **kwargs):
        logs = dict(logs)

        if self._loss_meter_count > 0 and "loss" in logs:
            count = float(self._loss_meter_count)
            logs["Train/Task_Loss"] = self._loss_meter_task / count
            logs["Train/KD_Loss"] = self._loss_meter_kd / count
            logs["Train/Total_Loss"] = self._loss_meter_total / count
            logs["Train/KD_Weight"] = self.kd_weight
            logs["Train/KD_Temperature"] = self.kd_temperature
            logs["Train/Common_Vocab_Size"] = float(
                self.student_common_ids.numel()
            )

            self._loss_meter_task = 0.0
            self._loss_meter_kd = 0.0
            self._loss_meter_total = 0.0
            self._loss_meter_count = 0

        return super().log(logs, *args, **kwargs)


# ======================================================================
# 10. Hidden-state 层级蒸馏 Trainer
#
# 三种消融模式：
#   align : Task + Full logits KD + Hidden-state Align
#   hier  : Task + Full logits KD + Hidden-state Hier relation
#   full  : Task + Full logits KD + Hidden Align + Hidden Hier
#
# Traditional KD 仍然在整个 gold Completion 的共同词表 logits 上计算。
# 两个创新层级损失则只读取 Completion 中第二次 L1/L2 特殊 token 的
# final-layer contextual hidden state。
# ======================================================================
class HiddenHierarchicalKDTrainer(TraditionalKDTrainer):
    """
    Traditional logits KD + 两个 hidden-state 层级损失。

    设：
      h_S^(k) : Student 第 k 层级特殊 token 第二次 occurrence 的 final hidden
      h_T^(k) : Teacher 对应位置 final hidden
      P        : 共享可训练线性投影，Student hidden -> Teacher hidden space
      hhat_S^(k) = P(h_S^(k))

    1) Hidden Align：
       L_align^(k) = alpha * MSE(hhat_S^(k), h_T^(k))
                   + (1-alpha) * [1 - cosine(hhat_S^(k), h_T^(k))]

       L_align = (w1 * L_align^L1 + w2 * L_align^L2) / (w1 + w2)

    2) Hidden Hier：
       delta_S = P(h_S^L2) - P(h_S^L1)
       delta_T = h_T^L2 - h_T^L1
       L_hier = MSE(delta_S, delta_T)

    注意：
      - Projector 在 L1/L2 间严格共享；
      - Teacher 全程冻结；
      - Teacher hidden 在线从同一次 Teacher decoder forward 中取得，
        与“预先离线保存相同 Teacher hidden”在数学定义上等价；
      - 不做全序列 hidden-state KD，只蒸馏两个层级特殊 token。
    """

    def __init__(
        self,
        *trainer_args,
        student_l1_token_id: int,
        student_l2_token_id: int,
        align_weight: float,
        hier_weight: float,
        align_alpha: float,
        l1_special_weight: float,
        l2_special_weight: float,
        **trainer_kwargs,
    ):
        self.student_l1_token_id = int(student_l1_token_id)
        self.student_l2_token_id = int(student_l2_token_id)
        self.align_weight = float(align_weight)
        self.hier_weight = float(hier_weight)
        self.align_alpha = float(align_alpha)
        self.l1_special_weight = float(l1_special_weight)
        self.l2_special_weight = float(l2_special_weight)

        if self.align_weight < 0 or self.hier_weight < 0:
            raise ValueError("❌ align_weight / hier_weight 不能小于 0。")
        if self.align_weight == 0 and self.hier_weight == 0:
            raise ValueError("❌ Align/Hier 不能同时为 0。")
        if not (0.0 <= self.align_alpha <= 1.0):
            raise ValueError(
                f"❌ align_alpha 必须位于 [0,1]，实际={self.align_alpha}"
            )
        if self.l1_special_weight < 0 or self.l2_special_weight < 0:
            raise ValueError("❌ L1/L2 special weight 不能小于 0。")
        if self.l1_special_weight + self.l2_special_weight <= 0:
            raise ValueError("❌ L1/L2 special weight 不能同时为 0。")

        self._meter_align_l1_mse = 0.0
        self._meter_align_l1_cos = 0.0
        self._meter_align_l1 = 0.0
        self._meter_align_l2_mse = 0.0
        self._meter_align_l2_cos = 0.0
        self._meter_align_l2 = 0.0
        self._meter_align = 0.0
        self._meter_hier = 0.0
        self._printed_hidden_hier_sanity = False

        super().__init__(*trainer_args, **trainer_kwargs)

        projector = get_hidden_projector(self.model)
        if projector is None:
            raise RuntimeError(
                "❌ HiddenHierarchicalKDTrainer 需要 model.hidden_projector。"
            )

        teacher_hidden_size = int(
            unwrap_training_model(self.teacher_model).config.hidden_size
        )
        if int(projector.out_features) != teacher_hidden_size:
            raise RuntimeError(
                "❌ Projector 输出维度与 Teacher hidden size 不一致："
                f"projector={projector.out_features}, teacher={teacher_hidden_size}"
            )

        print("\n" + "=" * 92)
        print("🎯 Traditional logits KD + Hidden-state Hierarchical KD")
        print(f"Full logits KD weight: {self.kd_weight}")
        print(f"Hidden Align weight:   {self.align_weight}")
        print(f"Hidden Hier weight:    {self.hier_weight}")
        print(f"Align alpha(MSE):       {self.align_alpha}")
        print(f"Align alpha(Cos):       {1.0 - self.align_alpha}")
        print(f"L1 special weight:      {self.l1_special_weight}")
        print(f"L2 special weight:      {self.l2_special_weight}")
        print(f"KD temperature:         {self.kd_temperature}")
        print(
            f"Shared Projector: R^{projector.in_features} -> "
            f"R^{projector.out_features}"
        )
        print("✅ Full KD 使用 logits；Align/Hier 使用 final hidden states")
        print("✅ L1/L2 共用同一个线性 Projector")
        print("=" * 92 + "\n")

    def _find_second_occurrence_positions(
        self,
        input_ids: torch.Tensor,
        token_id: int,
        name: str,
    ) -> torch.Tensor:
        positions = []
        for row_idx in range(input_ids.size(0)):
            row_positions = torch.nonzero(
                input_ids[row_idx].eq(token_id),
                as_tuple=False,
            ).flatten()
            if row_positions.numel() != 2:
                raise RuntimeError(
                    f"❌ batch row={row_idx} 的 {name} 应严格出现 2 次，"
                    f"实际位置={row_positions.detach().cpu().tolist()}"
                )
            positions.append(row_positions[1])
        return torch.stack(positions).to(
            device=input_ids.device,
            dtype=torch.long,
        )

    def _temperature_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        temperature = self.kd_temperature
        student_log_probs = F.log_softmax(
            student_logits.float() / temperature,
            dim=-1,
        )
        teacher_probs = F.softmax(
            teacher_logits.float() / temperature,
            dim=-1,
        )
        return F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        ) * (temperature ** 2)

    def _get_student_final_norm_module(self, model):
        """
        用 final RMSNorm 的 forward hook 只抓最后一层 hidden，避免
        output_hidden_states=True 保存所有层 hidden 导致显存大幅增加。
        """
        raw = unwrap_training_model(model)
        base_causal_lm = (
            raw.get_base_model()
            if hasattr(raw, "get_base_model")
            else raw
        )
        decoder = getattr(base_causal_lm, "model", None)
        if decoder is None:
            raise RuntimeError("❌ Student 找不到 Llama decoder (.model)。")
        final_norm = getattr(decoder, "norm", None)
        if final_norm is None:
            raise RuntimeError("❌ Student decoder 找不到 final norm。")
        return final_norm

    def _teacher_common_logits_and_special_hidden(
        self,
        teacher_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        valid_prediction_mask: torch.Tensor,
        l1_positions: torch.Tensor,
        l2_positions: torch.Tensor,
    ):
        """
        同一次 Teacher decoder forward 同时取得：
          1) Full-Completion KD 需要的 common-vocab logits；
          2) 第二次 L1/L2 特殊 token 的 final hidden state。
        """
        teacher = unwrap_training_model(self.teacher_model)
        decoder = getattr(teacher, "model", None)
        if decoder is None:
            raise RuntimeError("❌ Teacher 找不到 Llama decoder (.model)。")
        lm_head = teacher.get_output_embeddings()
        if lm_head is None:
            raise RuntimeError("❌ Teacher 找不到 lm_head。")

        batch_size = teacher_input_ids.size(0)
        all_common_logits = []
        all_l1_hidden = []
        all_l2_hidden = []

        with torch.inference_mode():
            for start in range(
                0,
                batch_size,
                self.teacher_forward_batch_size,
            ):
                end = min(
                    start + self.teacher_forward_batch_size,
                    batch_size,
                )

                chunk_input_ids = teacher_input_ids[start:end]
                chunk_attention_mask = attention_mask[start:end]
                chunk_valid_mask = valid_prediction_mask[start:end]
                chunk_l1_pos = l1_positions[start:end]
                chunk_l2_pos = l2_positions[start:end]

                actual_lengths = chunk_attention_mask.sum(dim=1)
                max_actual_len = int(actual_lengths.max().item())
                if max_actual_len < 2:
                    raise RuntimeError("❌ Teacher batch 出现长度不足 2 的样本。")

                if (chunk_l1_pos >= max_actual_len).any() or (
                    chunk_l2_pos >= max_actual_len
                ).any():
                    raise RuntimeError(
                        "❌ Teacher micro-batch 裁剪后特殊 token 位置越界。"
                    )

                chunk_input_ids = chunk_input_ids[:, :max_actual_len]
                chunk_attention_mask = chunk_attention_mask[:, :max_actual_len]
                chunk_valid_mask = chunk_valid_mask[:, : max_actual_len - 1]

                outputs = decoder(
                    input_ids=chunk_input_ids,
                    attention_mask=chunk_attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                hidden = outputs.last_hidden_state

                # Full KD logits
                prediction_hidden = hidden[:, :-1, :]
                selected_hidden = prediction_hidden[chunk_valid_mask]
                if selected_hidden.numel() > 0:
                    full_logits = lm_head(selected_hidden)
                    common_logits = full_logits.index_select(
                        dim=-1,
                        index=self.teacher_common_ids,
                    )
                    all_common_logits.append(common_logits)

                # 两个特殊 token 本身的 final hidden
                row_ids = torch.arange(
                    end - start,
                    device=hidden.device,
                    dtype=torch.long,
                )
                all_l1_hidden.append(hidden[row_ids, chunk_l1_pos, :])
                all_l2_hidden.append(hidden[row_ids, chunk_l2_pos, :])

        if not all_common_logits:
            raise RuntimeError("❌ 当前 batch 没有 Teacher Completion KD logits。")

        return (
            torch.cat(all_common_logits, dim=0),
            torch.cat(all_l1_hidden, dim=0),
            torch.cat(all_l2_hidden, dim=0),
        )

    def _hidden_align_component(
        self,
        projected_student: torch.Tensor,
        teacher_hidden: torch.Tensor,
    ):
        projected_student = projected_student.float()
        teacher_hidden = teacher_hidden.float()

        mse = F.mse_loss(
            projected_student,
            teacher_hidden,
            reduction="mean",
        )
        cos_loss = (
            1.0
            - F.cosine_similarity(
                projected_student,
                teacher_hidden,
                dim=-1,
                eps=1e-8,
            )
        ).mean()
        combined = (
            self.align_alpha * mse
            + (1.0 - self.align_alpha) * cos_loss
        )
        return combined, mse, cos_loss

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        del num_items_in_batch

        if "labels" not in inputs:
            raise RuntimeError("❌ KD 训练 batch 缺少 labels。")

        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # --------------------------------------------------------------
        # 1. 先定位 Completion 中第二次 L1/L2 特殊 token。
        # --------------------------------------------------------------
        l1_positions = self._find_second_occurrence_positions(
            input_ids,
            self.student_l1_token_id,
            "L1 token",
        )
        l2_positions = self._find_second_occurrence_positions(
            input_ids,
            self.student_l2_token_id,
            "L2 token",
        )
        if not torch.all(l1_positions < l2_positions):
            raise RuntimeError("❌ L1 第二次 occurrence 必须位于 L2 之前。")

        # --------------------------------------------------------------
        # 2. Student 正常 causal LM forward，同时用 hook 只抓 final hidden。
        # --------------------------------------------------------------
        captured = {}
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
            handle.remove()

        task_loss = student_outputs.loss
        if task_loss is None:
            raise RuntimeError("❌ Student forward 没有返回 task loss。")
        if "last_hidden" not in captured:
            raise RuntimeError("❌ Student final hidden hook 没有捕获到输出。")

        student_last_hidden = captured["last_hidden"]
        if not torch.is_tensor(student_last_hidden):
            raise TypeError("❌ Student final hidden hook 输出不是 Tensor。")

        row_ids = torch.arange(
            input_ids.size(0),
            device=input_ids.device,
            dtype=torch.long,
        )
        s_l1_hidden = student_last_hidden[row_ids, l1_positions, :]
        s_l2_hidden = student_last_hidden[row_ids, l2_positions, :]

        # --------------------------------------------------------------
        # 3. Traditional Full-Completion logits KD（定义保持不变）。
        # --------------------------------------------------------------
        shift_labels = labels[:, 1:]
        valid_prediction_mask = shift_labels.ne(-100)
        valid_count = int(valid_prediction_mask.sum().item())
        if valid_count <= 0:
            raise RuntimeError("❌ 当前 batch 没有 Completion supervision token。")

        # 两个特殊 token 所在位置的 logits 都必须属于 causal KD mask，
        # 因为它们分别预测 L1/L2 标签首 token。
        if not valid_prediction_mask[row_ids, l1_positions].all():
            raise RuntimeError("❌ L1 第二次 token 未落在 Completion KD mask。")
        if not valid_prediction_mask[row_ids, l2_positions].all():
            raise RuntimeError("❌ L2 第二次 token 未落在 Completion KD mask。")

        student_shift_logits = student_outputs.logits[:, :-1, :]
        student_selected_full_logits = student_shift_logits[
            valid_prediction_mask
        ]
        student_common_logits = student_selected_full_logits.index_select(
            dim=-1,
            index=self.student_common_ids,
        )

        teacher_input_ids = self._translate_student_ids_to_teacher(
            student_input_ids=input_ids,
            attention_mask=attention_mask,
        )

        (
            teacher_common_logits,
            t_l1_hidden,
            t_l2_hidden,
        ) = self._teacher_common_logits_and_special_hidden(
            teacher_input_ids=teacher_input_ids,
            attention_mask=attention_mask,
            valid_prediction_mask=valid_prediction_mask,
            l1_positions=l1_positions,
            l2_positions=l2_positions,
        )

        if student_common_logits.shape != teacher_common_logits.shape:
            raise RuntimeError(
                "❌ Student / Teacher common logits 形状不一致："
                f"student={tuple(student_common_logits.shape)}, "
                f"teacher={tuple(teacher_common_logits.shape)}"
            )

        full_kd_loss = self._temperature_kl(
            student_common_logits,
            teacher_common_logits,
        )

        # --------------------------------------------------------------
        # 4. 共享 Projector：Student 2048 -> Teacher 4096（按模型实际维度）。
        #    Projector 参数保持 float32，hidden 转 float32 后进入 projector。
        # --------------------------------------------------------------
        projector = get_hidden_projector(model)
        if projector is None:
            raise RuntimeError("❌ 训练中找不到 shared hidden_projector。")

        projected_s_l1 = projector(s_l1_hidden.float())
        projected_s_l2 = projector(s_l2_hidden.float())

        if projected_s_l1.shape != t_l1_hidden.shape:
            raise RuntimeError(
                "❌ L1 projected Student / Teacher hidden 形状不一致："
                f"student={tuple(projected_s_l1.shape)}, "
                f"teacher={tuple(t_l1_hidden.shape)}"
            )
        if projected_s_l2.shape != t_l2_hidden.shape:
            raise RuntimeError(
                "❌ L2 projected Student / Teacher hidden 形状不一致。"
            )

        # --------------------------------------------------------------
        # 5. Hidden Align：MSE + cosine，L1/L2 共享同一 Projector。
        # --------------------------------------------------------------
        zero = task_loss.new_zeros(())
        if self.align_weight > 0:
            align_l1, align_l1_mse, align_l1_cos = (
                self._hidden_align_component(projected_s_l1, t_l1_hidden)
            )
            align_l2, align_l2_mse, align_l2_cos = (
                self._hidden_align_component(projected_s_l2, t_l2_hidden)
            )

            special_weight_sum = (
                self.l1_special_weight + self.l2_special_weight
            )
            align_loss = (
                self.l1_special_weight * align_l1
                + self.l2_special_weight * align_l2
            ) / special_weight_sum
        else:
            align_l1 = zero
            align_l1_mse = zero
            align_l1_cos = zero
            align_l2 = zero
            align_l2_mse = zero
            align_l2_cos = zero
            align_loss = zero

        # --------------------------------------------------------------
        # 6. Hidden Hier：保持你原始创新公式，不改成 logits relation。
        # --------------------------------------------------------------
        if self.hier_weight > 0:
            delta_student = projected_s_l2 - projected_s_l1
            delta_teacher = t_l2_hidden.float() - t_l1_hidden.float()
            hier_loss = F.mse_loss(
                delta_student.float(),
                delta_teacher,
                reduction="mean",
            )
        else:
            hier_loss = zero

        # --------------------------------------------------------------
        # 7. 总损失。
        # --------------------------------------------------------------
        total_loss = (
            task_loss
            + self.kd_weight * full_kd_loss
            + self.align_weight * align_loss
            + self.hier_weight * hier_loss
        )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "❌ total loss NaN/Inf："
                f"task={task_loss.detach().float().item()}, "
                f"kd={full_kd_loss.detach().float().item()}, "
                f"align={align_loss.detach().float().item()}, "
                f"hier={hier_loss.detach().float().item()}"
            )

        # --------------------------------------------------------------
        # 8. 首 batch 打印。
        # --------------------------------------------------------------
        if not self._printed_hidden_hier_sanity:
            print("\n" + "=" * 100)
            print("✅ Traditional logits KD + Hidden-state Align/Hier 首批次检查")
            print(f"Student input_ids: {tuple(input_ids.shape)}")
            print(f"有效 Completion KD 位置: {valid_count}")
            print(f"Full KD common logits: {tuple(student_common_logits.shape)}")
            print(
                "L1 第二次 occurrence positions: "
                f"{l1_positions.detach().cpu().tolist()}"
            )
            print(
                "L2 第二次 occurrence positions: "
                f"{l2_positions.detach().cpu().tolist()}"
            )
            print(f"Student L1 hidden: {tuple(s_l1_hidden.shape)}")
            print(f"Student L2 hidden: {tuple(s_l2_hidden.shape)}")
            print(f"Projected L1 hidden: {tuple(projected_s_l1.shape)}")
            print(f"Teacher L1 hidden: {tuple(t_l1_hidden.shape)}")
            print(
                "✅ Full KD = entire Completion common-vocab logits KL"
            )
            print(
                "✅ Align = alpha*MSE(projected Student, Teacher) + "
                "(1-alpha)*(1-cosine)"
            )
            print(
                "✅ Hier = MSE(P(hS_L2)-P(hS_L1), hT_L2-hT_L1)"
            )
            print("✅ L1/L2 使用同一个共享线性 Projector")
            print(f"Task loss       : {task_loss.detach().float().item():.6f}")
            print(f"Full logits KD  : {full_kd_loss.detach().float().item():.6f}")
            print(f"Align L1 MSE    : {align_l1_mse.detach().float().item():.6f}")
            print(f"Align L1 CosLoss: {align_l1_cos.detach().float().item():.6f}")
            print(f"Align L1 Total  : {align_l1.detach().float().item():.6f}")
            print(f"Align L2 MSE    : {align_l2_mse.detach().float().item():.6f}")
            print(f"Align L2 CosLoss: {align_l2_cos.detach().float().item():.6f}")
            print(f"Align L2 Total  : {align_l2.detach().float().item():.6f}")
            print(f"Align loss      : {align_loss.detach().float().item():.6f}")
            print(f"Hidden Hier loss: {hier_loss.detach().float().item():.6f}")
            print(f"Total loss      : {total_loss.detach().float().item():.6f}")
            print(
                "公式: Total = Task + "
                f"{self.kd_weight}*FullLogitsKD + "
                f"{self.align_weight}*HiddenAlign + "
                f"{self.hier_weight}*HiddenHier"
            )
            print("=" * 100 + "\n")
            self._printed_hidden_hier_sanity = True

        # --------------------------------------------------------------
        # 9. 日志累计。
        # --------------------------------------------------------------
        if model.training:
            self._loss_meter_task += task_loss.detach().float().item()
            self._loss_meter_kd += full_kd_loss.detach().float().item()
            self._loss_meter_total += total_loss.detach().float().item()
            self._loss_meter_count += 1

            self._meter_align_l1_mse += align_l1_mse.detach().float().item()
            self._meter_align_l1_cos += align_l1_cos.detach().float().item()
            self._meter_align_l1 += align_l1.detach().float().item()
            self._meter_align_l2_mse += align_l2_mse.detach().float().item()
            self._meter_align_l2_cos += align_l2_cos.detach().float().item()
            self._meter_align_l2 += align_l2.detach().float().item()
            self._meter_align += align_loss.detach().float().item()
            self._meter_hier += hier_loss.detach().float().item()

        if return_outputs:
            return total_loss, student_outputs
        return total_loss

    def log(self, logs, *args, **kwargs):
        logs = dict(logs)
        if self._loss_meter_count > 0 and "loss" in logs:
            count = float(self._loss_meter_count)
            logs["Train/Task_Loss"] = self._loss_meter_task / count
            logs["Train/KD_Loss"] = self._loss_meter_kd / count
            logs["Train/Align_L1_MSE"] = self._meter_align_l1_mse / count
            logs["Train/Align_L1_CosLoss"] = self._meter_align_l1_cos / count
            logs["Train/Align_L1_Total"] = self._meter_align_l1 / count
            logs["Train/Align_L2_MSE"] = self._meter_align_l2_mse / count
            logs["Train/Align_L2_CosLoss"] = self._meter_align_l2_cos / count
            logs["Train/Align_L2_Total"] = self._meter_align_l2 / count
            logs["Train/Align_Loss"] = self._meter_align / count
            logs["Train/Hier_Hidden_MSE"] = self._meter_hier / count
            logs["Train/Total_Loss"] = self._loss_meter_total / count
            logs["Train/KD_Weight"] = self.kd_weight
            logs["Train/Align_Weight"] = self.align_weight
            logs["Train/Hier_Weight"] = self.hier_weight
            logs["Train/Align_Alpha"] = self.align_alpha
            logs["Train/KD_Temperature"] = self.kd_temperature
            logs["Train/Common_Vocab_Size"] = float(
                self.student_common_ids.numel()
            )

            self._loss_meter_task = 0.0
            self._loss_meter_kd = 0.0
            self._loss_meter_total = 0.0
            self._loss_meter_count = 0
            self._meter_align_l1_mse = 0.0
            self._meter_align_l1_cos = 0.0
            self._meter_align_l1 = 0.0
            self._meter_align_l2_mse = 0.0
            self._meter_align_l2_cos = 0.0
            self._meter_align_l2 = 0.0
            self._meter_align = 0.0
            self._meter_hier = 0.0

        return FullIOPeftTrainer.log(self, logs, *args, **kwargs)



# ======================================================================
# 9C. Offline Teacher cache: 不再加载 8B Teacher Transformer
# ======================================================================
class CachedGenerativeWOSDataset(GenerativeWOSDataset):
    """训练集额外返回稳定 sample_index，用于从离线 Teacher cache 精确取样本。"""

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["sample_index"] = int(idx)
        return item


class CachedTeacherDataCollator:
    """先让原 DataCollator pad 序列，再把 sample_index 作为 int64 tensor 放回 batch。"""

    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features):
        sample_indices = [int(x.pop("sample_index")) for x in features]
        batch = self.base_collator(features)
        batch["sample_index"] = torch.tensor(sample_indices, dtype=torch.long)
        return batch


class OfflineTeacherCache:
    """
    读取 extract_wos_teacher_cache.py 生成的只读 mmap cache。

    每个样本 i：
      start = offsets[i]
      end   = offsets[i+1]
      H_T^KD = teacher_completion_hidden[start:end]          [Ni, 4096]
      h_T^L1 = teacher_completion_hidden[start+l1_rel[i]]     [4096]
      h_T^L2 = teacher_completion_hidden[start+l2_rel[i]]     [4096]

    Teacher common lm_head 已按 common-vocab 的 Student-ID 顺序保存：
      W_T_common [Vcommon, 4096]
    因此：
      z_T_common = H_T^KD @ W_T_common^T (+ bias)
    恢复出来的仍是 Teacher logits，Traditional KD 定义不变。
    """

    def __init__(self, cache_dir: str, expected_num_samples: int):
        self.cache_dir = os.path.abspath(cache_dir)
        manifest_path = os.path.join(self.cache_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"❌ Teacher cache 缺少 manifest.json: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.num_samples = int(self.manifest["num_samples"])
        self.teacher_hidden_size = int(self.manifest["teacher_hidden_size"])
        self.common_vocab_size = int(self.manifest["common_vocab_size"])
        self.cache_dtype = str(self.manifest["cache_dtype"])

        if self.num_samples != int(expected_num_samples):
            raise RuntimeError(
                "❌ Teacher cache 样本数与当前训练集不一致："
                f"cache={self.num_samples}, train={expected_num_samples}"
            )

        files = self.manifest.get("files", {})

        def _path(key, fallback):
            name = files.get(key) or fallback
            path = os.path.join(self.cache_dir, name)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"❌ Teacher cache 缺少文件: {path}")
            return path

        self.offsets = np.load(_path("offsets", "offsets.npy"), mmap_mode="r")
        self.valid_counts = np.load(_path("valid_counts", "valid_counts.npy"), mmap_mode="r")
        self.l1_rel_offsets = np.load(_path("l1_rel_offsets", "l1_rel_offsets.npy"), mmap_mode="r")
        self.l2_rel_offsets = np.load(_path("l2_rel_offsets", "l2_rel_offsets.npy"), mmap_mode="r")
        self.input_hashes = np.load(_path("student_input_sha256", "student_input_sha256.npy"), mmap_mode="r")
        self.completion_hidden = np.load(
            _path("teacher_completion_hidden", "teacher_completion_hidden.npy"),
            mmap_mode="r",
        )
        self.student_common_ids = np.load(
            _path("student_common_ids", "student_common_ids.npy"),
            mmap_mode="r",
        )
        self.teacher_common_ids = np.load(
            _path("teacher_common_ids", "teacher_common_ids.npy"),
            mmap_mode="r",
        )
        self.common_lm_head = np.load(
            _path("teacher_common_lm_head", "teacher_common_lm_head.npy"),
            mmap_mode="r",
        )

        bias_name = files.get("teacher_common_lm_head_bias")
        self.common_lm_head_bias = None
        if bias_name:
            bias_path = os.path.join(self.cache_dir, bias_name)
            if not os.path.isfile(bias_path):
                raise FileNotFoundError(f"❌ manifest 声明了 lm_head bias，但文件不存在: {bias_path}")
            self.common_lm_head_bias = np.load(bias_path, mmap_mode="r")

        expected_total = int(self.manifest["total_completion_prediction_positions"])
        if self.offsets.shape != (self.num_samples + 1,):
            raise RuntimeError(f"❌ offsets shape 异常: {self.offsets.shape}")
        if self.valid_counts.shape != (self.num_samples,):
            raise RuntimeError(f"❌ valid_counts shape 异常: {self.valid_counts.shape}")
        if self.l1_rel_offsets.shape != (self.num_samples,):
            raise RuntimeError(f"❌ l1_rel_offsets shape 异常: {self.l1_rel_offsets.shape}")
        if self.l2_rel_offsets.shape != (self.num_samples,):
            raise RuntimeError(f"❌ l2_rel_offsets shape 异常: {self.l2_rel_offsets.shape}")
        if int(self.offsets[-1]) != expected_total:
            raise RuntimeError(
                f"❌ offsets[-1]={int(self.offsets[-1])} 与 manifest total={expected_total} 不一致。"
            )
        if self.completion_hidden.shape != (expected_total, self.teacher_hidden_size):
            raise RuntimeError(
                "❌ teacher_completion_hidden shape 异常："
                f"{self.completion_hidden.shape} != ({expected_total}, {self.teacher_hidden_size})"
            )
        if self.student_common_ids.shape != (self.common_vocab_size,):
            raise RuntimeError(f"❌ student_common_ids shape 异常: {self.student_common_ids.shape}")
        if self.teacher_common_ids.shape != (self.common_vocab_size,):
            raise RuntimeError(f"❌ teacher_common_ids shape 异常: {self.teacher_common_ids.shape}")
        if self.common_lm_head.shape != (self.common_vocab_size, self.teacher_hidden_size):
            raise RuntimeError(
                "❌ teacher_common_lm_head shape 异常："
                f"{self.common_lm_head.shape} != ({self.common_vocab_size}, {self.teacher_hidden_size})"
            )
        if self.common_lm_head_bias is not None and self.common_lm_head_bias.shape != (self.common_vocab_size,):
            raise RuntimeError(f"❌ Teacher common lm_head bias shape 异常: {self.common_lm_head_bias.shape}")

        print("\n" + "=" * 92)
        print("✅ Offline Teacher cache 加载与 shape 校验通过")
        print(f"Cache dir: {self.cache_dir}")
        print(f"Samples: {self.num_samples:,}")
        print(f"Total Completion KD positions: {expected_total:,}")
        print(f"Teacher hidden size: {self.teacher_hidden_size}")
        print(f"Common vocab: {self.common_vocab_size:,}")
        print(f"Hidden cache: {self.completion_hidden.shape}, {self.completion_hidden.dtype}")
        print(f"Teacher common lm_head: {self.common_lm_head.shape}, {self.common_lm_head.dtype}")
        print("✅ Student 训练阶段不会加载 8B Teacher Transformer")
        print("=" * 92 + "\n")

    @staticmethod
    def _hash_input_ids(input_ids: torch.Tensor, actual_len: int) -> str:
        ids = input_ids[:actual_len].detach().cpu().tolist()
        payload = ",".join(str(int(x)) for x in ids)
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    def verify_batch_layout(
        self,
        sample_indices: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        valid_prediction_mask: torch.Tensor,
        l1_positions: torch.Tensor,
        l2_positions: torch.Tensor,
    ) -> None:
        """严格验证当前 Student batch 与提取 cache 时的样本/position 布局完全一致。"""
        indices = sample_indices.detach().cpu().tolist()
        for row, sample_idx in enumerate(indices):
            sample_idx = int(sample_idx)
            if sample_idx < 0 or sample_idx >= self.num_samples:
                raise IndexError(f"❌ sample_index 越界: {sample_idx}")

            actual_len = int(attention_mask[row].sum().item())
            current_valid_count = int(valid_prediction_mask[row, : max(0, actual_len - 1)].sum().item())
            expected_valid_count = int(self.valid_counts[sample_idx])
            if current_valid_count != expected_valid_count:
                raise RuntimeError(
                    f"❌ sample {sample_idx} Completion KD position 数与 cache 不一致："
                    f"current={current_valid_count}, cache={expected_valid_count}"
                )

            valid_positions = torch.nonzero(
                valid_prediction_mask[row, : max(0, actual_len - 1)],
                as_tuple=False,
            ).flatten()
            pos_to_rel = {
                int(pos): rel
                for rel, pos in enumerate(valid_positions.detach().cpu().tolist())
            }
            current_l1_pos = int(l1_positions[row].item())
            current_l2_pos = int(l2_positions[row].item())
            if current_l1_pos not in pos_to_rel or current_l2_pos not in pos_to_rel:
                raise RuntimeError(f"❌ sample {sample_idx} L1/L2 特殊 token 不在 causal KD mask。")

            if int(pos_to_rel[current_l1_pos]) != int(self.l1_rel_offsets[sample_idx]):
                raise RuntimeError(
                    f"❌ sample {sample_idx} L1 relative offset 与 cache 不一致。"
                )
            if int(pos_to_rel[current_l2_pos]) != int(self.l2_rel_offsets[sample_idx]):
                raise RuntimeError(
                    f"❌ sample {sample_idx} L2 relative offset 与 cache 不一致。"
                )

            expected_hash_raw = self.input_hashes[sample_idx]
            if isinstance(expected_hash_raw, bytes):
                expected_hash = expected_hash_raw.decode("ascii")
            else:
                expected_hash = bytes(expected_hash_raw).decode("ascii")
            current_hash = self._hash_input_ids(input_ids[row], actual_len)
            if current_hash != expected_hash:
                raise RuntimeError(
                    f"❌ sample {sample_idx} Student input token SHA256 与 Teacher cache 不一致。\n"
                    f"current={current_hash}\ncache={expected_hash}\n"
                    "说明数据、Prompt、截断规则或 tokenizer 与提取时发生了变化。"
                )

    def fetch_batch_numpy(self, sample_indices: torch.Tensor):
        """按 DataLoader 当前 batch 顺序拼接缓存，顺序与 boolean mask flatten 完全一致。"""
        indices = [int(x) for x in sample_indices.detach().cpu().tolist()]
        all_hidden = []
        l1_hidden = []
        l2_hidden = []

        for sample_idx in indices:
            start = int(self.offsets[sample_idx])
            end = int(self.offsets[sample_idx + 1])
            count = end - start
            if count != int(self.valid_counts[sample_idx]):
                raise RuntimeError(f"❌ sample {sample_idx} offsets 与 valid_counts 不一致。")

            # np.array(copy=True) 避免把只读 memmap 直接暴露给 torch。
            sample_hidden = np.array(
                self.completion_hidden[start:end],
                dtype=np.float32,
                copy=True,
            )
            all_hidden.append(sample_hidden)

            l1_abs = start + int(self.l1_rel_offsets[sample_idx])
            l2_abs = start + int(self.l2_rel_offsets[sample_idx])
            l1_hidden.append(np.array(self.completion_hidden[l1_abs], dtype=np.float32, copy=True))
            l2_hidden.append(np.array(self.completion_hidden[l2_abs], dtype=np.float32, copy=True))

        return (
            np.concatenate(all_hidden, axis=0),
            np.stack(l1_hidden, axis=0),
            np.stack(l2_hidden, axis=0),
        )


class OfflineCachedKDTrainer(FullIOPeftTrainer):
    """
    统一离线 Teacher cache 蒸馏：

      Traditional KD:
        cached Teacher final hidden -> cached Teacher common lm_head -> Teacher logits
        -> temperature-scaled KL(Student common logits, Teacher common logits)

      Hidden Align:
        alpha*MSE(P(hS_k), hT_k) + (1-alpha)*(1-cosine(...)), k in {L1,L2}

      Hidden Hier:
        MSE(P(hS_L2)-P(hS_L1), hT_L2-hT_L1)

    不加载 Teacher 8B Transformer。Teacher common lm_head 只读驻留 GPU。
    """

    def __init__(
        self,
        *trainer_args,
        teacher_cache: OfflineTeacherCache,
        student_common_ids: List[int],
        kd_temperature: float,
        kd_weight: float,
        student_l1_token_id: int,
        student_l2_token_id: int,
        align_weight: float,
        hier_weight: float,
        align_alpha: float,
        l1_special_weight: float,
        l2_special_weight: float,
        **trainer_kwargs,
    ):
        self.teacher_cache = teacher_cache
        self._student_common_ids_cpu = [int(x) for x in student_common_ids]
        self.kd_temperature = float(kd_temperature)
        self.kd_weight = float(kd_weight)
        self.student_l1_token_id = int(student_l1_token_id)
        self.student_l2_token_id = int(student_l2_token_id)
        self.align_weight = float(align_weight)
        self.hier_weight = float(hier_weight)
        self.align_alpha = float(align_alpha)
        self.l1_special_weight = float(l1_special_weight)
        self.l2_special_weight = float(l2_special_weight)

        if self.kd_temperature <= 0:
            raise ValueError("❌ kd_temperature 必须 > 0。")
        if self.kd_weight < 0 or self.align_weight < 0 or self.hier_weight < 0:
            raise ValueError("❌ KD/Align/Hier weight 不能小于 0。")
        if not (0.0 <= self.align_alpha <= 1.0):
            raise ValueError("❌ align_alpha 必须位于 [0,1]。")
        if self.l1_special_weight < 0 or self.l2_special_weight < 0:
            raise ValueError("❌ L1/L2 special weight 不能小于 0。")
        if self.l1_special_weight + self.l2_special_weight <= 0:
            raise ValueError("❌ L1/L2 special weight 不能同时为 0。")

        self._loss_meter_task = 0.0
        self._loss_meter_kd = 0.0
        self._loss_meter_total = 0.0
        self._loss_meter_count = 0
        self._meter_align_l1_mse = 0.0
        self._meter_align_l1_cos = 0.0
        self._meter_align_l1 = 0.0
        self._meter_align_l2_mse = 0.0
        self._meter_align_l2_cos = 0.0
        self._meter_align_l2 = 0.0
        self._meter_align = 0.0
        self._meter_hier = 0.0
        self._printed_sanity = False

        super().__init__(*trainer_args, **trainer_kwargs)

        self.student_common_ids = torch.tensor(
            self._student_common_ids_cpu,
            dtype=torch.long,
            device=self.args.device,
        )

        if self.student_common_ids.numel() != self.teacher_cache.common_vocab_size:
            raise RuntimeError(
                "❌ Student common-vocab 数与 cache 不一致："
                f"student={self.student_common_ids.numel()}, cache={self.teacher_cache.common_vocab_size}"
            )

        cached_student_ids = np.asarray(self.teacher_cache.student_common_ids, dtype=np.int64)
        current_student_ids = np.asarray(self._student_common_ids_cpu, dtype=np.int64)
        if not np.array_equal(cached_student_ids, current_student_ids):
            raise RuntimeError("❌ 当前 Student common IDs 与 Teacher cache 保存的 IDs 不完全一致。")

        student_vocab_rows = unwrap_training_model(self.model).get_output_embeddings().weight.shape[0]
        if int(self.student_common_ids.max().item()) >= int(student_vocab_rows):
            raise RuntimeError("❌ student_common_ids 超出 Student lm_head 行数。")

        # Teacher common lm_head 只读驻留 GPU。cache 是 float16；在 bf16 训练时
        # 转成 bf16 更适合 A6000 上的矩阵乘。KL 前仍统一转 float32。
        target_dtype = torch.bfloat16 if self.args.bf16 else torch.float32
        self.teacher_common_lm_head = torch.from_numpy(
            np.array(self.teacher_cache.common_lm_head, copy=True)
        ).to(device=self.args.device, dtype=target_dtype)
        self.teacher_common_lm_head.requires_grad_(False)

        self.teacher_common_lm_head_bias = None
        if self.teacher_cache.common_lm_head_bias is not None:
            self.teacher_common_lm_head_bias = torch.from_numpy(
                np.array(self.teacher_cache.common_lm_head_bias, copy=True)
            ).to(device=self.args.device, dtype=target_dtype)
            self.teacher_common_lm_head_bias.requires_grad_(False)

        projector = get_hidden_projector(self.model)
        needs_projector = self.align_weight > 0 or self.hier_weight > 0
        if needs_projector:
            if projector is None:
                raise RuntimeError("❌ Hidden Align/Hier 开启，但 Student 没有 shared hidden_projector。")
            if int(projector.out_features) != int(self.teacher_cache.teacher_hidden_size):
                raise RuntimeError(
                    "❌ Projector 输出维度与 cached Teacher hidden size 不一致："
                    f"projector={projector.out_features}, teacher={self.teacher_cache.teacher_hidden_size}"
                )
        elif projector is not None:
            raise RuntimeError("❌ Traditional-only 模式不应创建 hidden_projector。")

        print("\n" + "=" * 100)
        print("🚀 Offline Teacher Cache KD Trainer 就绪")
        print("Teacher 8B Transformer: NOT LOADED")
        print(f"Teacher cache: {self.teacher_cache.cache_dir}")
        print(f"Teacher common lm_head on GPU: {tuple(self.teacher_common_lm_head.shape)}")
        print(f"Common vocab: {self.student_common_ids.numel():,}")
        print(f"KD temperature/weight: {self.kd_temperature}/{self.kd_weight}")
        print(f"Hidden Align weight: {self.align_weight}")
        print(f"Hidden Hier weight:  {self.hier_weight}")
        if projector is not None:
            print(f"Shared projector: R^{projector.in_features} -> R^{projector.out_features}")
        print("=" * 100 + "\n")

    def _find_second_occurrence_positions(self, input_ids, token_id: int, name: str):
        positions = []
        for row_idx in range(input_ids.size(0)):
            row_positions = torch.nonzero(input_ids[row_idx].eq(token_id), as_tuple=False).flatten()
            if row_positions.numel() != 2:
                raise RuntimeError(
                    f"❌ batch row={row_idx} 的 {name} 应严格出现 2 次，"
                    f"实际={row_positions.detach().cpu().tolist()}"
                )
            positions.append(row_positions[1])
        return torch.stack(positions).to(device=input_ids.device, dtype=torch.long)

    def _get_student_final_norm_module(self, model):
        raw = unwrap_training_model(model)
        base_causal_lm = raw.get_base_model() if hasattr(raw, "get_base_model") else raw
        decoder = getattr(base_causal_lm, "model", None)
        if decoder is None:
            raise RuntimeError("❌ Student 找不到 Llama decoder (.model)。")
        final_norm = getattr(decoder, "norm", None)
        if final_norm is None:
            raise RuntimeError("❌ Student decoder 找不到 final norm。")
        return final_norm

    def _temperature_kl(self, student_logits, teacher_logits):
        t = self.kd_temperature
        student_log_probs = F.log_softmax(student_logits.float() / t, dim=-1)
        teacher_probs = F.softmax(teacher_logits.float() / t, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (t ** 2)

    def _hidden_align_component(self, projected_student, teacher_hidden):
        projected_student = projected_student.float()
        teacher_hidden = teacher_hidden.float()
        mse = F.mse_loss(projected_student, teacher_hidden, reduction="mean")
        cos_loss = (
            1.0
            - F.cosine_similarity(
                projected_student,
                teacher_hidden,
                dim=-1,
                eps=1e-8,
            )
        ).mean()
        combined = self.align_alpha * mse + (1.0 - self.align_alpha) * cos_loss
        return combined, mse, cos_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch

        if "sample_index" not in inputs:
            raise RuntimeError("❌ Offline KD batch 缺少 sample_index。")
        sample_index = inputs.pop("sample_index")

        if "labels" not in inputs:
            raise RuntimeError("❌ KD batch 缺少 labels。")

        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        l1_positions = self._find_second_occurrence_positions(
            input_ids, self.student_l1_token_id, "L1 token"
        )
        l2_positions = self._find_second_occurrence_positions(
            input_ids, self.student_l2_token_id, "L2 token"
        )
        if not torch.all(l1_positions < l2_positions):
            raise RuntimeError("❌ L1 第二次 occurrence 必须位于 L2 之前。")

        need_student_hidden = self.align_weight > 0 or self.hier_weight > 0
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
            raise RuntimeError("❌ Student forward 没有返回 task loss。")

        shift_labels = labels[:, 1:]
        valid_prediction_mask = shift_labels.ne(-100)
        valid_count = int(valid_prediction_mask.sum().item())
        if valid_count <= 0:
            raise RuntimeError("❌ 当前 batch 没有 Completion supervision token。")

        row_ids = torch.arange(input_ids.size(0), device=input_ids.device, dtype=torch.long)
        if not valid_prediction_mask[row_ids, l1_positions].all():
            raise RuntimeError("❌ L1 第二次 token 未落在 Completion KD mask。")
        if not valid_prediction_mask[row_ids, l2_positions].all():
            raise RuntimeError("❌ L2 第二次 token 未落在 Completion KD mask。")

        # 每个 batch 都做严格 cache 对齐审计；若数据/Prompt/截断有任何变化立即停止。
        self.teacher_cache.verify_batch_layout(
            sample_indices=sample_index,
            input_ids=input_ids,
            attention_mask=attention_mask,
            valid_prediction_mask=valid_prediction_mask,
            l1_positions=l1_positions,
            l2_positions=l2_positions,
        )

        student_shift_logits = student_outputs.logits[:, :-1, :]
        student_selected_full_logits = student_shift_logits[valid_prediction_mask]
        student_common_logits = student_selected_full_logits.index_select(
            dim=-1,
            index=self.student_common_ids,
        )

        cached_kd_np, t_l1_np, t_l2_np = self.teacher_cache.fetch_batch_numpy(sample_index)
        if cached_kd_np.shape[0] != valid_count:
            raise RuntimeError(
                "❌ 当前 batch cached Teacher hidden 数与 Student causal mask 不一致："
                f"cache={cached_kd_np.shape[0]}, student={valid_count}"
            )

        cache_compute_dtype = self.teacher_common_lm_head.dtype
        cached_kd_hidden = torch.from_numpy(cached_kd_np).to(
            device=self.args.device,
            dtype=cache_compute_dtype,
            non_blocking=True,
        )

        # 关键：这里先把 cached final hidden 还原成 Teacher logits。
        # 因此 Traditional KD 仍是 logits KL，而不是 hidden MSE。
        teacher_common_logits = F.linear(
            cached_kd_hidden,
            self.teacher_common_lm_head,
            self.teacher_common_lm_head_bias,
        )

        if student_common_logits.shape != teacher_common_logits.shape:
            raise RuntimeError(
                "❌ Student / reconstructed Teacher common logits shape 不一致："
                f"student={tuple(student_common_logits.shape)}, "
                f"teacher={tuple(teacher_common_logits.shape)}"
            )

        full_kd_loss = self._temperature_kl(
            student_common_logits,
            teacher_common_logits,
        )

        zero = task_loss.new_zeros(())
        align_l1 = align_l1_mse = align_l1_cos = zero
        align_l2 = align_l2_mse = align_l2_cos = zero
        align_loss = zero
        hier_loss = zero
        s_l1_hidden = s_l2_hidden = None
        projected_s_l1 = projected_s_l2 = None

        if need_student_hidden:
            if "last_hidden" not in captured:
                raise RuntimeError("❌ Student final hidden hook 没有捕获到输出。")
            student_last_hidden = captured["last_hidden"]
            s_l1_hidden = student_last_hidden[row_ids, l1_positions, :]
            s_l2_hidden = student_last_hidden[row_ids, l2_positions, :]

            t_l1_hidden = torch.from_numpy(t_l1_np).to(
                device=self.args.device, dtype=torch.float32, non_blocking=True
            )
            t_l2_hidden = torch.from_numpy(t_l2_np).to(
                device=self.args.device, dtype=torch.float32, non_blocking=True
            )

            projector = get_hidden_projector(model)
            if projector is None:
                raise RuntimeError("❌ 训练中找不到 shared hidden_projector。")
            projected_s_l1 = projector(s_l1_hidden.float())
            projected_s_l2 = projector(s_l2_hidden.float())

            if projected_s_l1.shape != t_l1_hidden.shape:
                raise RuntimeError(
                    "❌ projected Student L1 与 cached Teacher L1 hidden shape 不一致："
                    f"student={tuple(projected_s_l1.shape)}, teacher={tuple(t_l1_hidden.shape)}"
                )
            if projected_s_l2.shape != t_l2_hidden.shape:
                raise RuntimeError("❌ projected Student L2 与 cached Teacher L2 hidden shape 不一致。")

            if self.align_weight > 0:
                align_l1, align_l1_mse, align_l1_cos = self._hidden_align_component(
                    projected_s_l1, t_l1_hidden
                )
                align_l2, align_l2_mse, align_l2_cos = self._hidden_align_component(
                    projected_s_l2, t_l2_hidden
                )
                weight_sum = self.l1_special_weight + self.l2_special_weight
                align_loss = (
                    self.l1_special_weight * align_l1
                    + self.l2_special_weight * align_l2
                ) / weight_sum

            if self.hier_weight > 0:
                delta_student = projected_s_l2 - projected_s_l1
                delta_teacher = t_l2_hidden - t_l1_hidden
                hier_loss = F.mse_loss(
                    delta_student.float(),
                    delta_teacher.float(),
                    reduction="mean",
                )

        total_loss = (
            task_loss
            + self.kd_weight * full_kd_loss
            + self.align_weight * align_loss
            + self.hier_weight * hier_loss
        )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "❌ total loss NaN/Inf："
                f"task={task_loss.detach().float().item()}, "
                f"kd={full_kd_loss.detach().float().item()}, "
                f"align={align_loss.detach().float().item()}, "
                f"hier={hier_loss.detach().float().item()}"
            )

        if not self._printed_sanity:
            print("\n" + "=" * 104)
            print("✅ Offline Teacher cache 首批次严格检查通过")
            print(f"sample_index: {sample_index.detach().cpu().tolist()}")
            print(f"Student input: {tuple(input_ids.shape)}")
            print(f"有效 Completion KD positions: {valid_count}")
            print(f"Cached Teacher KD hidden: {tuple(cached_kd_hidden.shape)}")
            print(f"Reconstructed Teacher common logits: {tuple(teacher_common_logits.shape)}")
            print(f"Student common logits: {tuple(student_common_logits.shape)}")
            print(f"L1 positions: {l1_positions.detach().cpu().tolist()}")
            print(f"L2 positions: {l2_positions.detach().cpu().tolist()}")
            print("✅ Teacher 8B Transformer 未加载；logits = cached hidden × cached lm_head")
            print("✅ Traditional KD 仍是 Full-Completion common-vocab temperature KL")
            if need_student_hidden:
                print(f"Student L1/L2 hidden: {tuple(s_l1_hidden.shape)} / {tuple(s_l2_hidden.shape)}")
                print(f"Projected hidden: {tuple(projected_s_l1.shape)}")
                print("✅ Align 使用原始 hidden MSE+cosine 公式")
                print("✅ Hier 使用原始 hidden relation MSE 公式")
            print(f"Task loss:  {task_loss.detach().float().item():.6f}")
            print(f"Logits KD:  {full_kd_loss.detach().float().item():.6f}")
            print(f"Align loss: {align_loss.detach().float().item():.6f}")
            print(f"Hier loss:  {hier_loss.detach().float().item():.6f}")
            print(f"Total loss: {total_loss.detach().float().item():.6f}")
            print(
                "公式: Total = Task + "
                f"{self.kd_weight}*FullLogitsKD + "
                f"{self.align_weight}*HiddenAlign + "
                f"{self.hier_weight}*HiddenHier"
            )
            print("=" * 104 + "\n")
            self._printed_sanity = True

        if model.training:
            self._loss_meter_task += task_loss.detach().float().item()
            self._loss_meter_kd += full_kd_loss.detach().float().item()
            self._loss_meter_total += total_loss.detach().float().item()
            self._loss_meter_count += 1
            self._meter_align_l1_mse += align_l1_mse.detach().float().item()
            self._meter_align_l1_cos += align_l1_cos.detach().float().item()
            self._meter_align_l1 += align_l1.detach().float().item()
            self._meter_align_l2_mse += align_l2_mse.detach().float().item()
            self._meter_align_l2_cos += align_l2_cos.detach().float().item()
            self._meter_align_l2 += align_l2.detach().float().item()
            self._meter_align += align_loss.detach().float().item()
            self._meter_hier += hier_loss.detach().float().item()

        if return_outputs:
            return total_loss, student_outputs
        return total_loss

    def log(self, logs, *args, **kwargs):
        logs = dict(logs)
        if self._loss_meter_count > 0:
            n = self._loss_meter_count
            logs["Train/Task_Loss"] = self._loss_meter_task / n
            logs["Train/KD_Loss"] = self._loss_meter_kd / n
            logs["Train/Align_L1_MSE"] = self._meter_align_l1_mse / n
            logs["Train/Align_L1_CosLoss"] = self._meter_align_l1_cos / n
            logs["Train/Align_L1_Total"] = self._meter_align_l1 / n
            logs["Train/Align_L2_MSE"] = self._meter_align_l2_mse / n
            logs["Train/Align_L2_CosLoss"] = self._meter_align_l2_cos / n
            logs["Train/Align_L2_Total"] = self._meter_align_l2 / n
            logs["Train/Align_Loss"] = self._meter_align / n
            logs["Train/Hier_Hidden_Loss"] = self._meter_hier / n
            logs["Train/Total_Loss"] = self._loss_meter_total / n
            logs["Train/KD_Weight"] = self.kd_weight
            logs["Train/Align_Weight"] = self.align_weight
            logs["Train/Hier_Weight"] = self.hier_weight
            logs["Train/Align_Alpha"] = self.align_alpha
            logs["Train/KD_Temperature"] = self.kd_temperature
            logs["Train/Common_Vocab_Size"] = float(self.student_common_ids.numel())

            self._loss_meter_task = 0.0
            self._loss_meter_kd = 0.0
            self._loss_meter_total = 0.0
            self._loss_meter_count = 0
            self._meter_align_l1_mse = 0.0
            self._meter_align_l1_cos = 0.0
            self._meter_align_l1 = 0.0
            self._meter_align_l2_mse = 0.0
            self._meter_align_l2_cos = 0.0
            self._meter_align_l2 = 0.0
            self._meter_align = 0.0
            self._meter_hier = 0.0

        return FullIOPeftTrainer.log(self, logs, *args, **kwargs)


# ======================================================================
# 10. Main
# ======================================================================
def main(args) -> None:
    set_all_seeds(args.seed)
    check_runtime_compatibility()

    mode = str(args.experiment_mode).strip().lower()
    if mode not in {"traditional", "align", "hier", "full"}:
        raise ValueError(
            f"❌ experiment_mode 必须是 traditional/align/hier/full，实际={mode!r}"
        )

    args.effective_align_weight = (
        float(args.align_weight) if mode in {"align", "full"} else 0.0
    )
    args.effective_hier_weight = (
        float(args.hier_weight) if mode in {"hier", "full"} else 0.0
    )

    if not args.save_dir:
        suffix = {
            "traditional": "checkpoints_wos_offline_logits_kd_seed42_lora16",
            "align": "checkpoints_wos_offline_logits_kd_hidden_align_seed42_lora16",
            "hier": "checkpoints_wos_offline_logits_kd_hidden_hier_seed42_lora16",
            "full": "checkpoints_wos_offline_logits_kd_hidden_align_hier_seed42_lora16",
        }[mode]
        args.save_dir = os.path.join(
            "/data1/lzp/code/hierarchical_distill", suffix
        ) + "/"

    if not args.run_name:
        args.run_name = {
            "traditional": "WOS-Offline-8Bto1B-LogitsKD-Seed42-Lora16",
            "align": "WOS-Offline-8Bto1B-LogitsKD-HiddenAlign-Seed42-Lora16",
            "hier": "WOS-Offline-8Bto1B-LogitsKD-HiddenHier-Seed42-Lora16",
            "full": "WOS-Offline-8Bto1B-LogitsKD-HiddenAlignHier-Seed42-Lora16",
        }[mode]

    args.save_dir = os.path.abspath(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    resume_checkpoint = resolve_resume_checkpoint(args)
    run_signature = build_run_signature(args)
    if resume_checkpoint:
        validate_resume_configuration(resume_checkpoint, run_signature)
        print(f"🔄 将从完整 WOS 蒸馏断点恢复: {resume_checkpoint}")
    else:
        print("🆕 开始新的 WOS 离线蒸馏实验。")

    # ------------------------------------------------------------------
    # A. taxonomy + label2id + clean split 严格校验
    # ------------------------------------------------------------------
    l1_label2id, l2_label2id, l2_to_l1 = load_and_validate_label_space(
        dict_path=args.dict_path,
        taxonomy_path=args.taxonomy_path,
        expected_num_l1=args.expected_num_l1,
        expected_num_l2=args.expected_num_l2,
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
    # B. Student tokenizer + 特殊 token
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
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
    # C. Offline Teacher cache：先于训练加载/校验；不加载 8B Teacher 模型
    # ------------------------------------------------------------------
    teacher_cache = OfflineTeacherCache(
        cache_dir=args.teacher_cache_dir,
        expected_num_samples=len(train_data),
    )

    manifest = teacher_cache.manifest
    if int(manifest.get("max_seq_length", -1)) != int(args.max_seq_length):
        raise RuntimeError(
            "❌ 当前 max_seq_length 与 Teacher cache 提取时不一致："
            f"train={args.max_seq_length}, cache={manifest.get('max_seq_length')}"
        )
    if int(manifest.get("reserve_tokens", -1)) != int(args.reserve_tokens):
        raise RuntimeError(
            "❌ 当前 reserve_tokens 与 Teacher cache 提取时不一致："
            f"train={args.reserve_tokens}, cache={manifest.get('reserve_tokens')}"
        )
    if str(manifest.get("system_prompt", "")) != str(args.system_prompt):
        raise RuntimeError("❌ 当前 system_prompt 与 Teacher cache 提取时不一致。")

    cached_student_common_ids = [
        int(x) for x in np.asarray(teacher_cache.student_common_ids).tolist()
    ]
    if not cached_student_common_ids:
        raise RuntimeError("❌ Teacher cache 的 student_common_ids 为空。")
    if max(cached_student_common_ids) >= len(tokenizer):
        raise RuntimeError("❌ Teacher cache 的 student_common_ids 超出当前 Student tokenizer。")

    # 四个 protected token 必须都保留在 cache common vocab 中。
    common_id_set = set(cached_student_common_ids)
    for role in ("ctx_start", "ctx_end", "l1", "l2"):
        if int(token_ids[role]) not in common_id_set:
            raise RuntimeError(
                f"❌ {role} token ID={token_ids[role]} 未进入离线 common vocab。"
            )

    with open(
        os.path.join(args.save_dir, "offline_teacher_cache_info.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "teacher_cache_dir": os.path.abspath(args.teacher_cache_dir),
                "teacher_cache_manifest": manifest,
                "traditional_kd_reconstructed_from_cached_hidden": True,
                "teacher_8b_transformer_loaded_during_student_training": False,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ------------------------------------------------------------------
    # D. 训练前正文审计
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
    # E. Train Dataset 带 sample_index；Validation callback 仍用原逻辑
    # ------------------------------------------------------------------
    train_dataset = CachedGenerativeWOSDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        reserve_tokens=args.reserve_tokens,
        system_prompt=args.system_prompt,
        split_name="训练",
    )

    # ------------------------------------------------------------------
    # F. Student 从 raw 1B 初始化；Teacher Transformer 不再加载
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
            "❌ tokenizer 长度与 Student input embedding 行数不一致："
            f"tokenizer={len(tokenizer)}, model={embedding_rows}。"
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    actually_tied = enable_full_input_output_training(model)

    # 只有 Hidden Align/Hier 开启时才创建 Projector。
    needs_projector = (
        args.effective_align_weight > 0 or args.effective_hier_weight > 0
    )
    if needs_projector:
        student_hidden_size = int(model.config.hidden_size)
        teacher_hidden_size = int(teacher_cache.teacher_hidden_size)
        model.hidden_projector = nn.Linear(
            student_hidden_size,
            teacher_hidden_size,
            bias=True,
            dtype=torch.float32,
        )
        model.hidden_projector.requires_grad_(True)
        print("\n" + "=" * 70)
        print("🔗 已创建 L1/L2 共享 Hidden-state Projector")
        print(f"P: R^{student_hidden_size} -> R^{teacher_hidden_size}")
        print("L1/L2 严格共享同一个 P")
        print("=" * 70 + "\n")
    else:
        print("✅ Traditional-only：不创建 hidden projector。")

    model.enable_input_require_grads()
    if not model.get_input_embeddings().weight.requires_grad:
        raise RuntimeError("❌ input embedding 未进入训练。")
    if not model.get_output_embeddings().weight.requires_grad:
        raise RuntimeError("❌ lm_head 未进入训练。")

    print("\n" + "=" * 82)
    print("🎯 WOS Offline Teacher-cache KD 配置")
    print(f"Mode: {mode}")
    print(f"Student init: {args.model_path}")
    print(f"Teacher cache: {args.teacher_cache_dir}")
    print("Teacher 8B Transformer: 不加载")
    print(f"input/lm_head 实际共享: {actually_tied}")
    print("LoRA: train | full input embedding/lm_head: train")
    print(f"KD temperature/weight: {args.kd_temperature}/{args.kd_weight}")
    print(f"Hidden Align weight: {args.effective_align_weight}")
    print(f"Hidden Hier weight:  {args.effective_hier_weight}")
    print(f"Align alpha(MSE/Cos): {args.align_alpha}/{1.0-args.align_alpha}")
    print("Traditional KD: cached hidden -> cached Teacher lm_head -> logits KL")
    print("Align/Hier: cached Teacher L1/L2 final hidden")
    print("=" * 82 + "\n")
    model.print_trainable_parameters()

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

    base_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=True,
    )
    collator = CachedTeacherDataCollator(base_collator)

    trainer = OfflineCachedKDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[swanlab_callback, f1_eval_callback],
        io_learning_rate=args.io_lr,
        projector_learning_rate=(args.projector_lr if needs_projector else None),
        teacher_cache=teacher_cache,
        student_common_ids=cached_student_common_ids,
        kd_temperature=args.kd_temperature,
        kd_weight=args.kd_weight,
        student_l1_token_id=token_ids["l1"],
        student_l2_token_id=token_ids["l2"],
        align_weight=args.effective_align_weight,
        hier_weight=args.effective_hier_weight,
        align_alpha=args.align_alpha,
        l1_special_weight=args.l1_special_weight,
        l2_special_weight=args.l2_special_weight,
    )

    print("\n" + "=" * 82)
    print(f"🚀 开始 WOS Offline KD: {mode.upper()}")
    print(
        "Loss = Task_CE + "
        f"{args.kd_weight}*FullLogitsKD + "
        f"{args.effective_align_weight}*HiddenAlign + "
        f"{args.effective_hier_weight}*HiddenHier"
    )
    print(f"Epoch={args.num_epochs} | batch={args.batch_size} | accum={args.accumulation_steps}")
    print(f"LoRA LR={args.lr} | IO LR={args.io_lr} | projector LR={args.projector_lr if needs_projector else 'N/A'}")
    print("=" * 82 + "\n")

    if trainer.is_world_process_zero():
        atomic_write_json(os.path.join(args.save_dir, "wos_distill_config.json"), {
            "run_signature": run_signature,
            "arguments": vars(args),
            "resolved_resume_checkpoint": resume_checkpoint,
        })
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    if trainer.is_world_process_zero():
        final_checkpoint_path = os.path.join(args.save_dir, "final_kd_checkpoint")
        save_peft_checkpoint_with_full_io(
            model=model,
            tokenizer=tokenizer,
            output_dir=final_checkpoint_path,
            overwrite=True,
        )

        best_checkpoint_path = os.path.join(args.save_dir, "best_kd_checkpoint")
        if not os.path.isdir(best_checkpoint_path):
            print("⚠️ KD 训练期间没有触发 best 保存，将 final 同步保存为 best。")
            save_peft_checkpoint_with_full_io(
                model=model,
                tokenizer=tokenizer,
                output_dir=best_checkpoint_path,
                overwrite=True,
            )

        if args.export_merged_model:
            final_model_path = os.path.join(args.save_dir, "final_model")
            best_model_path = os.path.join(args.save_dir, "best_kd_model")
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
# 11. 参数解析
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
            "WOS offline-cache mixed-space KD: raw Llama-3.2-1B Student; "
            "Traditional full-completion logits KD reconstructed from cached Teacher final hidden + lm_head; "
            "optional original hidden-state Align / Hier losses; no 8B Teacher Transformer during Student training."
        )
    )

    # ================= 路径参数 =================
    parser.add_argument("--model_path", type=str, default="/data1/lzp/code/llama-3.2_1b/")
    parser.add_argument("--teacher_cache_dir", type=str, default="/data1/lzp/code/hierarchical_distill/teacher_wos_hiddenstates/")
    parser.add_argument("--train_data_path", type=str, default="/data1/lzp/code/data/wos/train_formatted.jsonl")
    parser.add_argument("--val_data_path", type=str, default="/data1/lzp/code/data/wos/val_formatted.jsonl")
    parser.add_argument("--taxonomy_path", type=str, default="/data1/lzp/code/data/wos/wos.taxnomy")
    parser.add_argument("--dict_path", type=str, default="/data1/lzp/code/data/wos/label2id.json")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="不传则新实验；auto/true/latest 恢复最近完整断点；也可指定原 save_dir/checkpoints/checkpoint-N。",
    )

    # ================= 数据与随机种子 =================
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_num_l1", type=int, default=7)
    parser.add_argument("--expected_num_l2", type=int, default=134)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--reserve_tokens", type=int, default=300)

    # ================= 正文边界审计 =================
    parser.add_argument("--body_audit_print_samples", type=int, default=5)
    parser.add_argument("--body_audit_preview_chars", type=int, default=300)
    parser.add_argument("--body_audit_scan_limit", type=int, default=-1)

    # ================= 核心训练参数 =================
    parser.add_argument("--num_epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--io_lr", type=float, default=1.0e-5)
    parser.add_argument("--projector_lr", type=float, default=1.0e-4)

    # ================= Traditional logits KD =================
    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--kd_weight", type=float, default=1.0)

    # ================= 消融 / Hidden losses =================
    parser.add_argument(
        "--experiment_mode",
        type=str,
        choices=["traditional", "align", "hier", "full"],
        default="traditional",
    )
    parser.add_argument("--align_weight", type=float, default=1.0)
    parser.add_argument("--hier_weight", type=float, default=1.0)
    parser.add_argument("--align_alpha", type=float, default=0.5)
    parser.add_argument("--l1_special_weight", type=float, default=1.0)
    parser.add_argument("--l2_special_weight", type=float, default=1.0)

    # ================= Scheduler =================
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)

    # ================= LoRA =================
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # ================= 验证与保存 =================
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--loss_eval_batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--save_total_limit", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--export_merged_model", type=str2bool, default=True)

    # ================= 性能 =================
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dataloader_pin_memory", type=str2bool, default=True)
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")

    # ================= Prompt / SwanLab =================
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are a specialized academic classifier for the WOS two-level taxonomy. "
            "Follow the required output format exactly. Do not produce explanations or any extra conversational text."
        ),
    )
    parser.add_argument("--project_name", type=str, default="WOS-Offline-LogitsKD-Hidden-Hierarchical-KD")
    parser.add_argument("--run_name", type=str, default="")

    args = parser.parse_args()
    main(args)
