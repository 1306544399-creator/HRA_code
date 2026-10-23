# HRA_code
# 关于蒸馏脚本

| 文件 | 知识来源与区别 |
| --- | --- |
| [train_distill_mi_ma.py](train_distill_mi_ma.py) | （主实验）原始单层离线蒸馏；支持监督分类、logit、feature、attention 损失，并保存 macro/micro 对应模型 |
| [train_distill_mi_ma_cached_topk.py](train_distill_mi_ma_cached_topk.py) | （level-token的top-k探测实验脚本）从紧凑 FP32 缓存构造层级 token Top-K 目标；`--attention_topk` 支持正整数或 `all` |
| [train_distill_mi_ma_feature_ablation.py](train_distill_mi_ma_feature_ablation.py) | （feature-mean消融）`level_token` 与 `text_mean` 特征载体消融；分类仍使用层级 token |
| [train_distill_mi_ma_fullatt.py](train_distill_mi_ma_fullatt.py) | 单个 Transformer 层的完整有效 token 对 attention 蒸馏 |
| [train_distill_mi_ma_fullatt_ltot.py](train_distill_mi_ma_fullatt_ltot.py) | （level-token的fullatt蒸馏）`--full_attention_scope full/level_to_text`，默认 `full` |
| [train_distill_mi_ma_fullatt_scope.py](train_distill_mi_ma_fullatt_scope.py) | （当时主要为了做正文 token → 正文 token 的全量注意力蒸馏）`--full_attention_scope full/text_to_text`，默认 `full` |
| [train_distill_mi_ma_text_topk.py](train_distill_mi_ma_text_topk.py) | （这个是fp16保存了完整的att，然后选择的top-k，之后的top-k的选择可能会有点差别）增加 `matched_text_topk` 对照：使用末尾文本查询，教师 Top-K 加 `other` 桶 |
| [train_distill_mi_ma_text_topk1.py](train_distill_mi_ma_text_topk1.py) | 同类对照的另一实现：FP32 缓存/选取 → FP16 教师概率桶 → FP32 KL；保留为独立分支 |
| [teacher_bank_ops.py](teacher_bank_ops.py) | 多层分支共用的选层、配对与特征/attention 融合工具，不是独立训练入口 |
| [train_distill_mi_ma_multilayer.py](train_distill_mi_ma_multilayer.py) | `legacy/bank` 两种模式，支持单层、平均、多层一对一匹配 |
| [train_distill_mi_ma_multilayer_sampling.py](train_distill_mi_ma_multilayer_sampling.py) | 在多层蒸馏上增加固定、均匀及逐 epoch 随机选层 |

# 关于训练与评估

| 文件 | 用途 |
| --- | --- |
| [train_baseline.py](train_baseline.py) | 固定长度 padding 的分类基线，同时定义共用的数据集、模型与指标工具 |
| [train_baseline_collator.py](train_baseline_collator.py) | （我的教师学生主要用的这个教你）带独立 collator 的分类基线；`--padding_mode fixed/dynamic`，默认 `dynamic` |
| [eval_localconcat.py](eval_localconcat.py) | （评估脚本）对兼容的 LoRA 分类 checkpoint 进行独立评估，基于 `train_baseline` 的模型结构 |
| [train_generative_sft.py](train_generative_sft.py) | （正在跑的SFT脚本）生成式 HTC LoRA SFT，复用 collator 基线的数据/指标约定，生成按层组
