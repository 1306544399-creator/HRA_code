# HRA_code

- **单 Token 设计**：每一层使用一个层级特殊 Token。WOS 使用 L1/L2，NYT 使用 D1–D8，BGC 使用 D1–D4。
- **WOS 双 Token 设计**：每层分别使用 START、END 两个特殊 Token，用于对比不同层级表示载体。跨层级关系分别沿 START→START 和 END→END 两条分支建模。
- **空层处理**：NYT/BGC 的深层样本可能存在空标签 `[]`，但生成结构中的层级 Token 仍保留，以确保层级位置一致。

### 1 WOS：单 Token 主实验

| 文件 | 用途与区别 |
| --- | --- |
| [`WOS/prepare_data.py`](WOS/prepare_data.py) | （实验之前对数据集预处理）读取 WOS 原始数据和 taxonomy，规范化训练/验证/测试样本，生成统一的 `label2id.json` 及 `*_formatted.jsonl` |
| [`WOS/student_train_wos.py`](WOS/student_train_wos.py) | （学生单token训练）单 Token 生成式学生模型 SFT；使用每层一个特殊 Token 的层级标签输出格式 |
| [`WOS/student_test_wos.py`](WOS/student_test_wos.py) | （学生单token测试）独立测试单 Token SFT 学生模型，计算整体与 L1/L2 分层指标 |
| [`WOS/teacher_train_wos.py`](WOS/teacher_train_wos.py) | （教师单token训练）与学生使用一致层级输出结构的教师模型 SFT |
| [`WOS/teacher_test_wos.py`](WOS/teacher_test_wos.py) | （教师单token测试）独立测试单 Token SFT 教师模型 |
| [`WOS/extract_wos_teacher.py`](WOS/extract_wos_teacher.py) | （离线提取教师特征）离线缓存教师生成部分的隐状态、层级 Token 偏移量及公共词表输出层参数，为后续 KD / Align / Hier 提供教师知识 |
| [`WOS/distill_wos.py`](WOS/distill_wos.py) | **WOS 单 Token 蒸馏主脚本**；支持 `traditional`、`align`、`hier`、`full` 四种模式及损失权重消融 |
| [`WOS/distill_test_wos.py`](WOS/distill_test_wos.py) | 独立测试 WOS 单 Token 蒸馏学生模型 |
| [`WOS/analyze_distill_case.py`](WOS/analyze_distill_case.py) | （单token蒸馏后的case分析）比较教师与蒸馏学生的逐样本预测，分析 L1/L2 纠错、错误迁移等情况，并导出多工作表 Excel 报告 |

### 2 WOS：双 Token 对照实验

双 Token 分支在每层标签前后分别放置 START / END 层级 Token，以分析层级表示选择以及跨层级关系构建方式对蒸馏效果的影响。该分支与单 Token 分支的训练、提取、测试脚本应成套使用。

| 文件 | 用途与区别 |
| --- | --- |
| [`WOS/student_train_wos_2token.py`](WOS/student_train_wos_2token.py) | （学生双token训练）WOS 双 Token 学生 SFT；每层设置 START / END 一对特殊 Token |
| [`WOS/student_test_wos_2token.py`](WOS/student_test_wos_2token.py) | （学生双token测试）测试双 Token SFT 学生模型 |
| [`WOS/teacher_train_wos_2token.py`](WOS/teacher_train_wos_2token.py) | （教师双token训练）WOS 双 Token 教师 SFT，保持与对应学生分支一致的生成格式 |
| [`WOS/teacher_test_wos_2token.py`](WOS/teacher_test_wos_2token.py) | （教师双token测试）测试双 Token SFT 教师模型 |
| [`WOS/extract_wos_teacher_2token.py`](WOS/extract_wos_teacher_2token.py) | （教师双token提取特征）独立的双 Token 教师特征/Logits 离线缓存提取实现，记录四个层级 Token 对应的位置和隐状态信息 |
| [`WOS/distill_wos_2token.py`](WOS/distill_wos_2token.py) | **WOS 双 Token 蒸馏脚本**；内置 `cache` / `train` 两个子命令，支持四种蒸馏模式；Align 对齐四个 Token，Hier 匹配 START 与 END 两条跨层级差分关系 |
| [`WOS/distill_test_wos_2token.py`](WOS/distill_test_wos_2token.py) | 独立测试双 Token 蒸馏模型，并输出整体、分层及格式有效性等指标 |

> 双 Token 分支中，`distill_wos_2token.py` 自带缓存构建阶段；`extract_wos_teacher_2token.py` 是另一个独立提取入口。实际运行时应检查所选缓存格式与蒸馏训练入口的配置是否匹配，不要混用单 Token 和双 Token 缓存。

### 3 NYT：8 层层次文本分类

| 文件 | 用途与区别 |
| --- | --- |
| [`NYT/prepare_data.py`](NYT/prepare_data.py) | （实验之前对数据集预处理）根据原始 NYT 数据、taxonomy 和标签词表统一标签空间，生成清洗后的 JSONL 与 `label_space.json` |
| [`NYT/student_train.py`](NYT/student_train.py) | （学生训练）NYT 8 层特殊 Token 的学生模型生成式 SFT |
| [`NYT/student_test.py`](NYT/student_test.py) | （学生测试）独立测试学生 SFT 模型，评估整体及各深度的预测结果 |
| [`NYT/teacher_train.py`](NYT/teacher_train.py) | （教师训练）NYT 教师模型 SFT，生成格式与学生对应 |
| [`NYT/teacher_test.py`](NYT/teacher_test.py) | （教师测试）独立测试 NYT 教师模型 |
| [`NYT/extract_nyt_teacher.py`](NYT/extract_nyt_teacher.py) | 提取并缓存教师 Completion 隐状态、D1–D8 层级位置和公共词表输出层参数 |
| [`NYT/distill_nyt.py`](NYT/distill_nyt.py) | **NYT 全层级离线蒸馏主脚本**；支持 Traditional / Align / Hier / Full，Hier 使用 D1→D2 至 D7→D8 的相邻层级关系 |
| [`NYT/distill_masked.py`](NYT/distill_masked.py) | **NYT 层级关系 Mask 消融**；仅对相邻两层均有真实标签的样本计算对应 Hier 损失 |
| [`NYT/distill_test_nyt.py`](NYT/distill_test_nyt.py) | 通用于 NYT 全层级及 Masked 蒸馏结果的独立测试脚本 |

### 4 BGC：4 层层次文本分类

| 文件 | 用途与区别 |
| --- | --- |
| [`bgc/prepare_data.py`](bgc/prepare_data.py) | （实验之前对数据集预处理）根据 BGC 原始数据与 taxonomy 统一标签空间，保留文本原文，生成格式化 JSONL 与 `label_space.json` |
| [`bgc/student_train.py`](bgc/student_train.py) | （学生训练）BGC 4 层特殊 Token 的学生模型生成式 SFT |
| [`bgc/student_test.py`](bgc/student_test.py) | （学生测试）独立测试 BGC SFT 学生模型，输出整体和分层指标 |
| [`bgc/teacher_train.py`](bgc/teacher_train.py) | （教师训练）BGC 教师模型 SFT |
| [`bgc/teacher_test.py`](bgc/teacher_test.py) | （教师测试）独立测试 BGC 教师模型 |
| [`bgc/extract_bgc_teacher.py`](bgc/extract_bgc_teacher.py) | 提取教师 Completion 隐状态、D1–D4 相对位置与公共词表输出层参数，供离线蒸馏使用 |
| [`bgc/distill_bgc.py`](bgc/distill_bgc.py) | **BGC 全层级离线蒸馏主脚本**；支持四种蒸馏模式，Hier 使用 D1→D2 至 D3→D4 的相邻关系 |
| [`bgc/distill_masked.py`](bgc/distill_masked.py) | **BGC 层级关系 Mask 消融**；仅保留相邻两层均存在真实标签的 Hier 关系 |
| [`bgc/distill_test_bgc.py`](bgc/distill_test_bgc.py) | 通用于 BGC 全层级及 Masked 蒸馏结果的独立测试脚本 |

 设置蒸馏脚本中的 `--model_path`、`--teacher_cache_dir`、数据路径和 `--save_dir`，选择 `--experiment_mode` 与损失权重后训练
 
 比如1 Align + 0.1 Hier就是python WOS/distill_wos.py \
    --experiment_mode full \
    --kd_weight 1 \
    --align_weight 1 \
    --hier_weight 0.1 \
    --save_dir ./runs/wos_align1_hier01。

   只有traditional kd就是 python WOS/distill_wos.py \
    --experiment_mode traditional \
    --kd_weight 1 \
    --align_weight 0 \
    --hier_weight 0 \
    --save_dir ./runs/wos_traditional_1

    这里默认任务生成损失是1，可以更改traditional align hier的三个loss权重。

具体实验顺序是先运行 prepare_data.py➡学生教师的训练以及测试➡extract_*_teacher.py➡蒸馏脚本以及测试

——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————




## 1. 项目概览

| 项目 | 说明 |
| --- | --- |
| 研究任务 | 层次文本分类（HTC）：按层级生成标签，评估整体及各层分类性能 |
| 数据集 | WOS、NYT、BGC |
| 教师模型 | Llama-3.1-8B（经 SFT） |
| 学生模型 | Llama-3.2-1B |
| 训练方式 | 生成式 SFT + 离线知识蒸馏，使用 LoRA |
| 蒸馏目标 | 任务监督损失、Logits KD、层级 Token 隐状态对齐、跨层级关系蒸馏 |
| 主要指标 | Micro-F1、Macro-F1，以及分层级的 Micro-F1 / Macro-F1 |

各数据集在代码中的最大层级深度分别为：**WOS 2 层、NYT 8 层、BGC 4 层**。其中 WOS 还提供每层一个层级 Token 与每层一对 START/END Token 的对照实现。

> 说明：仓库主要提供实验脚本。原始数据、预训练模型、训练得到的权重及离线教师缓存需要在运行环境中自行准备；代码中的默认绝对路径来自原实验环境，复现时应修改为本地路径。

## 2. 方法与实验模式

### 2.1 层级 Token 与生成式预测

模型根据输入文本按层级生成标签。每个层级对应一个专用的 Llama reserved special token，用于区分层级位置，并提取该位置的最后一层上下文隐状态。教师知识提取与学生蒸馏使用 **Completion 中层级 Token 的第二次出现位置**，与 SFT 的 Prompt / Completion 约定保持一致。

- **单 Token 设计**：每一层使用一个层级特殊 Token。WOS 使用 L1/L2，NYT 使用 D1–D8，BGC 使用 D1–D4。
- **WOS 双 Token 设计**：每层分别使用 START、END 两个特殊 Token，用于对比不同层级表示载体。跨层级关系分别沿 START→START 和 END→END 两条分支建模。
- **空层处理**：NYT/BGC 的深层样本可能存在空标签 `[]`，但生成结构中的层级 Token 仍保留，以确保层级位置一致。

### 2.2 离线教师知识与损失设计

首先使用训练好的教师模型提取训练样本中生成部分的最终层隐状态，并保存公共词表对应的 `lm_head` 参数。蒸馏阶段在不加载完整教师 Transformer 的情况下，利用缓存隐状态与输出层权重重建教师 Logits，计算温度缩放的 KL 散度。因此，这部分仍属于 **Logits KD**，而非直接以隐藏状态代替 Logits 的蒸馏。

在传统蒸馏基础上，进一步加入：

- **Align（层级表示对齐）**：通过投影层映射学生隐状态，使其与教师对应层级 Token 的隐状态接近；对齐损失结合 MSE 与余弦距离。
- **Hier（跨层级关系蒸馏）**：对相邻层级隐状态的差分进行匹配，使学生学习教师表示中的层级变化关系。

总损失形式为：

$$
\mathcal{L}=\mathcal{L}_{\mathrm{task}}+\lambda_{\mathrm{KD}}\mathcal{L}_{\mathrm{KD}}+\lambda_{\mathrm{align}}\mathcal{L}_{\mathrm{align}}+\lambda_{\mathrm{hier}}\mathcal{L}_{\mathrm{hier}}.
$$

| `--experiment_mode` | 启用的损失 | 目的 |
| --- | --- | --- |
| `traditional` | Task + Logits KD | 传统输出分布蒸馏基线 |
| `align` | Task + Logits KD + Align | 验证层级隐状态对齐的贡献 |
| `hier` | Task + Logits KD + Hier | 验证相邻层级关系蒸馏的贡献 |
| `full` | Task + Logits KD + Align + Hier | 联合蒸馏方法 |

`--kd_weight`、`--align_weight`、`--hier_weight` 分别控制蒸馏损失权重；`--kd_temperature` 控制 Logits KD 温度；`--align_alpha` 控制 Align 中 MSE 与余弦损失的组合比例。具体默认值及其他训练参数以对应脚本为准。

### 2.3 全层级与 Masked Hier 对照

NYT 和 BGC 分别提供两种相邻层级关系蒸馏策略：

| 策略 | 脚本 | Hier 关系计算范围 |
| --- | --- | --- |
| 全层级 Hier | `NYT/distill_nyt.py`、`bgc/distill_bgc.py` | 按固定深度计算所有相邻层级之间的关系，包括空标签层级对应的特殊 Token |
| Masked / label-present Hier | `NYT/distill_masked.py`、`bgc/distill_masked.py` | 仅当某个样本的相邻两层 **均存在真实标签** 时，才计算该相邻层级的 Hier 损失 |

Masked 版本**只改变 Hier 关系损失的有效层级筛选条件**；若开启 Align，层级表示对齐仍按原有全层级策略执行。两种训练脚本均可使用本数据集对应的 `distill_test_*.py` 进行独立测试。

## 3. 代码说明

### 3.1 WOS：单 Token 主实验

| 文件 | 用途与区别 |
| --- | --- |
| [`WOS/prepare_data.py`](WOS/prepare_data.py) | （实验之前对数据集预处理）读取 WOS 原始数据和 taxonomy，规范化训练/验证/测试样本，生成统一的 `label2id.json` 及 `*_formatted.jsonl` |
| [`WOS/student_train_wos.py`](WOS/student_train_wos.py) | （学生单token训练）单 Token 生成式学生模型 SFT；使用每层一个特殊 Token 的层级标签输出格式 |
| [`WOS/student_test_wos.py`](WOS/student_test_wos.py) | （学生单token测试）独立测试单 Token SFT 学生模型，计算整体与 L1/L2 分层指标 |
| [`WOS/teacher_train_wos.py`](WOS/teacher_train_wos.py) | （教师单token训练）与学生使用一致层级输出结构的教师模型 SFT |
| [`WOS/teacher_test_wos.py`](WOS/teacher_test_wos.py) | （教师单token测试）独立测试单 Token SFT 教师模型 |
| [`WOS/extract_wos_teacher.py`](WOS/extract_wos_teacher.py) | （离线提取教师特征）离线缓存教师生成部分的隐状态、层级 Token 偏移量及公共词表输出层参数，为后续 KD / Align / Hier 提供教师知识 |
| [`WOS/distill_wos.py`](WOS/distill_wos.py) | **WOS 单 Token 蒸馏主脚本**；支持 `traditional`、`align`、`hier`、`full` 四种模式及损失权重消融 |
| [`WOS/distill_test_wos.py`](WOS/distill_test_wos.py) | 独立测试 WOS 单 Token 蒸馏学生模型 |
| [`WOS/analyze_distill_case.py`](WOS/analyze_distill_case.py) | （单token蒸馏后的case分析）比较教师与蒸馏学生的逐样本预测，分析 L1/L2 纠错、错误迁移等情况，并导出多工作表 Excel 报告 |

### 3.2 WOS：双 Token 对照实验

双 Token 分支在每层标签前后分别放置 START / END 层级 Token，以分析层级表示选择以及跨层级关系构建方式对蒸馏效果的影响。该分支与单 Token 分支的训练、提取、测试脚本应成套使用。

| 文件 | 用途与区别 |
| --- | --- |
| [`WOS/student_train_wos_2token.py`](WOS/student_train_wos_2token.py) | （学生双token训练）WOS 双 Token 学生 SFT；每层设置 START / END 一对特殊 Token |
| [`WOS/student_test_wos_2token.py`](WOS/student_test_wos_2token.py) | （学生双token测试）测试双 Token SFT 学生模型 |
| [`WOS/teacher_train_wos_2token.py`](WOS/teacher_train_wos_2token.py) | （教师双token训练）WOS 双 Token 教师 SFT，保持与对应学生分支一致的生成格式 |
| [`WOS/teacher_test_wos_2token.py`](WOS/teacher_test_wos_2token.py) | （教师双token测试）测试双 Token SFT 教师模型 |
| [`WOS/extract_wos_teacher_2token.py`](WOS/extract_wos_teacher_2token.py) | （教师双token提取特征）独立的双 Token 教师特征/Logits 离线缓存提取实现，记录四个层级 Token 对应的位置和隐状态信息 |
| [`WOS/distill_wos_2token.py`](WOS/distill_wos_2token.py) | **WOS 双 Token 蒸馏脚本**；内置 `cache` / `train` 两个子命令，支持四种蒸馏模式；Align 对齐四个 Token，Hier 匹配 START 与 END 两条跨层级差分关系 |
| [`WOS/distill_test_wos_2token.py`](WOS/distill_test_wos_2token.py) | 独立测试双 Token 蒸馏模型，并输出整体、分层及格式有效性等指标 |

> 双 Token 分支中，`distill_wos_2token.py` 自带缓存构建阶段；`extract_wos_teacher_2token.py` 是另一个独立提取入口。实际运行时应检查所选缓存格式与蒸馏训练入口的配置是否匹配，不要混用单 Token 和双 Token 缓存。

### 3.3 NYT：8 层层次文本分类

| 文件 | 用途与区别 |
| --- | --- |
| [`NYT/prepare_data.py`](NYT/prepare_data.py) | （实验之前对数据集预处理）根据原始 NYT 数据、taxonomy 和标签词表统一标签空间，生成清洗后的 JSONL 与 `label_space.json` |
| [`NYT/student_train.py`](NYT/student_train.py) | （学生训练）NYT 8 层特殊 Token 的学生模型生成式 SFT |
| [`NYT/student_test.py`](NYT/student_test.py) | （学生测试）独立测试学生 SFT 模型，评估整体及各深度的预测结果 |
| [`NYT/teacher_train.py`](NYT/teacher_train.py) | （教师训练）NYT 教师模型 SFT，生成格式与学生对应 |
| [`NYT/teacher_test.py`](NYT/teacher_test.py) | （教师测试）独立测试 NYT 教师模型 |
| [`NYT/extract_nyt_teacher.py`](NYT/extract_nyt_teacher.py) | 提取并缓存教师 Completion 隐状态、D1–D8 层级位置和公共词表输出层参数 |
| [`NYT/distill_nyt.py`](NYT/distill_nyt.py) | **NYT 全层级离线蒸馏主脚本**；支持 Traditional / Align / Hier / Full，Hier 使用 D1→D2 至 D7→D8 的相邻层级关系 |
| [`NYT/distill_masked.py`](NYT/distill_masked.py) | **NYT 层级关系 Mask 消融**；仅对相邻两层均有真实标签的样本计算对应 Hier 损失 |
| [`NYT/distill_test_nyt.py`](NYT/distill_test_nyt.py) | 通用于 NYT 全层级及 Masked 蒸馏结果的独立测试脚本 |

### 3.4 BGC：4 层层次文本分类

| 文件 | 用途与区别 |
| --- | --- |
| [`bgc/prepare_data.py`](bgc/prepare_data.py) | （实验之前对数据集预处理）根据 BGC 原始数据与 taxonomy 统一标签空间，保留文本原文，生成格式化 JSONL 与 `label_space.json` |
| [`bgc/student_train.py`](bgc/student_train.py) | （学生训练）BGC 4 层特殊 Token 的学生模型生成式 SFT |
| [`bgc/student_test.py`](bgc/student_test.py) | （学生测试）独立测试 BGC SFT 学生模型，输出整体和分层指标 |
| [`bgc/teacher_train.py`](bgc/teacher_train.py) | （教师训练）BGC 教师模型 SFT |
| [`bgc/teacher_test.py`](bgc/teacher_test.py) | （教师测试）独立测试 BGC 教师模型 |
| [`bgc/extract_bgc_teacher.py`](bgc/extract_bgc_teacher.py) | 提取教师 Completion 隐状态、D1–D4 相对位置与公共词表输出层参数，供离线蒸馏使用 |
| [`bgc/distill_bgc.py`](bgc/distill_bgc.py) | **BGC 全层级离线蒸馏主脚本**；支持四种蒸馏模式，Hier 使用 D1→D2 至 D3→D4 的相邻关系 |
| [`bgc/distill_masked.py`](bgc/distill_masked.py) | **BGC 层级关系 Mask 消融**；仅保留相邻两层均存在真实标签的 Hier 关系 |
| [`bgc/distill_test_bgc.py`](bgc/distill_test_bgc.py) | 通用于 BGC 全层级及 Masked 蒸馏结果的独立测试脚本 |

## 4. 实验流程

三个数据集的单 Token 实验遵循一致的总体顺序：

```text
原始数据 + 层级标签结构
          │
          ▼
     prepare_data.py
  ├─ train_formatted.jsonl
  ├─ val_formatted.jsonl
  ├─ test_formatted.jsonl
  └─ label2id.json / label_space.json
          │
          ├───────────────────────────┐
          ▼                           ▼
     Teacher SFT                  Student SFT
          │                           │
          ▼                           ▼
     Teacher Test                  Student Test
          │                           （独立基线）
          ▼
  extract_*_teacher.py
   离线教师知识缓存
          │
          ▼
   distill_*.py
   ├─ traditional
   ├─ align
   ├─ hier
   └─ full
          │
          ▼
  distill_test_*.py
          │
          ▼
   Micro-F1 / Macro-F1
   整体指标与各层级指标
```

**注意：**独立的 Student SFT 用于建立学生基线并与教师、蒸馏结果比较；当前蒸馏脚本从学生基础模型初始化并进行训练，不应将“先完成学生 SFT”理解为“蒸馏默认从学生 SFT 权重继续训练”。教师 SFT 权重则用于构建蒸馏缓存。

WOS 双 Token 实验使用对应的 `*_2token.py` 脚本；其蒸馏入口 `distill_wos_2token.py` 分为 `cache` 和 `train` 两阶段。

## 5. 运行说明

代码使用 Python，并依赖 PyTorch、Transformers、PEFT、NumPy、scikit-learn、tqdm、SwanLab 等组件；WOS 的案例分析还涉及 Excel 写入相关依赖。**实际 CUDA / PyTorch / Transformers / PEFT 版本需与所用模型及训练环境适配。**

1. 准备 WOS、NYT 或 BGC 的原始数据与标签层级文件，并准备教师/学生基础模型。
2. 修改对应 `prepare_data.py` 中的输入输出路径；WOS 预处理主要使用脚本内路径常量，NYT/BGC 预处理提供命令行路径参数。
3. 完成教师和学生 SFT，并运行相应测试脚本建立基线。
4. 使用对应的教师提取脚本生成离线缓存；确保教师、学生、数据、特殊 Token 和 Prompt / Completion 格式一致。
5. 设置蒸馏脚本中的 `--model_path`、`--teacher_cache_dir`、数据路径和 `--save_dir`，选择 `--experiment_mode` 与损失权重后训练，比如1 Align + 0.1 Hier就是python WOS/distill_wos.py \
    --experiment_mode full \
    --kd_weight 1 \
    --align_weight 1 \
    --hier_weight 0.1 \
    --save_dir ./runs/wos_align1_hier01。这里默认任务生成损失是1，可以更改traditional align hier的三个loss权重。
6. 使用**同一数据集、同一 Token 设计**的 `distill_test_*.py` 评估蒸馏结果。

可先通过以下命令查看部分脚本提供的参数：

```bash
python NYT/teacher_train.py --help
python NYT/extract_nyt_teacher.py --help
python NYT/distill_nyt.py --help
python NYT/distill_test_nyt.py --help

python bgc/distill_bgc.py --help
python bgc/distill_masked.py --help

python WOS/distill_wos.py --help
python WOS/distill_wos_2token.py cache --help
python WOS/distill_wos_2token.py train --help
```

以上命令仅用于查看参数，**不直接启动完整实验**。不同数据集及 WOS 单/双 Token 分支的缓存、权重和测试脚本不可直接互换；运行时请使用各脚本自身的参数名与实验目录。

## 6. 评估与分析

独立测试脚本主要统计：

- **整体 Micro-F1 / Macro-F1**：衡量所有层级标签的整体预测质量。
- **逐层 Micro-F1 / Macro-F1**：分析不同深度上的分类性能，观察教师、学生及蒸馏模型的层级差异。
- **生成格式和层级合法性**：部分测试脚本还记录解析成功率、有效标签组合、层级一致性等辅助指标。
- **WOS 案例分析**：`WOS/analyze_distill_case.py` 比较教师与蒸馏学生预测，分类统计学生修正教师错误、教师正确而学生错误等情况，并输出 Excel 明细。

## 7. 复现实验注意事项

- **标签映射一致性**：同一数据集的训练、验证、测试和蒸馏必须使用统一标签空间，避免标签 ID 不一致。
- **特殊 Token 与生成模板一致性**：教师、学生、提取缓存和蒸馏测试应保持相同的层级 Token 约定，尤其不能混用 WOS 单/双 Token 版本。
- **离线缓存一致性**：训练数据顺序、Tokenizer、截断方式及层级 Token 位置需要与缓存提取阶段严格对应。
- **空标签层级**：NYT/BGC 中 `[]` 层级的存在不代表该层级特殊 Token 被删除；是否计算相邻层级 Hier 由具体蒸馏版本决定。
- **测试数据与模型选择**：按验证集选择检查点，再在测试集上独立评估；确保论文比较表中的结果来自一致的测试协议。

---

本仓库用于层次文本分类中**输出分布知识、层级语义表示以及相邻层级结构关系**的蒸馏研究与消融分析。
