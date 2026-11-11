# PPO-CIS on ToxiCN（五分类器、两阶段级联、PPO-Clip）

复现论文 *PPO-CIS: A Deep Reinforcement Learning Framework for Real-Time Toxicity
Detection in Social Media*（Bodaghi, Fung & Schmitt, Knowledge-Based Systems 2026）
的强化学习过程，并把论文的 8 个分类器换成这个文件夹里的 5 个：

- **Stage 1（高吞吐）**：`tiny_bert`（BERT-Tiny）、`distil_roberta`（DistilRoBERTa）
- **Stage 2（高准确率）**：`bert`、`roberta_fusion`、`tke_bert`

Stage 1 只负责“高置信无毒 → 直接退出”；其余样本升级到 Stage 2 做最终判定。
这与论文的两阶段级联一致。单次最多调用 `2 + 3 = 5` 个分类器，实际策略通常只调用 1–2 个。

## 五个分类器

| 名称 | 阶段 | 预训练模型目录 | 架构 | 来源 |
|---|---|---|---|---|
| `tiny_bert` | 1 | `bert-tiny-chinese` | BERT-Tiny | `models.py :: TransformerClassifier` |
| `distil_roberta` | 1 | `chinese_roberta_L-4_H-256` | 蒸馏中文 RoBERTa | 同上 |
| `bert` | 2 | `bert-base-chinese` | `BertForSequenceClassification` | `bert-safe-chat/train.py` |
| `roberta_fusion` | 2 | `chinese-roberta-wwm-ext` | RoBERTa-wwm-ext + BiGRU + TextCNN | `CHSD/model.py :: RoBertFusion` |
| `tke_bert` | 2 | `bert-base-chinese`（与 `bert` 共用） | BERT + toxic/stance 知识 embedding + 两层 FFNN | `src/BERT.py` + `src/Models.py` |

说明：

- 两个吞吐变体都用 HuggingFace 现成的小模型，代码复用 `TransformerClassifier`，不需要新增模型类。
- `tke_bert` 在 BERT embedding 层加入：
  - `toxic_embeddings = nn.Embedding(6, 768)`：5 类脏词 + 无知识；
  - `stance_embeddings = nn.Embedding(2, 768)`：中立 / 攻击立场。
- 仓库没有 `attack_stance.json`，因此 `stance_ids` 全为 0（中立），其余结构不变。
- 模型统一从**本地目录**加载，目录约定见下面的“模型目录”一节。

## 模型目录（4 个目录对应 5 个分类器）

5 个分类器只需要 **4 个**预训练模型目录：`tke_bert` 不是另一个独立预训练模型，而是在
`bert-base-chinese` 的 embedding 上加了 toxic/stance 知识 embedding 和两层 FFNN，
所以它和 `bert` 共用同一个 `bert-base-chinese` 目录。

**直接改代码即可**。打开 `config.py`，只改这一行里的目录字符串：

```python
MODEL_ROOT = Path(os.environ.get("PPO_CIS_MODEL_ROOT", "D:/models/ppo_cis_models"))
#                                                      ↑ 换成你的模型根目录
```

目录结构：

```
D:/你的模型根目录/
  bert-base-chinese/          # bert + tke_bert 共用
  chinese-roberta-wwm-ext/    # roberta_fusion
  bert-tiny-chinese/          # tiny_bert
  chinese_roberta_L-4_H-256/  # distil_roberta
```

每个子目录就是一个标准 HuggingFace 模型目录（`config.json`、权重、`vocab.txt` /
tokenizer 文件）。改完 `MODEL_ROOT` 后不需要再动别的地方。

环境变量 `PPO_CIS_MODEL_ROOT` 和 `--bert_model` / `--roberta_model` /
`--tiny_bert_model` / `--distil_roberta_model` 只是可选的覆盖方式，直接改代码时不用管。

`config.resolve_model_name()` 只查找本地目录（`MODEL_ROOT/<模型文件夹名>`），
**不会联网下载**：目录不存在或缺 `config.json` / 权重文件时直接抛 `FileNotFoundError`，
训练/PPO 启动前还会一次性列出所有缺失的模型。旧 checkpoint 里记录的 `bert-base-chinese` /
`hfl/chinese-roberta-wwm-ext` 会被按文件夹名映射到对应本地目录。

## 损失函数

- 原 `ToxiCN_ex` 主实验用 `BCEWithLogitsLoss`；`CHSD`、`bert-safe-chat` 用 `CrossEntropyLoss`。
- 本管线五个分类器统一用 `nn.CrossEntropyLoss`（二分类下与 one-hot BCE 等价）。
- PPO 部分额外带一个**辅助交叉熵头**（`aux_coef=0.10`）：只在更新时用标签做辅助监督，
  推理不参与决策，不改变 PPO-Clip 主目标。

## 数据

- `ToxiCN_ex/ToxiCN/data/train.json`：8.0 MB，32,157 条
- `ToxiCN_ex/ToxiCN/data/test.json`：1.34 MB，5,323 条
- `ToxiCN_ex/ToxiCN/lexicon/*.json`：5 个脏词表

清洗：去空文本、去重、去掉 train/test 重叠文本；随后把 `train.json` 按 8:1:1 分层切分：

| split | 用途 | 规模（约） |
|---|---|---|
| `clf_train` | 训练五个分类器 | 25.6k |
| `clf_dev` | 分类器早停 | 3.2k |
| `ppo_train` | PPO 训练（分类器冻结） | 3.2k |
| `test` | 最终对比 | 5.3k |

预训练权重不在仓库内，运行时从上面的本地模型目录读取，不会联网下载。

## 强化学习建模（两阶段级联）

- **状态**：β 向量，长度 K=5。调用过的分类器填 `p(toxic)`，未调用填 `-1`（Eq. 1）。
- **动作**：`0..K-1` 调用分类器 k；`K` 判 toxic；`K+1` 判 nontoxic。
- **阶段**：
  - Stage 1：只输出“无毒”。融合后的无毒概率达阈值即可退出；**只要调用过任意
    一个模型，就允许随时升级到 Stage 2**，由策略自己决定何时跳过剩余小模型。
  - Stage 2：按同一阈值单侧判定——无毒概率达阈值判无毒，否则判有毒；调用过
    大模型后判定动作始终合法，策略可立即判定也可再调一个模型。
- **阈值门控**：Stage 1 退出与 Stage 2 判定共用一个 `threshold`（默认 0.70），
  作用在**当前阶段已调用模型的平均有毒概率**上（`1 - mean(p) >= threshold`）。
  不再使用 `min(p)`（那等于“挑最乐观的模型听”），也不再保留 `0.3~0.7` 死区。
- **奖励**：
  - 分类奖励 Eq. (1)：正确 `+r`，错误 `-iota*r`（`r=1, iota=2`）；
  - 时间奖励 Eq. (2)：改为逐样本线性惩罚 `-theta*t/u`（`u=1.5ms, theta=0.5`）；
  - 吞吐奖励 Eq. (6)：`psi_M = M / sum(t_i)`，仅作报告指标（`throughput_scale=0`）。
- **PPO-Clip**：`epsilon=0.2`、GAE(`gamma=0.99, lambda=0.95`)、
  value loss `0.5`、熵 `0.01`、辅助交叉熵 `0.10`、梯度裁剪 `0.5`；
  batch `256`、每个 batch 复用 `4` 次、共 `10` 个 epoch、MLP hidden `64`。
- **防塌缩**：rollout 时以 `explore_decide=0.10` 的概率强制选择判定动作，
  保证判定动作始终有样本量。若训练中仍然不早退，可以把 `--entropy_coef`
  调到 `0.05` 或把 `--explore_decide` 调到 `0.30`。

## 与论文的差异

1. 分类器数量：论文 8 个（4 种类型 × α/ψ），这里 5 个（2 个 ψ + 3 个 α）。
2. 数据集：论文用 ToxiGen / AugmenToxic，这里用 ToxiCN 中文数据。
3. 模型：论文的 α 版本是英文 `bert-base-uncased` / `roberta-base`，
   这里是中文 `bert-base-chinese` / `chinese-roberta-wwm-ext`，ψ 版本是中文 tiny 模型。
4. Eq. (2) 按秒设计，实际延迟只有毫秒级：`t << u` 时整个区间只变化 0.05，
   且在 `t -> u` 处发散。这里改成逐样本线性惩罚 `-theta*t/u`，方向和单调性
   与论文一致，但尺度可学（否则时间项对策略没有梯度）。
5. Eq. (7) 的 batch 调整项 `Delta` 定义不明确，未实现；只保留 Eq. (6)。
6. 延迟不是逐条 `batch=1` 计时，而是用实际 batch 吞吐折算单样本延迟。

## 运行

工作目录必须是 `ToxiCN_ex`：

```bash
cd ToxiCN_ex

# 完整流程：训练 5 个分类器 -> PPO -> 测试 -> 生成 CSV/MD/JSON/PNG
python -m ppo_cis.run_all

# 只训练两个新增的吞吐变体（已有三个 alpha checkpoint 会自动跳过）
python -m ppo_cis.run_all --step classifiers --only tiny_bert distil_roberta

# 只训练 / 评估 PPO（要求 5 个 checkpoint 已存在）
python -m ppo_cis.run_all --step ppo

# 只重新画图
python -m ppo_cis.run_all --step plot
```

离线机器：

```bash
python -m ppo_cis.run_all \
  --bert_model /path/to/bert-base-chinese \
  --roberta_model /path/to/chinese-roberta-wwm-ext \
  --tiny_bert_model /path/to/bert-tiny-chinese \
  --distil_roberta_model /path/to/chinese_roberta_L-4_H-256
```

## 输出（运行结束会一起生成）

```
ppo_cis/artifacts/
  classifiers/            # 5 个 .pt + metrics.json
  cache/                  # tokenized tensors（含 TKE 知识 ID）
  scores/                 # ppo_train.npz / test.npz（概率与延迟）
  ppo/policy.pt           # 策略 + 阶段划分 + 配置 + 训练历史
  reports/
    ppo_cis_results.csv   # 对比表格（Excel 可直接打开）
    ppo_cis_results.md    # Markdown 表格（可直接阅读/写进报告）
    ppo_cis_results.json  # 完整结果 + PPO 训练曲线
    ppo_cis_comparison.png# F1 / 延迟 / 调用次数 三联图
```

表格包含：5 个单分类器、顺序调用、固定阈值级联、多数投票、软投票、PPO-CIS，
字段为 Accuracy / Precision / Recall / F1 / 平均调用数 / 平均延迟(ms) / 吞吐。

## 运行时间预估

本机（Core Ultra 5 225H，14 核，31.4 GB RAM，**无 NVIDIA GPU**，CPU 版 torch）：

| 项目 | 预估 |
|---|---|
| 下载 4 个预训练模型 | 10–30 分钟（视网速） |
| `tiny_bert`（4 epochs） | 30–60 分钟 |
| `distil_roberta`（4 epochs） | 30–70 分钟 |
| 已有 `bert` / `roberta_fusion` / `tke_bert` checkpoint | 0（自动跳过） |
| 5 个分类器打分（ppo_train + test） | 5–10 分钟 |
| PPO 训练 + 评估 + 画图 | 15–30 分钟 |
| **本次增量合计** | **约 1.5–3 小时** |
| **五个分类器全部从头训练** | **约 15–18 小时** |

租用单卡 GPU：

| 显卡 | 增量（只训两个 tiny + PPO） | 全部从头 |
|---|---|---|
| RTX 4090 / A100 | 10–20 分钟 | 40–80 分钟 |
| RTX 3090 | 15–30 分钟 | 1–2 小时 |
| T4 | 40–70 分钟 | 2–4 小时 |

## 目录

| 路径 | 作用 |
|---|---|
| `ppo_cis/config.py` | 路径、5 个分类器规格与阶段划分、PPO/奖励超参 |
| `ppo_cis/data.py` | 清洗、切分、tokenizer 缓存、TKE 知识 ID |
| `ppo_cis/models.py` | BERT / RoBERTaFusion / TKE-BERT / tiny 通用 Transformer |
| `ppo_cis/train_classifiers.py` | 五个分类器训练（CrossEntropyLoss） |
| `ppo_cis/pipeline.py` | 冻结打分、两阶段环境、静态基线、Eq.(2)/Eq.(6) 奖励 |
| `ppo_cis/ppo.py` | Actor-Critic、rollout、GAE、PPO-Clip、评估 |
| `ppo_cis/train_ppo.py` | 训练 PPO、对比基线、写 CSV/MD/JSON/PNG |
| `ppo_cis/plot_results.py` | 读取 CSV 生成 F1 / 延迟 / 调用次数三联图 |
| `ppo_cis/run_all.py` | 一键入口 |
