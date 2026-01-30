# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research project on **Interactive Agentic Framework for Knowledge Extraction and Evaluation**, targeting **ICML 2026**. The goal is to explicitly characterize and measure the **knowledge boundaries** of black-box LLMs by deploying multiple agentic exploration strategies to push models beyond their "comfort zone" and systematically extract their latent knowledge.

**Core Research Questions:**
1. What does a model know in a given domain?
2. How much knowledge can be extracted at saturation?
3. Which exploration strategy is most efficient (Pareto-optimal)?
4. How does extractable knowledge scale with model size?

## Current Experiments (ICML 2026)

### Experiment 1: Pareto Frontier Analysis (Strategy Search)
**Goal**: Fix model to Llama-3.1-405B, compare 8 pipeline configurations to find Pareto-optimal strategy.

**Domains**: Transformer Architectures, Policy Gradient Methods, Algorithmic Fairness

**Configurations Tested**:
- P2_Sequential (baseline: iterative "What else?")
- P3_Reflection (self-critique loop)
- P4_Taxonomy_L2W2, L3W3, L5W5 (recursive taxonomy with different branching factors)
- P5_MultiProfile_N3, N10, N20 (multi-expert debate with different profile counts)

**Commands**:
```bash
# Run all trajectories (30 turns, with resume support)
python3 run_pareto_curves.py

# Evaluate trajectories (dedup + audit + compute yield ratios)
python3 evaluate_pareto_curves.py

# Plot individual domain curves
python3 plot_pareto_curves.py

# Plot aggregated Pareto frontier
python3 plot_aggregated_pareto.py
```

**Output Location**: `results/pareto_curves_405b/{domain}/{config}.trajectory.json`
**Important Results**: `results/*_old.json` and `results/plots/*_old.png` contain key experimental data.

### Experiment 2: Scaling Law Analysis (Cross-Model Comparison)
**Goal**: Use best strategy from Exp1 (P4_Taxonomy_L5W5) to compare Llama-3.1 family (8B, 70B, 405B).

**Hypothesis**: Extractable knowledge volume grows with model parameter count.

**Commands**:
```bash
# Run extraction for all models
python3 run_size_comparison.py

# Evaluate and compute scaling metrics
python3 evaluate_model_comparison.py
```

**Evaluation Pipeline** (`evaluate_model_comparison.py`):
1. **Domain串行处理**：避免API并发竞争，每处理完一个domain保存cache
2. **Model串行处理**：同domain内模型共享embedding cache
3. **三阶段去重** (per model):
   - Stage 1: Strict Dedup (embedding相似度 ≥0.92 → 直接合并)
   - Stage 2: Fuzzy Dedup (0.70-0.92 → LLM judge判断，批量并发)
   - Stage 3: Domain Audit (LLM判断是否为有效领域知识)
4. **Union构建**：合并所有模型的valid points，再次两阶段语义去重
5. **指标计算**：
   - Recall = |model_valid| / |union| (覆盖率)
   - Unique Discovery = 语义匹配后，该模型独有的知识点数量

**设计变更 (2025-01)**:
- 从"并行处理3个domain"改为"串行处理domain"，因为并行会导致API竞争和进度条混乱
- Union构建从字符串匹配改为语义两阶段去重
- Unique Discovery从字符串匹配改为embedding相似度计算
- 移除不必要的读锁，只保留写锁

**Output Location**: `results/llama31_scaling_law/{domain}/{model}/P4_Taxonomy_L5W5_raw.json`

### Experiment 3: RL Domain Specialization Analysis (Qwen Comparison)
**Goal**: Compare general-purpose vs domain-specialized (RL-tuned) models to understand the trade-off between specialization and knowledge breadth.

**Hypothesis**: RL fine-tuning improves accuracy in target domains but may sacrifice knowledge coverage and cause catastrophic forgetting in non-target domains.

**Models Compared**:
- Qwen 2.5-7B Instruct (general-purpose baseline)
- Qwen 2.5-7B Coder (RL-tuned for code/programming tasks)

**Strategy**: P4_Taxonomy_L5W5 (best from Exp1)

**Domains**:
- Deep Learning (code-adjacent)
- Machine Learning Systems (code-adjacent)
- Probabilistic Methods (non-code domain)

**Commands**:
```bash
# Run extraction for both models (reference implementation)
python3 run_evolution_comparison.py

# Evaluate and generate comparison report
python3 evaluate_qwen_comparison.py
```

**Key Findings**:

| Domain | Model | Raw | Unique | Valid | Recall | Accuracy | Discovery |
|--------|-------|-----|--------|-------|--------|----------|-----------|
| **Deep Learning** (target) | general | 4074 | 3239 | 1215 | 55.3% | **37.5%** | 716 |
|  | coder | 2098 | 1616 | 1189 | 56.2% | **73.6%** ↑ | 731 |
| **ML Systems** (target) | general | 3427 | 2702 | 1317 | 56.4% | **48.7%** | 817 |
|  | coder | 1772 | 1347 | 1091 | 52.0% | **81.0%** ↑ | 742 |
| **Probabilistic Methods** (non-target) | general | 3072 | 2348 | 1479 | **72.1%** | 63.0% | 1162 |
|  | coder | 1968 | 1306 | 622 | **32.6%** ↓ | 47.6% ↓ | 482 |

**Conclusions**:
1. **Target Domain Effect**: In RL-tuned domains (Deep Learning, ML Systems), the Coder model shows:
   - Dramatic accuracy improvements (37.5%→73.6%, 48.7%→81.0%)
   - Similar or slightly better recall (knowledge coverage maintained)
   - More efficient generation (fewer raw tokens needed)
   - **Implication**: RL training improves knowledge quality without increasing quantity

2. **Catastrophic Forgetting**: In non-target domains (Probabilistic Methods), the Coder model exhibits:
   - Severe recall degradation (72.1%→32.6%, -55% relative drop)
   - Accuracy decline (63.0%→47.6%)
   - 58% fewer unique discoveries (1162→482)
   - **Implication**: Domain specialization comes at the cost of general knowledge

3. **Specialization Trade-off**: RL fine-tuning creates a Pareto trade-off between domain expertise and knowledge breadth, suggesting that general-purpose models may be preferable for broad knowledge extraction tasks.

**Output Location**: `results/evolution_comparison/{domain}/{model}/P4_Taxonomy_L5W5.trajectory.json`

**Important Results**:
- `results/evolution_comparison/qwen_comparison_summary.txt`: Table summary
- `results/evolution_comparison/qwen_comparison_report.json`: Full metrics with sampled invalid/valid examples

### Experiment 4: Cross-Series Model Comparison (~7B)
**Goal**: Compare knowledge extraction across different model families at similar parameter counts (~7-8B).

**Hypothesis**: Different model series have distinct knowledge profiles due to training data and methodology differences.

**Models Compared** (4 models, ~7B each):
- meta/llama-3.1-8b-instruct (Llama 3.1 - US/Meta)
- qwen/qwen2.5-7b-instruct (Qwen 2.5 - CN/Alibaba)
- mistralai/mistral-7b-instruct-v0.3 (Mistral - EU)
- deepseek-ai/deepseek-r1-distill-qwen-7b (DeepSeek R1 - Reasoning-enhanced)

**Strategy**: P4_Taxonomy_L5W5 (best from Exp1)

**Commands**:
```bash
# Run extraction for all 4 models × 3 domains (12 parallel tasks)
python3 run_cross_series.py

# Evaluate and generate comparison report
python3 evaluate_cross_series.py
```

**Research Questions**:
1. Which model series discovers the most knowledge in each domain?
2. Do different series have different "unique discovery" patterns?
3. Does reasoning enhancement (R1 distillation) help knowledge extraction?

**Output Location**: `results/cross_series_7b/{domain}/{model}/P4_Taxonomy_L5W5_raw.json`

### Essential Setup Commands
```bash
# Environment
conda create -n ke python=3.10 -y && conda activate ke && pip install -r requirements.txt

# Local embedding server (optional, for better clustering)
bash launch_sglang.sh Qwen/Qwen3-Embedding-8B 30000 0.0.0.0 0

# Check API keys validity
python3 check_api_keys.py
```

## Architecture

### Research Framework (Paper Section 3)

The system is a closed-loop pipeline with three modules:

**1. Agentic Exploration Policies (Section 3.2)**
Four strategies to overcome the model's "comfort zone" effect:
- **P2: Sequential Associative Probing** (baseline) - Iterative "What else?" prompting
- **P3: Self-Reflective Refinement** - Critic-actor loop with self-audit
- **P4: Recursive Taxonomy Explorer** - Hierarchical decomposition with branching factor W and depth D
- **P5: Multi-Perspective Parallel Probing** - N expert personas working in parallel

Implementation: `pipelines/p2_sequential.py`, `p3_reflection.py`, `p4_taxonomy_explorer.py`, `p5_debate.py`

**2. Knowledge Processor (Section 3.3)**
Two-stage deduplication ensuring semantic uniqueness:
- **Stage 1: Vector Space Filtering** - Cosine similarity > 0.92 → immediate merge (using qwen3-8b-emb)
- **Stage 2: LLM-based Adjudication** - For ambiguous pairs (0.70 < sim < 0.92), use DeepSeek-V3.1 as judge

Implementation: `core/processor.py:86-116` (novelty mask), `processor.py:109-116` (LLM judge)

**3. Saturation-Based Extraction**
Each pipeline runs until knowledge growth saturates:
- Tracks novelty via embedding similarity against accumulated knowledge
- Stops when: growth < 1% OR efficiency < 10% OR novel_count < 3
- Supports trajectory recording and resumption for long-running experiments

Implementation: `BasePipeline.run()` at pipelines/base.py:61-155

### Evaluation Protocol (Paper Section 3.4)

**Pareto Knowledge Frontier**: Plot (Cost, Yield) curves where:
- Cost = cumulative tokens (generation + embedding + auditing)
- Yield = |K_ext| / |K_baseline| (ratio of unique valid atoms relative to strongest baseline)

**Domain Audit**: Uses DeepSeek-V3.1 with Bloom's Taxonomy rubric to filter:
- Valid: Factual, Conceptual, or Procedural knowledge
- Invalid: Meta-statements, generic fluff, structural fragments

Implementation: `core/judge.py:11-47` with scientific rubric

### Key Components

> 详细 API 文档见下方 **Core Framework** 部分

| 目录 | 文件 | 功能 |
|------|------|------|
| `agents/` | `clientpool.py` | 多 Key 轮询、自动重试 |
| | `call_agent.py` | LLM 生成封装、token 统计 |
| `core/` | `processor.py` | Embedding、两阶段去重 |
| | `judge.py` | Domain Audit (Bloom's Taxonomy) |
| | `cleaner.py` | Bullet point 提取 |
| `pipelines/` | `base.py` | 饱和循环、轨迹记录、断点续传 |
| | `p2_sequential.py` | Sequential baseline |
| | `p3_reflection.py` | Self-reflective refinement |
| | `p4_taxonomy_explorer.py` | Recursive taxonomy |
| | `p5_debate.py` | Multi-profile debate |

### Experiment-Specific Implementation Details

**Pareto Curves Experiment (run_pareto_curves.py)**:
- Configurations defined in `PARETO_CONFIGS` (line 22-31)
- Smart resume logic: skips completed runs, extends cut-off runs from 15→30 turns
- Saves 3 files per config: `.trajectory.json` (full state), `_raw.json` (points), `.emb.pkl` (vectors)
- Repair logic: regenerates missing embeddings if trajectory exists but PKL is lost

**Pareto Evaluation (evaluate_pareto_curves.py)**:
- Uses P4_Taxonomy_L2W2 as baseline anchor (normalized to 1.0)
- Audit cache: `results/trajectory_audit_cache.json` (shared across runs to save cost)
- Enforces monotonicity: knowledge discovered is never lost across turns
- Output: `results/all_domains_pareto_data.json` with yield_ratio curves

**Model Comparison (run_model_comparison.py)**:
- Only runs P4_Taxonomy_L5W5 (best from Pareto experiment)
- Models: llama-3.1-8b/70b/405b-instruct
- GenAgent explicitly sets `base_url="https://integrate.api.nvidia.com/v1"` per model

**Scaling Evaluation (evaluate_model_comparison.py)**:
- Builds **cross-model union** by processing models sequentially into shared union_nodes
- Two-stage dedup: strict match (>0.92) → fuzzy LLM check (0.70-0.92) → new node
- Fuzzy cache: `results/eval_fuzzy_cache.json` (stores LLM judge results for pairs)
- Audit cache: `results/eval_audit_cache.json` (stores domain relevance judgments)
- Computes: Recall (coverage of union), Accuracy (valid/total), Unique Discovery

## Paper Writing (ICML 2026)

**Location**: `paper/section/`

**Structure**:
- `0_abstract.tex`: Complete abstract describing the framework
- `1_intro.tex`: Motivation, challenges, contributions
- `3_method.tex`: Problem formulation, 4 policies (P2-P5), Knowledge Processor, Pareto metric
- `4_exp.tex`: Two-stage experimental design (Strategy Search + Scaling Law), partial results
- `fig_pareto_curve.tex`: Figure for aggregated Pareto frontier
- `thought.txt`: 5-question framework for intro (Chinese notes)

**Current Status**:
- Method section is complete with formal problem definition
- Exp section has setup and structure, needs results from running experiments
- Key findings to report:
  1. **Exp1**: P4_Taxonomy_L5W5 dominates Pareto frontier (optimal strategy identified)
  2. **Exp2**: Knowledge volume grows with model size (scaling law confirmed)
  3. **Exp3**: RL specialization improves target domain accuracy (+96% relative) but causes catastrophic forgetting in non-target domains (-55% recall)
  4. Different models show distinct "Unique Discovery" patterns

**Important Results Files**:
- `results/all_domains_pareto_data_old.json`: Exp1 curve data
- `results/plots/*_old.png`: Exp1 visualizations
- `results/evolution_comparison/qwen_comparison_report.json`: Exp3 RL specialization analysis
- Exp2 results pending (currently running)

## Core Framework

本项目的核心是一个**可扩展的知识抽取框架**，包含以下模块：

### 1. Pipeline 抽象 (`pipelines/`)

所有探索策略继承自 `BasePipeline`，实现 `get_next_step()` 方法：

```python
class BasePipeline(ABC):
    def __init__(self, agent, processor, model):
        self.agent = agent          # LLM 生成器
        self.processor = processor  # 知识处理器（embedding、去重）
        self.model = model          # 模型标识

    @abstractmethod
    async def get_next_step(self, query, history, current_points, turn) -> str:
        """返回本轮的 LLM 响应文本"""
        pass

    async def run(self, query, max_turns=30) -> Dict:
        """主循环：生成 → 清洗 → 去重 → 记录轨迹"""
        # 1. 调用 get_next_step() 获取响应
        # 2. KnowledgeCleaner.clean_bullet_points(response, min_length=30) 提取知识点
        # 3. processor.get_embeddings() + get_novelty_mask() 去重
        # 4. 记录到 trajectory，检查饱和条件
```

**已实现的策略**：
| Pipeline | 文件 | 描述 | 关键参数 |
|----------|------|------|----------|
| P2_Sequential | `p2_sequential.py` | 迭代式 "What else?" | - |
| P3_Reflection | `p3_reflection.py` | 自我批判循环 | - |
| P4_Taxonomy | `p4_taxonomy_explorer.py` | 递归分类树 | `l1_width`, `l2_width` |
| P5_Debate | `p5_debate.py` | 多专家辩论 | `num_profiles` |

### 2. Knowledge Processor (`core/processor.py`)

负责 embedding 计算和语义去重：

```python
processor = KnowledgeProcessor(
    client_pool=client_pool,           # LLM API 客户端池
    judge_model="deepseek-ai/deepseek-v3.1",  # 模糊去重的判断模型
    embed_model="Qwen/Qwen3-Embedding-8B",    # Embedding 模型
    embed_client_pool=embed_client_pool       # Embedding API 客户端池（本地服务）
)

# 核心方法
embeddings = await processor.get_embeddings(texts)      # 批量获取 embedding
is_novel = processor.get_novelty_mask(new_embs, old_embs, threshold=0.92)  # 新颖性判断
is_same = await processor._ask_llm_if_same(text1, text2)  # LLM 模糊判断
```

### 3. Domain Judge (`core/judge.py`)

基于 Bloom's Taxonomy 判断知识点是否有效：

```python
judge = DomainJudge(client_pool=client_pool, model="deepseek-ai/deepseek-v3.1")
results = await judge.check_batch(domain_query, points)  # 批量判断
```

**有效知识的标准**：
- Factual: 具体事实、数据、定义
- Conceptual: 概念关系、原理、理论
- Procedural: 方法、步骤、算法

**无效的类型**：
- Meta-statements: "这是一个重要的概念"
- Generic fluff: "深度学习很有用"
- Structural fragments: 不完整的句子

### 4. Knowledge Cleaner (`core/cleaner.py`)

从 LLM 输出中提取结构化知识点：

```python
KnowledgeCleaner.clean_bullet_points(text, min_length=5, require_space=True)
```

| 用途 | min_length | require_space | 说明 |
|------|------------|---------------|------|
| Taxonomy 解析 | 5 (default) | **False** | 允许单词如 "LSTM", "Fine-Tuning" |
| Knowledge Points | **30** | True (default) | 要求完整句子，过滤短片段 |

**处理流程**：
1. 移除 `<think>...</think>` 思考标签（DeepSeek 特有）
2. 逐行匹配 bullet pattern (`- `, `* `, `1. `, `1)`)
3. 过滤 meta-statements（以 "here are", "sure" 等开头）
4. 长度和空格检查
5. 去重返回

### 5. Client Pool (`agents/clientpool.py`)

多 API Key 轮询，自动重试：

```python
client_pool = MultiKeyClientPool(
    api_keys=["nvapi-xxx", "nvapi-yyy"],
    base_url="https://integrate.api.nvidia.com/v1"  # 或本地服务 URL
)
```

**错误处理**：429 (rate limit) → 切换 key 重试；403/401 → 标记 key 失效

### 6. Gen Agent (`agents/call_agent.py`)

LLM 生成封装，带 token 统计：

```python
agent = GenAgent(api_key=api_keys, model="meta/llama-3.1-8b-instruct")
response = await agent.generate(prompt)
print(agent.total_tokens)  # 累计 token 消耗
```

---

## Experiment Template

基于核心框架，实验脚本分为 **Run** 和 **Evaluate** 两部分。

### Run Script 结构 (`run_*.py`)

```python
# --- Configuration ---
MODELS = [...]                    # 待比较的模型列表
OUTPUT_BASE_DIR = "results/xxx"   # 输出目录
MAX_TURNS = 15                    # 最大轮数
TARGET_DOMAINS = {...}            # domain_id -> domain_query 映射

PIPELINE_CONFIG = {
    "id": "P4_Taxonomy_L5W5",
    "module": "pipelines.p4_taxonomy_explorer",
    "class": "RecursiveTaxonomyExplorer",
    "params": {"l1_width": 5, "l2_width": 5}
}

# --- Main Setup ---
async def main():
    # 1. Load API config
    with open("api.json") as f:
        api_data = json.load(f)

    # 2. Initialize client pools
    api_keys = api_data["api_keys"]
    client_pool = MultiKeyClientPool(api_keys=api_keys)
    embed_config = api_data.get("embed_config", {})

    # 3. Initialize KnowledgeProcessor (CRITICAL: must include embed_client_pool!)
    processor = KnowledgeProcessor(
        client_pool=client_pool,
        judge_model="deepseek-ai/deepseek-v3.1",
        embed_model=embed_config.get("model"),
        embed_client_pool=MultiKeyClientPool(
            api_keys=api_keys,
            base_url=embed_config.get("base_url")  # Local embedding server URL
        )
    )

    # 4. Create GenAgent per model
    gen_agent = GenAgent(api_key=api_keys, model=model_name)

    # 5. Instantiate pipeline and run
    pipeline = pipeline_class(gen_agent, processor, model=model_name, **params)
    await pipeline.run(domain_query, max_turns=MAX_TURNS)
```

**关键点**:
- `embed_client_pool` 必须传入，否则 embedding 会 fallback 到 NVIDIA API 导致 404
- `api.json` 中需要配置 `embed_config.base_url` 指向本地 sglang 服务

### 2. Evaluate Script 结构 (`evaluate_*.py`)

```python
# --- Thresholds ---
STRICT_MATCH = 0.92   # 严格去重阈值
FUZZY_LOW = 0.70      # 模糊去重下界

# --- Cache Paths ---
FUZZY_CACHE_PATH = "results/xxx_fuzzy_cache.json"   # LLM judge 结果缓存
AUDIT_CACHE_PATH = "results/xxx_audit_cache.json"   # Domain audit 结果缓存
```

### 3. 三阶段去重流程 (Per Model)

```
Raw Points (from _raw.json)
    │
    ▼ Stage 1: Strict Dedup (embedding similarity ≥ 0.92)
    │   - 计算所有点的 embedding
    │   - 贪心去重：保留第一个，标记所有 sim ≥ 0.92 的为重复
    │
    ▼ Stage 2: Fuzzy Dedup (0.70 ≤ similarity < 0.92)
    │   - 对每个点找最相似的祖先
    │   - 若 sim ∈ [0.70, 0.92)，查 fuzzy_cache 或调用 LLM judge
    │   - LLM 判断是否语义重复
    │
    ▼ Stage 3: Domain Audit
    │   - 查 audit_cache 或调用 DomainJudge
    │   - 判断是否为有效领域知识（基于 Bloom's Taxonomy）
    │
    ▼ Valid Points
```

### 4. Union 构建与指标计算

```python
# Union 构建：合并所有模型的 valid points，再做两阶段语义去重
all_valid_points = []
for model in models:
    all_valid_points.extend(model_valid_points[model])
union = build_union_with_dedup(all_valid_points)  # Strict + Fuzzy dedup

# 指标计算
for model in models:
    m_pts = model_valid_points[model]

    # Recall: 该模型覆盖了多少 union
    recall = len(m_pts) / len(union)

    # Unique Discovery: 该模型独有的知识点（其他模型都没有的）
    other_pts = [p for m in models if m != model for p in model_valid_points[m]]
    unique_count = 0
    for pt in m_pts:
        max_sim = max(cosine_similarity(pt, other) for other in other_pts)
        if max_sim < STRICT_MATCH:  # 没有语义匹配的
            unique_count += 1
```

### 5. Cleaner 配置

`KnowledgeCleaner.clean_bullet_points(text, min_length, require_space)`:

| 用途 | min_length | require_space | 说明 |
|------|------------|---------------|------|
| Taxonomy 解析 | 5 (default) | **False** | 允许单词如 "LSTM", "Fine-Tuning" |
| Knowledge Points | **30** | True (default) | 要求完整句子，过滤短片段 |

**在 pipeline 中**:
```python
# Taxonomy (L1/L2 categories)
KnowledgeCleaner.clean_bullet_points(res, require_space=False)

# Knowledge points (in BasePipeline.run)
KnowledgeCleaner.clean_bullet_points(response, min_length=30)
```

### 6. API 配置 (`api.json`)

```json
{
    "api_keys": ["nvapi-xxx", "nvapi-yyy"],
    "embed_config": {
        "model": "Qwen/Qwen3-Embedding-8B",
        "base_url": "http://localhost:30000/v1"
    }
}
```

### 7. 输出文件结构

```
results/{experiment_name}/
├── {domain}/
│   └── {model}/
│       ├── {pipeline_id}.trajectory.json  # 完整轨迹（含 internal_state）
│       ├── {pipeline_id}_raw.json         # 最终知识点列表
│       ├── {pipeline_id}.emb.pkl          # Embedding 缓存
│       └── {pipeline_id}.tokens.json      # Token 统计
├── {experiment}_fuzzy_cache.json          # LLM judge 缓存（跨模型共享）
├── {experiment}_audit_cache.json          # Domain audit 缓存（跨模型共享）
└── {experiment}_report.json               # 最终评估报告
```

### 8. 常见问题

**Q: Embedding 404 错误**
A: 检查 `embed_client_pool` 是否正确传入，`base_url` 是否指向本地 sglang 服务

**Q: Taxonomy 节点数不足 25**
A: 检查 cleaner 是否使用 `require_space=False`，否则单词术语会被过滤

**Q: 出现短片段（<30字符）的"知识点"**
A: 检查 `BasePipeline.run()` 中是否使用 `min_length=30`

**Q: Cache 没有生效**
A: 确保每个 domain 处理完后调用 `save_cache()`，避免中断丢失

## Important Thresholds

- **Saturation threshold**: 0.92 (strict match during generation)
- **Fuzzy threshold range**: 0.70-0.92 (LLM judge zone for ambiguous pairs)
- **Min growth ratio**: 0.01 (1% new knowledge per turn)
- **Min efficiency ratio**: 0.10 (10% of raw output must be novel)
- **Min novel count**: 3 (absolute minimum per turn)
- **Max turns**: 30 (extended from original 15)

## Data Organization

**Pareto Experiment**:
```
results/pareto_curves_405b/
├── {domain}/
│   ├── {ConfigID}.trajectory.json     # Full turn-by-turn state
│   ├── {ConfigID}_raw.json            # Final point list
│   └── {ConfigID}.emb.pkl             # Embedding cache
├── all_domains_pareto_data.json       # Evaluated yield curves
└── trajectory_audit_cache.json        # Domain audit results (shared)
```

**Scaling Experiment**:
```
results/model_comparison_l5w5/
├── {domain}/
│   └── {model}/
│       ├── P4_Taxonomy_L5W5.trajectory.json
│       ├── P4_Taxonomy_L5W5_raw.json
│       └── P4_Taxonomy_L5W5.emb.pkl
├── eval_fuzzy_cache.json              # Pairwise LLM judge cache
└── eval_audit_cache.json              # Domain relevance cache
```

**RL Specialization Experiment**:
```
results/evolution_comparison/
├── {domain}/
│   └── {model_folder}/                # qwen_qwen2.5-7b-instruct, qwen_qwen2.5-7b-coder-instruct
│       ├── P4_Taxonomy_L5W5.trajectory.json
│       ├── P4_Taxonomy_L5W5_raw.json
│       └── P4_Taxonomy_L5W5.emb.pkl
├── qwen_comparison_report.json        # Full metrics and sampled examples
├── qwen_comparison_summary.txt        # Summary table
├── evolution_fuzzy_cache.json         # Pairwise LLM judge cache
└── evolution_audit_cache.json         # Domain relevance cache
```
