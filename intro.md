# Introduction

### 1. What is the problem?
The core problem is how to systematically and comprehensively extract and quantify the boundaries of latent domain-specific knowledge embedded within Large Language Models (LLMs). Current evaluation paradigms rely heavily on static, multiple-choice datasets (e.g., MMLU), which are susceptible to data contamination and only measure reactive performance on predefined questions rather than proactively exploring the full extent of a model's internal knowledge base.

### 2. Why is it interesting and important?
As LLMs increasingly serve as the world's primary knowledge engines, understanding the limits of their internal knowledge is crucial for model transparency, safety alignment, and identifying specific capability gaps. Moving from "passive testing" to "active knowledge mining" allows researchers to define the "knowledge ceiling" of a model. Furthermore, quantifying knowledge saturation offers a more granular and intrinsic dimension to Scaling Laws—observing how knowledge density evolves as models scale from 8B to 70B and 405B parameters.

### 3. Why is it hard?
Probing the limits of a model's knowledge is challenging due to:
*   **Knowledge Saturation and Redundancy**: Naive sequential prompting often causes models to repeat themselves or provide superficial answers, failing to reach the "long-tail" or niche knowledge.
*   **Lack of Absolute Ground Truth**: For specialized fields (such as specific sub-categories of ICML), no absolute, comprehensive, and up-to-date "gold standard" encyclopedia exists to act as a benchmark.
*   **Semantic Deduplication at Scale**: Identifying truly unique knowledge points among tens of thousands of varied phrasings requires high-precision semantic processing to distinguish between rephrased existing points and genuine new insights.

### 4. Why hasn't it been solved before?
Previous research has largely focused on static benchmarks (which are inflexible and easily contaminated) or simple single-strategy agentic extraction. Our approach differs by introducing:
*   **Saturation-based Iterative Framework**: We utilize a dynamic feedback loop that measures the growth rate of novel embeddings to automatically detect when a model has reached its knowledge depletion point.
*   **Multi-strategy Pipeline Matrix**: Instead of a single prompt, we utilize a diverse set of cognitive strategies—such as **Recursive Taxonomy Explorer** (top-down structural decomposition) and **Multi-Agent Debate** (adversarial refinement)—to "squeeze" the model from multiple perspectives.
*   **Pseudo Ground Truth (PGT) Evaluation**: We solve the lack of ground truth by constructing a global union set of knowledge from all tested models and pipelines, enabling the calculation of **Relative Recall**.

### 5. What are the key components of my approach and results?
*   **Key Components**:
    *   **KnowledgeProcessor**: An optimized engine for high-concurrency vector-space deduplication and semantic analysis.
    *   **Persona-Driven Multi-Agent Pipelines**: Specialized strategies including P3 (Reflection), P4 (Taxonomy), and P5 (Adversarial Debate) that force models out of their default "safe" responses.
    *   **Domain Judge**: An automated auditing layer (using models like Llama 3.1 70B) to filter out noise and ensure domain relevance.
*   **Expected Results**:
    *   **Strategy Efficacy**: Structured pipelines (P4, P5) significantly outperform naive sequential baselines in terms of knowledge yield.
    *   **Scaling Insights**: A clear scaling effect where knowledge coverage and depth increase significantly as model parameters scale from 8B to 405B.
    *   **Domain Sensitivity**: Different model families exhibit distinct "knowledge signatures" across various sub-fields of machine learning.
*   **Limitations**:
    *   The framework is computationally intensive due to the multi-agent nature and recursive calls.
    *   The "Pseudo Ground Truth" is still an approximation, representing the collective lower-bound of current state-of-the-art model knowledge.
