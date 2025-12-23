# LLM Knowledge Extraction Evaluation Framework: Progress Summary

## 1. Project Objective
To build a dynamic, extensible framework that evaluates different **Knowledge Extraction Pipelines** (strategies) for a given LLM and domain. The framework establishes a "Pseudo Ground Truth" (Union Set) from all participating pipelines and calculates:
- **Recall (Coverage)**: How much of the model's total known knowledge in a domain can a single pipeline extract?
- **Accuracy (Relevance)**: What percentage of the extracted points are strictly relevant to the target domain?

## 2. Key Achievements & Current Implementation
- **Extensible Pipeline Architecture**: A modular system where new strategies (e.g., Reflection, Multi-turn, Agentic Search) can be added as standalone Python classes.
- **Sequential Multi-turn Experiment**: Implemented a staged extraction process (1 to 4 turns) to observe knowledge growth and convergence curves.
- **Hybrid Deduplication Logic**: A two-stage merging process:
    1. **Stage 1 (Embedding)**: High-confidence similarity matches (>= 0.92) are auto-merged.
    2. **Stage 2 (LLM Judge)**: Points in the "Grey Zone" (0.75 - 0.92) are adjudicated by a small LLM to determine semantic identity.
- **Judge-Before-Embed Filtering**: Irrelevant knowledge points are filtered out by a "Domain Judge" *before* they enter the Union Set, ensuring the denominator for Recall is pure.
- **High-Performance Adjudication**: Parallelized API calls using `asyncio.gather` to handle hundreds of knowledge points in seconds.
- **Full Traceability**: Every node in the Union Set tracks its original phrasing and contributing pipelines.

## 3. Critical Design Dilemmas & Examples
During implementation, we encountered several edge cases that challenge standard NLP approaches. These need further discussion for the final framework design:

### A. The "Semantic Trap" (Similarity vs. Logic)
*   **The Issue**: Embedding models often give high similarity scores to statements that share 90% of their vocabulary but are logically opposite.
*   **Example**:
    *   **Point A**: "A set of vectors is **linearly dependent** if at least one vector..."
    *   **Point B**: "A set of vectors is **linearly independent** if none of the vectors..."
    *   **Cosine Similarity**: **0.8143**
    *   **Current Solution**: We set the auto-merge threshold high (0.92) and use an LLM to distinguish these "negation" cases. A pure embedding-based system would have merged these, creating a logical error in the knowledge graph.

### B. Contextual Fragmentation
*   **The Issue**: The same mathematical concept is often described in different contexts (e.g., as a definition vs. a property).
*   **Example**:
    *   **Point A**: "A **basis** is a set of linearly independent vectors that span the space."
    *   **Point B**: "**Linear Independence**: A property where no vector in a set is a linear combination of others."
    *   **Challenge**: Should these be merged? Currently, our strict LLM judge says **"NO"** because A is about the *components of a basis*, while B is a *standalone property*. This leads to a larger Union Set and lower individual Recall.

### C. Phrasal Variance & Complexity
*   **The Issue**: Long, formal definitions vs. short, conceptual summaries often fall below traditional fixed thresholds.
*   **Example**:
    *   **Point A**: "LU decomposition is a factorization into a lower triangular matrix L and an upper triangular matrix U."
    *   **Point B**: "LU decomposition expresses a matrix as a product of two matrices, where the first contains lower entries and the second contains upper entries."
    *   **Cosine Similarity**: **~0.78** (This would be missed by a strict 0.85 or 0.90 threshold).
    *   **Current Solution**: Our "Grey Zone" starts at **0.75**, allowing these pairs to be sent to the LLM Judge for a semantic confirmation, rather than being discarded as different points.

### D. The Granularity of Knowledge
*   **The Issue**: Should a "title" merge with its "definition"?
*   **Example**:
    *   **Point A**: "Vector addition properties." (Summary)
    *   **Point B**: "Vector addition is commutative and associative." (Detail)
    *   **Dilemma**: If we merge them, we lose detail. If we don't, the Recall score is "penalized" for pipelines that provide more detail.

## 4. Preliminary Results (Case Study: Linear Algebra)
The following leaderboard was generated using our current hybrid framework. It demonstrates how multi-turn strategies significantly boost Recall compared to a one-shot baseline.

| Pipeline | Recall (Coverage) | Accuracy (Relevance) |
| :--- | :---: | :---: |
| Sequential_Turn_1 | 30.40% | 95.00% |
| Sequential_Turn_2 | 48.00% | 90.54% |
| Sequential_Turn_3 | 56.00% | 87.50% |
| **Sequential_Turn_4** | **58.40%** | **87.62%** |
| ReflectionPipeline | 50.40% | 90.91% |

**Key Observations:**
*   **Marginal Gains**: Moving from Turn 1 to Turn 2 provides the biggest jump in Recall (+17.6%), while moving from Turn 3 to Turn 4 shows diminishing returns (+2.4%).
*   **Accuracy vs. Depth**: Turn 1 (One-shot) has the highest accuracy (95%), but as we "force" the model to dig deeper in later turns, it starts producing slightly more irrelevant or repetitive content, leading to an accuracy dip.

## 5. Discussion Points for Meeting
1. **Adjudication Strictness**: Should we loosen the "EXACT same" requirement in the LLM judge to encourage more merging and higher Recall scores?
2. **Gold Standard vs. Union Set**: As we add more powerful pipelines (e.g., sub-topic decomposition), the Union Set will grow. Is it acceptable for existing pipelines' Recall to drop as the "Total Knowledge" denominator increases?
3. **Accuracy Definition**: Currently `Valid / Total_Raw`. Should we also penalize repetitive valid points in the Accuracy score?

