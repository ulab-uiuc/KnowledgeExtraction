import numpy as np
import asyncio
import json
import os
from typing import List, Dict, Any, Set, Optional, Tuple
from tqdm import tqdm
from agents.clientpool import safe_embed, safe_ask

class KnowledgeProcessor:
    def __init__(self, client_pool, threshold: float = 0.92, candidate_threshold: float = 0.82, embed_model: str = "nvidia/nv-embed-v1", embed_client_pool=None):
        self.client_pool = client_pool
        self.embed_client_pool = embed_client_pool or client_pool
        self.threshold = threshold  # Auto-merge above this
        self.candidate_threshold = candidate_threshold # Ask LLM between this and auto-merge
        self.embed_model = embed_model
        
        # Performance optimizations
        self.union_set: List[Dict[str, Any]] = []
        self.embedding_matrix: Optional[np.ndarray] = None # Shape: (N, D)
        self.llm_cache: Dict[str, bool] = {} # Key: sorted(text1, text2) -> bool
        self.embed_cache: Dict[str, np.ndarray] = {} # Global cache for embeddings
        self.llm_semaphore = asyncio.Semaphore(20) # Limit concurrent LLM calls

    async def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []
        
        # 1. Check cache
        needed_texts = []
        for t in texts:
            if t not in self.embed_cache:
                needed_texts.append(t)

        if needed_texts:
            batch_size = 64
            kwargs = {}
            if "nvidia" in self.embed_model.lower():
                kwargs["extra_body"] = {"input_type": "query", "truncate": "NONE"}

            # --- Optimization: Parallelize all embedding requests ---
            tasks = []
            for i in range(0, len(needed_texts), batch_size):
                batch = needed_texts[i:i + batch_size]
                # Enforce truncation to prevent overflow
                safe_batch = [t[:12000] for t in batch]
                tasks.append(safe_embed(
                    self.embed_client_pool,
                    model=self.embed_model,
                    messages=safe_batch,
                    **kwargs
                ))
            
            print(f"      [Embedding] Launching {len(tasks)} parallel batches to local server...")
            responses = await asyncio.gather(*tasks)
            
            # Store in cache
            idx = 0
            for resp in responses:
                for d in resp.data:
                    self.embed_cache[needed_texts[idx]] = np.array(d.embedding)
                    idx += 1

        return [self.embed_cache[t] for t in texts]

    def get_novelty_mask(self, new_embs: List[np.ndarray], pool_embs: List[np.ndarray], threshold: float) -> List[bool]:
        """
        Matrix-accelerated novelty detection using NumPy.
        Supports: 1. Intra-batch deduplication 2. Pool-based deduplication
        """
        if not new_embs:
            return []

        # 1. Prepare matrix for new points
        new_mat = np.stack(new_embs) # (N, D)
        new_mat = new_mat / (np.linalg.norm(new_mat, axis=1, keepdims=True) + 1e-9)
        
        N = len(new_embs)
        is_novel = [True] * N

        # 2. Intra-batch deduplication
        # Core: Each point only looks at points to its 'left' to avoid double-counting novel points in one turn
        if N > 1:
            self_sim_matrix = np.dot(new_mat, new_mat.T)
            # Use lower triangle (excluding diagonal) where i > j
            for i in range(1, N):
                if np.max(self_sim_matrix[i, :i]) > threshold:
                    is_novel[i] = False

        # 3. Pool deduplication (against history)
        if pool_embs:
            pool_mat = np.stack(pool_embs) # (M, D)
            pool_mat = pool_mat / (np.linalg.norm(pool_mat, axis=1, keepdims=True) + 1e-9)
            
            # Global similarity matrix (N, M)
            sim_matrix = np.dot(new_mat, pool_mat.T)
            max_sims_to_pool = np.max(sim_matrix, axis=1)
            
            for i in range(N):
                if max_sims_to_pool[i] > threshold:
                    is_novel[i] = False

        return is_novel

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    async def _ask_llm_if_same(self, text1: str, text2: str) -> bool:
        """
        Use a more capable model to judge if two points are semantically identical.
        Includes caching and semaphore for performance.
        """
        # 1. Check cache
        cache_key = "||".join(sorted([text1, text2]))
        if cache_key in self.llm_cache:
            return self.llm_cache[cache_key]

        prompt = f"""
        Do these two bullet points describe the same fundamental mathematical concept or property?
        
        Point A: {text1}
        Point B: {text2}
        
        Criteria for 'YES':
        1. They are semantically equivalent despite different phrasing.
        2. One is a slightly more detailed version of the same definition.
        3. They refer to the same mathematical theorem or property.
        
        Criteria for 'NO':
        1. They describe different concepts (e.g., Eigenvalue vs Eigenvector).
        2. They describe different properties of the same object.
        3. One is a general category and the other is a specific sub-topic.
        
        Answer only 'YES' or 'NO'.
        """
        
        async with self.llm_semaphore:
            try:
                response = await safe_ask(
                    self.client_pool,
                    model="meta/llama-3.1-70b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                raw_answer = response.choices[0].message.content.strip().upper()
                
                result = ("YES" in raw_answer) and ("NO" not in raw_answer)
                self.llm_cache[cache_key] = result
                return result
            except Exception:
                return False

    async def _pre_deduplicate_pipeline(self, pipeline_name: str, bullet_points: List[str]) -> Tuple[List[str], List[np.ndarray]]:
        """
        Fast internal deduplication: String match + High-threshold embedding match.
        Reduces the number of points before global merging.
        """
        if not bullet_points:
            return [], []
        
        # 1. Simple string deduplication (preserves order)
        unique_texts_ordered = list(dict.fromkeys(bullet_points))
        if len(unique_texts_ordered) < len(bullet_points):
            print(f"    [Pre-dedup] String deduplication: {len(bullet_points)} -> {len(unique_texts_ordered)}")
            
        embeddings = await self.get_embeddings(unique_texts_ordered)
        
        final_texts = []
        final_embeddings = []
        
        # High threshold for safe merging without LLM
        INTERNAL_THRESHOLD = 0.96 
        
        pbar = tqdm(total=len(unique_texts_ordered), desc="    Vector Dedup", leave=False)
        for text, emb in zip(unique_texts_ordered, embeddings):
            norm = np.linalg.norm(emb)
            if norm == 0: 
                pbar.update(1)
                continue
            
            is_new = True
            if final_embeddings:
                # Vectorized similarity check
                matrix = np.stack(final_embeddings)
                # Matrix is already normalized from previous iterations
                sims = np.dot(matrix, emb) / norm
                if np.max(sims) > INTERNAL_THRESHOLD:
                    is_new = False
            
            if is_new:
                final_texts.append(text)
                final_embeddings.append(emb / norm)
            pbar.update(1)
        pbar.close()
        
        if len(final_texts) < len(unique_texts_ordered):
            print(f"    [Pre-dedup] Vector deduplication: {len(unique_texts_ordered)} -> {len(final_texts)}")
            
        return final_texts, final_embeddings

    async def build_union_set(self, results_dir: str):
        """
        Optimized union set builder with NumPy matrix acceleration and pre-deduplication.
        """
        self.union_set = []
        self.embedding_matrix = None
        
        files = [f for f in os.listdir(results_dir) if f.endswith("_raw.json")]
        all_pipeline_data = {}
        for f in files:
            pipeline_name = f.replace("_raw.json", "") 
            with open(os.path.join(results_dir, f), "r") as f_in:
                all_pipeline_data[pipeline_name] = json.load(f_in)

        for pipeline_name in sorted(all_pipeline_data.keys()):
            raw_points = all_pipeline_data[pipeline_name]
            print(f"  Processing {pipeline_name} ({len(raw_points)} points)...")
            
            # Step 1: Pre-deduplicate
            bullet_points, embeddings = await self._pre_deduplicate_pipeline(pipeline_name, raw_points)
            
            # Step 2: Global Merge
            pbar = tqdm(total=len(bullet_points), desc="    Global Merge", leave=False)
            for text, emb in zip(bullet_points, embeddings):
                # emb is already normalized by _pre_deduplicate_pipeline
                match_idx = -1
                max_sim = -1
                
                if self.union_set:
                    # 2.1 Fast Matrix Similarity
                    sims = np.dot(self.embedding_matrix, emb)
                    
                    match_idx = int(np.argmax(sims))
                    max_sim = sims[match_idx]
                    
                    if max_sim >= self.threshold:
                        # Auto-merge
                        pass
                    elif max_sim >= self.candidate_threshold:
                        # LLM Check
                        if await self._ask_llm_if_same(text, self.union_set[match_idx]["representative_text"]):
                            # match_idx stays as is
                            pass
                        else:
                            match_idx = -1 # No match
                    else:
                        match_idx = -1 # No match
                
                if match_idx != -1:
                    # Merge into existing node
                    node = self.union_set[match_idx]
                    node["pipelines"].add(pipeline_name)
                    node["source_entries"].append({
                        "text": text,
                        "pipeline": pipeline_name,
                        "similarity": float(max_sim)
                    })
                    
                    # Update centroid incrementally
                    count = len(node["source_entries"])
                    current_sum = node["centroid_embedding"] * (count - 1)
                    new_sum = current_sum + emb
                    node["centroid_embedding"] = new_sum / np.linalg.norm(new_sum)
                    
                    # Sync to matrix
                    self.embedding_matrix[match_idx] = node["centroid_embedding"]
                else:
                    # Create new node
                    self.union_set.append({
                        "representative_text": text,
                        "centroid_embedding": emb,
                        "source_entries": [{
                            "text": text, 
                            "pipeline": pipeline_name,
                            "similarity": 1.0
                        }],
                        "pipelines": {pipeline_name}
                    })
                    # Update matrix
                    if self.embedding_matrix is None:
                        self.embedding_matrix = emb.reshape(1, -1)
                    else:
                        self.embedding_matrix = np.vstack([self.embedding_matrix, emb])
                pbar.update(1)
            pbar.close()
            
            print(f"    Union set size after {pipeline_name}: {len(self.union_set)}")

    def save_union_set(self, output_path: str):
        """Save the union set with detailed traceability."""
        serializable_set = []
        for item in self.union_set:
            serializable_set.append({
                "representative_text": item["representative_text"],
                "is_in_domain": item.get("is_in_domain", True), # Include audit result
                "pipelines_covered": list(item["pipelines"]),
                "occurrence_count": len(item["source_entries"]),
                "detailed_sources": item["source_entries"]
            })
        
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(serializable_set, f, indent=2, ensure_ascii=False)

    def get_union_size(self) -> int:
        return len(self.union_set)

    def get_pipeline_coverage(self, pipeline_name: str) -> List[int]:
        return [i for i, item in enumerate(self.union_set) if pipeline_name in item["pipelines"]]
