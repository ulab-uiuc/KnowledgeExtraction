import numpy as np
import asyncio
import json
import os
from typing import List, Dict, Any, Set
from agents.clientpool import safe_embed, safe_ask

class KnowledgeProcessor:
    def __init__(self, client_pool, threshold: float = 0.92, candidate_threshold: float = 0.75, embed_model: str = "nvidia/nv-embed-v1"):
        self.client_pool = client_pool
        self.threshold = threshold  # Auto-merge above this
        self.candidate_threshold = candidate_threshold # Ask LLM between this and auto-merge
        self.embed_model = embed_model
        # List of { 
        #   "representative_text": str, 
        #   "centroid_embedding": np.ndarray, 
        #   "source_entries": List[Dict], 
        #   "pipelines": Set[str] 
        # }
        self.union_set: List[Dict[str, Any]] = []

    async def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []
        
        batch_size = 64
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await safe_embed(
                self.client_pool,
                model=self.embed_model,
                messages=batch
            )
            all_embeddings.extend([np.array(d.embedding) for d in response.data])
        return all_embeddings

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    async def _ask_llm_if_same(self, text1: str, text2: str) -> bool:
        """
        Use a small model to judge if two points are semantically identical.
        Robust parsing: Must have 'YES' and must NOT have 'NO'.
        """
        prompt = f"""
        Are these two bullet points describing the EXACT same mathematical knowledge point?
        Point A: {text1}
        Point B: {text2}
        
        Answer only 'YES' or 'NO'. 
        If you are unsure, or if they describe slightly different aspects of the same topic, answer 'NO'.
        """
        try:
            response = await safe_ask(
                self.client_pool,
                model="meta/llama-3.2-3b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            raw_answer = response.choices[0].message.content.strip().upper()
            
            # Robust logic: 
            # 1. Answer must contain YES
            # 2. Answer must NOT contain NO (to handle "Yes, but actually No" or confusion)
            # 3. If neither or both, it defaults to False (No merge)
            has_yes = "YES" in raw_answer
            has_no = "NO" in raw_answer
            
            if has_yes and not has_no:
                return True
            return False
        except Exception:
            return False

    async def build_union_set(self, results_dir: str):
        """
        Load all raw JSON files, deduplicate using a hybrid Embedding + LLM approach.
        """
        self.union_set = []
        # Only process files ending with _raw.json
        files = [f for f in os.listdir(results_dir) if f.endswith("_raw.json")]
        
        all_pipeline_data = {}
        for f in files:
            # Strip both '.json' and '_raw' to match original pipeline name
            pipeline_name = f.replace("_raw.json", "") 
            with open(os.path.join(results_dir, f), "r") as f_in:
                all_pipeline_data[pipeline_name] = json.load(f_in)

        # Sort files to ensure deterministic merging order
        for pipeline_name in sorted(all_pipeline_data.keys()):
            bullet_points = all_pipeline_data[pipeline_name]
            embeddings = await self.get_embeddings(bullet_points)
            
            print(f"  Processing {pipeline_name} ({len(bullet_points)} points)...")
            
            for text, emb in zip(bullet_points, embeddings):
                match_idx = -1
                max_sim = -1
                candidates = [] # List of (idx, sim)
                
                # Phase 1: Embedding check
                for idx, existing in enumerate(self.union_set):
                    sim = self.cosine_similarity(emb, existing["centroid_embedding"])
                    if sim >= self.threshold:
                        if sim > max_sim:
                            max_sim = sim
                            match_idx = idx
                    elif sim >= self.candidate_threshold:
                        candidates.append((idx, sim))
                
                # Phase 2: LLM check for ambiguous cases (if no direct hit)
                if match_idx == -1 and candidates:
                    # Check top 3 candidates by similarity
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    for idx, sim in candidates[:3]:
                        if await self._ask_llm_if_same(text, self.union_set[idx]["representative_text"]):
                            match_idx = idx
                            max_sim = sim
                            break
                
                if match_idx != -1:
                    node = self.union_set[match_idx]
                    node["pipelines"].add(pipeline_name)
                    node["source_entries"].append({
                        "text": text,
                        "pipeline": pipeline_name,
                        "similarity": float(max_sim)
                    })
                    # Update centroid embedding
                    count = len(node["source_entries"])
                    node["centroid_embedding"] = (node["centroid_embedding"] * (count - 1) + emb) / count
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

    def save_union_set(self, output_path: str):
        """Save the union set with detailed traceability."""
        serializable_set = []
        for item in self.union_set:
            serializable_set.append({
                "representative_text": item["representative_text"],
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
