import numpy as np
import asyncio
from typing import List, Dict, Any
from agents.clientpool import safe_embed

class KnowledgeProcessor:
    def __init__(self, client_pool, threshold: float = 0.90, embed_model: str = "nvidia/llama-3.2-nv-embedqa-1b-v2"):
        self.client_pool = client_pool
        self.threshold = threshold
        self.embed_model = embed_model
        self.union_set: List[Dict[str, Any]] = [] # List of { "text": str, "embedding": np.ndarray }

    async def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []
        
        # Batch processing for embeddings
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
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    async def process_pipeline_output(self, pipeline_name: str, bullet_points: List[str]) -> List[int]:
        """
        Process output from a pipeline, update union set, and return indices of matched/new points.
        """
        embeddings = await self.get_embeddings(bullet_points)
        matched_indices = []

        for text, emb in zip(bullet_points, embeddings):
            found = False
            for idx, existing in enumerate(self.union_set):
                if self.cosine_similarity(emb, existing["embedding"]) >= self.threshold:
                    matched_indices.append(idx)
                    found = True
                    break
            
            if not found:
                new_idx = len(self.union_set)
                self.union_set.append({
                    "text": text,
                    "embedding": emb,
                    "pipelines": {pipeline_name}
                })
                matched_indices.append(new_idx)
            else:
                self.union_set[matched_indices[-1]]["pipelines"].add(pipeline_name)

        return matched_indices

    def get_union_size(self) -> int:
        return len(self.union_set)

