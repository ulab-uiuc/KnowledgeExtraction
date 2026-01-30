import numpy as np
import asyncio
import json
import os
import pickle
from typing import List, Dict, Any, Set, Optional, Tuple
from tqdm import tqdm
from agents.clientpool import safe_embed, safe_ask

class KnowledgeProcessor:
    def __init__(self, client_pool, threshold: float = 0.92, candidate_threshold: float = 0.70, embed_model: str = "nvidia/nv-embed-v1", embed_client_pool=None, judge_model: str = "deepseek-ai/deepseek-v3.1"):
        self.client_pool = client_pool
        self.embed_client_pool = embed_client_pool or client_pool
        self.threshold = threshold
        self.candidate_threshold = candidate_threshold
        self.embed_model = embed_model
        self.judge_model = judge_model
        self.embed_cache: Dict[str, np.ndarray] = {}
        self.total_tokens = 0 
        self.embed_semaphore = asyncio.Semaphore(100) # Increased for local SGLang performance
        self.llm_semaphore = asyncio.Semaphore(20)

    def load_embeddings(self, path: str):
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    self.embed_cache.update(data)
                    return len(data)
            except Exception as e:
                print(f"      [Cache] Warning: Failed to load {path}: {e}")
        return 0

    def save_embeddings(self, path: str):
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.embed_cache, f)
        except Exception as e:
            print(f"      [Cache] Warning: Failed to save {path}: {e}")

    async def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        if not texts: return []
        unique_needed = list(set(t for t in texts if t not in self.embed_cache))
        if unique_needed:
            batch_size = 128
            kwargs = {}
            if "nvidia" in self.embed_model.lower():
                kwargs["extra_body"] = {"input_type": "query", "truncate": "NONE"}
            async def wrapped_embed(batch):
                async with self.embed_semaphore:
                    for attempt in range(5): # Added retry logic
                        try:
                            res = await safe_embed(self.embed_client_pool, self.embed_model, batch, **kwargs)
                            if hasattr(res, 'usage') and res.usage:
                                self.total_tokens += res.usage.total_tokens
                            return batch, res
                        except Exception as e:
                            if attempt == 4:
                                print(f"\n      [Embedding Error] Failed after 5 attempts: {e}")
                                return batch, None
                            await asyncio.sleep(1.0 * (attempt + 1))
            tasks = [wrapped_embed(unique_needed[i:i + batch_size]) for i in range(0, len(unique_needed), batch_size)]
            results = await asyncio.gather(*tasks)
            for batch_texts, resp in results:
                if resp and hasattr(resp, 'data'):
                    for t, d in zip(batch_texts, resp.data):
                        self.embed_cache[t] = np.array(d.embedding)
        output = []
        for t in texts:
            if t in self.embed_cache:
                output.append(self.embed_cache[t])
            else:
                output.append(np.zeros(1024))
        return output

    def get_novelty_mask(self, new_embs: List[np.ndarray], pool_embs: List[np.ndarray], threshold: float) -> List[bool]:
        if not new_embs: return []
        new_mat = np.stack(new_embs)
        norms = np.linalg.norm(new_mat, axis=1, keepdims=True) + 1e-9
        new_mat = new_mat / norms
        N = len(new_embs)
        is_novel = [True] * N
        if N > 1:
            self_sim = np.dot(new_mat, new_mat.T)
            for i in range(1, N):
                if np.max(self_sim[i, :i]) > threshold: is_novel[i] = False
        if pool_embs:
            pool_mat = np.stack(pool_embs)
            pool_mat = pool_mat / (np.linalg.norm(pool_mat, axis=1, keepdims=True) + 1e-9)
            sim_matrix = np.dot(new_mat, pool_mat.T)
            max_sims = np.max(sim_matrix, axis=1)
            for i in range(N):
                if max_sims[i] > threshold: is_novel[i] = False
        return is_novel

    async def _ask_llm_if_same(self, text1: str, text2: str) -> bool:
        # Old simple prompt:
        # prompt = f"""Do these two describe the same knowledge? Respond ONLY YES or NO.\nA: {text1}\nB: {text2}"""
        
        # New expert-level redundancy check prompt:
        prompt = f"""You are a knowledge engineering expert. Compare two knowledge points and determine if one is redundant given the other.

Point A: {text1}
Point B: {text2}

Criteria for redundancy (YES):
1. They refer to the exact same concept using different wording.
2. One is a pure synonym of the other.

Criteria for uniqueness (NO):
1. One provides more specific details than the other (e.g., "Batch Normalization" vs "Group Normalization").
2. They are related but distinct concepts (e.g., "Gradients" vs "Hessians").
3. They describe different aspects of the same topic.

Does Point A and Point B refer to the same specific knowledge point such that one can be removed without losing ANY unique information?
Respond ONLY with 'YES' or 'NO'."""

        async with self.llm_semaphore:
            try:
                response = await safe_ask(self.client_pool, self.judge_model, [{"role": "user", "content": prompt}], temperature=0.0)
                return "YES" in response.choices[0].message.content.strip().upper()
            except: return False
