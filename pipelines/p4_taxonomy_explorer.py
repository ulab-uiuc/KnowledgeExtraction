import asyncio
import collections
from .base import BasePipeline
from core.cleaner import KnowledgeCleaner
from typing import List

class RecursiveTaxonomyExplorer(BasePipeline):
    """
    A top-down agentic pipeline that builds a taxonomy of the domain
    and extracts knowledge points from leaf nodes.
    
    L5W5 Version:
    - Parametrized widths for Level 1 and Level 2.
    - Turn 1: Build taxonomy and initial harvest.
    - Turn 2+: Iterative deepening on leaf nodes.
    """
    def __init__(self, agent, processor, model="meta/llama-3.1-8b-instruct", l1_width=5, l2_width=5):
        super().__init__(agent, processor, model)
        self.l1_width = l1_width
        self.l2_width = l2_width
        self.taxonomy = [] # Stores leaf node paths
        self.leaf_knowledge = collections.defaultdict(list) # leaf_path -> list of points
        self.semaphore = asyncio.Semaphore(50)

    def get_internal_state(self):
        return {
            "taxonomy": self.taxonomy,
            "leaf_knowledge": dict(self.leaf_knowledge)
        }

    def set_internal_state(self, state):
        self.taxonomy = state.get("taxonomy", [])
        self.leaf_knowledge = collections.defaultdict(list, state.get("leaf_knowledge", {})) 

    async def _safe_generate(self, prompt: str) -> str:
        """Helper to run generate within a semaphore."""
        async with self.semaphore:
            return await self.agent.generate(prompt)

    async def _build_taxonomy(self, query: str) -> List[str]:
        """Builds the taxonomy and returns a list of leaf node paths."""
        print(f"      [Taxonomy] Building level 1 sub-categories (Width: {self.l1_width})...")
        prompt_l1 = f"""You are a subject matter expert in '{query}'.

Task: Break down the domain '{query}' into exactly {self.l1_width} major, distinct sub-categories.

Output format: Output ONLY {self.l1_width} lines, one category name per line, starting with "- ". No descriptions, no explanations, no numbering.

Example format:
- Category A
- Category B
- Category C
- Category D
- Category E

Now list exactly {self.l1_width} sub-categories of '{query}':"""
        res_l1 = await self._safe_generate(prompt_l1)
        sub_cats_raw = KnowledgeCleaner.clean_bullet_points(res_l1, require_space=False)
        sub_cats = [c.split(':')[0].split(' - ')[0].strip() for c in sub_cats_raw][:self.l1_width]
        if len(sub_cats_raw) != self.l1_width:
            print(f"      [Taxonomy] WARNING: L1 returned {len(sub_cats_raw)} items (expected {self.l1_width})")

        print(f"      [Taxonomy] Expanding into level 2 sub-fields (Width: {self.l2_width}) for {len(sub_cats)} categories...")
        tasks = []
        for cat in sub_cats:
            prompt_l2 = f"""In the field of '{query}', the sub-topic '{cat}' can be further divided.

Task: List exactly {self.l2_width} specific sub-fields or key components of '{cat}'.

Output format: Output ONLY {self.l2_width} lines, one sub-field name per line, starting with "- ". No descriptions, no explanations.

Example format:
- Sub-field A
- Sub-field B
- Sub-field C
- Sub-field D
- Sub-field E

Now list exactly {self.l2_width} sub-fields of '{cat}':"""
            tasks.append(self._safe_generate(prompt_l2))
        
        res_l2_list = await asyncio.gather(*tasks)
        
        leaf_paths = []
        l2_issues = []
        for cat, res_l2 in zip(sub_cats, res_l2_list):
            subs_raw = KnowledgeCleaner.clean_bullet_points(res_l2, require_space=False)
            subs = subs_raw[:self.l2_width]
            if len(subs_raw) < self.l2_width:
                l2_issues.append(f"{cat}={len(subs_raw)}")
            for sub in subs:
                sub_clean = sub.split(':')[0].split(' - ')[0].strip()
                leaf_paths.append(f"{cat} -> {sub_clean}")

        if l2_issues:
            print(f"      [Taxonomy] WARNING: L2 shortfalls: {', '.join(l2_issues)}")

        return leaf_paths

    async def _process_leaf_turn1(self, query: str, leaf: str) -> str:
        """Initial harvest for a single leaf node."""
        prompt = f"""You are an expert in '{query}', specifically in '{leaf}'.

Task: List fundamental knowledge points about '{leaf}'. Each knowledge point must be:
1. A complete, self-contained factual statement
2. Technically precise and accurate
3. Specific to '{leaf}' (not generic statements)

Output format: Each line starts with "- " followed by a complete sentence. No headers, no explanations, no numbering.

Example of GOOD knowledge points:
- The backpropagation algorithm computes gradients by applying the chain rule from output to input layers.
- Batch normalization normalizes activations to have zero mean and unit variance within each mini-batch.
- Dropout randomly sets neurons to zero during training with probability p to prevent overfitting.

Example of BAD knowledge points (do NOT output these):
- Backpropagation (too short, not a sentence)
- This is important for training (vague, not specific)
- Neural networks are used in deep learning (too generic)

Now list knowledge points for '{leaf}':"""
        res = await self._safe_generate(prompt)
        points = KnowledgeCleaner.clean_bullet_points(res, min_length=30)  # Require complete sentences
        self.leaf_knowledge[leaf].extend(points)
        return f"### Knowledge for {leaf}:\n{res}"

    async def _process_leaf_reflection(self, query: str, leaf: str) -> str:
        """Reflect and find missing points for a single leaf node."""
        existing_points = self.leaf_knowledge[leaf]
        unique_points = list(dict.fromkeys(existing_points))
        points_str = "\n".join([f"- {p}" for p in unique_points])

        prompt = f"""You are an expert in '{query}', specifically in '{leaf}'.

Existing knowledge points we have collected:
{points_str}

Task: List NEW knowledge points about '{leaf}' that are NOT covered above. Each knowledge point must be:
1. A complete, self-contained factual statement
2. Technically precise and accurate
3. Different from all existing points above

Output format: Each line starts with "- " followed by a complete sentence. No headers, no explanations.

Now list NEW knowledge points for '{leaf}':"""

        res = await self._safe_generate(prompt)
        new_points = KnowledgeCleaner.clean_bullet_points(res, min_length=30)  # Require complete sentences
        self.leaf_knowledge[leaf].extend(new_points)
        return f"### Additional knowledge for {leaf}:\n{res}"

    async def get_next_step(self, query: str, history: List[str], current_points: List[str], turn: int) -> str:
        if turn == 1:
            self.taxonomy = await self._build_taxonomy(query)
            print(f"      [Taxonomy] Extracting from {len(self.taxonomy)} nodes...")
            tasks = [self._process_leaf_turn1(query, leaf) for leaf in self.taxonomy]
            results = await asyncio.gather(*tasks)
            return "\n\n".join(results)
        else:
            tasks = [self._process_leaf_reflection(query, leaf) for leaf in self.taxonomy]
            results = await asyncio.gather(*tasks)
            return "\n\n".join(results)
