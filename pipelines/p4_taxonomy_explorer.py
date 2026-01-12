import asyncio
import collections
from .base import BasePipeline
from core.cleaner import KnowledgeCleaner
from typing import List

class RecursiveTaxonomyExplorer(BasePipeline):
    """
    A top-down agentic pipeline that builds a taxonomy of the domain
    and extracts knowledge points from leaf nodes.
    
    Refactored Logic:
    - Turn 1: Build full taxonomy tree and extract knowledge from all leaf nodes (Broad Sweep).
    - Turn 2+: For EACH leaf node, reflect on its specific existing points and find missing details (Per-node Reflection).
    """
    def __init__(self, agent, processor, model="meta/llama-3.2-3b-instruct"):
        super().__init__(agent, processor, model)
        self.max_depth = 2
        self.taxonomy = [] # Stores leaf node paths
        self.leaf_knowledge = collections.defaultdict(list) # leaf_path -> list of points
        self.concurrency_limit = 20 # Optimized for 20 valid API keys
        self.semaphore = asyncio.Semaphore(self.concurrency_limit)

    async def _safe_generate(self, prompt: str) -> str:
        """Helper to run generate within a semaphore."""
        async with self.semaphore:
            return await self.agent.generate(prompt)

    async def _build_taxonomy(self, query: str) -> List[str]:
        """Builds the taxonomy and returns a list of leaf node paths."""
        print(f"      [Taxonomy] Building level 1 sub-categories...")
        prompt_l1 = (
            f"You are a subject matter expert in '{query}'. "
            f"Break down the domain '{query}' into 5-8 major, distinct sub-categories. "
            f"Provide ONLY the names of the categories as a simple bulleted list. Do not include descriptions."
        )
        res_l1 = await self._safe_generate(prompt_l1)
        sub_cats = KnowledgeCleaner.clean_bullet_points(res_l1)
        sub_cats = [c.split(':')[0].split(' - ')[0].strip() for c in sub_cats]
        print(f"      [Taxonomy] Found {len(sub_cats)} level 1 categories: {sub_cats}")
        
        if self.max_depth <= 1:
            return sub_cats

        print(f"      [Taxonomy] Expanding into level 2 sub-fields for {len(sub_cats)} categories...")
        tasks = []
        for cat in sub_cats:
            prompt_l2 = (
                f"In the context of the field '{query}', the sub-topic '{cat}' can be further divided. "
                f"Provide 3-5 more specific sub-fields or key components of '{cat}' as a bulleted list. "
                f"Provide ONLY the names of the sub-fields. Do not include descriptions."
            )
            tasks.append(self._safe_generate(prompt_l2))
        
        res_l2_list = await asyncio.gather(*tasks)
        
        leaf_paths = []
        for cat, res_l2 in zip(sub_cats, res_l2_list):
            subs = KnowledgeCleaner.clean_bullet_points(res_l2)
            subs = [s.split(':')[0].split(' - ')[0].strip() for s in subs]
            print(f"      [Taxonomy] Category '{cat}' has {len(subs)} sub-fields.")
            for sub in subs:
                leaf_paths.append(f"{cat} -> {sub}")
        
        return leaf_paths

    async def _process_leaf_turn1(self, query: str, leaf: str) -> str:
        """Initial harvest for a single leaf node."""
        prompt = (
            f"As an expert in '{query}', specifically focusing on '{leaf}', "
            f"list all fundamental, atomic knowledge points, definitions, and theorems. "
            f"Be extremely detailed and precise. Use bullet points."
        )
        res = await self._safe_generate(prompt)
        points = KnowledgeCleaner.clean_bullet_points(res)
        self.leaf_knowledge[leaf].extend(points)
        return f"### Knowledge for {leaf}:\n{res}"

    async def _process_leaf_reflection(self, query: str, leaf: str) -> str:
        """Reflect and find missing points for a single leaf node by focusing on its own history."""
        existing_points = self.leaf_knowledge[leaf]
        unique_points = list(dict.fromkeys(existing_points))
        points_str = "\n".join([f"- {p}" for p in unique_points])
        
        prompt = f"""You are an expert in '{query}', specifically focusing on the sub-topic '{leaf}'. 
We have already collected the following knowledge points for this specific sub-topic:

{points_str}

### Task
Identify and list ANY additional missing knowledge points, advanced theorems, or subtle nuances specifically related to '{leaf}' that are NOT already covered in the list above.
Focus on extreme depth and technical details. 

Provide ONLY the new, missing bullet points.
"""
        res = await self._safe_generate(prompt)
        new_points = KnowledgeCleaner.clean_bullet_points(res)
        self.leaf_knowledge[leaf].extend(new_points)
        return f"### Additional knowledge for {leaf}:\n{res}"

    async def get_next_step(self, query: str, history: List[str], current_points: List[str], turn: int) -> str:
        if turn == 1:
            # Step 1: Build the map (Categories and Sub-categories)
            self.taxonomy = await self._build_taxonomy(query)
            
            # Step 2: Initial harvest from all leaf nodes in parallel
            print(f"      [Taxonomy] Extracting knowledge from {len(self.taxonomy)} leaf nodes...")
            tasks = [self._process_leaf_turn1(query, leaf) for leaf in self.taxonomy]
            harvest_results = await asyncio.gather(*tasks)
            
            return "\n\n".join(harvest_results)
        
        else:
            # Turn 2+: Focus on per-node deepening
            print(f"      [Taxonomy] Turn {turn}: Reflecting on {len(self.taxonomy)} nodes individually...")
            tasks = [self._process_leaf_reflection(query, leaf) for leaf in self.taxonomy]
            reflection_results = await asyncio.gather(*tasks)
            
            return "\n\n".join(reflection_results)
