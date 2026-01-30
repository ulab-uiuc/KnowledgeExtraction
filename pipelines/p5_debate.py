import asyncio
from .base import BasePipeline
from core.cleaner import KnowledgeCleaner
from typing import List

class MultiProfileDebatePipeline(BasePipeline):
    def __init__(self, agent, processor, model="meta/llama-3.2-3b-instruct", num_profiles=5):
        super().__init__(agent, processor, model)
        self.num_profiles = num_profiles
        self.profiles = [] 
        self.semaphore = asyncio.Semaphore(20)

    async def _safe_generate(self, prompt: str) -> str:
        async with self.semaphore:
            return await self.agent.generate(prompt)

    def get_internal_state(self):
        return {"profiles": self.profiles}

    def set_internal_state(self, state):
        self.profiles = state.get("profiles", [])

    async def _generate_profiles(self, query: str) -> List[str]:
        print(f"      [MultiProfile] Generating {self.num_profiles} expert profiles...")
        prompt = f"Identify {self.num_profiles} distinct types of experts for '{query}'. 1 sentence per expert. Bullet points."
        res = await self._safe_generate(prompt)
        return [p.strip() for p in KnowledgeCleaner.clean_bullet_points(res)[:self.num_profiles]]

    async def _expert_extract(self, query: str, profile: str, current_points: List[str]) -> str:
        points_str = "\n".join([f"- {p}" for p in current_points[:30]])
        prompt = f"You are expert: {profile}. Expertise: '{query}'. Collected:\n{points_str}\nIdentify 5 new niche advanced knowledge points. Bullet points ONLY."
        return await self._safe_generate(prompt)

    async def get_next_step(self, query: str, history: List[str], current_points: List[str], turn: int) -> str:
        if not self.profiles:
            self.profiles = await self._generate_profiles(query)
            await self.record_snapshot("init", [], history)

        print(f"      [MultiProfile] Turn {turn}: {len(self.profiles)} experts working...")
        tasks = [self._expert_extract(query, p, current_points) for p in self.profiles]
        results = await asyncio.gather(*tasks)
        return "\n\n".join(results)
