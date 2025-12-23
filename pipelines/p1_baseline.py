from pipelines.base import BasePipeline
from typing import List
from core.cleaner import KnowledgeCleaner

class BaselinePipeline(BasePipeline):
    async def run(self, query: str) -> List[str]:
        prompt = f"""
        List all the knowledge points you know about '{query}'.
        Provide your answer as a list of bullet points.
        Each point should be a single atomic fact.
        """
        
        response = await self.agent.generate(prompt)
        return KnowledgeCleaner.clean_bullet_points(response)

