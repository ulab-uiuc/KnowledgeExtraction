from pipelines.base import BasePipeline
from typing import List
import re

class BaselinePipeline(BasePipeline):
    async def run(self, query: str) -> List[str]:
        prompt = f"""
        List all the knowledge points you know about '{query}'.
        Provide your answer as a list of bullet points.
        Each point should be a single atomic fact.
        """
        
        response = await self.agent.generate(prompt)
        # Simple parsing logic for bullet points
        points = re.findall(r'^\s*[\-\*\u2022]\s*(.*)', response, re.MULTILINE)
        if not points:
            # Fallback if regex fails: split by lines and filter
            points = [line.strip("-* ") for line in response.split('\n') if line.strip()]
        
        return [p.strip() for p in points if p.strip()]

