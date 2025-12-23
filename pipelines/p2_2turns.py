from pipelines.base import BasePipeline
from typing import List
from core.cleaner import KnowledgeCleaner

class MultiTurn2Pipeline(BasePipeline):
    """
    Asks the model 2 times (initial + 1 follow-up) to provide information.
    """
    async def run(self, query: str) -> List[str]:
        all_points = []
        
        # Round 1
        prompt = f"List all atomic knowledge points about '{query}' in bullet points."
        response = await self.agent.generate(prompt)
        all_points.extend(KnowledgeCleaner.clean_bullet_points(response))
        
        # Round 2
        follow_up = "What else? Please provide more specific and in-depth points that were not mentioned above."
        combined_prompt = f"User: {prompt}\nAssistant: {response}\nUser: {follow_up}"
        response2 = await self.agent.generate(combined_prompt)
        all_points.extend(KnowledgeCleaner.clean_bullet_points(response2))
        
        return list(dict.fromkeys(all_points))

