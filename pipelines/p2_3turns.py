from pipelines.base import BasePipeline
from typing import List
from core.cleaner import KnowledgeCleaner

class MultiTurn3Pipeline(BasePipeline):
    """
    Asks the model 3 times (initial + 2 follow-ups) to provide information.
    """
    async def run(self, query: str) -> List[str]:
        all_points = []
        history = []
        
        prompt = f"List all atomic knowledge points about '{query}' in bullet points."
        response = await self.agent.generate(prompt)
        all_points.extend(KnowledgeCleaner.clean_bullet_points(response))
        history.append(f"User: {prompt}\nAssistant: {response}")
        
        for _ in range(2):
            follow_up = "What else? Please provide more specific and in-depth points that were not mentioned above."
            context = "\n".join(history)
            combined_prompt = f"{context}\nUser: {follow_up}"
            response = await self.agent.generate(combined_prompt)
            all_points.extend(KnowledgeCleaner.clean_bullet_points(response))
            history.append(f"User: {follow_up}\nAssistant: {response}")
            
        return list(dict.fromkeys(all_points))

