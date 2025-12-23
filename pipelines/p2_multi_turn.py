from pipelines.base import BasePipeline
from typing import List
from core.cleaner import KnowledgeCleaner

class MultiTurnPipeline(BasePipeline):
    """
    Asks the model multiple times to provide more information.
    """
    async def run(self, query: str) -> List[str]:
        all_points = []
        
        # Initial request
        prompt = f"List all atomic knowledge points about '{query}' in bullet points."
        response = await self.agent.generate(prompt)
        all_points.extend(KnowledgeCleaner.clean_bullet_points(response))
        
        # Follow-up
        follow_up = "What else? Please provide more specific and in-depth points that were not mentioned above."
        # Note: Since our GenAgent is currently stateless, we'll construct a combined prompt
        # In a real multi-turn, we'd use a chat history. 
        # For this baseline, we'll simulate it by providing the previous context.
        combined_prompt = f"User: {prompt}\nAssistant: {response}\nUser: {follow_up}"
        response2 = await self.agent.generate(combined_prompt)
        all_points.extend(KnowledgeCleaner.clean_bullet_points(response2))
        
        # Deduplication is already handled in cleaner, but we do a final pass
        return list(dict.fromkeys(all_points))  # Preserve order while removing duplicates

