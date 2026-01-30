from .base import BasePipeline
from typing import List

class SequentialPipeline(BasePipeline):
    """
    Continually asks "What else?" until the model repeats itself.
    """
    async def get_next_step(self, query: str, history: List[str], current_points: List[str], turn: int) -> str:
        if turn == 1:
            full_prompt = f"List all atomic knowledge points about '{query}' in bullet points."
        else:
            prompt = "What else? Please provide more specific and in-depth points that were not mentioned above."
            # Combine full history into context to act as a baseline
            context = "\n".join(history)
            full_prompt = f"{context}\nUser: {prompt}"
        
        response = await self.agent.generate(full_prompt)
        return response

