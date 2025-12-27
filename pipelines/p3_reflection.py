from .base import BasePipeline
from typing import List

class ReflectionPipeline(BasePipeline):
    """
    Asks the model to reflect on its own output and find missing points.
    """
    async def get_next_step(self, query: str, history: List[str], current_points: List[str], turn: int) -> str:
        if turn == 1:
            prompt = f"List all atomic knowledge points about '{query}' in bullet points."
        else:
            # Provide ALL current points for reflection
            points_str = "\n".join([f"- {p}" for p in current_points]) 
            prompt = f"""Based on the points already listed below, identify and list ANY additional missing knowledge points related to '{query}'. 
            Focus on obscure details, advanced theories, or specific examples not yet covered.
            
            Existing points:
            {points_str}
            """
        
        response = await self.agent.generate(prompt)
        return response
