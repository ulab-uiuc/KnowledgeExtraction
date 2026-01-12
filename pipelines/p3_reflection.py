from .base import BasePipeline
from typing import List

class ReflectionPipeline(BasePipeline):
    """
    Refines knowledge based on the 'Reflexion' framework (Shinn et al., 2023).
    Includes a self-criticism step before generating new points, with full history awareness.
    """
    async def get_next_step(self, query: str, history: List[str], current_points: List[str], turn: int) -> str:
        if turn == 1:
            return await self.agent.generate(f"List all atomic knowledge points about '{query}' in bullet points.")
        
        # Deduplicate to keep context clean and informative
        unique_points = list(dict.fromkeys(current_points))
        points_str = "\n".join([f"- {p}" for p in unique_points])
        
        prompt = f"""You are a senior subject matter expert in '{query}'.
We have already extracted the following {len(unique_points)} unique knowledge points:
{points_str}

### Step 1: Self-Criticism
Analyze the coverage above. What advanced theories, subtle edge cases, or fundamental principles of '{query}' are missing, or described too superficially? 

### Step 2: New Extraction
Based on your analysis, list ONLY the new, missing knowledge points in bullet points. 

### CONSTRAINTS
- DO NOT repeat any points or concepts already mentioned above.
- Focus on depth and obscurity.
- Use the standard bullet point format (- Point).
"""
        response = await self.agent.generate(prompt)
        return response
