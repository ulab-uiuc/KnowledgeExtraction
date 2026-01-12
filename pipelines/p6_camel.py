from .base import BasePipeline
from typing import List

class CamelExplorer(BasePipeline):
    """
    Optimized P6: Multi-Persona Perspective Expansion.
    Uses 'Ensemble of Personas' (Chan et al., 2023) to maximize knowledge coverage.
    Instead of a student-teacher dialogue, it uses 5 diverse expert perspectives.
    """
    async def get_next_step(self, query: str, history: List[str], current_points: List[str], turn: int) -> str:
        unique_points = list(dict.fromkeys(current_points))
        points_str = "\n".join([f"- {p}" for p in unique_points])
        
        # Step 1: Persona Sampling - Identify 5 extremely diverse expert roles
        persona_prompt = f"""You are a Strategic Knowledge Coordinator. 
Your goal is to fully extract all knowledge about '{query}'.
Identify 5 extremely diverse and distinct expert personas who would have unique perspectives on '{query}'.
For example, if the topic is 'AI', personas could include: a Hardware Architect, an Ethics Philosopher, a Kernel Developer, a Product Manager, and a Theoretical Physicist.

List only the 5 personas, each with a one-sentence description of their unique angle."""
        personas = await self.agent.generate(persona_prompt)
        
        # Step 2: Parallel Multi-Perspective Extraction
        extraction_prompt = f"""You are a panel of 5 diverse experts:
{personas}

Existing knowledge already collected:
{points_str}

Each expert, please identify and list 5-8 atomic knowledge points about '{query}' that are unique to your professional perspective and NOT mentioned above.
Focus on highly specific, niche, or interdisciplinary details that only your persona would know.
Provide your points in a clear, bulleted list."""
        
        initial_extractions = await self.agent.generate(extraction_prompt)
        
        # Step 3: Blindspot Coordination
        coordinator_prompt = f"""As the Coordinator, you have seen the existing knowledge and the new contributions from the 5 experts:
Experts' New Contributions:
{initial_extractions}

Your task: Identify 3 'Blindspots' that even these 5 experts missed. 
Then, provide a final list of all new atomic knowledge points extracted from this session.
DO NOT repeat anything from the existing list:
{points_str}

Provide ONLY the final new bullet points."""
        
        final_response = await self.agent.generate(coordinator_prompt)
        
        # Return initial + final to maximize the cleaner's yield
        return f"{initial_extractions}\n\n{final_response}"
