from pipelines.base import BasePipeline
from typing import List
from core.cleaner import KnowledgeCleaner

class ReflectionPipeline(BasePipeline):
    """
    Asks the model to provide information, reflect on it, and refine.
    """
    async def run(self, query: str) -> List[str]:
        # Step 1: Draft
        prompt = f"List all atomic knowledge points about '{query}' in bullet points."
        draft = await self.agent.generate(prompt)
        draft_points = KnowledgeCleaner.clean_bullet_points(draft)
        
        # Step 2: Reflect
        draft_summary = '\n'.join(f'- {p}' for p in draft_points[:10])
        reflect_prompt = f"""
        Here is a list of knowledge points about '{query}':
        {draft_summary}
        
        Critique this list. What key concepts or specific technical details are missing or underdeveloped?
        Just list the missing points in bullet points.
        """
        refinement = await self.agent.generate(reflect_prompt)
        refinement_points = KnowledgeCleaner.clean_bullet_points(refinement)
        
        # Step 3: Combine (deduplication handled in cleaner)
        all_points = draft_points + refinement_points
        return list(dict.fromkeys(all_points))  # Final deduplication pass

