from .base import BasePipeline
from typing import List

class DebatePipeline(BasePipeline):
    """
    Optimized Multi-Agent Debate.
    Theorist + Practitioner propose, Critic challenges, and experts refine.
    Inspired by 'Multi-Agent Debate' (Du et al., 2023).
    """
    async def get_next_step(self, query: str, history: List[str], current_points: List[str], turn: int) -> str:
        unique_points = list(dict.fromkeys(current_points))
        points_str = "\n".join([f"- {p}" for p in unique_points])
        
        # 1. Theorist & Practitioner Initial Proposals
        proposal_prompt = f"""You are a panel of two world-class experts in '{query}':
- Expert A (Theoretical Academic Expert): Focuses on formal definitions, axioms, proofs, and fundamental principles.
- Expert B (Applied Specialist/Engineer): Focuses on implementation nuances, real-world constraints, and edge cases.

We have already collected these knowledge points:
{points_str}

Each expert, please propose 5-8 unique, advanced points that are NOT mentioned above.
Provide your responses clearly labeled as Expert A and Expert B.
"""
        initial_res = await self.agent.generate(proposal_prompt)
        
        # 2. The Critic's Challenge
        critic_prompt = f"""You are a cynical and highly pedantic Peer Reviewer. 
You just saw these proposed points for '{query}' by two experts:
{initial_res}

Critique these points. Are they too superficial? What deep technical nuances, subtle mathematical properties, or 'dark corners' of the field did they fail to address? 
Challenge them to provide even more atomic and obscure details.
"""
        critic_challenge = await self.agent.generate(critic_prompt)
        
        # 3. Final Expert Synthesis
        synthesis_prompt = f"""Experts, the Critic has challenged you with the following feedback:
"{critic_challenge}"

Based on this challenge and your initial thoughts, provide a final, integrated list of highly advanced, atomic knowledge points. 
Focus on what is STILL missing from the global list:
{points_str}

Provide ONLY the new, missing bullet points.
"""
        final_response = await self.agent.generate(synthesis_prompt)
        
        # Return everything to ensure maximum extraction by the cleaner
        return f"{initial_res}\n\n{final_response}"
