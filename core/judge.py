import asyncio
from agents.clientpool import safe_ask
from typing import List

class DomainJudge:
    def __init__(self, client_pool, model: str = "meta/llama-3.2-3b-instruct"):
        self.client_pool = client_pool
        self.model = model

    async def is_in_domain(self, query: str, knowledge_point: str) -> bool:
        prompt = f"""
        Domain: {query}
        Knowledge Point: {knowledge_point}
        
        Is the above knowledge point strictly relevant to the given domain? 
        Answer only 'YES' or 'NO'. 
        If it's partially relevant but primarily belongs to another field, answer 'NO'.
        """
        
        try:
            response = await safe_ask(
                self.client_pool,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            raw_answer = response.choices[0].message.content.strip().upper()
            
            # Robust parsing logic:
            has_yes = "YES" in raw_answer
            has_no = "NO" in raw_answer
            
            if has_yes and not has_no:
                return True
            return False
        except Exception:
            return False

    async def check_batch(self, query: str, points: List[str]) -> List[bool]:
        """
        Parallelize judge calls using asyncio.gather for high performance.
        """
        tasks = [self.is_in_domain(query, point) for point in points]
        return await asyncio.gather(*tasks)

