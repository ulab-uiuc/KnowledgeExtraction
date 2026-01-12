import asyncio
from agents.clientpool import safe_ask
from typing import List
from tqdm import tqdm

class DomainJudge:
    def __init__(self, client_pool, model: str = "meta/llama-3.1-8b-instruct"):
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
        Includes a progress bar.
        """
        pbar = tqdm(total=len(points), desc="Auditing")
        
        async def tracked_is_in_domain(q, p):
            res = await self.is_in_domain(q, p)
            pbar.update(1)
            return res

        tasks = [tracked_is_in_domain(query, point) for point in points]
        results = await asyncio.gather(*tasks)
        pbar.close()
        return results

