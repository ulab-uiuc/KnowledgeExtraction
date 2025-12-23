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
        """
        
        response = await safe_ask(
            self.client_pool,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        answer = response.choices[0].message.content.strip().upper()
        return "YES" in answer

    async def check_batch(self, query: str, points: List[str]) -> List[bool]:
        results = []
        # Simple sequential check, can be improved with gather if needed
        for point in points:
            results.append(await self.is_in_domain(query, point))
        return results

