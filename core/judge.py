import asyncio
from agents.clientpool import safe_ask
from typing import List

class DomainJudge:
    def __init__(self, client_pool, model: str = "deepseek-ai/deepseek-v3.1", concurrency: int = 30):
        self.client_pool = client_pool
        self.model = model
        self.semaphore = asyncio.Semaphore(concurrency)

    async def is_in_domain(self, query: str, knowledge_point: str) -> bool:
        prompt = f"""
        Role: Senior Research Scientist in {query}.
        Task: Evaluate if the given "Knowledge Point" constitutes a substantive piece of technical knowledge according to the following scientific rubric.

        A valid "Knowledge Point" must fall into one of these three categories:
        1. **Factual Knowledge**: Precise definitions of terms, specific technical details, or discrete bits of information specific to {query}.
        2. **Conceptual Knowledge**: Principles, theories, models, or the interrelationships between basic elements.
        3. **Procedural Knowledge**: Algorithms, techniques, methods, specific implementation steps, or mathematical formulas.

        Criteria for 'YES' (Must meet ALL):
        - **Truth/Accuracy**: The statement must be factually correct and recognized in the field of {query}.
        - **Substantive Content**: It must provide actual information. Reject meta-statements like "Researchers study X".
        - **Technical Precision**: It should use domain-specific language correctly. 
        - **Completeness**: It must be a full, coherent sentence or proposition.

        Criteria for 'NO' (Reject these):
        - **Introductory Fluff**: Generic statements like "Data quality is important".
        - **Meta-Knowledge**: Statements about the field development (e.g., "The field of {query} has grown").
        - **Structural Fragments**: Headers, titles, or list pointers.

        Knowledge Point: "{knowledge_point}"

        Decision: Respond with ONLY 'YES' or 'NO'.
        """
        async with self.semaphore:
            try:
                response = await safe_ask(
                    self.client_pool,
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                raw_answer = response.choices[0].message.content.strip().upper()
                return "YES" in raw_answer and "NO" not in raw_answer
            except Exception as e:
                raise e

    async def check_batch(self, query: str, points: List[str]) -> List[bool]:
        tasks = [self.is_in_domain(query, p) for p in points]
        return await asyncio.gather(*tasks)
