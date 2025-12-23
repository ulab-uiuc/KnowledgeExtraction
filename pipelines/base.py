from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

class BasePipeline(ABC):
    def __init__(self, agent, model: str = "meta/llama-3.2-3b-instruct"):
        self.agent = agent
        self.model = model

    @abstractmethod
    async def run(self, query: str) -> List[str]:
        """
        Run the knowledge extraction strategy.
        Should return a list of bullet points (strings).
        """
        pass

