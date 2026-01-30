# Developer Guide: Adding New Knowledge Extraction Pipelines

This framework is designed to be easily extensible. To add a new knowledge extraction strategy (Pipeline), you only need to create a new Python file in this directory and implement a single class.

## How it Works
The framework uses **Auto-discovery**. Any class that inherits from `BasePipeline` in this folder (excluding `base.py`) will be automatically picked up and evaluated by `main.py`.

The execution follows a **Saturation-based Logic**:
1. The framework calls your pipeline's `get_next_step` repeatedly.
2. It cleans the output into atomic knowledge points.
3. It calculates embeddings to check for "Novelty".
4. It automatically stops when your pipeline starts repeating itself or fails to produce new information (Saturation).

---

## Steps to Add a Pipeline

### 1. Create a new file
Create a file like `pipelines/p4_my_awesome_strategy.py`.

### 2. Implement the Class
Your class must inherit from `BasePipeline` and implement the `get_next_step` method.

```python
from .base import BasePipeline
from typing import List

class MyAwesomePipeline(BasePipeline):
    """
    Brief description of your strategy.
    """
    async def get_next_step(self, query: str, history: List[str], current_points: List[str], turn: int) -> str:
        # Implementation logic here
        pass
```

### 3. Parameters Explained
- `query` (str): The domain name (e.g., "Linear Algebra").
- `history` (List[str]): The full conversation history of the current run (useful for maintaining context).
- `current_points` (List[str]): All unique, cleaned knowledge points extracted by **this** pipeline so far.
- `turn` (int): The current iteration number (starts at 1).

---

## Implementation Examples

### Simple Sequential Prompting
If you just want to ask "What else?" in a loop:
```python
async def get_next_step(self, query: str, history: List[str], current_points: List[str], turn: int) -> str:
    if turn == 1:
        prompt = f"List everything you know about {query}."
    else:
        prompt = "Provide more points not mentioned above."
    
    # Simple call to the agent
    return await self.agent.generate(prompt)
```

### Complex Agentic Workflow
You can implement complex logic within `get_next_step`. Since it is an `async` function, you can perform multiple internal LLM calls or orchestrate sub-agents.

```python
async def get_next_step(self, query: str, history: List[str], current_points: List[str], turn: int) -> str:
    # Step 1: Brainstorm sub-topics
    plan = await self.agent.generate(f"Break down {query} into 5 sub-topics.")
    
    # Step 2: Deep dive into one sub-topic
    # (In a real case, you might use 'turn' to rotate through topics)
    response = await self.agent.generate(f"Detail everything about topic X in {query}...")
    
    # Just return the final string containing bullet points
    return response 
```

## Requirements
- Your class **must** return a string containing bullet points (e.g., lines starting with `-`, `*`, or numbers).
- Do **not** worry about deduplication or domain filtering; the core framework handles this globally after your pipeline finishes its "saturation run".

