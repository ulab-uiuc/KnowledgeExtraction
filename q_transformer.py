from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, List, Dict, Any, Optional, Iterable, Tuple
import json
import time
import re
from openai import OpenAI
import random
import hashlib
import re
import html
from openai import AsyncOpenAI
import asyncio
import numpy as np
from clientpool import MultiKeyClientPool, safe_ask, safe_embed

REWRITE_PROMPT = """
You are a Question/Instruction Consolidation Agent.

We provide:
1) A list of the original questions/instructions.
2) A list of the expanded (rewritten) versions of those same questions/instructions.

Your task:
- Only operate on the expanded questions/instructions.
- Identify expansions that express the same or similar meaning.
- Merge and deduplicate them into a concise set of clearer, more generalizable forms.
- When merging, you MUST preserve the original speaker's person perspective.

Rules:
1. Do not introduce new content not implied by the original intent.
2. If multiple expanded questions share the same intent, merge them into ONE question/instruction.
3. Output only the consolidated set, formatted as a sequence of <Q></Q> blocks.

Here are the original questions (for reference only, do NOT rewrite these):
{ORIGINAL_QUESTIONS}

Here are the expanded questions (these are what you will merge/deduplicate):
{INPUT_QUESTIONS}

Now output the final consolidated questions below as <Q></Q> blocks:
"""


SYSTEM_PROMPT = """
You are a Question Rewriting Agent.

Your task:
- Rewrite the user's question to make it clearer, more precise, and more structured.
- You may break the question into multiple sub-questions *if it helps clarity*.
- **Keep the same speaker perspective, intention, and tone.** 
- **Do NOT answer the question.** 
- **Do NOT add new assumptions.**

Output format:
- Put each rewritten question inside <Q>...</Q>
- The output should still be in the form of questions, not explanations.

Here is the user's original question:
{USER_QUESTION}
Now rewrite the question(s) as <Q></Q> blocks:
"""

DEEPEN_PROMPT = """
You are a Follow-Up Deepening Agent.

Your task:
- Given the user's original question and the assistant's answer, generate a structured set of deeper follow-up questions the user might naturally ask next.
- The follow-up questions must be self-contained: each question should explicitly include the necessary context from the original question and answer so that it can be understood and answered independently, without referring back to the conversation.
- The follow-up questions should aim to clarify, refine, or push the reasoning further along dimensions such as:
  1) Purpose / Goals.
  2) Constraints / Trade-offs.
  3) Assumptions / Missing Context.
  4) Evaluation / Comparison.
  5) Next Steps / Actionable Details.

Rules:
- Do NOT answer the follow-up questions.
- Write each follow-up question in the tone of the original user, as if the user is directly asking the assistant.
- Do NOT switch to assistant voice or meta-commentary.
- Do NOT introduce new, unrelated topics; only deepen questions based on what is implied in the original question and answer.
- Each question should be concise but fully contextually complete.

Output Format:
- Each follow-up question wrapped in <Q></Q>.

Original Question:
{USER_QUESTION}

Assistant's Answer:
{ASSISTANT_ANSWER}

Now generate the self-contained, deeper follow-up questions:
"""


_Q_PATTERN = re.compile(r"<\s*Q\s*>(.*?)<\s*/\s*Q\s*>", re.IGNORECASE | re.DOTALL)


# ---- 工具函数：把任意嵌套结构展平成 List[str]，并清洗 ----
from typing import Iterable, List

def _flatten_to_str_list(items: Iterable) -> List[str]:
    out = []
    stack = [items]
    while stack:
        x = stack.pop()
        if x is None:
            continue
        if isinstance(x, (list, tuple)):
            stack.extend(x)
            continue
        if isinstance(x, dict) and "question" in x:
            x = x["question"]
        x = str(x).strip()
        if x:
            out.append(x)
    # 保序去重，可选
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


class QTransformerAgent:
    def __init__(
        self,
        api_key: List[str],
        # model: str = "meta/llama-3.2-3b-instruct",
        # model: str = "meta/llama-3.1-8b-instruct",
        model: str = "qwen/qwen2.5-7b-instruct",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 4096,
    ):
        self.client = MultiKeyClientPool(
            api_keys=api_key,
            base_url=base_url
        )
        self.api_key = api_key
        self.system_prompt = SYSTEM_PROMPT
        self.rewrite_prompt = REWRITE_PROMPT
        self.deepen_prompt = DEEPEN_PROMPT
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # embedding dedupe threshold
        self.threshold = 0.90

    # ---- 强化 embed：再做一次展平与分批 ----
    async def embed(self, texts) -> list[list[float]]:
        texts = _flatten_to_str_list(texts)
        if not texts:
            return []
        assert all(isinstance(t, str) for t in texts), f"embed expects List[str], got {[type(t) for t in texts[:3]]}"

        BATCH = 256
        all_vecs = []
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i+BATCH]
            resp = await safe_embed(
                self.client,
                model="nvidia/llama-3.2-nv-embedqa-1b-v2",
                messages=chunk,
            )
            all_vecs.extend([d.embedding for d in resp.data])
        return all_vecs

    def cosine_sim(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # ---- 修正 dedupe_questions：先展平，再与 embeddings 对齐 ----
    async def dedupe_questions(self, questions, threshold=0.93):
        # 展平成一维 List[str] 并清洗
        questions = _flatten_to_str_list(questions)
        if not questions:
            return []

        embeddings = await self.embed(questions)   # 这里确保 embed 也只吃 List[str]

        kept = []
        kept_embs = []
        for q, e in zip(questions, embeddings):
            is_duplicate = False
            for ke_emb in kept_embs:
                if self.cosine_sim(e, ke_emb) >= threshold:
                    # 打印时可以简短些，避免超长日志
                    # print(f"去重: {q[:30]}... 相似度 {self.cosine_sim(e, ke_emb):.4f} >= {threshold}")
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(q)
                kept_embs.append(e)

        return kept

    def parse_q_blocks(self, text: str, dedupe: bool = True):
        chunks = [html.unescape(m.strip()) for m in _Q_PATTERN.findall(text)]

        if not dedupe:
            return chunks

        # 去重（保持出现顺序）
        seen = set()
        out = []
        for c in chunks:
            key = " ".join(c.lower().split())
            if key not in seen:
                seen.add(key)
                out.append(c)
        return out

    def format_as_q_blocks(self, questions):
        """
        将解析后的内容重新生成为多对<Q></Q>格式
        """
        return "\n".join(f"<Q>{q}</Q>" for q in questions)


    # ---- 修正 deepen：类型注解 + 注入 answer 到提示 + 展平 q_blocks ----
    async def deepen(self, problem, answer, n_sample=4) -> list[str]:
        # 把 answer 放进 messages，有助于生成更相关的 follow-ups
        sys_content = self.deepen_prompt.replace("{USER_QUESTION}", problem)
        # 如果你的 prompt 还支持 {ASSISTANT_ANSWER}，一起替换
        sys_content = sys_content.replace("{ASSISTANT_ANSWER}", str(answer))

        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": f"Based on the above background, please list the executable in-depth follow-up questions."}
        ]

        Tasks = [safe_ask(self.client, self.model, messages, self.temperature, self.top_p, self.max_tokens) for _ in range(n_sample)]
        completions = await asyncio.gather(*Tasks)

        all_questions = []
        for completion in completions:
            for choice in completion.choices:
                text = choice.message.content or ""
                q_blocks = self.parse_q_blocks(text, dedupe=True)  # 可能返回 List[str] 或嵌套
                all_questions.append(q_blocks)

        # 展平一次，避免把 List[List[str]] 传给 dedupe/embeddings
        all_questions = _flatten_to_str_list(all_questions)

        deduped = await self.dedupe_questions(all_questions, threshold=self.threshold)
        return deduped



    async def generate(self, problem, n_sample=4) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt.replace("{USER_QUESTION}", problem)},
        ]

        Tasks = [safe_ask(self.client, self.model, messages, self.temperature, self.top_p, self.max_tokens) for _ in range(n_sample)]
        completions = await asyncio.gather(*Tasks)
        all_questions = []
        for completion in completions:
            for choice in completion.choices:
                text = choice.message.content
                q_blocks = self.parse_q_blocks(text, dedupe=True)
                if q_blocks:
                    all_questions.extend(q_blocks)

        deduped = await self.dedupe_questions(all_questions, threshold=self.threshold)
        return deduped

    async def generate_batch(
            self,
            problems,
            n_sample: int = 4,
            concurrency: int = 5,
        ):
            sem = asyncio.Semaphore(concurrency)

            async def _gen_one(q):
                async with sem:
                    return await self.generate(q, n_sample=n_sample)

            async def _row(row):
                return await asyncio.gather(*(_gen_one(q) for q in row))

            # 判断维度
            if isinstance(problems[0], list):
                return await asyncio.gather(*(_row(row) for row in problems))
            else:
                return await asyncio.gather(*(_gen_one(q) for q in problems))

    # -------- 批量并发（可选） --------
    async def deepen_batch(
        self,
        problems: List[str],
        answers: List[str],
        n_sample: int = 4,
        concurrency: int = 5,
    ) -> List[List[str]]:
        """
        并发跑多条问题；用 semaphore 控制并发度，避免打爆 QPS。
        """
        sem = asyncio.Semaphore(concurrency)
        # print(f"n_sample={n_sample}")
        async def _one(p: str, a: str):
            async with sem:
                return await self.deepen(p, a, n_sample=n_sample)

        return await asyncio.gather(*[_one(p, a) for p, a in zip(problems, answers)])


async def main():
    with open("api.json") as f:
        api_data = json.load(f)
    api_key = api_data["api_keys"]
    agent = QTransformerAgent(api_key=api_key)
    result = await agent.generate("estimate the future population of the world in 2050")
    # print(result)

    batch = await agent.generate_batch(
        ["write a poem about AI advancements in 2024", "please write a poem about AI advancements in 2024"], concurrency=3, n_sample=4
    )

    
    # print(batch)

    # batch = await agent.deepen_batch(
    #     [
    #         "What are the latest advancements in artificial intelligence in 2024?",
    #         "Explain the theory of relativity."
    #     ],
    #     [
    #         "In 2024, significant advancements in artificial intelligence include the development of more sophisticated natural language processing models, improvements in computer vision technologies, and the integration of AI in various industries such as healthcare, finance, and transportation. Additionally, there has been a focus on ethical AI and ensuring transparency in AI decision-making processes.",
    #         "The theory of relativity, developed by Albert Einstein, consists of two main parts: special relativity and general relativity. Special relativity deals with the physics of objects moving at constant speeds, particularly those approaching the speed of light, and introduces concepts such as time dilation and length contraction. General relativity extends these ideas to include gravity, describing it as the curvature of spacetime caused by mass and energy."
    #     ],
    #     concurrency=2,
    #     n_sample=4
    # )
    print(batch)
    print(len(batch[0]))

if __name__ == "__main__":
    asyncio.run(main())
