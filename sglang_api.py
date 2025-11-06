"""
Conversation Tree Scheduler with sglang
--------------------------------------
A production‑ready skeleton that builds a conversation tree:
1) Transform a root problem into sub‑questions
2) Answer sub‑questions with an LLM
3) Judge answers and select the best
4) Optionally call external APIs as tools
5) Aggregate a final answer and persist a memory tree to disk

Features
- sglang pipelines for each agent role
- Async concurrency over sub‑questions
- Simple tool‑calling example via Python functions
- Streaming hooks for live tokens
- JSON save and load for the tree

Note
- Replace `MODEL_NAME` and `OPENAI_API_KEY` usage according to your backend.
- sglang currently supports multiple backends including OpenAI/VLLM. Adjust the runtime init accordingly.
"""
from __future__ import annotations
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# ---------- sglang imports ----------
# You may need: pip install sglang
from sglang import function, R, S  # Core DSL
from sglang import Runtime  # Runtime to execute functions

# ---------- Configuration ----------
MODEL_NAME = os.environ.get("SGLANG_MODEL", "gpt-4o-mini")
MAX_SUBQS = 6
N_SAMPLES_PER_SUBQ = 2
CONCURRENCY = 8
TEMPERATURE_GEN = 0.7
TEMPERATURE_Q = 0.3
TEMPERATURE_JUDGE = 0.0
TREE_PATH = os.environ.get("TREE_PATH", "conversation_tree.json")

# ---------- Example external tools ----------
# In real usage, replace these with your own business logic. These run in Python and can be called from pipelines.
import random
import requests

def call_weather_api(city: str) -> str:
    """Example tool. Replace with your API.
    This function must be fast and resilient.
    """
    try:
        # Dummy demo fallback
        # You can switch to a real API call, for example:
        # resp = requests.get(f"https://wttr.in/{city}?format=3", timeout=4)
        # if resp.ok: return resp.text
        sample = ["晴", "多云", "小雨", "雷阵雨", "阴"]
        return random.choice(sample)
    except Exception as e:
        return f"<tool_error:{e}>"

# ---------- sglang pipelines ----------
# 1) Sub-question generator
@function
def q_transformer(s: S, problem: str, max_subqs: int = MAX_SUBQS):
    s += (
        "你是一个任务分解专家。给定一个复杂问题。请输出不超过{max_subqs}个相互独立、可并行求解的子问题。"
        "要求：每个子问题具体、带可执行目标、避免重复。只输出列表。"
    )
    s += f"复杂问题：{problem}\n子问题列表：\n"
    subq_text = s.collect(R.UntilStop(), name="subq_text")  # free-form capture
    return subq_text

# 2) Sub-question answerer
@function
def answer_agent(s: S, subq: str):
    s += (
        "你是精准回答助手。请针对下面的子问题给出直接、可验证、分步的答案。"
        "如需要数据或接口。请清楚指明来源或用工具结果。"
    )
    s += f"子问题：{subq}\n回答：\n"
    ans = s.collect(R.UntilStop(), name="ans")
    return ans

# 3) Judge pipeline
@function
def judge_agent(s: S, subq: str, candidate_a: str, candidate_b: str):
    s += (
        "你是一个评审。根据‘准确性、完整性、可执行性、引用或工具使用的可信度’标准打分并给出胜者。"
        "输出JSON：{winner: 'A'或'B', reason: '...'}"
    )
    s += f"子问题：{subq}\n候选A：{candidate_a}\n候选B：{candidate_b}\nJSON："
    out = s.collect(R.Json(), name="judgement")
    return out

# 4) Aggregator pipeline
@function
def aggregator(s: S, problem: str, qa_pairs: List[Tuple[str, str]]):
    s += (
        "你是总结器。请基于已验证的子问答对。给出对原问题的最终解答。"
        "需要：结构化分段。列出关键结论与可执行Next Steps。"
    )
    s += f"原问题：{problem}\n"
    s += "子问答对：\n"
    for q, a in qa_pairs:
        s += f"- Q: {q}\n- A: {a}\n"
    s += "\n最终回答：\n"
    final = s.collect(R.UntilStop(), name="final_answer")
    return final

# ---------- Helper: parse list bullets back into python list ----------
def parse_list(text: str, max_items: int) -> List[str]:
    items = []
    for line in text.splitlines():
        line = line.strip(" -*•\t")
        if not line:
            continue
        # stop if line looks like header
        if len(items) >= max_items:
            break
        items.append(line)
    # dedupe while keeping order
    seen = set()
    uniq = []
    for x in items:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq[:max_items]

# ---------- Memory Tree ----------
@dataclass
class TreeNode:
    question: str
    answers: List[str] = field(default_factory=list)
    best_answer: str | None = None
    children: List["TreeNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answers": self.answers,
            "best_answer": self.best_answer,
            "children": [c.to_dict() for c in self.children],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TreeNode":
        node = TreeNode(
            question=d["question"],
            answers=d.get("answers", []),
            best_answer=d.get("best_answer"),
            children=[TreeNode.from_dict(x) for x in d.get("children", [])],
        )
        return node

class ConversationTree:
    def __init__(self):
        self.roots: Dict[str, TreeNode] = {}

    def get_or_create_root(self, problem: str) -> TreeNode:
        if problem not in self.roots:
            self.roots[problem] = TreeNode(question=problem)
        return self.roots[problem]

    def save(self, path: str = TREE_PATH):
        data = {k: v.to_dict() for k, v in self.roots.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str = TREE_PATH) -> "ConversationTree":
        ct = ConversationTree()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                ct.roots[k] = TreeNode.from_dict(v)
        return ct

# ---------- Scheduler ----------
class AgentScheduler:
    def __init__(self, runtime: Runtime):
        self.rt = runtime
        self.tree = ConversationTree()

    async def transform(self, problem: str) -> List[str]:
        out = await self.rt.run(q_transformer, problem=problem, max_subqs=MAX_SUBQS, temperature=TEMPERATURE_Q)
        subq_text = out["subq_text"]
        subqs = parse_list(subq_text, MAX_SUBQS)
        return subqs

    async def answer_one(self, subq: str, n_samples: int = N_SAMPLES_PER_SUBQ) -> List[str]:
        sem = asyncio.Semaphore(CONCURRENCY)
        async def _once() -> str:
            async with sem:
                # Optional example: if subq mentions weather, call tool first and append
                tool_note = ""
                if "天气" in subq or "weather" in subq.lower():
                    city = "北京"
                    w = call_weather_api(city)
                    tool_note = f"\n[工具] {city} 天气: {w}\n"
                res = await self.rt.run(answer_agent, subq=subq + tool_note, temperature=TEMPERATURE_GEN)
                return res["ans"].strip()
        return await asyncio.gather(*[_once() for _ in range(n_samples)])

    async def judge_two(self, subq: str, a: str, b: str) -> Tuple[str, Dict[str, Any]]:
        out = await self.rt.run(judge_agent, subq=subq, candidate_a=a, candidate_b=b, temperature=TEMPERATURE_JUDGE)
        judgement = out["judgement"]
        winner = judgement.get("winner", "A").upper()
        return (a if winner == "A" else b), judgement

    async def aggregate(self, problem: str, qa_pairs: List[Tuple[str, str]]) -> str:
        out = await self.rt.run(aggregator, problem=problem, qa_pairs=qa_pairs, temperature=0.2)
        return out["final_answer"].strip()

    async def handle_problem(self, problem: str, rounds: int = 1) -> Dict[str, Any]:
        root = self.tree.get_or_create_root(problem)

        # 1) transform into sub-questions
        subqs = await self.transform(problem)
        if not subqs:
            return {"error": "No sub-questions generated"}

        # 2) answer each sub-question with sampling and judging
        qa_pairs: List[Tuple[str, str]] = []
        for subq in subqs:
            candidates = await self.answer_one(subq)
            best = candidates[0]
            if len(candidates) >= 2:
                best, judge_json = await self.judge_two(subq, candidates[0], candidates[1])
            node = TreeNode(question=subq, answers=candidates, best_answer=best)
            root.children.append(node)
            qa_pairs.append((subq, best))

        # 3) aggregate final answer
        final_answer = await self.aggregate(problem, qa_pairs)
        root.best_answer = final_answer

        # Persist
        self.tree.save(TREE_PATH)

        return {
            "problem": problem,
            "sub_questions": subqs,
            "qa_pairs": qa_pairs,
            "final_answer": final_answer,
            "saved_to": TREE_PATH,
        }

# ---------- Streaming demo hooks ----------
# For true token streaming. you can supply callbacks in your Runtime backend configuration.
# Here we show a minimal wrapper that prints elapsed times.
class Timer:
    def __init__(self):
        self.t0 = time.time()
    def tick(self, label: str = ""):  # no-op for now. replace with token callbacks when backend supports
        dt = time.time() - self.t0
        print(f"[{dt:5.2f}s] {label}")

# ---------- Entrypoint ----------
async def main():
    # Initialize sglang runtime
    # For OpenAI compatible backends. set env OPENAI_API_KEY and pick a model.
    rt = Runtime(model=MODEL_NAME)
    sched = AgentScheduler(rt)

    problem = "如何把复杂需求分解为可执行子任务。并给出可验证的交付标准与里程碑"

    t = Timer()
    result = await sched.handle_problem(problem, rounds=1)
    t.tick("done")

    print("\n=== RESULT ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Tree saved to: {result['saved_to']}")

if __name__ == "__main__":
    asyncio.run(main())
