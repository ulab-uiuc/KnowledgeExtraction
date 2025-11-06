from openai import OpenAI
import json
import random
from call_agent import GenAgent
from judge_agent import JudgeAgent
from q_transformer import QTransformerAgent
import asyncio
from typing import Dict, Tuple, List, Any, Optional
from treelib import Tree


import asyncio
from typing import List, Tuple, Optional, Dict, Union
from treelib import Tree

StrOrList = Union[str, List[str]]

def ensure_list_str(x: StrOrList) -> List[str]:
    """把可能的 str 或 [str] 或 [[str]] 规范化为一维 List[str]."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        out: List[str] = []
        for item in x:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, list):
                for y in item:
                    if isinstance(y, str):
                        out.append(y)
                    else:
                        out.append(str(y))
            else:
                out.append(str(item))
        return out
    return [str(x)]

class AgentScheduler:
    def __init__(
        self,
        gen_agent,               # async def generate(q: str) -> str
        judge_agent,
        q_transformer_agent,     # async def generate_batch(problems: List[str], n_sample: int) -> List[List[str]]
                                 # async def deepen_batch(questions: List[str], answers: List[str], n_sample: int) -> Union[List[str], List[List[str]]]
        answer_timeout: Optional[float] = None,
    ):
        self.gen_agent = gen_agent
        self.judge_agent = judge_agent
        self.q_transformer_agent = q_transformer_agent
        self.answer_timeout = answer_timeout
        self.memory_tree: Dict[str, Tree] = {}

    async def handle_problem(
        self,
        problem: str,
        rounds: int = 3,
        n_sample_first: int = 2,
        n_sample_next: int = 2,
        deepen_sample: int = 2,
        max_children_per_node: int = 3,
    ) -> Tree:
        # 初始化树
        if problem not in self.memory_tree:
            t = Tree()
            t.create_node(tag="Root", identifier="root", data={"question": problem})
            self.memory_tree[problem] = t
        tree = self.memory_tree[problem]

        # 第0轮. 从根问题生成多个子问
        sub_questions_nested: List[List[str]] = await self.q_transformer_agent.generate_batch(
            [problem], n_sample=n_sample_first
        )
        sub_questions = sub_questions_nested[0] if sub_questions_nested else []

        for j, q in enumerate(sub_questions):
            q_id = f"q_0_{j}"
            tree.create_node(tag=f"Q0-{j}", identifier=q_id, parent="root", data={"question": q})

        print(f"Initial sub-questions generated: {len(sub_questions)}")
        parent_question_ids = [f"q_0_{j}" for j in range(len(sub_questions))]
        iteration = 0

        while iteration < rounds:
            print(f"--- Iteration {iteration} ---")
            print(f"Total parent questions: {len(parent_question_ids)}")
            # 为当前轮问题并发生成答案
            async def gen_one(q: str) -> Tuple[str, str]:
                if self.answer_timeout:
                    ans = await asyncio.wait_for(self.gen_agent.generate(q), timeout=self.answer_timeout)
                else:
                    ans = await self.gen_agent.generate(q)
                return q, ans

            current_questions: List[str] = [tree.get_node(qid).data["question"] for qid in parent_question_ids]
            answers: List[Tuple[str, str]] = await asyncio.gather(*(gen_one(q) for q in current_questions))

            # 答案挂到对应问题下
            answer_ids: List[str] = []
            for idx, q_id in enumerate(parent_question_ids):
                _, ans_text = answers[idx]
                a_id = f"a_{iteration}_{idx}"
                tree.create_node(tag=f"A{iteration}-{idx}", identifier=a_id, parent=q_id, data={"answer": ans_text})
                answer_ids.append(a_id)

            iteration += 1
            if iteration >= rounds:
                break

            # —— 关键修改点：逐父节点地深化并展开，以保持对齐且保证传入 generate_batch 的是 str —— 
            new_parent_question_ids: List[str] = []
            # 对每个父“问题-答案”对，分别深化，再基于每个深化母问题生成下一层子问
            # 这样避免把二维 deepen 结果直接拼成一个 problems 传入 generate_batch
            for i, q_id in enumerate(parent_question_ids):
                parent_answer_id = answer_ids[i]
                q_text = tree.get_node(q_id).data["question"]
                a_text = tree.get_node(parent_answer_id).data["answer"]

                # 深化：可能返回 str 或 [str] 或 [[str]]
                deepened = await self.q_transformer_agent.deepen_batch([q_text], [a_text], n_sample=deepen_sample)
                # 规范化为一维 List[str]
                deepened_list = ensure_list_str(deepened)

                # 针对每个“深化母问题”单独调用 generate_batch，这样 generate_batch 的输入一定是 List[str]
                for k, dq in enumerate(deepened_list):
                    batch_out: List[List[str]] = await self.q_transformer_agent.generate_batch([dq], n_sample=n_sample_next)
                    child_questions = batch_out[0] if batch_out else []
                    # 限制宽度
                    if max_children_per_node is not None and len(child_questions) > max_children_per_node:
                        child_questions = child_questions[:max_children_per_node]

                    for j, cq in enumerate(child_questions):
                        q_child_id = f"q_{iteration}_{i}_{k}_{j}"
                        tree.create_node(
                            tag=f"Q{iteration}-{i}-{k}-{j}",
                            identifier=q_child_id,
                            parent=parent_answer_id,   # 子问挂在该父答案节点下
                            data={"question": cq},
                        )
                        new_parent_question_ids.append(q_child_id)

            if max_children_per_node is not None:
                parent_question_ids = new_parent_question_ids[:max_children_per_node]
            else:
                parent_question_ids = new_parent_question_ids

            print(f"New parent questions for next round: {len(parent_question_ids)}")

        return tree



def main():
    with open("api.json") as f:
        api_data = json.load(f)

    gen_agent = GenAgent(api_key=api_data["api_keys"], max_tokens=4096)
    judge_agent = JudgeAgent(api_key=api_data["api_keys"], max_tokens=4096)
    q_transformer_agent = QTransformerAgent(api_key=api_data["api_keys"], max_tokens=4096)

    # 假设你已有 gen_agent, judge_agent, q_transformer_agent
    scheduler = AgentScheduler(gen_agent, judge_agent, q_transformer_agent, answer_timeout=120.0)
    tree = asyncio.run(scheduler.handle_problem("How to convert complex requirements into executable subtasks", rounds=3))
    tree.show()  # 控制台打印树形结构

if __name__ == "__main__":
    main()
    # Example usage of AgentScheduler
