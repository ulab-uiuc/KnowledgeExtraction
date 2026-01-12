import asyncio
import json
import os
import importlib
import inspect
from agents.clientpool import MultiKeyClientPool
from agents.call_agent import GenAgent
from core.processor import KnowledgeProcessor
from core.judge import DomainJudge
from pipelines.base import BasePipeline

# --- 实验配置 ---
CATEGORIES = {
    "Deep Learning": [
        "Transformer Architectures",
        "Generative Models",
        "Deep Learning Theory"
    ],
    "Reinforcement Learning": [
        "Markov Decision Processes",
        "Policy Gradient Methods",
        "Hierarchical Reinforcement Learning"
    ],
    "Trustworthy ML": [
        "Algorithmic Fairness",
        "Adversarial Robustness",
        "Model Interpretability"
    ]
}

LLAMA_MODELS = [
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-405b-instruct"
]

PIPELINES_TO_RUN = [
    "p2_sequential",
    "p3_reflection",
    "p4_taxonomy_explorer",
    "p5_debate"
]

JUDGE_MODEL = "meta/llama-3.1-8b-instruct"

async def run_single_experiment(category_name, sub_category, model_name, client_pool, api_keys):
    """运行单个 (Area + Model) 下的所有 Pipelines"""
    query_id = sub_category.lower().replace(" ", "_")
    cat_id = category_name.lower().replace(" ", "_")
    output_dir = os.path.join("results", cat_id, query_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n>>> Running: [{category_name}] {sub_category} | Model: {model_name}")

    # 1. 初始化通用组件
    # 注意：这里的 GenAgent 需要传入 model_name
    gen_agent = GenAgent(api_key=api_keys, model=model_name)
    
    # Embedding 暂时使用默认
    processor = KnowledgeProcessor(
        client_pool=client_pool,
        embed_client_pool=client_pool,
        embed_model="nvidia/nv-embed-v1"
    )
    
    judge = DomainJudge(client_pool=client_pool, model=JUDGE_MODEL)

    # 2. 加载选定的 Pipelines
    active_pipelines = []
    for p_file in PIPELINES_TO_RUN:
        module_name = f"pipelines.{p_file}"
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BasePipeline) and obj != BasePipeline:
                    # 传入指定的 model 供 pipeline 内部使用
                    active_pipelines.append((name, obj(gen_agent, processor, model=model_name)))
        except Exception as e:
            print(f"  Error loading pipeline {p_file}: {e}")

    # 3. 运行每个 Pipeline 并保存原始结果
    pipeline_raw_outputs = {}
    model_slug = model_name.split("/")[-1] # e.g., llama-3.1-8b-instruct

    for name, pipeline in active_pipelines:
        raw_file = os.path.join(output_dir, f"{model_slug}_{name}_raw.json")
        
        # 如果文件已存在则跳过（方便断点续传）
        if os.path.exists(raw_file):
            print(f"  Pipeline {name} already exists. Skipping.")
            with open(raw_file, "r") as f:
                pipeline_raw_outputs[name] = json.load(f)
            continue

        print(f"  Running pipeline: {name}...")
        try:
            raw_points = await pipeline.run(sub_category)
            pipeline_raw_outputs[name] = raw_points
            
            with open(raw_file, "w") as f:
                json.dump(raw_points, f, indent=2)
            print(f"    Finished. Points: {len(raw_points)}")
        except Exception as e:
            print(f"    Error in {name}: {e}")

    # 4. 构建当前模型在当前 Area 下的 Union Set 并进行审计 (Post-Audit)
    # 这一步可以选择在所有运行结束后批量处理，或者现在处理。
    # 为了能实时看到进度，我们在这里处理当前模型的局部并集。
    
    # 注意：build_union_set 目前逻辑是从文件夹读取，我们需要稍微调整或手动调用
    # 这里我们只做生成，后续审计可以单独脚本跑，防止 API 压力过大
    print(f"  Extraction for {model_slug} on {sub_category} done.")

async def main():
    # 加载 API Key
    if not os.path.exists("api.json"):
        print("Error: api.json not found.")
        return
        
    with open("api.json") as f:
        api_data = json.load(f)
    api_keys = api_data["api_keys"]
    client_pool = MultiKeyClientPool(api_keys=api_keys)

    # 遍历所有大类、小类、模型
    for cat_name, sub_cats in CATEGORIES.items():
        for sub_cat in sub_cats:
            for model_name in LLAMA_MODELS:
                await run_single_experiment(cat_name, sub_cat, model_name, client_pool, api_keys)

if __name__ == "__main__":
    asyncio.run(main())
