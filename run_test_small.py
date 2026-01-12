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

# --- 测试用配置 (小规模) ---
CATEGORIES = {
    "Test Category": [
        "Transformer Architectures"
    ]
}

LLAMA_MODELS = [
    "meta/llama-3.1-8b-instruct"
]

PIPELINES_TO_RUN = [
    "p2_sequential",
    "p3_reflection"
]

JUDGE_MODEL = "meta/llama-3.1-8b-instruct"

async def run_single_experiment(category_name, sub_category, model_name, client_pool, api_keys):
    """运行单个 (Area + Model) 下的所有 Pipelines"""
    query_id = sub_category.lower().replace(" ", "_")
    cat_id = category_name.lower().replace(" ", "_")
    output_dir = os.path.join("results", cat_id, query_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n>>> [TEST] Running: [{category_name}] {sub_category} | Model: {model_name}")

    # 1. 初始化通用组件
    gen_agent = GenAgent(api_key=api_keys, model=model_name)
    
    processor = KnowledgeProcessor(
        client_pool=client_pool,
        embed_client_pool=client_pool,
        embed_model="nvidia/nv-embed-v1"
    )
    
    # 2. 加载选定的 Pipelines
    active_pipelines = []
    for p_file in PIPELINES_TO_RUN:
        module_name = f"pipelines.{p_file}"
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BasePipeline) and obj != BasePipeline:
                    active_pipelines.append((name, obj(gen_agent, processor, model=model_name)))
        except Exception as e:
            print(f"  Error loading pipeline {p_file}: {e}")

    # 3. 运行每个 Pipeline
    model_slug = model_name.split("/")[-1]

    for name, pipeline in active_pipelines:
        raw_file = os.path.join(output_dir, f"{model_slug}_{name}_raw.json")
        print(f"  Running pipeline: {name}...")
        try:
            # 只跑 1 轮或者较少的次数以便快速测试，但 pipeline.run 内部逻辑是自动饱和的
            # 我们直接运行，观察日志输出
            raw_points = await pipeline.run(sub_category)
            
            with open(raw_file, "w") as f:
                json.dump(raw_points, f, indent=2)
            print(f"    Finished. Points: {len(raw_points)}")
        except Exception as e:
            print(f"    Error in {name}: {e}")

async def main():
    if not os.path.exists("api.json"):
        print("Error: api.json not found.")
        return
        
    with open("api.json") as f:
        api_data = json.load(f)
    api_keys = api_data["api_keys"]
    client_pool = MultiKeyClientPool(api_keys=api_keys)

    for cat_name, sub_cats in CATEGORIES.items():
        for sub_cat in sub_cats:
            for model_name in LLAMA_MODELS:
                await run_single_experiment(cat_name, sub_cat, model_name, client_pool, api_keys)

    print("\n[TEST] Small scale run completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
