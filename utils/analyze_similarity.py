import asyncio
import json
import os
import numpy as np
from agents.clientpool import MultiKeyClientPool
from core.processor import KnowledgeProcessor

import argparse

async def main():
    # 0. Parse arguments
    parser = argparse.ArgumentParser(description="Analyze Embedding Similarity Distribution")
    parser.add_argument("--query", type=str, default="Linear Algebra", help="Domain to analyze")
    parser.add_argument("--is_code", action="store_true", help="Whether the domain is code-related")
    args = parser.parse_args()

    # 1. 加载配置
    if not os.path.exists("api.json"):
        print("错误: 未找到 api.json 文件")
        return
        
    with open("api.json") as f:
        api_data = json.load(f)
    
    # 3. Determine Embedding Strategy
    domain_query = args.query
    is_code_domain = args.is_code
    config_key = "code_embed_config" if is_code_domain else "embed_config"
    embed_config = api_data.get(config_key, api_data.get("embed_config", {}))
    
    if not embed_config:
        print(f"错误: api.json 中未发现 {config_key} 配置")
        return

    # 2. 初始化处理器
    target_keys = embed_config.get("api_keys", api_data["api_keys"])
    embed_client_pool = MultiKeyClientPool(
        api_keys=target_keys, 
        base_url=embed_config.get("base_url")
    )
    # 这里我们只需要 Embedding 功能
    processor = KnowledgeProcessor(
        client_pool=None, 
        embed_client_pool=embed_client_pool,
        embed_model=embed_config.get("model")
    )

    # 3. 读取所有原始知识点 (去重后的唯一文本集)
    query_id = domain_query.lower().replace(" ", "_")
    output_dir = os.path.join("results", query_id)
    
    if not os.path.exists(output_dir):
        print(f"错误: 目录 {output_dir} 不存在，请先运行 main.py 生成数据")
        return

    all_points = set()
    for filename in os.listdir(output_dir):
        if filename.endswith("_raw.json"):
            with open(os.path.join(output_dir, filename), "r") as f:
                points = json.load(f)
                all_points.update(points)
    
    unique_points = list(all_points)
    print(f"找到唯一知识点总数: {len(unique_points)}")
    if len(unique_points) < 2:
        print("知识点太少，无法进行两两对比")
        return

    # 4. 获取 Embedding
    print(f"正在调用 {embed_config.get('model')} 获取向量...")
    embeddings = await processor.get_embeddings(unique_points)
    
    # 5. 计算两两相似度
    print("正在计算两两余弦相似度分布...")
    similarities = []
    num_points = len(embeddings)
    
    # 这是一个 O(N^2) 的计算，对于几百个点非常快
    for i in range(num_points):
        for j in range(i + 1, num_points):
            sim = processor.cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)
    
    similarities = np.array(similarities)
    
    # 6. 输出分析报告
    print("\n" + "="*60)
    print(f"Similarity Distribution Report: {embed_config.get('model')}")
    print(f"Domain: {domain_query}")
    print("="*60)
    print(f"对比对数 (Pairs):   {len(similarities)}")
    print(f"最小值 (Min):       {np.min(similarities):.4f}")
    print(f"最大值 (Max):       {np.max(similarities):.4f}")
    print(f"平均值 (Mean):      {np.mean(similarities):.4f}")
    print(f"中位数 (Median):    {np.median(similarities):.4f}")
    print("-" * 40)
    print("百分位数分布 (Percentiles):")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}th 分位数: {np.percentile(similarities, p):.4f}")
    
    print("-" * 40)
    print("区间占比 (Range Distribution):")
    ranges = [(0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.0)]
    for low, high in ranges:
        count = np.sum((similarities >= low) & (similarities < high))
        print(f"  [{low:.2f} - {high:.2f}): {count:5d} 对 ({count/len(similarities):.2%})")
    print("="*60)
    print("建议：观察 0.85 以上的占比。如果 Qwen3 区分度高，你会发现极高分 (0.95+) 的对数显著减少。")

if __name__ == "__main__":
    asyncio.run(main())


