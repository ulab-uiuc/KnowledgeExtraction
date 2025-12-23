import asyncio
import json
import importlib
import os
import inspect
from agents.call_agent import GenAgent
from pipelines.base import BasePipeline

async def test_output():
    with open("api.json") as f:
        api_data = json.load(f)
    api_keys = api_data["api_keys"]
    gen_agent = GenAgent(api_key=api_keys)
    
    query = "Linear Algebra"
    query_id = query.lower().replace(" ", "_")
    output_dir = os.path.join("results", query_id)
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline_instances = []
    # ... (discovery code stays same) ...
    
    all_results = {}
    
    for name, instance in pipeline_instances:
        print(f"--- Pipeline: {name} ---")
        try:
            points = await instance.run(query)
            all_results[name] = points
            # Save individual pipeline result
            with open(os.path.join(output_dir, f"{name}.json"), "w") as f:
                json.dump(points, f, indent=2)
            
            print(f"Extracted {len(points)} points. Saved to {name}.json")
            # ... (print sample code stays same) ...

        except Exception as e:
            print(f"Error running {name}: {e}")
        print("\n")

if __name__ == "__main__":
    asyncio.run(test_output())

