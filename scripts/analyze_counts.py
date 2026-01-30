import json
import os

def count_points(directory):
    stats = {}
    for filename in os.listdir(directory):
        if filename.endswith("_raw.json"):
            parts = filename.replace("_raw.json", "").split("_")
            model = parts[0]
            pipeline = parts[1]
            
            with open(os.path.join(directory, filename), 'r') as f:
                data = json.load(f)
                stats[filename] = len(data)
    return stats

dirs = [
    "results/deep_learning/transformer_architectures",
    "results/deep_learning/generative_models"
]

all_stats = {}
for d in dirs:
    all_stats[d] = count_points(d)

print(json.dumps(all_stats, indent=2))
