<div align="center">

<img src="docs/static/images/logo.png" width="80%">

## An Interactive Agentic Framework for Deep Knowledge Extraction

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Paper (arXiv)](https://arxiv.org/abs/2602.00959) | [Project Page](https://ulab-uiuc.github.io/KnowledgeExtraction/) | [Code](https://github.com/ulab-uiuc/KnowledgeExtraction)

</div>

## Overview

<p align="center">
  <img src="docs/static/images/framework.png" width="92%">
</p>

This framework systematically extracts and quantifies the latent knowledge of black-box LLMs through interactive agentic exploration.

## Installation

```bash
# Clone the repository
git clone https://github.com/ulab-uiuc/KnowledgeExtraction.git
cd KnowledgeExtraction

# Create conda environment
conda create -n ke python=3.10 -y
conda activate ke

# Install dependencies
pip install -r requirements.txt
```

## Prepare API Key(s)

We use NVIDIA NIM API for LLM inference. Follow these steps to get your API key:

1. **Register**: Go to [NVIDIA NIM](https://build.nvidia.com/nim) and create an account
2. **Get API Key**: Navigate to the API Keys section and create a new key
3. **Create config file**: Create `api.json` in the project root:

```json
{
    "api_keys": [
        "nvapi-xxxxxxx-your-first-key-xxxx",
        "nvapi-yyyyyyy-your-second-key-yyyy"
    ],
    "embed_config": {
        "model": "Qwen/Qwen3-Embedding-8B",
        "base_url": "http://localhost:30000/v1"
    }
}
```

> **Tip**: You can add multiple API keys (from multiple accounts) to distribute the load and avoid rate limits.

## Launch Embedding Server (Optional)

For better deduplication performance, launch a local embedding server:

```bash
bash launch_sglang.sh Qwen/Qwen3-Embedding-8B 30000 0.0.0.0 0
```

## Quick Start

```bash
python main.py --domain "Deep Learning" --model "meta/llama-3.1-70b-instruct"
```

## Reproduce Experiments

We provide scripts to reproduce all experiments from the paper:

| Experiment | Description | Command |
|------------|-------------|---------|
| **Exp1: Strategy Search** | Pareto frontier analysis | `python scripts/run_pareto_curves.py` |
| **Exp2: Scaling Law** | Cross-scale comparison (8B/70B/405B) | `python scripts/run_size_comparison.py` |
| **Exp3: Specialization** | General vs RL-tuned models | `python scripts/run_evolution_comparison.py` |
| **Exp4: Cross-Series** | Different model families (~7B) | `python scripts/run_cross_series.py` |

### Example: Reproduce Experiment 1

```bash
# Run extraction with all strategies
python scripts/run_pareto_curves.py

# Evaluate and generate plots
python scripts/evaluate_pareto_curves.py
python scripts/plot_aggregated_pareto.py
```

## Project Structure

```
KnowledgeExtraction/
├── agents/                 # LLM client and API handling
│   ├── call_agent.py      # Generation wrapper with token tracking
│   └── clientpool.py      # Multi-key rotation and retry logic
├── core/                   # Core processing modules
│   ├── processor.py       # Embedding and deduplication
│   ├── judge.py           # Domain relevance auditing
│   ├── evaluator.py       # Evaluation metrics computation
│   └── cleaner.py         # Bullet point extraction
├── pipelines/              # Exploration strategies
│   ├── base.py            # Base pipeline with saturation detection
│   ├── p2_sequential.py   # Sequential probing
│   ├── p3_reflection.py   # Self-reflective refinement
│   ├── p4_taxonomy_explorer.py  # Recursive taxonomy
│   └── p5_debate.py       # Multi-perspective debate
├── scripts/                # Experiment scripts
│   ├── run_*.py           # Experiment runners
│   ├── evaluate_*.py      # Evaluation pipelines
│   └── plot_*.py          # Visualization scripts
└── utils/                  # Utility functions
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{yang2025probing,
  title={Probing the Knowledge Boundary: An Interactive Agentic Framework for Deep Knowledge Extraction},
  author={Yang, Yuheng and Zhu, Siqi and Feng, Tao and Liu, Ge and You, Jiaxuan},
  journal={arXiv preprint arXiv:2602.00959},
  year={2025}
}
```

## License

This project is licensed under the MIT License.
