# Knowledge Extraction Framework

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Interactive Agentic Framework for Knowledge Extraction and Evaluation**

[Paper]() | [Project Page]()

## Overview

This framework systematically extracts and quantifies the latent knowledge of black-box LLMs through interactive agentic exploration. We address three key research questions:

1. **What does a model know** in a given domain?
2. **How much knowledge** can be extracted at saturation?
3. **Which exploration strategy** is most efficient (Pareto-optimal)?

## Key Features

- **4 Exploration Strategies**: Sequential, Self-Reflection, Recursive Taxonomy, Multi-Perspective Debate
- **3-Stage Knowledge Processor**: Vector filtering → LLM adjudication → Domain auditing
- **Saturation-based Stopping**: Automatic detection of knowledge exhaustion
- **Cross-model Evaluation**: Scaling laws, specialization trade-offs, training data effects

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

## Quick Start

### 1. Configure API Keys

Create `api.json` in the project root:

```json
{
    "api_keys": ["your-nvidia-api-key"],
    "embed_config": {
        "model": "Qwen/Qwen3-Embedding-8B",
        "base_url": "http://localhost:30000/v1"
    }
}
```

### 2. Launch Embedding Server (Optional)

```bash
bash launch_sglang.sh Qwen/Qwen3-Embedding-8B 30000 0.0.0.0 0
```

### 3. Run Knowledge Extraction

```bash
python main.py --domain "Deep Learning" --model "meta/llama-3.1-70b-instruct"
```

## Experiments

We provide scripts to reproduce all experiments from the paper:

| Experiment | Description | Run | Evaluate |
|------------|-------------|-----|----------|
| **Exp1: Strategy Search** | Pareto frontier analysis | `scripts/run_pareto_curves.py` | `scripts/evaluate_pareto_curves.py` |
| **Exp2: Scaling Law** | Cross-scale comparison (8B/70B/405B) | `scripts/run_size_comparison.py` | `scripts/evaluate_model_comparison.py` |
| **Exp3: Specialization** | General vs RL-tuned models | `scripts/run_evolution_comparison.py` | `scripts/evaluate_qwen_comparison.py` |
| **Exp4: Cross-Series** | Different model families (~7B) | `scripts/run_cross_series.py` | `scripts/evaluate_cross_series.py` |

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
│   └── cleaner.py         # Bullet point extraction
├── pipelines/              # Exploration strategies
│   ├── base.py            # Base pipeline with saturation detection
│   ├── p2_sequential.py   # Sequential "What else?" probing
│   ├── p3_reflection.py   # Self-reflective refinement
│   ├── p4_taxonomy_explorer.py  # Recursive taxonomy decomposition
│   └── p5_debate.py       # Multi-perspective debate
├── scripts/                # Experiment scripts
│   ├── run_*.py           # Extraction runners
│   └── evaluate_*.py      # Evaluation pipelines
├── main.py                 # Main entry point
└── requirements.txt
```

## Citation

```bibtex
@article{knowledge-extraction-2025,
  title={Interactive Agentic Framework for Knowledge Extraction and Evaluation},
  author={},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License.
