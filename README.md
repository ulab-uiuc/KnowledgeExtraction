# LLM Knowledge Extraction Evaluation Framework

This framework is designed to evaluate and compare different **Knowledge Extraction Pipelines** (strategies) for a given Large Language Model (LLM) and domain. It establishes a "Pseudo Ground Truth" (Union Set) by aggregating results from multiple pipelines and calculates performance metrics like **Recall (Coverage)** and **Accuracy (Relevance)**.

---

## 1. Getting Started

### Prerequisites
- **Conda** installed on your system.
- Access to **NVIDIA API** (or other OpenAI-compatible endpoints).
- (Optional) A GPU server for local embedding deployment.

### Environment Setup
Create and activate the environment using the provided `requirements.txt`:

```bash
# Create the environment
conda create -n ke python=3.10 -y
conda activate ke

# Install dependencies
pip install -r requirements.txt
```

### Configuration
Create an `api.json` file in the root directory with your API keys and model configurations:

```json
{
  "api_keys": ["your-nvidia-api-key-1", "your-nvidia-api-key-2"],
  "embed_config": {
    "base_url": "http://localhost:30000/v1",
    "model": "Qwen/Qwen3-Embedding-8B"
  },
  "code_embed_config": {
    "base_url": "https://integrate.api.nvidia.com/v1",
    "model": "nvidia/nv-embedcode-7b-v1"
  }
}
```

---

## 2. Core Workflow

The framework operates in two main phases:

### Phase 1: Saturation Generation
Each active pipeline runs repeatedly until it reaches **Saturation** (i.e., when new information generated drops below a novelty threshold).

### Phase 2: Global Evaluation & Auditing
1. **Deduplication**: Aggregates all raw points and uses a hybrid "Embedding + LLM Judge" approach to build a unique Union Set.
2. **Domain Audit**: Uses a high-quality LLM to filter out points irrelevant to the target domain.
3. **Metric Calculation**: Computes Recall and Accuracy for each pipeline based on the audited Union Set.

---

## 3. How to Run

### Main Execution
Run the full generation and evaluation loop for a specific domain:

```bash
# For general text domains (e.g., Math, Science)
python3 main.py --query "Linear Algebra"

# For code-specific domains (switches to code-optimized embedding)
python3 main.py --query "Python Decorators" --is_code
```

### Evaluation Only
If you have already generated raw data and want to re-run the deduplication or audit with different thresholds/models:

```bash
python3 evaluate_only.py --query "Linear Algebra"
```

---

## 4. Advanced Features

### Local Embedding Deployment
To use the advanced **Qwen3-Embedding-8B** model locally for better semantic clustering:

```bash
# Usage: bash launch_sglang.sh [MODEL_PATH] [PORT] [HOST] [GPU_ID]
bash launch_sglang.sh Qwen/Qwen3-Embedding-8B 30000 0.0.0.0 0
```

### Analysis Tools (`utils/`)
- `analyze_similarity.py`: Check the distribution of cosine similarities in your current domain.
- `verify_thresholds.py`: Sample pairs and use 405B models to verify if your embedding thresholds are accurate.
- `list_nvidia_models.py`: List all available models from your API provider.

```bash
PYTHONPATH=. python3 utils/analyze_similarity.py --query "Linear Algebra"
```

---

## 5. Collaboration & Contribution

### Adding New Pipelines
The framework uses **Auto-discovery**. To contribute a new extraction strategy:
1. Navigate to the `pipelines/` directory.
2. Follow the instructions in [pipelines/README_DEVELOPER.md](pipelines/README_DEVELOPER.md).
3. Create your class inheriting from `BasePipeline`, and it will be automatically included in the next `main.py` run.

### Project Structure
- `agents/`: LLM client pooling and safe-call wrappers.
- `core/`: Core logic for deduplication (`processor.py`), auditing (`judge.py`), and cleaning (`cleaner.py`).
- `pipelines/`: Modular extraction strategies.
- `results/`: JSON outputs categorized by domain.
- `utils/`: Diagnostic and calibration scripts.

---

## 6. License
Apache-2.0 License.
