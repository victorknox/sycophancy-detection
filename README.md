# Sycophancy Detection via Representation Engineering

Detection of sycophantic behavior in large language models using white-box methods. This project implements activation-based detection approaches from representation engineering to identify when LLMs give sycophantic (user-pleasing rather than truthful) responses.

## Overview

**Sycophancy** is when an AI assistant agrees with or validates a user's stated beliefs, preferences, or biases rather than providing objective, truthful responses. This is problematic because it can reinforce misconceptions and reduce the utility of AI assistants.

This repository provides tools to:
1. **Measure** how often a model exhibits sycophantic behavior
2. **Detect** sycophantic responses using activation-based methods
3. **Compare** different detection approaches (diff-in-means, linear probes, LLM-as-judge)

### Key Findings

| Method | Test Accuracy | Test AUROC | Notes |
|--------|--------------|------------|-------|
| Random Baseline | ~50% | ~50% | Sanity check |
| LLM-as-Judge | 60-75% | 0.692 | Black-box baseline |
| **Diff-in-Means** | **85-95%** | **0.90-0.98** | Best performing |
| Linear Probe | 80-92% | 0.88-0.96 | Comparable to diff-in-means |

## Installation

```bash
# Clone the repository
git clone https://github.com/victorknox/sycophancy-detection.git
cd sycophancy-detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- ~5GB disk space for model weights

## Quick Start

### 1. Prepare Data

Download and process the dataset from HuggingFace:

```bash
python data_prep.py --n-train 1000 --n-test 500
```

This creates balanced train/test splits with contrastive pairs in `data/`.

### 2. Run Experiments

**Layer Sweep (Diff-in-Means)** - Find the best layer for detection:
```bash
python run_layer_sweep.py --model Qwen/Qwen2.5-3B-Instruct
```

**Linear Probe** - Train logistic regression on activations:
```bash
python run_linear_probe.py --model Qwen/Qwen2.5-3B-Instruct
```

**Model Sycophancy Rate** - Measure how often the model is sycophantic:
```bash
python run_model_sycophancy_rate.py --model Qwen/Qwen2.5-3B-Instruct
```

**LLM-as-Judge Baseline** - Black-box detection baseline:
```bash
python run_llm_judge_baseline.py --model Qwen/Qwen2.5-3B-Instruct
```

## Methods

### Diff-in-Means (Representation Engineering)

The core method learns a "sycophancy direction" in the model's activation space:

1. **Extract activations** from a specific layer for sycophantic (label=1) and non-sycophantic (label=0) responses
2. **Compute direction**: `v = mean(h | sycophantic) - mean(h | non-sycophantic)`
3. **Score new responses** by projecting their activations onto this direction
4. **Classify** using an optimal threshold learned on the training set

This approach is inspired by [Representation Engineering](https://arxiv.org/abs/2310.01405).

### Linear Probe

Trains a logistic regression classifier directly on the model's hidden activations:
- Uses mean-pooled activations from each layer
- Provides interpretable decision boundary
- Generally comparable performance to diff-in-means

### LLM-as-Judge

Uses the same model (or a larger model) to judge whether responses are sycophantic:
- Zero-shot prompting with structured JSON output
- Serves as a black-box baseline
- Lower accuracy than white-box methods

## Project Structure

```
sycophancy-detection/
├── data_prep.py              # Dataset preparation
├── run_layer_sweep.py        # Diff-in-means layer sweep experiment
├── run_linear_probe.py       # Linear probe experiment
├── run_model_sycophancy_rate.py  # Model sycophancy measurement
├── run_llm_judge_baseline.py # LLM-as-judge baseline
├── run_random_baseline.py    # Random baseline
├── utils.py                  # Shared utilities
├── requirements.txt          # Dependencies
├── data/                     # Prepared data (gitignored)
└── results/                  # Experiment results (gitignored)
```

## Configuration Options

### Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--seed` | Random seed for reproducibility | 42 |
| `--model` | HuggingFace model name | `Qwen/Qwen2.5-3B-Instruct` |
| `--output-dir` | Directory for results | `results` |

### Layer Sweep / Linear Probe

| Argument | Description | Default |
|----------|-------------|---------|
| `--layers` | Specific layers to test | Auto-select |
| `--train-instances` | Training set size | 2000 |
| `--test-instances` | Test set size | 1000 |

### Supported Models

Any HuggingFace causal LM works. Tested models:
- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct`

## Dataset

Uses the [Anthropic Model-Written Evals](https://huggingface.co/datasets/Anthropic/model-written-evals) political typology sycophancy dataset:

- **Format**: Each example contains a political biography prompt and two answer options
- **Labels**:
  - `label=1`: Sycophantic response (matches user's stated political views)
  - `label=0`: Non-sycophantic response (objective/contrarian)
- **Size**: ~1000 unique prompts available

Example:
```
Prompt: "I am a [political biography]. Which of these statements do you agree with?
(A) [Statement aligned with user's views]
(B) [Contrarian statement]"

Sycophantic: Model chooses (A) to please the user
Non-sycophantic: Model chooses (B) objectively
```

## Results Interpretation

### Layer Sweep Results

```
Layer    Train Acc    Test Acc     Test F1      Test AUROC
0        0.6234       0.5890       0.5723       0.6145
8        0.8912       0.8456       0.8401       0.9123
16       0.9534       0.9123       0.9098       0.9678      <-- BEST
24       0.9345       0.8967       0.8934       0.9534
```

- **Best layer** is typically in the middle-to-late layers (50-75% depth)
- Early layers capture low-level features, not useful for concept detection
- Very late layers may have task-specific information that hurts generalization

### Sycophancy Rate Results

```
Sycophancy Rate: 0.7234
(362/500 prompts chose sycophantic answer)
```

This measures the model's inherent tendency to be sycophantic before any detection/mitigation.

## Troubleshooting

**Out of Memory:**
- Reduce `--train-instances` and `--test-instances`
- Use `--use-4bit` flag for LLM judge with large models
- Use a smaller model variant

**Slow Extraction:**
- Use a GPU (10-100x faster than CPU)
- Reduce the number of layers tested with `--layers`

**Low Accuracy:**
- Ensure balanced labels in data (check with `data_prep.py` output)
- Try different layers (middle layers often work best)
- Increase training set size

## Citation

If you use this code, please cite:

```bibtex
@software{sycophancy_detection,
  title = {Sycophancy Detection via Representation Engineering},
  author = {Vamshi Krishna Bonagiri, Aryaman Bahl},
  year = {2024},
  url = {https://github.com/victorknox/sycophancy-detection}
}
```

### Related Work

- [Representation Engineering](https://arxiv.org/abs/2310.01405) - Zou et al., 2023
- [Towards Understanding Sycophancy in LLMs](https://arxiv.org/abs/2310.13548) - Sharma et al., 2023
- [Anthropic Model-Written Evals](https://huggingface.co/datasets/Anthropic/model-written-evals)

## License

MIT License - see LICENSE file for details.
