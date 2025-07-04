# Gemma Benchmarking Suite

A comprehensive benchmarking framework for evaluating Google's Gemma models and comparing them with other open-source language models on standard academic benchmarks.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/google)

## Overview

This benchmarking suite provides a unified framework to:

- **Evaluate language models** on standard academic benchmarks (MMLU, GSM8K, HumanEval, ARC, TruthfulQA)
- **Compare performance** across different model families (Gemma, Mistral, Llama, and any HuggingFace model)
- **Measure efficiency** including latency, throughput, and memory usage
- **Generate comprehensive visualizations** with statistical analysis
- **Support quantization** for memory-efficient evaluation
- **Create publication-ready reports** with leaderboards and charts

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/heilcheng/gemma-benchmark.git
cd gemma-benchmark
```

### 2. Installation Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Environment

```bash
cd src
python -m venv venv
source venv/bin/activate  # Linux/Mac
# For Windows PowerShell:
# venv\Scripts\Activate.ps1
# For Windows CMD:
# venv\Scripts\activate.bat
cd ..
```

### 4. Run Benchmark

To run with the default comprehensive configuration:

```bash
python src/scripts/run_benchmark.py \
    --config src/config/benchmark_config.yaml \
    --output-dir results/ \
    --log-level INFO 
```

To run with the advanced custom configuration showcasing custom script loaders:

```bash
python src/scripts/run_benchmark.py \
    --config src/config/advanced_custom_benchmark_config.yaml \
    --output-dir results_custom/ \
    --log-level DEBUG
```

The framework features detailed logging, writing to a timestamped file in `run_logs/` by default (e.g., `run_logs/benchmark_YYYYMMDD_HHMMSS.log`)

## Examples & Getting Started

### Quick Test

You can quickly test the framework by running a basic benchmark with the default configuration. This provides a great starting point for evaluating the framework's capabilities.

### Generating a Default Configuration

To get started with a base configuration, you can use the `generate_default_config.py` script located in the `src/` directory:

```bash
python src/generate_default_config.py
```

This will create a `default_benchmark_config.yaml` file in `src/config/` which you can then customize.

### Visualizing Benchmark Results

An interactive dashboard is available to explore the benchmark results.

1. Ensure you have run a benchmark and results are saved (e.g., in `benchmarks_output/benchmark_results.json`).
2. Run the Streamlit application:
```bash
streamlit run visualize/dashboard.py
```

This will launch the dashboard in your web browser.

## 📋 Prerequisites

### Hardware Requirements

| Model Size | Min VRAM | Recommended | Quantized Memory |
|------------|----------|-------------|------------------|
| 2B params  | 4GB      | 8GB         | ~2GB            |
| 7-9B params| 8GB      | 16GB        | ~5GB            |
| 13B params | 16GB     | 24GB        | ~8GB            |
| 27B params | 24GB     | 32GB+       | ~14GB           |

### Software Requirements

- Python 3.8 or higher
- CUDA 11.8+ (for GPU acceleration)
- 50GB+ disk space (for models and datasets)

### HuggingFace Setup

1. Create account at [huggingface.co](https://huggingface.co)
2. Accept model licenses (e.g., [Gemma](https://huggingface.co/google/gemma-2-2b))
3. Generate access token with read permissions

## 🛠️ Installation

### Standard Installation

```bash
# Clone repository
git clone https://github.com/heilcheng/gemma-benchmark.git
cd gemma-benchmark

# Install dependencies
pip install -r requirements.txt
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### Docker Installation

```bash
# Build Docker image
docker build -t gemma-benchmark .

# Run with GPU support
docker run --gpus all -v $(pwd)/results:/app/results gemma-benchmark \
  python -m gemma_benchmark.scripts.run_benchmark --config /app/configs/benchmark_config.yaml
```

## 📊 Supported Benchmarks

| Benchmark | Type | Description | Metrics |
|-----------|------|-------------|---------|
| **MMLU** | Knowledge | 57 subjects covering STEM, humanities, and more | Accuracy per subject |
| **GSM8K** | Math Reasoning | Grade school math word problems | Exact match accuracy |
| **HumanEval** | Code Generation | Python programming problems | Pass@k rates |
| **ARC** | Science Reasoning | Science questions (Easy/Challenge sets) | Multiple choice accuracy |
| **TruthfulQA** | Truthfulness | Questions testing for common misconceptions | MC accuracy, truthfulness |
| **Efficiency** | Performance | Speed and resource utilization | Tokens/sec, memory, latency |

## 🤖 Supported Models

### Recent Open-Weight Models
- **Microsoft Phi Series**: Phi-3 Mini (3.8B), Phi-2 (2.7B)
- **Qwen Series**: Qwen2-1.5B, Qwen2-7B, Qwen2-14B
- **Mistral AI**: Mistral-7B-v0.1, Mixtral-8x7B
- **Meta Llama**: Llama-2-7b-chat, Llama-2-13b-chat
- **Google Gemma**: Gemma-2B, Gemma-7B (requires authentication)

### Model Features
- **4-bit Quantization**: Efficient GPU usage for large models
- **Auto Device Selection**: Automatic CPU/GPU detection
- **Memory Optimization**: Smart caching and memory management
- **Batch Processing**: Configurable batch sizes for optimal performance

### Example Configuration
```yaml
models:
  phi-3-mini:
    type: huggingface
    model_id: "microsoft/Phi-3-mini-4k-instruct"
    quantization: true
    
  qwen2-7b:
    type: huggingface
    model_id: "Qwen/Qwen2-7B"
    quantization: true
    
  mistral-7b:
    type: huggingface
    model_id: "mistralai/Mistral-7B-v0.1"
    quantization: true
```

## 🔧 Configuration

Create a YAML configuration file to define your benchmark:

```yaml
# Example configuration
models:
  gemma-2b:
    type: gemma
    size: 2b
    variant: it
    quantization: true  # Enable 4-bit quantization
    cache_dir: ./cache/models

tasks:
  mmlu:
    type: mmlu
    subset: all  # or specific subject like "mathematics"
    shot_count: 5
    
  gsm8k:
    type: gsm8k
    shot_count: 8
    
  efficiency:
    type: efficiency
    output_lengths: [128, 256, 512]

evaluation:
  runs: 1  # Number of runs for statistical analysis
  batch_size: auto

output:
  path: ./results
  visualize: true
  export_formats: [json, yaml]

hardware:
  device: auto  # auto, cuda, cpu
  precision: bfloat16
```

## 📈 Usage Examples

### Google Colab Notebook
For the easiest experience, use our interactive Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/heilcheng/gemma-benchmark/blob/main/gemma_benchmark_colab.ipynb)

The notebook provides:
- **No Authentication Required**: Uses publicly available models
- **Interactive Visualizations**: Real-time charts and comparisons
- **Easy Model Selection**: Choose from pre-configured model lists
- **Export Results**: Download results in multiple formats
- **Cloud GPU**: Free GPU acceleration on Google Colab

### Basic Evaluation

```bash
# Evaluate a single model on one task
python -m gemma_benchmark.scripts.run_benchmark \
  --config configs/benchmark_config.yaml \
  --models gemma-2b \
  --tasks mmlu
```

### Multi-Model Comparison

```bash
# Compare multiple models across all tasks
python -m gemma_benchmark.scripts.run_benchmark \
  --config configs/benchmark_config.yaml \
  --models gemma-2b gemma-9b mistral-7b \
  --tasks mmlu gsm8k humaneval \
  --visualize
```

### Custom Model Evaluation

```python
from gemma_benchmark.core.benchmark import GemmaBenchmark

# Initialize benchmark
benchmark = GemmaBenchmark("my_config.yaml")

# Load specific models and tasks
benchmark.load_models(["gemma-2b"])
benchmark.load_tasks(["mmlu"])

# Run evaluation
results = benchmark.run_benchmarks()

# Save results
benchmark.save_results("results.yaml")
```

### Advanced Analysis

```python
from gemma_benchmark.visualization.charts import BenchmarkVisualizer

# Create comprehensive visualizations
visualizer = BenchmarkVisualizer("./results", style="publication")
visualizer.create_performance_overview(results)
visualizer.create_efficiency_analysis(results)
visualizer.create_statistical_analysis(results, multi_run_data)
```

## 📊 Output and Visualization

The framework generates comprehensive results including:

### Performance Reports
- Model comparison heatmaps
- Task-specific accuracy charts
- Subject-level breakdowns (for MMLU)
- Efficiency metrics (latency, throughput, memory)

### Statistical Analysis
- Confidence intervals
- Variance analysis
- Statistical significance testing
- Multi-run aggregation

### Output Structure
```
results/
├── 20250108_143022/
│   ├── results.yaml              # Raw benchmark results
│   ├── summary.json              # Aggregated metrics
│   ├── visualizations/
│   │   ├── performance_overview.png
│   │   ├── efficiency_analysis.png
│   │   ├── mmlu_comparison.png
│   │   └── statistical_analysis.png
│   └── executive_summary.md      # Human-readable report
```

## 🔐 Authentication

The framework supports HuggingFace authentication for accessing gated models:

```bash
# Option 1: Environment variable
export HF_TOKEN=hf_your_token_here

# Option 2: Interactive setup
python -m gemma_benchmark.auth

# Option 3: HuggingFace CLI
huggingface-cli login
```

## ⚡ Performance Optimization

### Memory Optimization
```yaml
models:
  large-model:
    quantization: true  # 4-bit quantization
    device_map: auto    # Automatic device mapping
    max_memory: {0: "15GB", 1: "15GB"}  # Multi-GPU
```

### Speed Optimization
```yaml
hardware:
  precision: bfloat16
  torch_compile: true  # PyTorch 2.0 compilation
  mixed_precision: true
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_core.py -v
pytest tests/test_tasks.py -v

# Run with coverage
pytest --cov=gemma_benchmark tests/
```

## 🚨 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Enable quantization or reduce batch size |
| `Repository not found` | Check HF token and model access permissions |
| `No module named 'flash_attn'` | Optional dependency - ignore or install separately |
| Authentication errors | Ensure HF_TOKEN is set and has read permissions |

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Project Structure

```
gemma-benchmark/
├── gemma_benchmark/
│   ├── core/           # Core orchestration and model loading
│   ├── tasks/          # Benchmark implementations
│   ├── utils/          # Metrics, validation, data downloading
│   ├── visualization/  # Charts and reporting
│   └── scripts/        # CLI entry points
├── configs/            # Example configurations
├── tests/              # Test suite
└── examples/           # Usage examples
```

## 🤝 Contributing

Contributions are welcome! Please see our contributing guidelines.

### Adding New Models

1. Create a loader in `gemma_benchmark/core/model_loader.py`
2. Register the model type
3. Add configuration examples

### Adding New Benchmarks

1. Implement task in `gemma_benchmark/tasks/`
2. Inherit from `AbstractBenchmark`
3. Register with `BenchmarkFactory`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this benchmarking suite in your research, please cite:

```bibtex
@software{gemma_benchmarking_suite,
  author = {Hailey Cheng},
  title = {Gemma Benchmarking Suite: A Comprehensive Evaluation Framework},
  year = {2025},
  url = {https://github.com/heilcheng/gemma-benchmark},
  version = {1.0.0}
}
```

## 🙏 Acknowledgments

- Google Research for open-sourcing Gemma models
- HuggingFace for model hosting and datasets infrastructure
- The open-source community for benchmark datasets and evaluation methodologies

---

**Note**: This is an academic research tool. Please ensure you have appropriate permissions and compute resources before running large-scale evaluations.