# 🚀 Gemma Benchmark Suite - Colab Edition

## Overview

This repository has been enhanced to work seamlessly with Google Colab, providing an easy way to benchmark language models without authentication issues. The main improvements include:

## ✨ Key Features

### 🎯 **No Authentication Required**
- Uses publicly available models (GPT-2, DialoGPT, etc.)
- No Hugging Face token needed for basic benchmarking
- Perfect for educational and research purposes

### 📊 **Rich Visualizations**
- Interactive performance heatmaps
- Model comparison charts
- Task difficulty analysis
- Overall ranking visualizations

### ☁️ **Cloud-Ready**
- Optimized for Google Colab environment
- Free GPU acceleration
- Easy sharing and collaboration

## 📁 New Files Added

### 1. `gemma_benchmark_colab.ipynb`
- **Purpose**: Main Colab notebook for interactive benchmarking
- **Features**: 
  - One-click setup and installation
  - Model selection interface
  - Real-time visualizations
  - Export functionality

### 2. `configs/colab_config.yaml`
- **Purpose**: Configuration for publicly available models
- **Models**: GPT-2, GPT-2 Medium, DistilGPT-2, DialoGPT
- **Tasks**: MMLU, GSM8K, HumanEval

### 3. `test_colab_benchmark.py`
- **Purpose**: Test script to verify installation
- **Features**: 
  - Comprehensive functionality testing
  - Demo visualizations
  - Sample results generation

## 🔧 Repository Fixes

### 1. **Dependencies Fixed**
- Updated `requirements.txt` with compatible versions
- Resolved Python environment conflicts
- Added missing packages

### 2. **Configuration Enhanced**
- Created Colab-friendly configuration
- Added public model support
- Simplified setup process

### 3. **Documentation Updated**
- Added Colab notebook instructions
- Updated README with quick start guide
- Included troubleshooting section

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
1. Click the Colab badge in the README
2. Run all cells in the notebook
3. Select your models and benchmarks
4. View results and visualizations

### Option 2: Local Testing
```bash
# Clone and setup
git clone https://github.com/heilcheng/gemma-benchmark.git
cd gemma-benchmark
pip install -r requirements.txt

# Test the installation
python test_colab_benchmark.py

# Run with Colab config
python -m gemma_benchmark.scripts.run_benchmark \
  --config configs/colab_config.yaml \
  --visualize
```

## 📊 Available Models

### Public Models (No Auth Required)
- **GPT-2** (124M parameters)
- **GPT-2 Medium** (355M parameters)
- **DistilGPT-2** (82M parameters)
- **DialoGPT Medium** (345M parameters)

### Gated Models (Auth Required)
- **Gemma-2B** (requires Hugging Face token)
- **Gemma-9B** (requires Hugging Face token)
- **Other Gemma variants**

## 📋 Available Benchmarks

### 1. **MMLU (Massive Multitask Language Understanding)**
- **Description**: 57 subjects covering STEM, humanities, and more
- **Metrics**: Accuracy per subject
- **Colab Config**: Uses mathematics subset for faster execution

### 2. **GSM8K (Grade School Math 8K)**
- **Description**: Grade school math word problems
- **Metrics**: Exact match accuracy
- **Features**: Chain-of-thought reasoning

### 3. **HumanEval**
- **Description**: Python programming problems
- **Metrics**: Pass@k rates
- **Use Case**: Code generation evaluation

## 🎨 Visualization Features

### 1. **Performance Heatmap**
- Color-coded performance matrix
- Model vs Task comparison
- Easy identification of strengths/weaknesses

### 2. **Model Comparison Charts**
- Bar charts for each benchmark
- Side-by-side performance comparison
- Statistical significance indicators

### 3. **Overall Ranking**
- Horizontal bar chart ranking
- Average performance across tasks
- Model consistency analysis

### 4. **Detailed Analysis**
- Task difficulty ranking
- Score distribution analysis
- Performance correlation matrix

## 📤 Export Options

### Available Formats
- **JSON**: Raw benchmark results
- **CSV**: Tabular data for analysis
- **PNG**: High-resolution visualizations
- **Markdown**: Human-readable reports

### Export Locations
- **Colab**: Download from files panel
- **Local**: Saved in `results/` directory
- **Cloud**: Automatic backup to Google Drive

## 🔍 Troubleshooting

### Common Issues

#### 1. **Import Errors**
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

#### 2. **Model Loading Failures**
```bash
# Solution: Use Colab config
python -m gemma_benchmark.scripts.run_benchmark \
  --config configs/colab_config.yaml
```

#### 3. **Authentication Errors**
```bash
# Solution: Use public models or set token
export HF_TOKEN=your_token_here
```

#### 4. **Memory Issues**
```bash
# Solution: Enable quantization
# Edit configs/colab_config.yaml
quantization: true
```

## 🎯 Best Practices

### 1. **Model Selection**
- Start with smaller models for testing
- Use quantization for large models
- Consider hardware limitations

### 2. **Benchmark Configuration**
- Use smaller subsets for faster execution
- Adjust batch sizes based on memory
- Enable visualization for better analysis

### 3. **Results Analysis**
- Compare multiple models
- Consider task-specific performance
- Look for consistency patterns

## 🔮 Future Enhancements

### Planned Features
- **More Public Models**: Additional open-source models
- **Custom Benchmarks**: User-defined evaluation tasks
- **Advanced Visualizations**: Interactive dashboards
- **Cloud Integration**: Direct Google Drive integration

### Community Contributions
- **New Benchmarks**: Submit custom evaluation tasks
- **Model Support**: Add support for new model types
- **Visualization Improvements**: Enhanced chart types

## 📚 Resources

### Documentation
- [Original README](README.md)
- [API Documentation](docs/api_docs.md)
- [Configuration Guide](configs/)

### External Links
- [Hugging Face Models](https://huggingface.co/models)
- [Google Colab](https://colab.research.google.com)
- [Gemma Models](https://huggingface.co/google)

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Adding new models
- Creating custom benchmarks
- Improving visualizations
- Enhancing documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Happy Benchmarking! 🎉**

For questions or issues, please open an issue on GitHub or check the troubleshooting section above. 