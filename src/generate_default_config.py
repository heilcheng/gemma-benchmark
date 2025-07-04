#!/usr/bin/env python3
"""
Generate default benchmark configuration with recent open-weight models.
"""

import yaml
import os
from pathlib import Path

def generate_default_config():
    """Generate a default benchmark configuration."""
    
    config = {
        "models": {
            # Small models for quick testing
            "phi-3-mini": {
                "type": "huggingface",
                "model_id": "microsoft/Phi-3-mini-4k-instruct",
                "size": "3.8B",
                "quantization": True,
                "cache_dir": "cache/models"
            },
            "phi-2": {
                "type": "huggingface",
                "model_id": "microsoft/Phi-2",
                "size": "2.7B",
                "quantization": True,
                "cache_dir": "cache/models"
            },
            "qwen2-1.5b": {
                "type": "huggingface",
                "model_id": "Qwen/Qwen2-1.5B",
                "size": "1.5B",
                "quantization": True,
                "cache_dir": "cache/models"
            },
            # Medium models for comprehensive testing
            "qwen2-7b": {
                "type": "huggingface",
                "model_id": "Qwen/Qwen2-7B",
                "size": "7B",
                "quantization": True,
                "cache_dir": "cache/models"
            },
            "mistral-7b": {
                "type": "huggingface",
                "model_id": "mistralai/Mistral-7B-v0.1",
                "size": "7B",
                "quantization": True,
                "cache_dir": "cache/models"
            }
        },
        "tasks": {
            "mmlu": {
                "type": "mmlu",
                "subset": "mathematics",  # Use smaller subset for faster execution
                "shot_count": 5,
                "temperature": 0.0
            },
            "gsm8k": {
                "type": "gsm8k",
                "shot_count": 5,
                "use_chain_of_thought": True
            },
            "humaneval": {
                "type": "humaneval",
                "shot_count": 0,
                "temperature": 0.2
            },
            "arc": {
                "type": "arc",
                "shot_count": 3,
                "temperature": 0.0
            },
            "truthfulqa": {
                "type": "truthfulqa",
                "shot_count": 0,
                "temperature": 0.0
            }
        },
        "evaluation": {
            "runs": 1,
            "batch_size": "auto",
            "statistical_tests": False
        },
        "output": {
            "path": "benchmarks_output",
            "visualize": True,
            "export_formats": ["json", "csv"]
        },
        "hardware": {
            "device": "auto",
            "precision": "float16",
            "quantization": True
        }
    }
    
    # Create config directory if it doesn't exist
    config_dir = Path("src/config")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Save default configuration
    config_path = config_dir / "default_benchmark_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Default configuration generated: {config_path}")
    print(f"📝 Configuration includes:")
    print(f"  • Models: {len(config['models'])} recent open-weight models")
    print(f"  • Tasks: {len(config['tasks'])} benchmark tasks")
    print(f"  • Quantization: Enabled for efficient GPU usage")
    print(f"  • Output: Results saved to {config['output']['path']}")
    
    return config_path

def generate_advanced_config():
    """Generate an advanced configuration with more models and options."""
    
    config = {
        "models": {
            # Small models
            "phi-3-mini": {
                "type": "huggingface",
                "model_id": "microsoft/Phi-3-mini-4k-instruct",
                "size": "3.8B",
                "quantization": True,
                "cache_dir": "cache/models"
            },
            "phi-2": {
                "type": "huggingface",
                "model_id": "microsoft/Phi-2",
                "size": "2.7B",
                "quantization": True,
                "cache_dir": "cache/models"
            },
            "qwen2-1.5b": {
                "type": "huggingface",
                "model_id": "Qwen/Qwen2-1.5B",
                "size": "1.5B",
                "quantization": True,
                "cache_dir": "cache/models"
            },
            # Medium models
            "qwen2-7b": {
                "type": "huggingface",
                "model_id": "Qwen/Qwen2-7B",
                "size": "7B",
                "quantization": True,
                "cache_dir": "cache/models"
            },
            "mistral-7b": {
                "type": "huggingface",
                "model_id": "mistralai/Mistral-7B-v0.1",
                "size": "7B",
                "quantization": True,
                "cache_dir": "cache/models"
            },
            # Large models (require more GPU memory)
            "llama2-7b": {
                "type": "huggingface",
                "model_id": "meta-llama/Llama-2-7b-chat-hf",
                "size": "7B",
                "quantization": True,
                "cache_dir": "cache/models"
            },
            "gemma-2b": {
                "type": "huggingface",
                "model_id": "google/gemma-2b-it",
                "size": "2B",
                "quantization": True,
                "cache_dir": "cache/models"
            }
        },
        "tasks": {
            "mmlu": {
                "type": "mmlu",
                "subset": "all",  # Full MMLU dataset
                "shot_count": 5,
                "temperature": 0.0
            },
            "gsm8k": {
                "type": "gsm8k",
                "shot_count": 5,
                "use_chain_of_thought": True
            },
            "humaneval": {
                "type": "humaneval",
                "shot_count": 0,
                "temperature": 0.2
            },
            "arc": {
                "type": "arc",
                "shot_count": 3,
                "temperature": 0.0
            },
            "truthfulqa": {
                "type": "truthfulqa",
                "shot_count": 0,
                "temperature": 0.0
            }
        },
        "evaluation": {
            "runs": 3,  # Multiple runs for statistical analysis
            "batch_size": "auto",
            "statistical_tests": True
        },
        "output": {
            "path": "benchmarks_output",
            "visualize": True,
            "export_formats": ["json", "csv", "html"]
        },
        "hardware": {
            "device": "auto",
            "precision": "float16",
            "quantization": True
        }
    }
    
    # Create config directory if it doesn't exist
    config_dir = Path("src/config")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Save advanced configuration
    config_path = config_dir / "advanced_custom_benchmark_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Advanced configuration generated: {config_path}")
    print(f"📝 Advanced configuration includes:")
    print(f"  • Models: {len(config['models'])} models including large ones")
    print(f"  • Tasks: Full benchmark suite")
    print(f"  • Runs: {config['evaluation']['runs']} for statistical analysis")
    print(f"  • Statistical tests: Enabled")
    
    return config_path

if __name__ == "__main__":
    print("🔧 Generating benchmark configurations...")
    print("=" * 50)
    
    # Generate default configuration
    default_config = generate_default_config()
    
    print("\n" + "=" * 50)
    
    # Generate advanced configuration
    advanced_config = generate_advanced_config()
    
    print("\n" + "=" * 50)
    print("🎉 Configuration generation complete!")
    print("\n📋 Usage:")
    print(f"  • Default config: python src/scripts/run_benchmark.py --config {default_config}")
    print(f"  • Advanced config: python src/scripts/run_benchmark.py --config {advanced_config}")
    print("\n💡 Tips:")
    print("  • Start with the default config for quick testing")
    print("  • Use the advanced config for comprehensive evaluation")
    print("  • Enable quantization for efficient GPU usage")
    print("  • Adjust batch sizes based on your hardware") 