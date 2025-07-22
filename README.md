# ESE (Embedding Size Exploration) Project

This project implements and evaluates embedding models with various configurations including MoE (Mixture of Experts), ESE (Embedding Size Exploration), and different backbone models. The project provides comprehensive training, evaluation, and visualization tools for sentence embedding models.

## 📁 Project Structure

```
ESE/
├── train_moe.py              # Main training entry point
├── run_train.py              # Automated training and evaluation script
├── eval_nli_main.py          # General evaluation script
├── eval_nli_main_v2.py       # Evaluation with Max Avg functionality
├── eval_ese_layers.py        # Evaluation with plotting capabilities
├── config/                   # Configuration files
│   ├── bge_base.yaml
│   ├── bge_ese.yaml
│   ├── bge_moe_ese.yaml
│   ├── qwen_base.yaml
│   ├── qwen_moe_ese.yaml
│   ├── uae_base.yaml
│   ├── uae_ese.yaml
│   └── uae_moe_ese.yaml
├── modeling/                 # Custom model implementations
│   ├── configuration_bert_moe.py
│   ├── configuration_qwen_moe.py
│   ├── modeling_bert_moe.py
│   └── modeling_qwem_moe.py
├── SentEval/                # SentEval evaluation framework
└── requirements.txt          # Python dependencies
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download SentEval datasets
cd SentEval/data/downstream/
bash download_dataset.sh
cd ../../..
```

### 2. Basic Training

```bash
# Train with default configuration
python train_moe.py --config config/bge_base.yaml

# Train with custom parameters
python train_moe.py --config config/bge_moe_ese.yaml --epochs 5 --batch_size 32
```

### 3. Automated Training and Evaluation

```bash
# Run automated training and evaluation pipeline
python run_train.py
```

## 📋 Main Components

### 1. `train_moe.py` - Main Training Entry Point

The primary training script that supports:
- **Multiple backbone models**: BGE, Qwen, UAE
- **MoE (Mixture of Experts)**: Multi-expert architectures
- **ESE (Embedding Size Exploration)**: Dynamic embedding size optimization
- **LoRA fine-tuning**: Parameter-efficient training
- **Multiple loss functions**: Cosine, IBN, Angle losses

**Key Features:**
- YAML configuration support
- Multi-GPU training
- Gradient accumulation
- Model checkpointing
- Comprehensive logging

**Usage:**
```bash
python train_moe.py --config config/bge_moe_ese.yaml --epochs 5 --batch_size 32
```

### 2. `run_train.py` - Automated Training Pipeline

Automates the complete training and evaluation workflow:
- **Hyperparameter grid search**: Systematic parameter exploration
- **Batch training**: Multiple model configurations
- **Automatic evaluation**: Post-training assessment
- **Error handling**: Robust execution with logging

**Configuration:**
```python
CURRENT_EXPERIMENT = {
    "config": "bge_moe_ese.yaml",
    "epochs": 1,
    "top_k": [2],
    "num_experts": [4],
    # Add more hyperparameters to test
}
```

### 3. `eval_nli_main.py` - General Evaluation

Standard evaluation script for:
- **SentEval tasks**: STS, NLI, classification tasks
- **Layer-wise evaluation**: Performance across different layers
- **Multiple pooling strategies**: CLS, last, average, max
- **LLM support**: Large language model evaluation

**Usage:**
```bash
python eval_nli_main.py --model_name_or_path ./train_result/model --out_dir ./eval_results
```

### 4. `eval_nli_main_v2.py` - Enhanced Evaluation with Max Avg

Extended evaluation with:
- **Max Average functionality**: Optimized embedding size selection
- **Detailed scoring**: Layer-wise performance analysis
- **Size optimization**: Automatic embedding dimension tuning
- **Comprehensive reporting**: Detailed CSV outputs

**Features:**
- Automatic embedding size optimization
- Layer-wise performance tracking
- Best size identification per layer
- Detailed score reporting

### 5. `eval_ese_layers.py` - Visualization and Analysis

Advanced evaluation with plotting capabilities:
- **Performance visualization**: Layer-wise score plots
- **Embedding size analysis**: Size vs. performance relationships
- **Multi-model comparison**: Comparative analysis
- **Interactive plots**: Matplotlib-based visualizations

**Features:**
- Embedding size vs. performance plots
- Layer-wise performance analysis
- Multi-model comparison charts
- Automated plot generation

## ⚙️ Configuration Files

The `config/` directory contains YAML configuration files for different model setups:

### Base Configurations
- `bge_base.yaml`: Standard BGE model training
- `qwen_base.yaml`: Standard Qwen model training  
- `uae_base.yaml`: Standard UAE model training

### ESE Configurations
- `bge_ese.yaml`: BGE with Embedding Size Exploration
- `qwen_ese.yaml`: Qwen with Embedding Size Exploration
- `uae_ese.yaml`: UAE with Embedding Size Exploration

### MoE Configurations
- `bge_moe_ese.yaml`: BGE with MoE and ESE
- `qwen_moe_ese.yaml`: Qwen with MoE and ESE
- `uae_moe_ese.yaml`: UAE with MoE and ESE

## 🔧 Model Architectures

### MoE (Mixture of Experts)
- **Multi-expert routing**: Dynamic expert selection
- **Gating mechanism**: Learned routing decisions
- **Parameter efficiency**: Shared vs. expert parameters

### ESE (Embedding Size Exploration)
- **Dynamic sizing**: Adaptive embedding dimensions
- **Compression**: KL-based size reduction
- **Performance optimization**: Size vs. quality trade-offs

### Supported Backbones
- **BGE**: BAAI/bge-base-en-v1.5
- **Qwen**: Qwen2.5-7B-Instruct
- **UAE**: UAE-Large-V1

## 📊 Evaluation Tasks

The project evaluates models on SentEval tasks:

### Semantic Textual Similarity (STS)
- STS12, STS13, STS14, STS15, STS16
- STSBenchmark

### Natural Language Inference (NLI)
- SNLI
- SICK

### Classification Tasks
- MR (Movie Review)
- CR (Customer Review)
- SUBJ (Subjectivity)
- MPQA
- SST (Stanford Sentiment Treebank)

### Other Tasks
- TREC (Question Classification)
- MRPC (Microsoft Research Paraphrase Corpus)

## 🎯 Usage Examples

### 1. Basic Training
```bash
# Train BGE model with ESE
python train_moe.py --config config/bge_ese.yaml --epochs 5

# Train Qwen model with MoE
python train_moe.py --config config/qwen_moe_ese.yaml --epochs 3
```

### 2. Automated Experimentation
```bash
# Run automated training pipeline
python run_train.py
```

### 3. Model Evaluation
```bash
# Standard evaluation
python eval_nli_main.py --model_name_or_path ./train_result/model --out_dir ./eval_results

# Enhanced evaluation with Max Avg
python eval_nli_main_v2.py --model_name_or_path ./train_result/model --out_dir ./eval_results

# Evaluation with visualization
python eval_ese_layers.py --model_name_or_path ./train_result/model --out_dir ./eval_results
```

## 📈 Results and Outputs

### Training Outputs
- **Model checkpoints**: Best and latest checkpoints
- **Training logs**: Detailed training progress
- **Configuration backups**: Experiment settings

### Evaluation Outputs
- **Performance tables**: PrettyTable formatted results
- **CSV files**: Detailed score breakdowns
- **Visualization plots**: Performance analysis charts
- **Error logs**: Failed experiment tracking

## 🔍 Key Features

### Advanced Training Features
- **Gradient accumulation**: Memory-efficient training
- **Mixed precision**: FP16 training support
- **LoRA integration**: Parameter-efficient fine-tuning
- **Multi-loss training**: Combined loss functions

### Evaluation Features
- **Layer-wise analysis**: Performance across layers
- **Size optimization**: Automatic embedding dimension tuning
- **Multi-model comparison**: Comparative analysis
- **Visualization**: Performance plotting

### Experimental Features
- **Hyperparameter search**: Automated parameter exploration
- **Batch experimentation**: Multiple configuration testing
- **Error recovery**: Robust experiment execution
- **Comprehensive logging**: Detailed experiment tracking

## 🛠️ Dependencies

Key dependencies include:
- `torch>=2.6.0`: PyTorch framework
- `transformers>=4.51.3`: Hugging Face transformers
- `datasets>=2.21.0`: Dataset handling
- `matplotlib>=3.10.1`: Visualization
- `prettytable>=3.16.0`: Result formatting
- `sentence-transformers>=5.0.0`: Sentence embedding utilities

See `requirements.txt` for complete dependency list.

## 📝 License

This project is part of the ESE (Embedding Size Exploration) research initiative.

## 🤝 Contributing

For questions or contributions, please refer to the project documentation or contact the development team.

---

**Note**: Make sure to download the SentEval datasets before running evaluations:
```bash
cd SentEval/data/downstream/
bash download_dataset.sh
``` 