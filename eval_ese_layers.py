# -*- coding: utf-8 -*-
"""
To run it successfully, Please download the senteval data first.
$ cd SentEval/data/downstream/
$ bash download_dataset.sh
"""

import sys
import os
import logging

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
import torch
import argparse
import numpy as np
from prettytable import PrettyTable
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel
from espresso import Pooler
import json
import csv
import matplotlib.pyplot as plt
from transformers import AutoConfig

from billm import Qwen2ForCausalLM

# Import SentEval
sys.path.insert(0, './SentEval')
import senteval  # type: ignore

PATH_TO_DATA = './SentEval/data'


def create_batcher(args, model, tokenizer, backbone, layer_index=None, embedding_size=None):
    """Create a generic batcher function to avoid code duplication"""

    def batcher(params, batch, max_length=None):
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]
        sentences = [' '.join(s) for s in batch]
        
        if args.prompt_template:
            sentences = [args.prompt_template.format(text=s) for s in sentences]
            
        tok = tokenizer(sentences, padding='longest', max_length=args.max_length, truncation=True, return_tensors='pt')
        for k, v in tok.items():
            tok[k] = v.to(backbone.device)
            
        with torch.no_grad():
            outputs = model(tok, layer_index=layer_index)
            
        outputs = outputs[:, args.embedding_start:]
        if embedding_size is not None:
            return outputs[:, :embedding_size].float().detach().cpu().numpy()
        return outputs.float().detach().cpu().numpy()
        
    return batcher

def get_senteval_params(args):
    """Get standard parameters for SentEval"""
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 16}
    params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': args.eval_batch_size, 'tenacity': 5, 'epoch_size': 4}
    return params

def prepare(params, samples):
    """Empty prepare function, required by SentEval but not used here"""
    return

def evaluate_task(se, task):
    """Evaluate a single task and return the score"""
    result = se.eval(task)
    if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
        score = result['all']['spearman']['all'] * 100
    else:
        score = result['test']['spearman'].correlation * 100
    return score, result

def evaluate_single_layer(args, model, tokenizer, backbone, tasks, layer_index, embedding_size):
    """Evaluate the performance of a single layer and embedding size"""
    batcher = create_batcher(args, model, tokenizer, backbone, layer_index, embedding_size)
    params = get_senteval_params(args)
    results = {}
    for task in tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        _, result = evaluate_task(se, task)
        results[task] = result
    score = results['STSBenchmark']['test']['spearman'].correlation * 100
    score = round(score, 0) 
    
    return score

def plot_layer_results(results_matrix_list, embedding_sizes, layer_indices, out_dir, model_names):
    """
    Plot the relationship between embedding sizes and scores for each real layer index.
    Each results_matrix in results_matrix_list corresponds to one model.
    """
    plots_dir = os.path.join(out_dir, "embedding_plots")
    os.makedirs(plots_dir, exist_ok=True)

    colors = ['#8B4513', '#1E90FF', '#228B22', '#FF4500', '#6A5ACD']
    n_layers = len(layer_indices)
    cols = min(3, n_layers)
    rows = (n_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_layers > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # 计算每一行的 y 轴范围
    row_y_limits = []
    for row in range(rows):
        min_y, max_y = float('inf'), float('-inf')
        for col in range(cols):
            i = row * cols + col
            if i >= n_layers:
                continue
            for results_matrix in results_matrix_list:
                scores = results_matrix[i]
                scores = [s for s in scores if not np.isnan(s)]
                if scores:
                    min_y = min(min_y, min(scores))
                    max_y = max(max_y, max(scores))
        y_range = max_y - min_y
        if y_range == 0:
            y_range = 10
        row_y_limits.append((max(0, min_y - 0.1 * y_range), max_y + 0.1 * y_range))

    for i, layer_idx in enumerate(layer_indices):
        ax = axes[i]
        row = i // cols
        y_min, y_max = row_y_limits[row]

        for j, results_matrix in enumerate(results_matrix_list):
            scores = results_matrix[i]
            scores = [0 if np.isnan(s) else s for s in scores]
            ax.plot(
                embedding_sizes, scores, marker='s', color=colors[j % len(colors)],
                linestyle='-', markersize=6, linewidth=2, label=model_names[j]
            )

        ax.set_title(f"Layer = {layer_idx}", fontsize=12)
        ax.set_ylim([y_min, y_max])
        ax.set_xlim([0, max(embedding_sizes) + 50])
        ax.grid(True, linestyle='--', alpha=0.5)

        if i % cols == 0:
            ax.set_ylabel("Spearman's", fontsize=10)
        if i == 0:
            ax.legend(fontsize=9)

    for i in range(n_layers, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "all_layers_embedding_comparison.png"), dpi=300, bbox_inches='tight')

    # Save individual plots
    for i, layer_idx in enumerate(layer_indices):
        plt.figure(figsize=(6, 4))
        for j, results_matrix in enumerate(results_matrix_list):
            scores = results_matrix[i]
            scores = [0 if np.isnan(s) else s for s in scores]
            plt.plot(
                embedding_sizes, scores, marker='s', color=colors[j % len(colors)],
                linestyle='-', markersize=6, linewidth=2, label=model_names[j]
            )

        plt.title(f"Layer = {layer_idx}", fontsize=12)
        plt.ylabel("Spearman's", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)

        all_scores = [
            score for results_matrix in results_matrix_list
            for score in results_matrix[i] if not np.isnan(score)
        ]
        max_score = max(all_scores)
        min_score = min(all_scores)
        y_range = max_score - min_score
        if y_range == 0:
            y_range = 10
        plt.ylim([max(0, min_score - 0.1 * y_range), max_score + 0.1 * y_range])
        plt.xlim([0, max(embedding_sizes) + 50])

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"layer_{layer_idx}_embedding_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    print(f"Plots have been saved to {plots_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_llm", type=int, default=0, choices=[0, 1], help="Is it a large language model. Default: 0")
    parser.add_argument("--pooling_strategy", type=str, default='cls', help="Pooling strategy")
    parser.add_argument("--layer_size", type=int, default=12, help="Number of layers to evaluate")
    parser.add_argument("--embedding_start", type=int, default=0, help="Embedding start position")
    parser.add_argument("--model_name_or_path_list", type=str, default=None, help="Comma-separated list of model names or paths")
    # parser.add_argument("--prompt_template", type=str, default="Represent following sentence for general embedding: {text} <|end_of_text|>", help="Prompt template")
    parser.add_argument("--prompt_template", type=str, default=None, help="Prompt template")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument('--lora_weight', type=str, default=None, help="LoRA weight path")
    parser.add_argument('--embedding_size_list', type=str, default="75,150,225,300,375,450,525,600,675,750", help="Comma-separated list of embedding sizes to evaluate")
    parser.add_argument('--out_dir', type=str, default="evl_res/plot", help="Directory to save output files")
    parser.add_argument('--eval_batch_size', type=int, default=32, help="Eavluation batch size")

    args = parser.parse_args()

    # Convert strings "none", "true", "false" to their actual Python types
    for arg in vars(args):
        val = getattr(args, arg)
        if isinstance(val, str) and val.lower() in {"none", "true", "false"}:
            setattr(args, arg, {"none": None, "true": True, "false": False}[val.lower()])

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "parser_para.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print("Using device:", device)

    # 获取模型列表
    if args.model_name_or_path_list:
        model_names = args.model_name_or_path_list.split(",")
    else:
        # model_names = ["WhereIsAI/ese-qwen-0.5b-nli", "Qwen/Qwen1.5-0.5B"]
        model_names = ["models/github_para_best", "models/github_para_last", "BAAI/bge-base-en-v1.5"]
    model_results_matrix = []

    for model_name in model_names:
        print(f"\nEvaluating model: {model_name}")
        model_out_dir = os.path.join(args.out_dir, model_name.split("/")[-1].replace("-", "_"))
        os.makedirs(model_out_dir, exist_ok=True)

        # Initialize model, tokenizer, and Pooler
        if args.is_llm:
            if "qwen" in model_name.lower():
                backbone = Qwen2ForCausalLM.from_pretrained(
                    model_name, output_hidden_states=True, torch_dtype=torch.float16, device_map='auto').to(device)
            else:
                backbone = AutoModelForCausalLM.from_pretrained(
                    model_name, output_hidden_states=True, torch_dtype=torch.float16, device_map='auto').to(device)
        else:
            backbone = AutoModel.from_pretrained(
                model_name, output_hidden_states=True).to(device)

        if args.is_llm and args.lora_weight:
            backbone = PeftModel.from_pretrained(
                backbone, args.lora_weight, torch_dtype=torch.float16, device_map='auto')
            backbone.print_trainable_parameters()

        model = Pooler(backbone, pooling_strategy=args.pooling_strategy)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if args.is_llm and "llama" in model_name.lower():
            tokenizer.pad_token = tokenizer.eos_token

        # Get number of model layers
        if hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'layer'):
            n_layers = len(backbone.encoder.layer)
        elif hasattr(backbone, 'transformer') and hasattr(backbone.transformer, 'h'):
            n_layers = len(backbone.transformer.h)
        elif hasattr(backbone, 'model') and hasattr(backbone.model, 'layers'):
            n_layers = len(backbone.model.layers)
        else:
            config = AutoConfig.from_pretrained(model_name)
            n_layers = config.num_hidden_layers
        if n_layers is None:
            raise ValueError("Cannot determine the number of layers in the model.")

        # Multi-embedding-size evaluation
        embedding_sizes = [int(x) for x in args.embedding_size_list.split(",")]
        layer_indices = list(range(n_layers-args.layer_size+1, n_layers+1))
        tasks = ['STSBenchmark']

        results_matrix = []
        for layer_index in layer_indices:
            row = []
            for emb_size in embedding_sizes:
                score = evaluate_single_layer(args, model, tokenizer, backbone, tasks, layer_index, emb_size)
                print(f"Model {model_name}, Layer {layer_index}, Embedding Size {emb_size} => Avg Score: {score:.2f}")
                row.append(score)
            results_matrix.append(row)

        # Save results to CSV
        csv_file = os.path.join(model_out_dir, "STS_benchmark_result.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["layer_index"] + [f"emb_{e}" for e in embedding_sizes]
            writer.writerow(header)
            for i, row in enumerate(results_matrix):
                writer.writerow([layer_indices[i]] + row)
        print(f"\nEvaluation results for model {model_name} saved to {csv_file}")
        model_results_matrix.append(results_matrix)

    # Plot results
    plot_layer_results(model_results_matrix, embedding_sizes, layer_indices, args.out_dir, model_names)

if __name__ == "__main__":
    main()