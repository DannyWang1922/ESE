# -*- coding: utf-8 -*-

""" To run it successfully, Please download the senteval data first.

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

# Import SentEval
sys.path.insert(0, './SentEval')
import senteval  # type: ignore

PATH_TO_DATA = './SentEval/data'

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def save_table_as_csv(task_names, scores, out_dir): 
    csv_path = os.path.join(out_dir, "main_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(task_names)
        writer.writerow(scores)

def create_batcher(args, model, tokenizer, backbone, layer_index=None, embedding_size=None):
    """Create a generic batcher function to avoid code duplication"""
    if layer_index is None:
        layer_index = args.layer_index
    if embedding_size is None:
        embedding_size = args.embedding_size
    
    def batcher(params, batch, max_length=None):
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]
        sentences = [' '.join(s) for s in batch]
        
        if args.prompt_template:
            sentences = [args.prompt_template.format(text=s) for s in sentences]

        if args.is_llm and "llama" in args.model_name_or_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            
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

def get_senteval_params():
    """Get standard parameters for SentEval"""
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 16}
    params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}
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

def evaluate_layers(layer_indices, args, model, tokenizer, backbone, tasks):
    """Evaluate the performance of multiple layers"""
    layer_scores = []
    nan_layers = []  

    for layer_index in layer_indices:
        batcher = create_batcher(args, model, tokenizer, backbone, layer_index)
        params = get_senteval_params()
        try:
            results = {}
            for task in tasks:
                se = senteval.engine.SE(params, batcher, prepare)
                score, _ = evaluate_task(se, task)
                results[task] = score
            avg_score = sum(results.values()) / len(results)
            
            if np.isnan(avg_score):
                print(f"Layer {layer_index}: Avg STS Score = NaN (skipping)")
                nan_layers.append(layer_index)
            else:
                layer_scores.append(avg_score)
                print(f"Layer {layer_index}: Avg STS Score = {avg_score:.2f}")
        
        except Exception as e:
            print(f"Error in layer {layer_index}: {str(e)} (skipping)")
            nan_layers.append(layer_index)
    
    # Record problematic layers to file
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "nan_layers.txt"), "w") as f:
        f.write(f"Model: {args.model_name_or_path}\n")
        f.write(f"Total layers: {len(layer_indices)}\n")
        f.write(f"Problematic layers (NaN values): {len(nan_layers)}\n")
        f.write("Layer indices with NaN values: " + ", ".join(map(str, nan_layers)) + "\n")

    return layer_scores

def evaluate_single_layer(args, model, tokenizer, backbone, tasks, layer_index, embedding_size):
    """Evaluate the performance of a single layer and embedding size"""
    batcher = create_batcher(args, model, tokenizer, backbone, layer_index, embedding_size)
    params = get_senteval_params()
    try:
        results = {}
        for task in tasks:
            se = senteval.engine.SE(params, batcher, prepare)
            score, _ = evaluate_task(se, task)
            results[task] = score
        avg_score = sum(results.values()) / len(results)
        return avg_score
    except Exception as e:
        print(f"Error at Layer {layer_index}, Embedding Size {embedding_size}: {e}")
        return float('nan')

def plot_layer_results(results_matrix_list, embedding_sizes, n_layers, out_dir):
    """
    Plot the relationship between embedding sizes and scores for each layer.
    Each results_matrix in results_matrix_list corresponds to one line in the plot.
    """
    # Create a directory to save the plots
    plots_dir = os.path.join(out_dir, "embedding_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    colors = ['#8B4513', '#1E90FF', '#228B22', '#FF4500', '#6A5ACD']  # Define a color palette for different results matrices
    rows = (n_layers + 2) // 3  # Determine the layout of subplots
    cols = min(3, n_layers)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))  # Create subplots for all layers
    if n_layers > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Create a subplot for each layer
    for layer_idx in range(n_layers):
        ax = axes[layer_idx]
        
        # Plot each results_matrix as a separate line in the same layer's subplot
        for i, results_matrix in enumerate(results_matrix_list):
            scores = results_matrix[layer_idx]  # Get the scores for the current layer
            scores = [0 if np.isnan(s) else s for s in scores]  # Replace NaN values with 0
            ax.plot(
                embedding_sizes, scores, marker='s', color=colors[i % len(colors)],
                linestyle='-', markersize=6, linewidth=2, label=f"Results {i + 1}"
            )
        
        ax.set_title(f"Layer = {layer_idx + 1}", fontsize=12)  # Set the title and labels
        
        # Set the y-axis range based on the data
        all_scores = [
            score for results_matrix in results_matrix_list
            for score in results_matrix[layer_idx] if not np.isnan(score)
        ]
        max_score = max(all_scores)
        min_score = min(all_scores)
        y_range = max_score - min_score
        if y_range == 0:  # Avoid range issues when all values are the same
            y_range = 10
        ax.set_ylim([max(0, min_score - 0.1 * y_range), max_score + 0.1 * y_range])

        ax.set_xlim([0, max(embedding_sizes) + 50])  # Set the x-axis range
        
        ax.grid(True, linestyle='--', alpha=0.7)  # Add grid lines for better readability
        
        if layer_idx % cols == 0:  # Add y-axis label only for the leftmost subplot in each row
            ax.set_ylabel("Spearman's", fontsize=12)
        
        if layer_idx == 0:  # Add legend to the first subplot
            ax.legend(fontsize=10)
    
    # Remove unused subplots
    for i in range(n_layers, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()  # Adjust the layout
    plt.savefig(os.path.join(plots_dir, "all_layers_embedding_comparison.png"), dpi=300, bbox_inches='tight')  # Save the combined plot
    
    # Save individual plots for each layer
    for layer_idx in range(n_layers):
        plt.figure(figsize=(6, 4))
        
        for i, results_matrix in enumerate(results_matrix_list):
            scores = results_matrix[layer_idx]
            scores = [0 if np.isnan(s) else s for s in scores]
            
            plt.plot(
                embedding_sizes, scores, marker='s', color=colors[i % len(colors)],
                linestyle='-', markersize=6, linewidth=2, label=f"Results {i + 1}"
            )
        
        plt.title(f"Layer = {layer_idx + 1}", fontsize=12)
        plt.ylabel("Spearman's", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        # Set the y-axis range
        all_scores = [
            score for results_matrix in results_matrix_list
            for score in results_matrix[layer_idx] if not np.isnan(score)
        ]
        max_score = max(all_scores)
        min_score = min(all_scores)
        y_range = max_score - min_score
        if y_range == 0:  # Avoid range issues when all values are the same
            y_range = 10
        plt.ylim([max(0, min_score - 0.1 * y_range), max_score + 0.1 * y_range])
        plt.xlim([0, max(embedding_sizes) + 50])
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"layer_{layer_idx + 1}_embedding_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots have been saved to {plots_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_llm", type=int, default=0, choices=[0, 1], help="Is it a large language model. Default: 0")
    parser.add_argument("--pooling_strategy", type=str, default='cls', help="Pooling strategy")
    parser.add_argument("--layer_index", type=int, default=-1, help="Layer index to evaluate")
    parser.add_argument("--embedding_start", type=int, default=0, help="Embedding start position")
    parser.add_argument("--embedding_size", type=int, default=None, help="Embedding size")
    parser.add_argument("--model_name_or_path", type=str, default="BAAI/bge-base-en-v1.5", help="Model name or path")
    parser.add_argument("--prompt_template", type=str, default="Represent following sentence for general embedding: {text} <|end_of_text|>", help="Prompt template")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--mode", type=str, choices=['dev', 'test', 'fasttest'], default='test', help="Evaluation mode")
    parser.add_argument("--task_set", type=str, choices=['sts', 'transfer', 'full', 'na'], default='sts', help="Task set")
    parser.add_argument('--load_kbit', type=int, choices=[4, 8, 16], default=16, help="Quantization precision")
    parser.add_argument('--avg', action='store_true', help="Compute average score")
    parser.add_argument('--lora_weight', type=str, default=None, help="LoRA weight path")
    parser.add_argument('--pretrained_model_path', type=str, default=None, help="Pretrained model path")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Checkpoint path")
    parser.add_argument('--embedding_sizes', type=str, default="32,64,128,256", help="Comma-separated list of embedding sizes to evaluate")
    parser.add_argument('--multi_embedding_eval', type=int, default=1, help="Enable multi-embedding-size evaluation")
    parser.add_argument('--out_dir', type=str, default="eval_result", help="Directory to save output files") 
    parser.add_argument('--main_table_eval', type=int, default=0, help="Enable multi-embedding-size evaluation")

    args = parser.parse_args()
    if args.pretrained_model_path == 'None':
        args.pretrained_model_path = None
    
    os.makedirs(args.out_dir, exist_ok=True)  
    with open(os.path.join(args.out_dir, "parser_para.json"), "w") as f:
        json.dump(vars(args), f, indent=4)


    device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print("Using device:", device)
    
    # Initialize model, tokenizer, and Pooler
    if args.is_llm:
        backbone = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, output_hidden_states=True, torch_dtype=torch.float16, device_map='auto').to(device)
    else:
        backbone = AutoModel.from_pretrained(
            args.model_name_or_path, output_hidden_states=True).to(device)

    if args.is_llm and args.lora_weight:
        backbone = PeftModel.from_pretrained(
            backbone, args.lora_weight, torch_dtype=torch.float16, device_map='auto',)
        backbone.print_trainable_parameters()

    model = Pooler(backbone, pooling_strategy=args.pooling_strategy)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Get number of model layers
    if hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'layer'):
        n_layers = len(backbone.encoder.layer)
    elif hasattr(backbone, 'transformer') and hasattr(backbone.transformer, 'h'):
        n_layers = len(backbone.transformer.h)
    elif hasattr(backbone, 'model') and hasattr(backbone.model, 'layers'):
        n_layers = len(backbone.model.layers)
    else:
        raise ValueError("Cannot determine the number of layers in the model.")
    
    # Set tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        if args.mode == 'dev':
            args.tasks = ['STSBenchmark-dev']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    
    # Main table
    if args.main_table_eval:
        batcher = create_batcher(args, model, tokenizer, backbone)
        params = get_senteval_params()
        
        results = {}
        for task in args.tasks:
            se = senteval.engine.SE(params, batcher, prepare)
            _, result = evaluate_task(se, task)
            results[task] = result

        task_names = ['Model'] + ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS-B', 'SICK-R', 'Avg.']
        scores = [args.model_name_or_path]
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
            scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
        scores.append("%.2f" % (results['STSBenchmark']['test']['spearman'].correlation * 100))
        scores.append("%.2f" % (results['SICKRelatedness']['test']['spearman'].correlation * 100))
        avg = sum([float(s) for s in scores[1:]]) / (len(scores)-1) # Subtract model name
        scores.append("%.2f" % avg)

        # Compute average performance of non-final layers
        print("\n[≺ Avg.] Computing average STS performance of all non-final layers...")
        layer_indices = list(range(n_layers - 1))
        sts_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        layer_scores = evaluate_layers(layer_indices, args, model, tokenizer, backbone, sts_tasks)
        pavg = sum(layer_scores) / len(layer_scores)
        scores.append("%.2f" % pavg)
        task_names.append("≺ Avg.")

        print_table(task_names, scores)
        save_table_as_csv(task_names, scores, args.out_dir)

    # Multi-embedding-size evaluation
    if args.multi_embedding_eval:
        embedding_sizes = [int(x) for x in args.embedding_sizes.split(",")]
        layer_indices = list(range(n_layers))
        tasks = ['STSBenchmark']

        results_matrix = []
        results_matrix_list = []
        for layer_index in layer_indices:
            row = []
            for emb_size in embedding_sizes:
                score = evaluate_single_layer(args, model, tokenizer, backbone, tasks, layer_index, emb_size)
                print(f"Layer {layer_index}, Embedding Size {emb_size} => Avg Score: {score:.2f}")
                row.append(score)
            results_matrix.append(row)

        # Save results to CSV
        csv_file = os.path.join(args.out_dir, "STS_benchmark_result.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["layer_index"] + [f"emb_{e}" for e in embedding_sizes]
            writer.writerow(header)
            for i, row in enumerate(results_matrix):
                writer.writerow([i] + row)
        print(f"\nEvaluation results saved to {csv_file}")
        results_matrix_list.append(results_matrix)
    
    plot_layer_results(results_matrix_list, embedding_sizes, n_layers, args.out_dir)


if __name__ == "__main__":
    main()