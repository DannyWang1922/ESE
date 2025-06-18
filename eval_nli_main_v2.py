# -*- coding: utf-8 -*-

""" To run it successfully, Please download the senteval data first.

$ cd SentEval/data/downstream/
$ bash download_dataset.sh
"""

import sys
import os
import logging
from datetime import datetime

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
from transformers import AutoConfig
from modeling.modeling_bert_moe import BertMoEModel

# from billm import Qwen2ForCausalLM

# Import SentEval
sys.path.insert(0, './SentEval')
import senteval  # type: ignore

PATH_TO_DATA = './SentEval/data'

def print_table(task_names, scores, out_dir=None):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)
    
    if out_dir:  # save to txt file
        result_file_path = os.path.join(out_dir, "main_result.txt")
        with open(result_file_path, "w") as f:
            f.write(f"Evaluation Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n") # Add timestamps and separators
            f.write("="*80 + "\n\n")
            f.write(str(tb))
            f.write("\n")
        print(f"\nMain results saved to: {result_file_path}")

def save_table_with_detailed_scores(task_names, scores, layer_default_scores, layer_best_scores, layer_all_scores, out_dir):
    """Save CSV file containing detailed score information"""
    csv_path = os.path.join(out_dir, "main_table_with_sizes.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(task_names)
        writer.writerow(scores)
        writer.writerow([])
        writer.writerow(["Layer", "Default Score", "Best Score", "Best Size"])
        
        for i, (default_score, best_score) in enumerate(zip(layer_default_scores, layer_best_scores)):
            layer_idx = i + 1  # layer_indices 从 1 开始
            # 找到最佳分数对应的 size
            best_size = None
            for size, score in layer_all_scores.get(layer_idx, []):
                if abs(score - best_score) < 0.01:  # 浮点数比较
                    best_size = size
                    break
            writer.writerow([f"Layer {layer_idx}", f"{default_score:.2f}", f"{best_score:.2f}", best_size])

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

def get_senteval_params(args):
    """Get standard parameters for SentEval"""
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': args.batch_size}
    params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': args.batch_size*4, 'tenacity': 5, 'epoch_size': 4}
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

def evaluate_layers_with_sizes(layer_indices, embedding_sizes, args, model, tokenizer, backbone, tasks):
    """评估多个层在不同 embedding_size 下的性能"""
    layer_best_scores = []  # Store the best score for each layer
    layer_all_scores = {}   # Store all score for each layer
    layer_default_scores = []  # Store the full embedding size score for each layer
    nan_layers = []
    
    # Assuming that the last embedding is the complete embedding size
    full_embedding_size = embedding_sizes[-1]

    os.makedirs(args.out_dir, exist_ok=True)
    score_file_path = os.path.join(args.out_dir, "layer_scores_with_sizes.txt")

    with open(score_file_path, "w") as score_file:
        score_file.write(f"Model: {args.model_name_or_path}\n")
        score_file.write(f"Embedding sizes: {embedding_sizes}\n")
        score_file.write(f"Full embedding size: {full_embedding_size}\n\n")

        for layer_index in layer_indices:
            layer_scores_for_sizes = []
            best_score = -1
            best_size = None
            default_score = None
            
            for emb_size in embedding_sizes:
                batcher = create_batcher(args, model, tokenizer, backbone, layer_index, emb_size)
                params = get_senteval_params(args)
                scores = []
                results = {}
                
                for task in tasks:
                    se = senteval.engine.SE(params, batcher, prepare)
                    _, result = evaluate_task(se, task)
                    results[task] = result

                for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                scores.append("%.2f" % (results['STSBenchmark']['test']['spearman'].correlation * 100))
                scores.append("%.2f" % (results['SICKRelatedness']['test']['spearman'].correlation * 100))
                avg_score = sum([float(s) for s in scores]) / len(scores)
                
                if not np.isnan(avg_score):
                    layer_scores_for_sizes.append((emb_size, avg_score))
                    if avg_score > best_score:
                        best_score = avg_score
                        best_size = emb_size
                    # Record the score for the full embedding size
                    if emb_size == full_embedding_size:
                        default_score = avg_score
                    print(f"Layer {layer_index}, Size {emb_size}: Avg STS Score = {avg_score:.2f}")
                    score_file.write(f"Layer {layer_index}, Size {emb_size}: Scores = {scores}, Avg = {avg_score:.2f}\n")
                else:
                    print(f"Layer {layer_index}, Size {emb_size}: Avg STS Score = NaN (skipping)")
                    score_file.write(f"Layer {layer_index}, Size {emb_size}: NaN\n")
            
            if best_score > -1:
                layer_best_scores.append(best_score)
                layer_all_scores[layer_index] = layer_scores_for_sizes
                if default_score is not None:
                    layer_default_scores.append(default_score)
                score_file.write(f"Layer {layer_index} Best: Size {best_size}, Score = {best_score:.2f}\n")
                score_file.write(f"Layer {layer_index} Default (size {full_embedding_size}): Score = {default_score:.2f}\n\n")
            else:
                nan_layers.append(layer_index)
                score_file.write(f"Layer {layer_index}: All NaN\n\n")

    return layer_best_scores, layer_all_scores, layer_default_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_llm", type=int, default=0, choices=[0, 1], help="Is it a large language model. Default: 0")
    parser.add_argument("--pooling_strategy", type=str, default='cls', help="Pooling strategy")
    parser.add_argument("--layer_index", type=int, default=-1, help="Layer index to evaluate")
    parser.add_argument("--embedding_start", type=int, default=0, help="Embedding start position")
    parser.add_argument("--embedding_size", type=int, default=None, help="Embedding size")
    parser.add_argument("--model_name_or_path", type=str, default="BAAI/bge-base-en-v1.5", help="Model name or path") # Qwen/Qwen1.5-0.5B, WhereIsAI/ese-qwen-0.5b-nli, BAAI/bge-base-en-v1.5, WhereIsAI/UAE-Large-V1 
    parser.add_argument("--prompt_template", type=str, default=None)
    # parser.add_argument("--prompt_template", type=str, default="Represent following sentence for general embedding: {text} <|end_of_text|>", help="Prompt template")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--mode", type=str, choices=['dev', 'test', 'fasttest'], default='test', help="Evaluation mode")
    parser.add_argument("--task_set", type=str, choices=['sts', 'transfer', 'full', 'na'], default='sts', help="Task set")
    parser.add_argument('--lora_weight', type=str, default=None, help="LoRA weight path")
    parser.add_argument('--out_dir', type=str, default="evl_res/main", help="Directory to save output files")
    parser.add_argument('--batch_size', type=int, default=512, help="Eavluation batch size")
    parser.add_argument('--is_moe', type=int, default=1, help="Eavluation batch size")
    parser.add_argument('--embedding_size_list', type=str, default="32,64,128,256,512,640,768", 
                        help="Comma-separated list of embedding sizes to evaluate")

    args = parser.parse_args()
    
    # Convert strings "none", "true", "false" to their actual Python types
    for arg in vars(args):
        val = getattr(args, arg)
        if isinstance(val, str) and val.lower() in {"none", "true", "false"}:
            setattr(args, arg, {"none": None, "true": True, "false": False}[val.lower()])

    # 解析 embedding_size_list
    embedding_sizes = [int(x.strip()) for x in args.embedding_size_list.split(',')]

    os.makedirs(args.out_dir, exist_ok=True)  
    with open(os.path.join(args.out_dir, "parser_para.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print("Using device:", device)
    
    # Initialize model, tokenizer, and Pooler
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    if args.is_moe:
        backbone = BertMoEModel.from_pretrained(
                args.model_name_or_path, output_hidden_states=True, torch_dtype=torch.float16, device_map='auto').to(device)
    else:
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
    
    
    # Get number of model layers
    if hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'layer'):
        n_layers = len(backbone.encoder.layer)
    elif hasattr(backbone, 'transformer') and hasattr(backbone.transformer, 'h'):
        n_layers = len(backbone.transformer.h)
    elif hasattr(backbone, 'model') and hasattr(backbone.model, 'layers'):
        n_layers = len(backbone.model.layers)
    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        n_layers = config.num_hidden_layers
    if n_layers is None:
        raise ValueError("Cannot determine the number of layers in the model.")
    print("Number of layers in the model:", n_layers)
    
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
    batcher = create_batcher(args, model, tokenizer, backbone)
    params = get_senteval_params(args)
    
    results = {}
    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        _, result = evaluate_task(se, task)
        results[task] = result

    task_names = ['Model', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS-B', 'SICK-R', 'Avg.']
    scores = [args.model_name_or_path]
    for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
        scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
    scores.append("%.2f" % (results['STSBenchmark']['test']['spearman'].correlation * 100))
    scores.append("%.2f" % (results['SICKRelatedness']['test']['spearman'].correlation * 100))
    avg = sum([float(s) for s in scores[1:]]) / (len(scores)-1) # Subtract model name
    scores.append("%.2f" % avg)

    # Calculate the average performance of each layer under different embeddings sizes
    print("\n[Max Avg. & ≺ Avg.] Computing STS performance across embedding sizes for all non-final layers...")
    layer_indices = list(range(1, n_layers))  # Exclude the last layer
    sts_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    
    # Obtain the best score and default score for each layer
    layer_best_scores, layer_all_scores, layer_default_scores = evaluate_layers_with_sizes(
        layer_indices, embedding_sizes, args, model, tokenizer, backbone, sts_tasks
    )
    
    # Calculate Max Avg. (the average of the best embeddings size results for non final layers)
    max_avg = sum(layer_best_scores) / len(layer_best_scores) if layer_best_scores else 0
    
    # Calculate ≺ Avg. (using the average of the full embeddings size)
    pavg = sum(layer_default_scores) / len(layer_default_scores) if layer_default_scores else 0
    
    print(f"\n≺ Avg. (default embedding size): {pavg:.2f}")
    print(f"Max Avg. (best embedding size per layer): {max_avg:.2f}")
    
    # Update Table - Add Two New Columns
    scores.append("%.2f" % pavg)
    scores.append("%.2f" % max_avg)
    task_names.extend(['≺ Avg.', 'Max Avg.'])

    # Ensure that the lengths of scores and task_name are consistent
    assert len(scores) == len(task_names), f"Length mismatch: scores={len(scores)}, task_names={len(task_names)}"
    print_table(task_names, scores, args.out_dir)
    save_table_with_detailed_scores(task_names, scores, layer_default_scores, layer_best_scores, layer_all_scores, args.out_dir)

if __name__ == "__main__":
    main()