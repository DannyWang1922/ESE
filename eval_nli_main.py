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
from transformers import AutoConfig

# from modeling.transformers_modeling_qwen2 import Qwen2ForCausalLM
# from transformers import Qwen2ForCausalLM

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

def evaluate_layers(layer_indices, args, model, tokenizer, backbone, tasks):
    """Evaluate the performance of multiple layers"""
    layer_scores = []
    nan_layers = []  

    for layer_index in layer_indices:
        batcher = create_batcher(args, model, tokenizer, backbone, layer_index)
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
        avg_score = sum([float(s) for s in scores]) / (len(scores))
        
        if np.isnan(avg_score):
            print(f"Layer {layer_index}: Avg STS Score = NaN (skipping)")
            nan_layers.append(layer_index)
        else:
            layer_scores.append(avg_score)
            print(f"Layer {layer_index}: Avg STS Score = {avg_score:.2f}")
    
    # Record problematic layers to file
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "nan_layers.txt"), "w") as f:
        f.write(f"Model: {args.model_name_or_path}\n")
        f.write(f"Total layers: {len(layer_indices)}\n")
        f.write(f"Problematic layers (NaN values): {len(nan_layers)}\n")
        f.write("Layer indices with NaN values: " + ", ".join(map(str, nan_layers)) + "\n")

    return layer_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_llm", type=int, default=0, choices=[0, 1], help="Is it a large language model. Default: 0")
    parser.add_argument("--pooling_strategy", type=str, default='cls', help="Pooling strategy")
    parser.add_argument("--layer_index", type=int, default=-1, help="Layer index to evaluate")
    parser.add_argument("--embedding_start", type=int, default=0, help="Embedding start position")
    parser.add_argument("--embedding_size", type=int, default=None, help="Embedding size")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen1.5-0.5B-Chat", help="Model name or path")
    # parser.add_argument("--model_name_or_path", type=str, default="BAAI/bge-base-en-v1.5", help="Model name or path")
    parser.add_argument("--prompt_template", type=str, default="Represent following sentence for general embedding: {text} <|end_of_text|>", help="Prompt template")
    # parser.add_argument("--prompt_template", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--mode", type=str, choices=['dev', 'test', 'fasttest'], default='test', help="Evaluation mode")
    parser.add_argument("--task_set", type=str, choices=['sts', 'transfer', 'full', 'na'], default='sts', help="Task set")
    parser.add_argument('--lora_weight', type=str, default=None, help="LoRA weight path")
    parser.add_argument('--out_dir', type=str, default="evl_res/main", help="Directory to save output files")
    parser.add_argument('--eval_batch_size', type=int, default=256, help="Eavluation batch size")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)  
    with open(os.path.join(args.out_dir, "parser_para.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print("Using device:", device)
    
    # Initialize model, tokenizer, and Pooler
    if args.is_llm:
        backbone = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, output_hidden_states=True, torch_dtype=torch.float16, device_map='auto').to(device)
        # backbone = Qwen2ForCausalLM.from_pretrained(
        #     args.model_name_or_path, output_hidden_states=True, torch_dtype=torch.float16, device_map='auto').to(device)
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
    layer_indices = list(range(1,n_layers))
    sts_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    layer_scores = evaluate_layers(layer_indices, args, model, tokenizer, backbone, sts_tasks)
    pavg = sum(layer_scores) / len(layer_scores)
    scores.append("%.2f" % pavg)
    task_names.append("≺ Avg.")

    print_table(task_names, scores)
    save_table_as_csv(task_names, scores, args.out_dir)

if __name__ == "__main__":
    main()