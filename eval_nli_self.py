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
import fcntl
import time
import argparse
from prettytable import PrettyTable
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel
from espresso import Pooler
import numpy as np

# Import SentEval
sys.path.insert(0, './SentEval')
import senteval # type: ignore


PATH_TO_DATA = './SentEval/data'


# def evaluate_layers(layer_indices, args, model, tokenizer, backbone, tasks):
#     layer_scores = []

#     def prepare(params, samples):
#         return

#     for layer_index in layer_indices:
#         def batcher(params, batch, max_length=None):
#             if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
#                 batch = [[word.decode('utf-8') for word in s] for s in batch]
#             sentences = [' '.join(s) for s in batch]
#             if args.prompt_template:
#                 sentences = [args.prompt_template.format(text=s) for s in sentences]
#             tok = tokenizer(sentences, padding='longest', max_length=args.max_length, truncation=True, return_tensors='pt')
#             for k, v in tok.items():
#                 tok[k] = v.to(backbone.device)
#             with torch.no_grad():
#                 outputs = model(tok, layer_index=layer_index)
#             outputs = outputs[:, args.embedding_start:]
#             if args.embedding_size is not None:
#                 return outputs[:, :args.embedding_size].float().detach().cpu().numpy()
#             return outputs.float().detach().cpu().numpy()

#         params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 16}
#         params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}

#         results = {}
#         for task in tasks:
#             se = senteval.engine.SE(params, batcher, prepare)
#             result = se.eval(task)
#             if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
#                 score = result['all']['spearman']['all'] * 100
#             else:
#                 score = result['test']['spearman'].correlation * 100
#             results[task] = score

#         avg_score = sum(results.values()) / len(results)
#         layer_scores.append(avg_score)
#         print(f"Layer {layer_index}: Avg STS Score = {avg_score:.2f}")

#     return layer_scores

def evaluate_layers(layer_indices, args, model, tokenizer, backbone, tasks):
    layer_scores = []
    nan_layers = []  

    def prepare(params, samples):
        return

    for layer_index in layer_indices:
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
            if args.embedding_size is not None:
                return outputs[:, :args.embedding_size].float().detach().cpu().numpy()
            return outputs.float().detach().cpu().numpy()

        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 16}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}

        try:
            results = {}
            for task in tasks:
                se = senteval.engine.SE(params, batcher, prepare)
                result = se.eval(task)
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    score = result['all']['spearman']['all'] * 100
                else:
                    score = result['test']['spearman'].correlation * 100
                results[task] = score

            avg_score = sum(results.values()) / len(results)
            
            # 检查avg_score是否为NaN
            if np.isnan(avg_score):
                print(f"Layer {layer_index}: Avg STS Score = NaN (skipping)")
                nan_layers.append(layer_index)
            else:
                layer_scores.append(avg_score)
                print(f"Layer {layer_index}: Avg STS Score = {avg_score:.2f}")
        
        except Exception as e:
            print(f"Error in layer {layer_index}: {str(e)} (skipping)")
            nan_layers.append(layer_index)
    
    # 将问题层记录到文件中
    with open("nan_layers.txt", "w") as f:
        f.write(f"Model: {args.model_name_or_path}\n")
        f.write(f"Total layers: {len(layer_indices)}\n")
        f.write(f"Problematic layers (NaN values): {len(nan_layers)}\n")
        f.write("Layer indices with NaN values: " + ", ".join(map(str, nan_layers)) + "\n")
        
        # 计算问题层的百分比
        nan_percentage = (len(nan_layers) / len(layer_indices)) * 100 if layer_indices else 0
        f.write(f"Percentage of problematic layers: {nan_percentage:.2f}%\n")

    return layer_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_llm", type=int, default=0, choices=[0, 1], help="Whether the model is a LLM. Default: 0")
    parser.add_argument("--pooling_strategy", type=str, default='cls')
    parser.add_argument("--layer_index", type=int, default=-1)
    parser.add_argument("--embedding_start", type=int, default=0)
    parser.add_argument("--embedding_size", type=int, default=None)
    parser.add_argument("--model_name_or_path", type=str, default="BAAI/bge-base-en-v1.5", help="Transformers' model name or path")
    parser.add_argument("--prompt_template", type=str, default="Represent following sentence for general embedding: {text} <|end_of_text|>")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--mode", type=str, choices=['dev', 'test', 'fasttest'], default='test')
    parser.add_argument("--task_set", type=str, choices=['sts', 'transfer', 'full', 'na'], default='sts')
    parser.add_argument('--load_kbit', type=int, choices=[4, 8, 16], default=16)
    parser.add_argument('--avg', action='store_true')
    parser.add_argument('--lora_weight', type=str, default=None)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)

    args = parser.parse_args()
    # print('>>> args:', args)
    if args.pretrained_model_path == 'None':
        args.pretrained_model_path = None

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    if args.is_llm:
        backbone = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, output_hidden_states=True, torch_dtype=torch.float16, device_map='auto').to(device)
    else:
        backbone = AutoModel.from_pretrained(
            args.model_name_or_path, output_hidden_states=True).to(device)

    if args.is_llm and args.lora_weight:
        backbone = PeftModel.from_pretrained(
            backbone,
            args.lora_weight,
            torch_dtype=torch.float16,
            # device_map={'': 0},
            device_map='auto',
        )
        backbone.print_trainable_parameters()

    model = Pooler(backbone, pooling_strategy=args.pooling_strategy)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Get the layers number
    if hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'layer'):
        n_layers = len(backbone.encoder.layer)
    elif hasattr(backbone, 'transformer') and hasattr(backbone.transformer, 'h'):
        n_layers = len(backbone.transformer.h)
    elif hasattr(backbone, 'model') and hasattr(backbone.model, 'layers'):
        n_layers = len(backbone.model.layers)
    else:
        raise ValueError("Cannot determine the number of layers in the model.")

    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]
        if args.prompt_template:
            sentences = [args.prompt_template.format(text=s) for s in sentences]

        if max_length == 500:
            sentences = [tokenizer.decode(tokenizer.encode(s, add_special_tokens=False)[:max_length]) for s in sentences]
            max_length = 512

        if args.is_llm and "llama" in args.model_name_or_path.lower():
            tokenizer.pad_token = tokenizer.eos_token

        tok = tokenizer(
            sentences,
            padding='longest',
            max_length=args.max_length,
            truncation=True,
            return_tensors='pt')
        for k, v in tok.items():
            tok[k] = v.to(backbone.device)
        with torch.no_grad():
            outputs = model(tok, layer_index=args.layer_index)
        outputs = outputs[:,  args.embedding_start:]
        if args.embedding_size is not None:
            return outputs[:, :args.embedding_size].float().detach().cpu().numpy()
        return outputs.float().detach().cpu().numpy()

    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 16}
    params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}

    print("------ %s ------" % (args.mode))
    results = {}
    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # 计算主表格（最后一层）
    task_names = ['Model'] + ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS-B', 'SICK-R', 'Avg.']  # Added 'Avg.' here
    scores = [args.model_name_or_path]
    for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
        scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
    scores.append("%.2f" % (results['STSBenchmark']['test']['spearman'].correlation * 100))
    scores.append("%.2f" % (results['SICKRelatedness']['test']['spearman'].correlation * 100))
    avg = sum([float(s) for s in scores[1:]]) / 7
    scores.append("%.2f" % avg)

    # 计算 ≺ Avg.
    print("\n[≺ Avg.] Calculating average STS performance across all non-final layers...")
    layer_indices = list(range(n_layers - 1))
    layer_scores = evaluate_layers(layer_indices, args, model, tokenizer, backbone, ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness'])
    pavg = sum(layer_scores) / len(layer_scores)
    scores.append("%.2f" % pavg)
    task_names.append("≺ Avg.")

    # 输出总表格
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

if __name__ == "__main__":
    main()