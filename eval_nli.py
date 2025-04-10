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


def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


def lock_and_write_file(file_path, content):
    with open(file_path, 'a') as file:
        while True:
            try:
                # Acquire an exclusive lock (non-blocking)
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Perform your write operations here
                file.write(content + '\n')
                file.flush()

            except IOError as e:
                print("File is locked by another process. Can't write.")
                time.sleep(1)
            finally:
                # Release the lock
                fcntl.flock(file, fcntl.LOCK_UN)
                break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_llm", type=int, default=0, choices=[0, 1], help="Whether the model is a LLM. Default: 0")
    parser.add_argument("--pooling_strategy", type=str, default='cls')
    parser.add_argument("--layer_index", type=int, default=-1)
    parser.add_argument("--embedding_start", type=int, default=0)
    parser.add_argument("--embedding_size", type=int, default=None)
    parser.add_argument("--model_name_or_path", type=str, default="BAAI/bge-base-en-v1.5", help="Transformers' model name or path")
    # parser.add_argument("--prompt_template", type=str,
    #                     default="Represent following sentence for general embedding: {text} <|end_of_text|>",
    #                     help="Prompt template")
    parser.add_argument("--prompt_template", type=str,
                        default=None,
                        help="Prompt template")
    parser.add_argument("--max_length", type=int, default=512, help="max len")
    parser.add_argument("--mode", type=str,
                        choices=['dev', 'test', 'fasttest'],
                        default='test',
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str,
                        choices=['sts', 'transfer', 'full', 'na'],
                        default='sts',
                        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument('--load_kbit', type=int,
                        choices=[4,8,16],
                        default=16,
                        help="Load model in kbit")

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

    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        if args.mode == 'dev':
            args.tasks = ['STSBenchmark-dev']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size': 8}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 32,
                                         'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 16}  # 16
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,  # 64
                                         'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
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

    results = {}
    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark-dev']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        if args.checkpoint_path is not None:
            # evaluate checkpoints on dev
            if os.path.exists(os.path.join(args.checkpoint_path, 'dev_results')):
                max_scores = 0
                with open(os.path.join(args.checkpoint_path, 'dev_results'), 'r') as f:
                    for i in f:
                        max_scores = max(max_scores, float(i.split()[1]))
            else:
                max_scores = 0

            # save best checkpoint
            if float(scores[-1]) >= max_scores:
                import shutil
                if args.lora_weight is not None:
                    shutil.copytree(args.lora_weight, os.path.join(args.checkpoint_path, 'best_model'), dirs_exist_ok=True)
                else:
                    shutil.copytree(args.model_name_or_path, os.path.join(args.checkpoint_path, 'best_model'), dirs_exist_ok=True)

            # log dev results
            with open(os.path.join(args.checkpoint_path, 'dev_results'), 'a') as f:
                prefix = args.mask_embedding_sentence_template if not args.avg else 'avg'
                line = prefix + ' ' +str(scores[-1]) + ' ' + \
                    args.lora_weight if args.lora_weight is not None else args.model_name_or_path
                f.write( line + '\n')

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        if args.task_set in ['transfer', 'full']:
            print_table(task_names, scores)


    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        if args.task_set in ['sts', 'full']:
            print(f'embedding_start={args.embedding_start}')
            print(f'embedding_size={args.embedding_size}')
            print(f'layer_index={args.layer_index}')
            print_table(task_names, scores)
        
        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        if args.task_set in ['transfer', 'full']:
            print_table(task_names, scores)


if __name__ == "__main__":
    main()
