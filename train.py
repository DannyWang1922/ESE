# -*- coding: utf-8 -*-

import json
import os
import logging
import argparse
import random

import numpy as np
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from datasets import load_dataset

from espresso import AnglE, AngleDataTokenizer
from datasets import concatenate_datasets
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Espresso')

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default="BAAI/bge-base-en-v1.5",
                    help='Specify model name or path to set transformer backbone, required')
parser.add_argument('--pretrained_model_path', type=str, default=None,
                    help='Specify pretrained model path to load pretrained model, default None')
parser.add_argument('--pretrained_lora_path', type=str, default=None,
                    help='Specify pretrained lora path to load lora, default None')
parser.add_argument('--train_name_or_path', type=str, default=None,
                    help='Specify huggingface datasets name or local file path for train set, required')
parser.add_argument('--train_subset_name', type=str, default=None,
                    help='Specify huggingface datasets subset name for train set, default None')
parser.add_argument('--train_split_name', type=str, default='train',
                    help='Specify huggingface datasets split name for train set, default `train`')
parser.add_argument('--valid_name_or_path', type=str, default="mteb/stsbenchmark-sts",
                    help='Specify huggingface datasets name or local file path for valid set, default None.')
parser.add_argument('--valid_subset_name', type=str, default=None,
                    help='Specify huggingface datasets subset name for valid set, default None')
parser.add_argument('--valid_split_name', type=str, default='train',
                    help='Specify huggingface datasets split name for valid set, default `train`')
parser.add_argument('--prompt_template', type=str, default=None,
                    help='Specify prompt_template like "xxx: {text}", default None.'
                         'This prompt will be applied for all text columns.'
                         'If you want to specify different prompts for different text columns, please specify it manually.')
parser.add_argument('--save_dir', type=str, default="train_res",
                    help='Specify save dir, default None')
parser.add_argument('--seed', type=int, default=-1,
                    help='Specify random seed, default -1')
parser.add_argument('--dataset_seed', type=int, default=None,
                    help='Specify dataset random seed, default None')
parser.add_argument('--workers', type=int, default=16,
                    help='Specify dataset workers, default 2')
parser.add_argument('--cosine_w', type=float, default=1.0,
                    help='Specify weight for cosine loss, default 1.0')
parser.add_argument('--ibn_w', type=float, default=1.0,
                    help='Specify weight for ibn loss, default 1.0')
parser.add_argument('--angle_w', type=float, default=1.0,
                    help='Specify weight for angle loss, default 1.0')
parser.add_argument('--angle_tau', type=float, default=20.0,
                    help='Specify angle_tau, default 20.0')
parser.add_argument('--cosine_tau', type=float, default=20.0,
                    help='Specify cosine_tau, defaut 20.0')
parser.add_argument('--ibn_tau', type=float, default=20.0,
                    help='Specify ibn_tau, defaut 20.0')
parser.add_argument('--apply_lora', type=int, default=0, choices=[0, 1],
                    help='Specify lora flag, choices [0, 1], default 0')
parser.add_argument('--load_kbit', type=int, default=None, choices=[4, 8, 16],
                    help='Specify kbit training, choices [4, 8, 16], default None')
parser.add_argument('--lora_r', type=int, default=32,
                    help='Specify lora_r, defaut 32')
parser.add_argument('--lora_alpha', type=int, default=32,
                    help='Specify lora_alpha, defaut 32')
parser.add_argument('--lora_dropout', type=float, default=0.1,
                    help='Specify lora_dropout, defaut 0.1')
parser.add_argument('--lora_target_modules', type=str, default=None,
                    help='Specify lora_target_modules. comma serves as the splitter, such as `W,b`. Defaut None')
parser.add_argument('--learning_rate', type=float, default=5e-5,
                    help='Specify learning_rate, defaut 1e-5')
parser.add_argument('--warmup_steps', type=int, default=100,
                    help='Specify warmup_steps, defaut 100')
parser.add_argument('--logging_steps', type=int, default=100,
                    help='Specify logging_steps, defaut 100')
parser.add_argument('--pooling_strategy', type=str, default='cls',
                    help='Specify pooling_strategy from [`cls`, `last`, `avg`, `cls_avg`, `max`], default `cls`')
parser.add_argument('--tokenizer_padding_side', type=str, default=None, choices=['left', 'right'],
                    help='specify tokenizer padding side from [`left`, `right`], default None')
parser.add_argument('--epochs', type=int, default=10, help='Specify epochs, default 10')
parser.add_argument('--max_steps', type=int, default=-1,
                    help='Specify max steps, default -1 (Automatically calculated from epochs)')
parser.add_argument('--save_steps', type=int, default=1000, help='Specify save_steps, default 1000')
parser.add_argument('--batch_size', type=int, default=32, help='Specify batch size, default 32')
parser.add_argument('--maxlen', type=int, default=512, help='Specify max length, default 512')
parser.add_argument('--streaming', action='store_true', default=False,
                    help='Flag to enable streaming mode, default False')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help='Specify gradient_accumulation_steps, default 1')
parser.add_argument('--torch_dtype', type=str, default=None, choices=['auto', 'float32', 'float16', 'bfloat16'],
                    help='Specify torch_dtype from [`auto`, `float32`, `float16`, `bfloat16`], default None')
parser.add_argument('--fp16', type=bool, default=None, choices=[0, 1],
                    help='Specify fp16, choices [0, 1], default None')
parser.add_argument('--push_to_hub', type=int, default=0, choices=[0, 1], help='Specify push_to_hub, default 0')
parser.add_argument('--hub_private_repo', type=int, default=1, choices=[0, 1],
                    help='Specify hub_private_repo, default 1')
parser.add_argument('--hub_model_id', type=str, default=None,
                    help='Specify hub_model_id, default None, format like organization/model_id')
# configure LLM
parser.add_argument('--is_llm', type=int, default=0, choices=[0, 1],
                    help='Specify is_llm, choices [0, 1], defaut 0')
parser.add_argument('--apply_billm', type=int, default=0, choices=[0, 1],
                    help='Specify apply_billm, choices [0, 1], defaut 0')
parser.add_argument('--billm_model_class', type=str, default=None,
                    help='Specify billm model class name, default None')
# configure ESE
parser.add_argument('--apply_ese', type=int, default=1, choices=[0, 1],
                    help='Specify apply_ese to support Espresso Sentence Embedding training, default 0')
parser.add_argument('--ese_kl_temperature', type=float, default=1.0,
                    help='Specify KL temperature for ese, default 1.0')
parser.add_argument('--ese_compression_size', type=int, default=128,
                    help='Specify compression size for ese, default 128')
# configure teacher alignment
parser.add_argument('--teacher_name_or_path', type=str, default=None,
                    help='Specify model_name_or_path for teacher alignment, default None')
parser.add_argument('--teacher_pooling_strategy', type=str, default='cls',
                    help='Specify pooling strategy for teacher from [`cls`, `last`, `avg`, `cls_avg`, `max`], default `cls`')  # NOQA
# configure wandb
parser.add_argument('--wandb_project', type=str, default="ESE", help='Specify WANDB_PROJECT, default None')
parser.add_argument('--wandb_log_model', type=str, default="false", help='Specify WANDB_LOG_MODEL, default None')

parser.add_argument('--config', type=str, default=None, help='Path to YAML config file.')

# Pre-parse config parameters
config_args, remaining_argv = parser.parse_known_args()

# If a config file is specified, read and update the parser's default values
if config_args.config is not None:
    with open(config_args.config, 'r') as f:
        config_data = yaml.safe_load(f)
        for key, value in config_data.items():
            if any(a.dest == key for a in parser._actions):
                parser.set_defaults(**{key: value})
            else:
                logger.warning(f"Unknown config key `{key}` in config file.")

args = parser.parse_args()
# Convert strings "none", "true", "false" to their actual Python types
for arg in vars(args):
    if arg == "wandb_log_model":
        continue  # ignore wandb_log_model
    val = getattr(args, arg)
    if isinstance(val, str) and val.lower() in {"none", "true", "false"}:
        setattr(args, arg, {"none": None, "true": True, "false": False}[val.lower()])
logger.info(f'Args: {args}')

if args.seed is not None and args.seed > 0:
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

if args.wandb_project is not None:
    import wandb
    logger.info('Set up wandb...')
    os.environ['WANDB_PROJECT'] = args.wandb_project
    os.environ['WANDB_LOG_MODEL'] = args.wandb_log_model

    wandb.login()

if args.torch_dtype == 'float32':
    args.torch_dtype = torch.float32
elif args.torch_dtype == 'float16':
    args.torch_dtype = torch.float16
elif args.torch_dtype == 'bfloat16':
    args.torch_dtype = torch.bfloat16

apply_bfloat16 = None
if args.torch_dtype == torch.bfloat16:
    apply_bfloat16 = True

lora_config = {
    'r': args.lora_r,
    'lora_alpha': args.lora_alpha,
    'lora_dropout': args.lora_dropout,
}
if args.lora_target_modules is not None:
    lora_config['target_modules'] = [v.strip() for v in args.lora_target_modules.split(',') if v.strip()]

os.makedirs(args.save_dir, exist_ok=True)
with open(os.path.join(args.save_dir, "parser_para.json"), "w") as f:
    json.dump(vars(args), f, indent=4)

def main():
    model = AnglE(args.model_name_or_path,
                  max_length=args.maxlen,
                  pretrained_model_path=args.pretrained_model_path,
                  pretrained_lora_path=args.pretrained_lora_path,
                  pooling_strategy=args.pooling_strategy,
                  train_mode=True,
                  apply_lora=args.apply_lora,
                  lora_config_kwargs=lora_config,
                  load_kbit=args.load_kbit,
                  torch_dtype=args.torch_dtype,
                  apply_bfloat16=apply_bfloat16,
                  tokenizer_padding_side=args.tokenizer_padding_side,
                  is_llm=args.is_llm,
                  apply_billm=args.apply_billm,
                  billm_model_class=args.billm_model_class)

    if args.train_name_or_path is not None:
        dataset_path_list = [ds for ds in args.train_name_or_path.split(",")]
    else:
        dataset_path_list = ["nyu-mll/multi_nli", "stanfordnlp/snli"]

    all_train_datasets = []
    for dataset_path in dataset_path_list:
        if os.path.exists(dataset_path):
            ds = load_dataset('json',
                            data_files=[dataset_path],
                            num_proc=args.workers,
                            streaming=args.streaming)
        else:
            ds = load_dataset(dataset_path,
                            args.train_subset_name,
                            num_proc=args.workers,
                            streaming=args.streaming)
        ds = ds.map(lambda obj: {"text1": str(obj["premise"]), "text2": str(obj["hypothesis"]), "label": obj["label"]})
        ds = ds.select_columns(["text1", "text2", "label"])
        logger.info(f'Training dataset overview {dataset_path}:')
        all_train_datasets.append(ds[args.train_split_name])
        print(ds)
    
    logger.info('All datasets loaded. Concatenating...')
    if args.streaming:
        from itertools import chain
        combined_dataset = all_train_datasets[0]
        for ds in all_train_datasets[1:]:
            combined_dataset = combined_dataset.concatenate(ds)
    else:
        combined_dataset = concatenate_datasets(all_train_datasets)
    print("Combined_dataset: \n", combined_dataset)

    logger.info('Processing train...')
    train_ds = combined_dataset.shuffle(args.dataset_seed).map(
        AngleDataTokenizer(model.tokenizer, model.max_length, prompt_template=args.prompt_template),
        num_proc=args.workers)
    
    valid_ds = None
    if valid_ds is None and args.valid_name_or_path is not None:
        logger.info('Validation detected, processing validation...')
        if os.path.exists(args.valid_name_or_path):
            valid_ds = load_dataset('json', data_files=[args.valid_name_or_path], num_proc=args.workers)
        else:
            if args.valid_subset_name is not None:
                valid_ds = load_dataset(args.valid_name_or_path, args.valid_subset_name, num_proc=args.workers)
            else:
                valid_ds = load_dataset(args.valid_name_or_path, num_proc=args.workers)
        
        valid_ds = valid_ds.map(lambda obj: {"text1": str(obj["sentence1"]), 
                                             "text2": str(obj['sentence2']), 
                                             "label": float(obj.get("score", 0.0)) / 5.0})  # normalize to [0, 1]})
        valid_ds = valid_ds.select_columns(["text1", "text2", "label"])
        logger.info(f'Test dataset overview {args.valid_name_or_path}:')
        print(valid_ds, "\n")   

        valid_ds = valid_ds[args.valid_split_name].map(
            AngleDataTokenizer(model.tokenizer, model.max_length, prompt_template=args.prompt_template),
            num_proc=args.workers)

    argument_kwargs = {}
    if args.push_to_hub:
        assert args.hub_model_id is not None, 'Please specify hub_mode_id via --hub_model_id xxx'
        argument_kwargs['push_to_hub'] = True
        argument_kwargs['hub_private_repo'] = bool(args.hub_private_repo)
        argument_kwargs['hub_model_id'] = args.hub_model_id
    if args.wandb_project is not None:
        argument_kwargs['report_to'] = 'wandb'
    if args.max_steps > 0:
        argument_kwargs['max_steps'] = args.max_steps

    trainer_kwargs = None
    if args.teacher_name_or_path is not None:
        trainer_kwargs = {
            'teacher_name_or_path': args.teacher_name_or_path,
            'teacher_pooling_strategy': args.teacher_pooling_strategy,
        }
    if args.apply_ese:
        trainer_kwargs = trainer_kwargs or {}
        trainer_kwargs = dict(trainer_kwargs, **{
            'ese_kl_temperature': args.ese_kl_temperature,
            'ese_compression_size': args.ese_compression_size,
        })

    model.fit(
        train_ds=train_ds,
        valid_ds=valid_ds,
        output_dir=args.save_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        loss_kwargs={
            'cosine_w': args.cosine_w,
            'ibn_w': args.ibn_w,
            'angle_w': args.angle_w,
            'cosine_tau': args.cosine_tau,
            'ibn_tau': args.ibn_tau,
            'angle_tau': args.angle_tau,
        },
        fp16=args.fp16,
        argument_kwargs=argument_kwargs,
        apply_ese=args.apply_ese,
        apply_aoe=args.apply_aoe,
        trainer_kwargs=trainer_kwargs,
        eval_steps =args.save_steps,
        save_total_limit = 1,
    )


if __name__ == '__main__':
    main()
