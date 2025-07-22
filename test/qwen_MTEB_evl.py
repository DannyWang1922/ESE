# -*- coding: utf-8 -*-
"""
使用MTEB评估Qwen3-Embedding-0.6B模型在STS任务上的表现
"""

import os
import json
import torch
import torch.nn.functional as F
import argparse
from datetime import datetime
from typing import List, Dict, Any
import mteb
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
from prettytable import PrettyTable


def last_token_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def create_instruction_prompt(task_type, instruction_template, query_text):
    """
    创建带instruction的prompt
    
    Args:
        task_type: 任务类型 (如 "STS", "Retrieval", "Classification")
        instruction_template: instruction模板 (如 "Instruct: {}\nQuery:")
        query_text: 原始查询文本
    
    Returns:
        formatted_prompt: 格式化后的prompt
    """
    # 根据任务类型获取instruction
    if task_type in ["STS", "PairClassification"]:
        instruction = "Retrieve semantically similar text"
    elif task_type in ["Retrieval"]:
        instruction = "Retrieve relevant passages for the given query"
    elif task_type in ["Classification"]:
        instruction = "Classify the given text"
    else:
        instruction = "Generate embedding for the given text"
    
    # 格式化instruction
    formatted_instruction = instruction_template.format(instruction)
    
    # 拼接最终prompt
    final_prompt = f"{formatted_instruction}{query_text}"
    
    return final_prompt


class QwenEmbeddingModel:
    """
    Qwen3-Embedding-0.6B模型的封装类，用于MTEB评估
    """
    
    def __init__(self, model_name: str, device: str, use_instruction: bool = False, 
                 task_type: str = "STS", instruction_template: str = "Instruct: {}\nQuery: "):
        self.model_name = model_name
        self.device = device
        
        self.use_instruction = use_instruction
        self.task_type = task_type
        self.instruction_template = instruction_template

        # 加载模型和tokenizer，使用官方推荐的配置
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left') # 官方设置
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).to(self.device)
        
        print(f"Loading model: {model_name} successfully!")
    
    def encode(self, sentences: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """
        编码句子为embedding向量，使用官方推荐的方法
        
        Args:
            sentences: 待编码的句子列表
            batch_size: 批处理大小
            **kwargs: 其他参数
            
        Returns:
            编码后的embedding矩阵
        """
        all_embeddings = []
        
        # 如果使用instruction，对每个句子应用instruction
        if self.use_instruction:
            processed_sentences = []
            for sentence in sentences:
                processed_sentence = create_instruction_prompt(
                    self.task_type, 
                    self.instruction_template, 
                    sentence
                )
                processed_sentences.append(processed_sentence)
            sentences = processed_sentences
        
        # 分批处理
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            
            # Tokenize，使用官方推荐的配置
            inputs = self.tokenizer(batch_sentences, padding=True, truncation=True, max_length=512,return_tensors="pt")
            inputs.to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


def get_sts_tasks() -> List[str]:
    sts_tasks = ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICK-R"]
    return sts_tasks

def run_mteb_evaluation(model_name: str, 
                       output_dir: str,
                       batch_size: int,
                       device: str,
                       use_instruction: bool = False,
                       task_type: str = "STS",
                       instruction_template: str = "Instruct: {}\nQuery: ") -> List[Any]:
    """
    运行MTEB评估
    
    Args:
        model_name: 模型名称
        output_dir: 输出目录
        batch_size: 批处理大小
        device: 设备类型
        use_instruction: 是否使用instruction
        task_type: 任务类型
        instruction_template: instruction模板
        
    Returns:
        评估结果列表 (TaskResult对象列表)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化模型
    model = QwenEmbeddingModel(
        model_name, 
        device, 
        use_instruction=use_instruction,
        task_type=task_type,
        instruction_template=instruction_template
    )
    
    # 获取STS任务 - 使用新的API
    sts_task_names = get_sts_tasks()
    tasks = mteb.get_tasks(tasks=sts_task_names, languages=["eng"])
    
    print(f"Prepare to evaluate the following STS tasks: {sts_task_names}")
    
    # 创建MTEB评估器 - 使用新的API
    evaluation = mteb.MTEB(tasks=tasks)
    
    # 运行评估 - 使用encode_kwargs传递batch_size
    print("Begin MTEB evaluation...")
    results = evaluation.run(
        model, 
        output_folder=output_dir,
        eval_splits=["test"],  # 使用测试集
        encode_kwargs={'batch_size': batch_size},
        verbosity=2,
        overwrite_results=True
    )
    return results


def print_results_table(results: List[Any], output_dir: str = None, use_instruction: bool = False, 
                       task_type: str = "STS", instruction_template: str = ""):
    """
    打印和保存结果表格
    
    Args:
        results: 评估结果列表 (MTEB返回的是TaskResult对象列表)
        output_dir: 输出目录
        use_instruction: 是否使用了instruction
        task_type: 任务类型
        instruction_template: instruction模板
    """
    # 创建表格
    table = PrettyTable()
    table.field_names = ["Task", "Metric", "Score"]
    
    task_scores = []
    
    # MTEB返回的是TaskResult对象列表
    for task_result in results:
        task_name = task_result.task_name
        scores = task_result.scores
        
        # 获取测试集结果
        test_scores = None
        if "test" in scores:
            test_scores = scores["test"]
        elif "dev" in scores:
            test_scores = scores["dev"]
        else:
            # 如果没有test或dev，取第一个可用的split
            test_scores = list(scores.values())[0] if scores else None
        
        if test_scores:
            # test_scores 是一个列表，包含多个分数字典
            # 我们需要找到主要的分数指标
            main_score = None
            metric_name = "N/A"
            
            # 遍历所有分数字典
            for score_dict in test_scores:
                if isinstance(score_dict, dict):
                    # 寻找主要指标
                    if "main_score" in score_dict:
                        main_score = score_dict["main_score"]
                        metric_name = "Main Score"
                        break
                    elif "spearman" in score_dict:
                        main_score = score_dict["spearman"]
                        metric_name = "Spearman"
                        break
                    elif "pearson" in score_dict:
                        main_score = score_dict["pearson"]
                        metric_name = "Pearson"
                        break
                    else:
                        # 寻找任何数值类型的分数
                        for key, value in score_dict.items():
                            if isinstance(value, (int, float)) and value is not None:
                                main_score = value
                                metric_name = key
                                break
                        if main_score is not None:
                            break
            
            if main_score is not None:
                table.add_row([task_name, metric_name, f"{main_score:.4f}"])
                task_scores.append((task_name, main_score))
    
    # 计算平均分
    if task_scores:
        avg_score = np.mean([score for _, score in task_scores])
        table.add_row(["Average", "Score", f"{avg_score:.4f}"])
    
    # 打印表格
    print("\n" + "="*60)
    print("MTEB STS Tasks Evaluation Results")
    if use_instruction:
        print(f"Instruction Mode: Enabled (Task Type: {task_type})")
    print("="*60)
    print(table)
    
    # 保存结果到文件
    if output_dir:
        result_file = os.path.join(output_dir, "evaluation_summary.txt")
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"MTEB STS Tasks Evaluation Results\n")
            f.write(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: Qwen/Qwen3-Embedding-0.6B\n")
            if use_instruction:
                f.write(f"Instruction Mode: Enabled\n")
                f.write(f"Task Type: {task_type}\n")
                f.write(f"Instruction Template: {instruction_template}\n")
            else:
                f.write(f"Instruction Mode: Disabled\n")
            f.write("="*60 + "\n")
            f.write(str(table))
            f.write("\n")
        
        print(f"\nDetailed results saved to: {result_file}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="使用MTEB评估Qwen3-Embedding-0.6B模型在STS任务上的表现")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="模型名称或路径")
    parser.add_argument("--output_dir", type=str, default="mteb_results", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=128, help="批处理大小")
    parser.add_argument("--use_instruction", type=int, default=1, help="是否使用instruction")
    parser.add_argument("--task_type", type=str, default="STS", help="任务类型 (STS, Retrieval, Classification)")
    parser.add_argument("--instruction_template", type=str, default="Instruct: {}\nQuery: ", help="instruction模板")
    
    args = parser.parse_args()
    
    # 自动检测设备
    if torch.cuda.is_available():
        args.device = "cuda"
    elif torch.backends.mps.is_available():
        args.device = "mps"
    else:
        args.device = "cpu"
    
    print("="*60)
    print("MTEB STS Tasks Evaluation for Qwen3-Embedding-0.6B")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {args.device}")
    if args.use_instruction:
        print(f"Use Instruction: {args.use_instruction}")
        print(f"Task Type: {args.task_type}")
    print("="*60)
    
    try:
        # 运行评估
        results = run_mteb_evaluation(
            model_name=args.model_name,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
            use_instruction=args.use_instruction,
            task_type=args.task_type,
            instruction_template=args.instruction_template
        )
        
        # 打印结果
        print_results_table(results, args.output_dir, args.use_instruction, args.task_type, args.instruction_template)
        
        print(f"\nFull results saved to: {args.output_dir}")
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
