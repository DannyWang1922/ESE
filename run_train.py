import subprocess
import sys
import itertools

# Training configuration
nv_cmd = "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=1"

# ============ 实验配置区域 - 只需修改这里 ============
# 当前实验的超参数配置
CURRENT_EXPERIMENT = {
    "config": "bge_moe_ese.yaml",  # 使用的配置文件
    # "config": "uae_base.yaml",  # 使用的配置文件
    "epochs": 1,                        # 训练轮数,
    "top_k": [2],
    "num_experts": [4],
    
    # 当前实验中要尝试的超参数及其取值范围（以下参数根据需要可以注释掉不需要的）
    # "last_layer_loss_weight": [0.05, 0.1],
    # "learning_rate": [5e-6, 1e-6],
    # "loss_decay_type": [0, 2],
    # "prior_layers_weight": [0.6, 0.7, 0.8],
    # "max_train_samples": [1000],
}
# ================================================

def generate_train_commands():
    train_cmd_list = []
    
    # 获取所有要实验的超参数及其取值
    params_to_test = {}
    for param, values in CURRENT_EXPERIMENT.items():
        if isinstance(values, list):
            params_to_test[param] = values
    
    if not params_to_test:
        # 如果没有要测试的超参数，就只运行一次基础配置
        save_dir = f"train_result/{CURRENT_EXPERIMENT['config'].replace('.yaml', '')}"
        cmd = f"{nv_cmd} python train_moe.py --save_dir {save_dir} --config config/{CURRENT_EXPERIMENT['config']} --epochs {CURRENT_EXPERIMENT['epochs']}"
        train_cmd_list.append((cmd, save_dir))
    else:
        # 获取所有参数值的组合
        param_names = list(params_to_test.keys())
        param_values_list = list(params_to_test.values())
        
        # 生成所有可能的参数组合
        for combination in itertools.product(*param_values_list):
            # 构建参数字符串
            param_str = ""
            save_dir_suffix = ""
            
            for param_name, value in zip(param_names, combination):
                param_str += f" --{param_name} {value}"
                save_dir_suffix += f"_{param_name}_{value}"
            
            save_dir = f"train_result/{CURRENT_EXPERIMENT['config'].replace('.yaml', '')}{save_dir_suffix}"
            cmd = f"{nv_cmd} python train_moe.py --save_dir {save_dir} --config config/{CURRENT_EXPERIMENT['config']} --epochs {CURRENT_EXPERIMENT['epochs']}{param_str}"
            train_cmd_list.append((cmd, save_dir))
    
    return train_cmd_list

def generate_eval_commands(trained_models):
    eval_cmd_list = []
    for cmd, save_dir in trained_models:
        model_path = f"{save_dir}/best-checkpoint"
        out_dir = f"evl_res/{save_dir.split('/')[-1]}"

        if "qwen" in model_path.lower():
            pooling_strategy = "last"
            is_llm = 1
        else:
            pooling_strategy = "cls"
            is_llm = 0

        if "moe" in save_dir.lower():
            is_moe = "1"
        else:
            is_moe = "0"
            
        cmd = f"{nv_cmd} python eval_nli_main.py --model_name_or_path {model_path} --out_dir {out_dir} --is_moe {is_moe} --pooling_strategy {pooling_strategy} --is_llm {is_llm}"
        eval_cmd_list.append(cmd)
    return eval_cmd_list

def run_commands(cmd_list, is_eval=False):
    """执行命令列表"""
    for cmd in cmd_list:
        if isinstance(cmd, tuple):
            cmd = cmd[0]  # 提取训练命令
            
        print(f"\nRunning {'evaluation' if is_eval else 'training'} command: {cmd}\n")
        try:
            result = subprocess.run(cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
            print(f"Command completed successfully: {cmd}")
            print("=" * 100)
        except subprocess.CalledProcessError as e:
            print(f"Command execution failure: {cmd}")
            with open("error_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"FAILED CMD: {cmd}\n")
                log_file.write(f"Error message:\n{str(e)}\n")
                log_file.write("=" * 100 + "\n")
            if not is_eval:
                print("Stopping execution due to training failure")
                sys.exit(1)

def main():
    # 生成训练命令
    train_commands = generate_train_commands()
    # print("\nTraining Commands:")
    # for cmd, _ in train_commands:
    #     print(cmd)
    
    # 运行训练
    run_commands(train_commands)
    
    # 生成并运行评估命令
    eval_commands = generate_eval_commands(train_commands)
    # print("\nEvaluation Commands:")
    # for cmd in eval_commands:
    #     print(cmd)

    # 运行评估
    run_commands(eval_commands, is_eval=True)

if __name__ == "__main__":
    main()
