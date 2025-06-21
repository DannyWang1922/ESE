import subprocess
import sys

# Training configuration
nv_cmd = "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0"

# ============ 实验配置区域 - 只需修改这里 ============
# 当前实验的超参数配置
CURRENT_EXPERIMENT = {
    "config": "bge_moe_ese_all.yaml",  # 使用的配置文件
    "epochs": 3,                        # 训练轮数
    "last_layer_loss3_weight": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    
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
    
    # 获取要实验的超参数及其取值
    param_to_test = None
    param_values = None
    for param, values in CURRENT_EXPERIMENT.items():
        if isinstance(values, list):
            param_to_test = param
            param_values = values
            break
    
    if param_to_test is None:
        # 如果没有要测试的超参数，就只运行一次基础配置
        save_dir = f"train_result/{CURRENT_EXPERIMENT['config'].replace('.yaml', '')}"
        cmd = f"{nv_cmd} python train_moe.py --save_dir {save_dir} --config config/{CURRENT_EXPERIMENT['config']} --epochs {CURRENT_EXPERIMENT['epochs']}"
        train_cmd_list.append((cmd, save_dir))
    else:
        # 针对要测试的超参数生成命令
        for value in param_values:
            save_dir = f"train_result/{CURRENT_EXPERIMENT['config'].replace('.yaml', '')}_{param_to_test}_{value}"
            cmd = f"{nv_cmd} python train_moe.py --save_dir {save_dir} --config config/{CURRENT_EXPERIMENT['config']} --epochs {CURRENT_EXPERIMENT['epochs']} --{param_to_test} {value}"
            train_cmd_list.append((cmd, save_dir))
    
    return train_cmd_list

def generate_eval_commands(trained_models):
    eval_cmd_list = []
    for cmd, save_dir in trained_models:
        model_path = f"{save_dir}/best-checkpoint"
        out_dir = f"evl_res/{save_dir.split('/')[-1]}"
        
        # 根据模型类型设置is_moe参数
        if "moe" in save_dir.lower():
            is_moe = "1"
        else:
            is_moe = "0"
        
        cmd = f"{nv_cmd} python eval_nli_main_v2.py --model_name_or_path {model_path} --out_dir {out_dir} --is_moe {is_moe}"
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
    print("\nTraining Commands:")
    # for cmd, _ in train_commands:
    #     print(cmd)
    
    # 运行训练
    run_commands(train_commands)
    
    # 生成并运行评估命令
    eval_commands = generate_eval_commands(train_commands)
    print("\nEvaluation Commands:")
    # for cmd in eval_commands:
    #     print(cmd)
    
    run_commands(eval_commands, is_eval=True)

if __name__ == "__main__":
    main()
