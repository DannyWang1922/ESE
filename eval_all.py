import os
from pathlib import Path
import subprocess


def get_subdirectories(directory_path):
    try:
        path = Path(directory_path)
        subdirs = [item.name for item in path.iterdir() 
                   if item.is_dir() and not item.name.startswith('.')]
        return subdirs
    except (OSError, FileNotFoundError) as e:
        print(f"错误: {e}")
        return []


res_dir = "."  # 当前目录
subdirs = get_subdirectories(res_dir)
print(subdirs)

nv_cmd = "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0"
cmd_list = []
for subdir in subdirs:
    cmd = f"{nv_cmd} python eval_nli_main.py --is_llm 0 --pooling_strategy cls  --batch_size 256 --model_name_or_path train_result/{subdir}/best-checkpoint --out_dir evl_res/{subdir} --is_moe 1"
    print(cmd)
    cmd_list.append(cmd)

for cmd in cmd_list:
    print(f"\nRunning cmd: {cmd}\n")
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print(result.stdout)
        print(f"Command Done: {cmd}")
        print("=" * 100)
    except subprocess.CalledProcessError as e:
        print(f"Command execution failure: {cmd}")
        with open("error_log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"FAILED CMD: {cmd}\n")
            log_file.write("=" * 100 + "\n")