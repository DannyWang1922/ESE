import subprocess
import sys

cmd_list = [
    "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train_moe.py --config config/bert_base.yaml",
    "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train_moe.py --config config/bert_moe_base_10.yaml",
    "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train_moe.py --config config/bert_ese.yaml",
    "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train_moe.py --config config/bert_moe_ese_10.yaml",
    "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train_moe.py --config config/bert_moe_base_all.yaml",
    "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train_moe.py --config config/bert_moe_ese_all.yaml",
]

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
            log_file.write(f"RETURN CODE: {e.returncode}\n")
            log_file.write(f"STDOUT:\n{e.stdout}\n")
            log_file.write(f"STDERR:\n{e.stderr}\n")
            log_file.write("=" * 100 + "\n")
