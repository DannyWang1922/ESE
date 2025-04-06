import subprocess
import sys

cmd_list = [
    'NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train_self.py --model_name_or_path BAAI/bge-base-en-v1.5  --apply_ese 0 --save_dir result_beg',
    'NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train_self.py --model_name_or_path BAAI/bge-base-en-v1.5  --apply_ese 1 --save_dir result_beg_ese',
]

for cmd in cmd_list:
    print(f"\nRunning cmd: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print(f"Command Done: {cmd}\n")
        print("=" * 100)
    except subprocess.CalledProcessError as e:
        print(f"Command execution failure: {e}")
        print("Error message:")
        print(e.stderr)
