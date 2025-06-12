import subprocess
import sys

# cmd_list = [
#     "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train_moe.py --config config/bge_base.yaml",
#     "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train_moe.py --config config/bge_ese.yaml",
#     "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train_moe.py --config config/bge_moe_base_all.yaml",
#     "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train_moe.py --config config/bge_moe_ese_all.yaml",
#     "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train_moe.py --config config/bge_moe_base_11.yaml",
#     "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python train_moe.py --config config/bge_moe_ese_11.yaml"
# ]

config = "bge_moe_ese_all.yaml"
learning_rate_list = [1e-4, 5e-5, 5e-6, 1e-6]
# num_experts = [16, 8, 4]
# moe_expert_intermediate_size_list = [512, 256]
nv_cmd = "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0"

# cmd_list = []
# for lr in learning_rate_list:
#     for num_expert in num_experts:
#         for moe_expert_intermediate_size in moe_expert_intermediate_size_list:
#             save_dir = f"output/bge_moe_ese_all_{lr}_{num_expert}_{moe_expert_intermediate_size}"
#             cmd = f"{nv_cmd} python train_moe.py --learning_rate {lr} --num_experts {num_expert} --moe_expert_intermediate_size {moe_expert_intermediate_size} --save_dir {save_dir} --config config/{config}"
#             cmd_list.append(cmd)

cmd_list = []
for lr in learning_rate_list:
        save_dir = f"output/bge_moe_ese_all_{lr}"
        cmd = f"{nv_cmd} python train_moe.py --learning_rate {lr} --save_dir {save_dir} --config config/{config}"
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
