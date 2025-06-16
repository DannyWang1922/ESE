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

config_list = ["bge_moe_ese_all.yaml"]
# ese_compression_size_list =[32, 64, 128, 256, 512, 640, 768]
# moe_expert_intermediate_size_list = [256, 512]
top_k = 4
epoch = 5
experts_num_list = [4, 8]
# experts_num_list = [16]
nv_cmd = "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0"

cmd_list = []
for config in config_list:
    for experts_num in experts_num_list:
        if config == "bge_moe_ese_all.yaml":
            save_dir = f"train_result/bge_moe_ese_all_{top_k}_{experts_num}"
        cmd = f"{nv_cmd} python train_moe.py --save_dir {save_dir} --config config/{config} --num_experts {experts_num} --epochs {epoch} --top_k {top_k}"
        print(cmd)
        # cmd_list.append(cmd)

# for cmd in cmd_list:
#     print(f"\nRunning cmd: {cmd}\n")
#     try:
#         result = subprocess.run(cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
#         print(result.stdout)
#         print(f"Command Done: {cmd}")
#         print("=" * 100)
#     except subprocess.CalledProcessError as e:
#         print(f"Command execution failure: {cmd}")
#         with open("error_log.txt", "a", encoding="utf-8") as log_file:
#             log_file.write(f"FAILED CMD: {cmd}\n")
#             log_file.write("=" * 100 + "\n")
