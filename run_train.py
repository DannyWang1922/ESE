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
ese_compression_size_list =[32, 64, 128, 256, 512, 640, 768]
moe_expert_intermediate_size_list = [256, 512]
nv_cmd = "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0"

cmd_list = []
for config in config_list:
    for moe_expert_intermediate_size in moe_expert_intermediate_size_list:
        for ese_compression_size in ese_compression_size_list:
            if config == "bge_base.yaml":
                save_dir = f"train_result/bge_base_{moe_expert_intermediate_size}_{ese_compression_size}"
            elif config == "bge_ese.yaml":
                save_dir = f"train_result/bge_ese_{moe_expert_intermediate_size}_{ese_compression_size}"
            elif config == "bge_moe_ese_all.yaml":
                save_dir = f"train_result/bge_moe_ese_all_{moe_expert_intermediate_size}_{ese_compression_size}"
            cmd = f"{nv_cmd} python train_moe.py --save_dir {save_dir} --config config/{config} --moe_expert_intermediate_size {moe_expert_intermediate_size} --ese_compression_size {ese_compression_size} --moe_expert_compressed_size {ese_compression_size}"
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
