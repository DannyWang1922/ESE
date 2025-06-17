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

nv_cmd = "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0"
config_list = ["bge_moe_ese_all.yaml"]

epoch = 5
loss_decay_type_list = [0, 2]
prior_layers_weight_list = [0.1, 0.2]

# loss_decay_type_list = [2]
# prior_layers_weight_list = [0.3, 0.4, 0.5]


cmd_list = []
for config in config_list:
    for loss_decay_type in loss_decay_type_list:
        for prior_layers_weight in prior_layers_weight_list:
            if config == "bge_moe_ese_all.yaml":
                save_dir = f"train_result/bge_moe_ese_all_{loss_decay_type}_{prior_layers_weight}"
            cmd = f"{nv_cmd} python train_moe.py --save_dir {save_dir} --config config/{config}  --epochs {epoch} --loss_decay_type {loss_decay_type} --prior_layers_weight {prior_layers_weight}"
            print(cmd)
            if loss_decay_type == 0:
                break
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
