import subprocess
import sys
from turtle import st

# NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=1 python eval_nli_main.py --is_llm 0 --pooling_strategy cls  --batch_size 512 --model_name_or_path output/bge_base_30w/best-checkpoint --out_dir evl_res/bge_base_30w --is_moe 0


# model_name_or_path = ["bge_ese"]
model_name_or_path = ["bge_moe_ese_all"]

# moe_expert_intermediate_size_list = [256, 512]
# ese_compression_size_list =[32, 64, 128, 256, 512, 640, 768]

nv_cmd = "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0"
batch_size = 512
embedding_size_list = "32,64,128,256,512,640,768"
cmd_list = []
loss_decay_type_list = [0, 2]
prior_layers_weight_list = [0.1, 0.2, 0.3, 0.4, 0.5]


for model_name in model_name_or_path:
    for loss_decay_type in loss_decay_type_list:
        for prior_layers_weight in prior_layers_weight_list:
            loss_decay_type = str(loss_decay_type)
            prior_layers_weight = str(prior_layers_weight)

            model_name_or_path = f"train_result/{loss_decay_type}_{prior_layers_weight}/best-checkpoint"
            out_dir = f"evl_res/{model_name}_{loss_decay_type}_{prior_layers_weight}"
            
            plot_out_dir = out_dir+"/plot"

            if "qwen" not in model_name.lower():
                is_llm = "0"
                pooling_strategy = "cls"
            else:
                is_llm = "1"
                pooling_strategy = "mean"

            if "moe" in model_name.lower():
                is_moe = "1"
            else:
                is_moe = "0"
            
            cmd = nv_cmd + f" python eval_nli_main.py --model_name_or_path {model_name_or_path} --out_dir {out_dir} --is_moe {is_moe}"
            # cmd = nv_cmd + f" python eval_ese_layers.py --model_name_or_path {model_name_or_path} --out_dir {plot_out_dir} --is_moe {is_moe} --embedding_size_list {embedding_size_list}"
            print(cmd+ "\n")
            # cmd_list.append(cmd)
            if loss_decay_type == "0":
                break
            


for cmd in cmd_list:
    print(f"\nRunning cmd: {cmd}\n")
    try:
        # subprocess.run(cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print(f"Command Done: {cmd}")
        print("=" * 100)
    except subprocess.CalledProcessError as e:
        print(f"Command execution failure: {e}")
        print("Error message:")
        print(e.stderr)
