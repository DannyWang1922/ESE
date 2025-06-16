import subprocess
import sys
from turtle import st

# NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=1 python eval_nli_main.py --is_llm 0 --pooling_strategy cls  --batch_size 512 --model_name_or_path output/bge_base_30w/best-checkpoint --out_dir evl_res/bge_base_30w --is_moe 0


# model_name_or_path = ["bge_ese"]
model_name_or_path = ["bge_moe_ese_all"]


moe_expert_intermediate_size_list = [256, 512]
ese_compression_size_list =[32, 64, 128, 256, 512, 640, 768]

nv_cmd = "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=1"
batch_size = 512
embedding_size_list = "32,64,128,256,512,640,768"
cmd_list = []

for model_name in model_name_or_path:
    for moe_expert_intermediate_size in moe_expert_intermediate_size_list:
        for ese_compression_size in ese_compression_size_list:
            moe_expert_intermediate_size = str(moe_expert_intermediate_size)
            ese_compression_size = str(ese_compression_size)
            model_name_or_path = f"train_result/{model_name}_{moe_expert_intermediate_size}_{ese_compression_size}/best-checkpoint"
            out_dir = f"evl_res/{model_name}_{moe_expert_intermediate_size}_{ese_compression_size}/plot"
          
            plot_out_dir = "evl_res/"+ model_name+"/plot"

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
            
            # cmd = nv_cmd + f" python eval_nli_main.py --is_llm {is_llm} --pooling_strategy {pooling_strategy}  --batch_size {batch_size} --model_name_or_path {model_name_or_path} --out_dir {out_dir} --is_moe {is_moe}"
            cmd = nv_cmd + f" python eval_ese_layers.py --is_llm {is_llm} --pooling_strategy {pooling_strategy}  --batch_size {batch_size} --model_name_or_path {model_name_or_path} --out_dir {out_dir} --is_moe {is_moe} --embedding_size_list {embedding_size_list}"
            # print(cmd+ "\n")
            cmd_list.append(cmd)
            


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
