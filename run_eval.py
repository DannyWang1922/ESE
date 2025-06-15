import subprocess
import sys

# NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=1 python eval_nli_main.py --is_llm 0 --pooling_strategy cls  --batch_size 512 --model_name_or_path output/bge_base_30w/best-checkpoint --out_dir evl_res/bge_base_30w --is_moe 0


# model_name_or_path = ["bge_ese"]
model_name_or_path = ["bge_moe_ese_all"]
# lr_rate_list = ["1e-04", "5e-05", "5e-06", "1e-06"]
wd_list = [2, 4, 6, 8]
nv_cmd = "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=1"
batch_size = 512

for model_name in model_name_or_path:
    for lr in  wd_list:
        lr = str(lr)
        model_name_or_path = "train_result/" + model_name + "_" + lr + "/best-checkpoint"
        out_dir = "evl_res/" + model_name + "_" + lr
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
        
        cmd = nv_cmd + f" python eval_nli_main.py --is_llm {is_llm} --pooling_strategy {pooling_strategy}  --batch_size {batch_size } --model_name_or_path {model_name_or_path} --out_dir {out_dir} --is_moe {is_moe}"
        print(cmd+ "\n")
    




# ese_sts_model_list = ese_model+","+base_model+","+baseline_model
# main_out_dir = "evl_res/"+ model_name

# if "bge" in model_name.lower():
#     baseline_model = "BAAI/bge-base-en-v1.5"
# elif "uae" in model_name.lower():
#     baseline_model = "WhereIsAI/UAE-Large-V1"
# elif "qwen" in model_name.lower():
#     baseline_model = "Qwen/Qwen1.5-0.5B"
# elif "bert" in model_name.lower():
#     baseline_model = "google-bert/bert-base-uncased"
# else:
#     raise ValueError(f"Unknown model name: {model_name}")


# cmd_list = [
#         f"python eval_nli_main.py --is_llm {is_llm} --pooling_strategy {pooling_strategy}  --model_name_or_path {base_model} --out_dir {main_out_dir}",
#         f"python eval_nli_main.py --is_llm {is_llm} --pooling_strategy {pooling_strategy}  --model_name_or_path {ese_model} --out_dir {main_out_dir}",
#         f"python eval_ese_layers.py --is_llm {is_llm} --pooling_strategy {pooling_strategy}  --model_name_or_path_list {ese_sts_model_list} --out_dir {plot_out_dir}"
#     ]

# for cmd in cmd_list:
#     cmd = nv_cmd + " " + cmd
#     print(f"\nRunning cmd: {cmd}\n")
#     try:
#         # subprocess.run(cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
#         print(f"Command Done: {cmd}")
#         print("=" * 100)
#     except subprocess.CalledProcessError as e:
#         print(f"Command execution failure: {e}")
#         print("Error message:")
#         print(e.stderr)
