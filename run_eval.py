import subprocess
import sys

base_model  = "models/bert_base" # models/beg_defualt_para, models/uae_github_para, models/qwen_default_para
model_name = base_model.split("/")[-1]
ese_model = base_model+"_ese"

nv_cmd = "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0"

if "bge" in model_name.lower():
    baseline_model = "BAAI/bge-base-en-v1.5"
elif "uae" in model_name.lower():
    baseline_model = "WhereIsAI/UAE-Large-V1"
elif "qwen" in model_name.lower():
    baseline_model = "Qwen/Qwen1.5-0.5B"
elif "bert" in model_name.lower():
    baseline_model = "google-bert/bert-base-uncased"
else:
    raise ValueError(f"Unknown model name: {model_name}")

ese_sts_model_list = ese_model+","+base_model+","+baseline_model

main_out_dir = "evl_res/"+ model_name
plot_out_dir = "evl_res/"+ model_name+"/plot"

if "qwen" not in base_model.lower():
    is_llm = "0"
    pooling_strategy = "cls"
else:
    is_llm = "1"
    pooling_strategy = "mean"

cmd_list = [
        f"python eval_nli_main.py --is_llm {is_llm} --pooling_strategy {pooling_strategy}  --model_name_or_path {base_model} --out_dir {main_out_dir}",
        f"python eval_nli_main.py --is_llm {is_llm} --pooling_strategy {pooling_strategy}  --model_name_or_path {ese_model} --out_dir {main_out_dir}",
        f"python eval_ese_layers.py --is_llm {is_llm} --pooling_strategy {pooling_strategy}  --model_name_or_path_list {ese_sts_model_list} --out_dir {plot_out_dir}"
    ]

for cmd in cmd_list:
    cmd = nv_cmd + " " + cmd
    print(f"\nRunning cmd: {cmd}\n")
    try:
        # subprocess.run(cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print(f"Command Done: {cmd}")
        print("=" * 100)
    except subprocess.CalledProcessError as e:
        print(f"Command execution failure: {e}")
        print("Error message:")
        print(e.stderr)
