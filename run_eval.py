import subprocess
import sys

# model_name_or_path  = "models/beg_defult_para"
model_name_or_path  = "models/uae_default_para"
# model_name_or_path  = "models/qwen_default_para"

nv_cmd = "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0"

model_name = model_name_or_path.split("/")[-1]
best_model = model_name_or_path + "/best"
last_model = model_name_or_path + "/last"

# BAAI/bge-base-en-v1.5", "WhereIsAI/UAE-Large-V1", "Qwen/Qwen1.5-0.5B"
if "bge" in model_name.lower():
    baseline_model = "BAAI/bge-base-en-v1.5"
elif "uae" in model_name.lower():
    baseline_model = "WhereIsAI/UAE-Large-V1"
elif "qwen" in model_name.lower():
    baseline_model = "Qwen/Qwen1.5-0.5B"
else:
    raise ValueError(f"Unknown model name: {model_name}")

ese_sts_model_list = best_model+","+last_model+","+baseline_model

main_out_dir = "evl_res/"+ model_name+ "/main_" + model_name +"_"
plot_out_dir = "evl_res/"+ model_name+"/plot"

if "qwen" in model_name_or_path.lower():
    is_llm = "1"
    pooling_strategy = "mean"
    prompt_template = "Represent following sentence for general embedding: {text} <|end_of_text|>"
    cmd_list = [
        f"python eval_nli_main.py --is_llm {is_llm} --pooling_strategy {pooling_strategy} --prompt_template {prompt_template} --model_name_or_path {best_model} --out_dir {main_out_dir + best_model.split('/')[-1]}",
        f"python eval_nli_main.py --is_llm {is_llm} --pooling_strategy {pooling_strategy} --prompt_template {prompt_template} --model_name_or_path {last_model} --out_dir {main_out_dir + last_model.split('/')[-1]}",
        f"python eval_ese_sts_bench.py --is_llm {is_llm} --pooling_strategy {pooling_strategy}  --prompt_template {prompt_template} --model_name_or_path_list {ese_sts_model_list} --out_dir {plot_out_dir}"
    ]
else:
    is_llm = "0"
    pooling_strategy = "cls"
    cmd_list = [
        f"python eval_nli_main.py --is_llm {is_llm} --pooling_strategy {pooling_strategy} --model_name_or_path {best_model} --out_dir {main_out_dir + best_model.split('/')[-1]}",
        f"python eval_nli_main.py --is_llm {is_llm} --pooling_strategy {pooling_strategy} --model_name_or_path {last_model} --out_dir {main_out_dir + last_model.split('/')[-1]}",
        f"python eval_ese_sts_bench.py --is_llm {is_llm} --pooling_strategy {pooling_strategy} --model_name_or_path_list {ese_sts_model_list} --out_dir {plot_out_dir}"
    ]

for cmd in cmd_list:
    cmd = nv_cmd + " " + cmd
    print(f"\nRunning cmd: {cmd}\n")
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print(f"Command Done: {cmd}")
        print("=" * 100)
    except subprocess.CalledProcessError as e:
        print(f"Command execution failure: {e}")
        print("Error message:")
        print(e.stderr)
