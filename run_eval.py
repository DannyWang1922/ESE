import subprocess
import sys

model_name_or_path  = "models/beg_defult_para"
pooling_strategy = "cls"
is_llm = "0"

model_name = model_name_or_path.split("/")[-1]
best_model = model_name_or_path + "/best"
last_model = model_name_or_path + "/last"

# BAAI/bge-base-en-v1.5", "WhereIsAI/UAE-Large-V1", "Qwen/Qwen1.5-0.5B"
ese_sts_model_list = best_model+","+last_model+","+"BAAI/bge-base-en-v1.5"

main_out_dir = "evl_res/main_" + model_name +"_"
plot_out_dir = "evl_res/plot"


cmd_list = [
    f"python eval_nli_main.py --is_llm {is_llm} --pooling_strategy {pooling_strategy}  --model_name_or_path {best_model} --out_dir {main_out_dir + best_model.split('/')[-1]}",
    f"python eval_nli_main.py --is_llm {is_llm} --pooling_strategy {pooling_strategy}  --model_name_or_path {last_model} --out_dir {main_out_dir + last_model.split('/')[-1]}",
    f"python eval_ese_sts_bench.py --is_llm {is_llm} --pooling_strategy {pooling_strategy}  --model_name_or_path_list {ese_sts_model_list} --out_dir {plot_out_dir}"
]

for cmd in cmd_list:
    print(f"\nRunning cmd: {cmd}")
    try:
        # subprocess.run(cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print(f"Command Done: {cmd}\n")
        print("=" * 100)
    except subprocess.CalledProcessError as e:
        print(f"Command execution failure: {e}")
        print("Error message:")
        print(e.stderr)
