import subprocess
import sys
from turtle import st

model_name_or_path = ["bge_moe_ese_all"]
nv_cmd = "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0"
batch_size = 512
embedding_size_list = "32,64,128,256,512,640,768"
cmd_list = []
# loss_decay_type_list = [0, 2]
# prior_layers_weight_list = [0.6, 0.7, 0.8]
last_layer_loss_weight = [1.25, 1.5, 1.75, 2.0, 5.0, 10]

for model_name in model_name_or_path:
    for last_layer_loss_weight in last_layer_loss_weight:
        last_layer_loss_weight = str(last_layer_loss_weight)

        model_name_or_path = f"train_result/{model_name}_{last_layer_loss_weight}/best-checkpoint"
        out_dir = f"evl_res/{model_name}_{last_layer_loss_weight}"
        
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
        
        cmd = nv_cmd + f" python eval_nli_main_v2.py --model_name_or_path {model_name_or_path} --out_dir {out_dir} --is_moe {is_moe}"
        # cmd = nv_cmd + f" python eval_ese_layers.py --model_name_or_path {model_name_or_path} --out_dir {plot_out_dir} --is_moe {is_moe} --embedding_size_list {embedding_size_list}"
        print(cmd+ "\n")
        # cmd_list.append(cmd)
        


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
