from modeling.modeling_bert_moe import BertMoEModel
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print("Using device:", device)

model_name_or_path = "train_result/bert_moe_ese_11/best-checkpoint"
# model_name_or_path = "train_result/bert_moe_ese_all/best-checkpoint"
# model_name_or_path = "train_result/bert_moe_base_11/best-checkpoint"
# model_name_or_path = "train_result/bert_moe_base_all/best-checkpoint"

model = BertMoEModel.from_pretrained(
	model_name_or_path, output_hidden_states=True, torch_dtype=torch.float16, device_map='auto').to(device)

if hasattr(model.bert_moe.encoder, "expert_metrics"):
    print("\nExpert Utilization Metrics:")
    expert_metrics = model.encoder.expert_metrics

    # 打印 MoE 使用的层
    if "moe_layers_used" in expert_metrics:
        print(f"MoE layers used: {expert_metrics['moe_layers_used']}")

    for key, value in expert_metrics.items():
        if key != "moe_layers_used":  # 避免重复打印
            print(f"{key}: {value}")