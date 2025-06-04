import torch

def compare_layer_weights(model1, model2, layer_name="embeddings.word_embeddings.weight", rtol=1e-5, atol=1e-8):
    param1 = dict(model1.named_parameters()).get(layer_name, None)
    param2 = dict(model2.named_parameters()).get(layer_name, None)

    if param1 is None or param2 is None:
        print(f"Layer '{layer_name}' not found in one or both models.")
        return
    
    if param1.shape != param2.shape: # Determine if the shapes are the same
        print(f"Shape mismatch: {param1.shape} vs {param2.shape}")
        return

    if torch.equal(param1, param2): # Check if they are completely equal
        print(f"Layer '{layer_name}' weights are exactly the same.")
    elif torch.allclose(param1, param2, rtol=rtol, atol=atol):
        print(f"Layer '{layer_name}' weights are numerically close (within tolerance).")
    else:
        diff = torch.abs(param1 - param2).max()
        print(f"Layer '{layer_name}' weights differ. Max absolute difference: {diff.item()}")

def check_model_weights(model):
    for name, param in model.named_parameters():
        if param is None:
            status = "No weights (None)"
        elif torch.all(param == 0):
            status = "All-zero weights"
        elif not param.requires_grad:
            status = "Frozen (requires_grad=False)"
        else:
            status = "Has pretrained weights" if param.abs().sum() > 0 else "Possibly uninitialized"
        
        print(f"{name:60} : {status}")