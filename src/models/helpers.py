import torch
import torch.nn as nn
import json
import os

def load_class_names(json_path):
    """Loads class names from a COCO annotation JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create dictionary {0: 'name', 1: 'name', ...}
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    return categories

def patch_model_architecture(wrapper_model, num_classes):
    """
    Manually resizes the classification heads of RFDETRBase.
    This fixes the shape mismatch error (19 classes vs 91 default).
    """
    print(f"🔧 Patching model architecture for {num_classes} classes...")
    
    # 1. Find the inner model (LWDETR or similar) inside the wrapper
    inner_model = getattr(getattr(wrapper_model, 'model', None), 'model', None)
    if inner_model is None:
        inner_model = getattr(wrapper_model, 'model', None)
    if inner_model is None:
        inner_model = wrapper_model
        
    # 2. Patch the main Class Embedding layer
    if hasattr(inner_model, 'class_embed') and isinstance(inner_model.class_embed, nn.Linear):
        in_features = inner_model.class_embed.in_features
        inner_model.class_embed = nn.Linear(in_features, num_classes)
    elif hasattr(inner_model, 'class_embed') and isinstance(inner_model.class_embed, nn.ModuleList):
         for i, layer in enumerate(inner_model.class_embed):
             if isinstance(layer, nn.Linear):
                 inner_model.class_embed[i] = nn.Linear(layer.in_features, num_classes)

    # 3. Patch the Transformer Encoder Auxiliary Heads
    if hasattr(inner_model, 'transformer') and hasattr(inner_model.transformer, 'enc_out_class_embed'):
        enc_heads = inner_model.transformer.enc_out_class_embed
        if isinstance(enc_heads, nn.ModuleList):
            for i, layer in enumerate(enc_heads):
                if isinstance(layer, nn.Linear):
                    enc_heads[i] = nn.Linear(layer.in_features, num_classes)
                    
    return wrapper_model

def smart_load_weights(model, checkpoint_path, device):
    """Loads weights into the model, handling nested wrappers automatically."""
    print(f"🔄 Loading weights from {checkpoint_path}...")
    
    try:
        # weights_only=False is needed for your specific checkpoint format
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint file: {e}")
    
    # Get the state dict (sometimes it's inside a 'model' key)
    state_dict = checkpoint.get('model', checkpoint)
    
    # Potential locations for the actual PyTorch model inside the wrapper
    potential_targets = [
        getattr(getattr(model, 'model', None), 'model', None), # Deep wrapper
        getattr(model, 'model', None),       # Standard wrapper
        model                                # Direct model
    ]
    
    success = False
    for target in potential_targets:
        if target is None or not hasattr(target, 'load_state_dict'):
            continue
            
        try:
            # strict=False allows loading even if some minor keys don't match
            target.load_state_dict(state_dict, strict=False)
            target.to(device)
            target.eval()
            success = True
            print("✅ Weights loaded successfully.")
            break
        except Exception:
            continue
            
    if not success:
        raise RuntimeError("Could not find a valid place to load weights in the model wrapper.")
