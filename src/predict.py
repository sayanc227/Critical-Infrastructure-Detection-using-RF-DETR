import torch
import torch.nn as nn
from rfdetr import RFDETRBase
from PIL import Image
import supervision as sv
import json
import os
import numpy as np
import sys
import argparse
from tqdm import tqdm

# ==========================================
# 1. HELPER FUNCTIONS FOR ARCHITECTURE
# ==========================================

def patch_model_for_custom_classes(wrapper_model, num_classes):
    """
    Manually resize the classification heads to match custom dataset size.
    Prevents shape mismatch errors (e.g., 91 vs 19 classes).
    """
    print(f"🔧 Patching model architecture for {num_classes} classes...")

    # Find the inner model
    inner_model = getattr(getattr(wrapper_model, 'model', None), 'model', None)
    if inner_model is None:
        inner_model = getattr(wrapper_model, 'model', None)
    if inner_model is None:
        inner_model = wrapper_model

    # Patch the main Class Embedding layer
    if hasattr(inner_model, 'class_embed') and isinstance(inner_model.class_embed, nn.Linear):
        in_features = inner_model.class_embed.in_features
        inner_model.class_embed = nn.Linear(in_features, num_classes)
    elif hasattr(inner_model, 'class_embed') and isinstance(inner_model.class_embed, nn.ModuleList):
         for i, layer in enumerate(inner_model.class_embed):
             if isinstance(layer, nn.Linear):
                 inner_model.class_embed[i] = nn.Linear(layer.in_features, num_classes)

    # Patch Transformer Encoder Auxiliary Heads
    if hasattr(inner_model, 'transformer') and hasattr(inner_model.transformer, 'enc_out_class_embed'):
        enc_heads = inner_model.transformer.enc_out_class_embed
        if isinstance(enc_heads, nn.ModuleList):
            for i, layer in enumerate(enc_heads):
                if isinstance(layer, nn.Linear):
                    enc_heads[i] = nn.Linear(layer.in_features, num_classes)
    return wrapper_model

def load_weights_smart(wrapper_model, state_dict, device):
    """
    Smartly injects weights into the nested PyTorch modules.
    """
    potential_targets = [
        ("model.model.model", getattr(getattr(wrapper_model, 'model', None), 'model', None)),
        ("model.model", getattr(wrapper_model, 'model', None)),
        ("model", wrapper_model)
    ]

    success = False
    for name, target_obj in potential_targets:
        if target_obj is None or not hasattr(target_obj, 'load_state_dict'):
            continue
        try:
            print(f"🔄 Attempting to load weights into: {name}...")
            msg = target_obj.load_state_dict(state_dict, strict=False)
            print(f"✅ Success! Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
            target_obj.to(device)
            target_obj.eval()
            success = True
            break
        except Exception as e:
            print(f"⚠️  Failed to load into {name}: {e}")

    if not success:
        print("\n❌ CRITICAL ERROR: Could not find a valid place to load weights.")
        sys.exit(1)

# ==========================================
# 2. MAIN EXECUTION LOGIC
# ==========================================

def run_prediction(args):
    # --- Step A: Load Class Names ---
    if not os.path.exists(args.json):
        print(f"❌ Error: JSON file not found at {args.json}")
        return

    with open(args.json, 'r') as f:
        data = json.load(f)
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    num_classes = len(categories)
    print(f"✅ Loaded {num_classes} classes.")

    # --- Step B: Initialize & Patch Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        model = RFDETRBase(num_classes=num_classes)
    except:
        model = RFDETRBase()
    
    patch_model_for_custom_classes(model, num_classes)

    # --- Step C: Load Checkpoint ---
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found at {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    load_weights_smart(model, state_dict, device)

    # --- Step D: Process Images ---
    os.makedirs(args.output, exist_ok=True)
    
    # Handle single image or directory
    if os.path.isfile(args.source):
        image_list = [args.source]
    else:
        extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        image_list = [os.path.join(args.source, f) for f in os.listdir(args.source) if f.lower().endswith(extensions)]

    print(f"🚀 Processing {len(image_list)} images...")
    
    bbox_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    for img_path in tqdm(image_list):
        image_pil = Image.open(img_path).convert("RGB")
        
        try:
            detections = model.predict(image_pil, threshold=args.conf)
        except Exception as e:
            print(f"Skipping {img_path} due to error: {e}")
            continue

        # Annotation
        labels = [f"{categories.get(cid, f'ID:{cid}')} {conf:.2f}" 
                  for cid, conf in zip(detections.class_id, detections.confidence)]
        
        image_np = np.array(image_pil)
        annotated = bbox_annotator.annotate(scene=image_np.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        # Save result
        save_path = os.path.join(args.output, os.path.basename(img_path))
        Image.fromarray(annotated).save(save_path)

    print(f"✅ All results saved to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF-DETR Custom Inference Script")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument('--json', type=str, required=True, help="Path to COCO JSON for class names")
    parser.add_argument('--source', type=str, required=True, help="Path to image or directory of images")
    parser.add_argument('--output', type=str, default="/content/results", help="Output directory")
    parser.add_argument('--conf', type=float, default=0.35, help="Confidence threshold")
    
    args = parser.parse_args()
    run_prediction(args)
