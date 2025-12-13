import argparse
import os
import torch
import numpy as np
import sys
from PIL import Image
import supervision as sv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rfdetr import RFDETRBase
except ImportError:
    print("❌ Error: 'rfdetr' library not found.")
    sys.exit(1)

# Import our custom helper functions
from models.helpers import load_class_names, patch_model_architecture, smart_load_weights

def parse_args():
    parser = argparse.ArgumentParser(description="Run RF-DETR Inference on images.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth checkpoint file')
    parser.add_argument('--source', type=str, required=True, help='Path to folder containing images')
    parser.add_argument('--annotations', type=str, required=True, help='Path to _annotations.coco.json (for class names)')
    parser.add_argument('--output', type=str, default='runs/detect', help='Folder to save results')
    parser.add_argument('--conf', type=float, default=0.35, help='Confidence threshold')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
    
    # 2. Load Class Names
    print(f"Loading class names from {args.annotations}...")
    try:
        categories = load_class_names(args.annotations)
        num_classes = len(categories)
        print(f"✅ Loaded {num_classes} classes.")
    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        return

    # 3. Initialize & Patch Model
    print("Initializing model...")
    try:
        model = RFDETRBase(num_classes=num_classes)
    except TypeError:
        # Fallback if specific init isn't supported
        model = RFDETRBase()
        
    # Apply our custom fixes from helpers.py
    patch_model_architecture(model, num_classes)
    smart_load_weights(model, args.checkpoint, device)

    # 4. Processing Loop
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    if not os.path.exists(args.source):
         print(f"❌ Source directory not found: {args.source}")
         return
         
    images = [f for f in os.listdir(args.source) if f.lower().endswith(valid_exts)]
    
    if not images:
        print(f"❌ No images found in {args.source}")
        return

    print(f"🚀 Starting inference on {len(images)} images...")
    
    bbox_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    for filename in images:
        img_path = os.path.join(args.source, filename)
        
        try:
            image_pil = Image.open(img_path).convert("RGB")
            
            # Predict
            detections = model.predict(image_pil, threshold=args.conf)
            
            # Create labels
            labels = [
                f"{categories.get(class_id, class_id)} {confidence:.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
            
            # Annotate
            image_np = np.array(image_pil)
            annotated_image = bbox_annotator.annotate(scene=image_np.copy(), detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            
            # Save
            save_path = os.path.join(args.output, f"pred_{filename}")
            Image.fromarray(annotated_image).save(save_path)
            print(f"  Saved: {save_path}")
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print(f"✅ Inference complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
