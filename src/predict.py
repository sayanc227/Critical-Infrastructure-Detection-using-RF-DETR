import os
import torch
import cv2
import argparse
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# This assumes the RF-DETR repo is in the python path
import sys
sys.path.append('/content/Critical-Infrastructure-Detection-using-RF-DETR')

# Import the model building logic from the source
# Note: Adjust these imports if the folder structure is different
from src.models import build_model 
from src.util.slconfig import SLConfig
from src.util.visualizer import COCOVisualizer
from src.datasets import build_dataset

def main(args):
    # 1. Load Configuration from Checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_config_path = os.path.join(os.path.dirname(args.checkpoint), 'config.json') # Usually alongside pth
    
    # If config not found, we use the args passed in
    # For RF-DETR, we typically rebuild the model architecture
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. Build Model
    # We use the arguments stored in the checkpoint or passed via CLI
    model, criterion, postprocessors = build_model(args)
    model.to(args.device)
    
    # 3. Load Weights
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 4. Prepare Transformation
    transform = T.Compose([
        T.Resize(args.resolution),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 5. Process Images
    source_path = Path(args.source)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_paths = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png')) + list(source_path.glob('*.jpeg'))
    
    print(f"Found {len(image_paths)} images. Starting inference...")

    with torch.no_grad():
        for img_p in image_paths:
            # Load image
            img_raw = Image.open(img_p).convert("RGB")
            w, h = img_raw.size
            img = transform(img_raw).unsqueeze(0).to(args.device)
            
            # Inference
            outputs = model(img)
            
            # Process results (BBoxes & Scores)
            orig_target_sizes = torch.tensor([[h, w]]).to(args.device)
            results = postprocessors['bbox'](outputs, orig_target_sizes)[0]
            
            # Filter by confidence
            scores = results['scores']
            labels = results['labels']
            boxes = results['boxes']
            
            keep = scores > args.conf
            filt_scores = scores[keep]
            filt_labels = labels[keep]
            filt_boxes = boxes[keep]
            
            # Visualizing
            vsl = COCOVisualizer()
            # Convert PIL to CV2 for saving if needed, or use the visualizer
            output_img = vsl.visualize(img_raw, {
                'boxes': filt_boxes,
                'labels': filt_labels,
                'scores': filt_scores
            }, args.class_names)
            
            save_name = output_path / img_p.name
            output_img.save(save_name)
            print(f"Saved: {save_name}")

if __name__ == "__main__":
    # Standard RF-DETR args to avoid the UnboundLocalError in their main.py
    parser = argparse.ArgumentParser('RF-DETR inference')
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--source', default='', type=str)
    parser.add_argument('--output', default='results', type=str)
    parser.add_argument('--conf', default=0.35, type=float)
    parser.add_argument('--resolution', default=560, type=int)
    # Add hidden defaults required by build_model to prevent crashes
    parser.add_argument('--num_queries', default=300, type=int)
    parser.add_argument('--encoder', default='dinov2_windowed_small', type=str)
    parser.add_argument('--num_classes', default=19, type=int) # From your logs
    # ... add other architecture args if build_model fails ...
    
    args = parser.parse_args()
    # Apply your specific class names from the log
    args.class_names = ['Airport_Runway', 'Bridge', 'Cargo_Ship', 'Cooling_Tower', 'Dam', 'Electrical_Substation', 'Energy Storage Infrastructure', 'Mobile Tower', 'Nuclear_Reactor', 'Oil Refinery', 'Satellite_Dish /Ground_Station', 'Seaport', 'Shipping Containers', 'Solar_Power_Plant', 'Thermal Power Plant', 'Transmission Tower', 'Water Tower', 'Wind Turbine', 'mobile harbour cranes']
    
    main(args)
