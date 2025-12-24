import argparse
from rfdetr import RFDETRBase
from PIL import Image
import os

def main(args):
model = RFDETRBase()


model.train(
dataset_dir=args.dataset_dir,
epochs=0,
resume=args.checkpoint
)


image = Image.open(args.image_path).convert("RGB")
detections = model.predict(image, threshold=args.threshold)


print("Detections:")
for cls, conf, box in zip(detections.class_id, detections.confidence, detections.xyxy):
print(f"Class {cls} | Conf {conf:.2f} | Box {box}")




if __name__ == "__main__":
parser = argparse.ArgumentParser(description="Run inference with RF-DETR")
parser.add_argument("--dataset_dir", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--image_path", required=True)
parser.add_argument("--threshold", type=float, default=0.3)


main(parser.parse_args())
