import argparse
import os
from PIL import Image
from rfdetr import RFDETRBase
import supervision as sv

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

def main(args):
    model = RFDETRBase()

    # Load trained weights (RF-DETR correct way)
    model.train(
        dataset_dir=args.dataset_dir,
        epochs=0,
        resume=args.checkpoint
    )

    os.makedirs(args.output_dir, exist_ok=True)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    for fname in os.listdir(args.input_dir):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue

        img_path = os.path.join(args.input_dir, fname)
        image = Image.open(img_path).convert("RGB")

        detections = model.predict(image, threshold=args.threshold)

        labels = [
            f"{cls}:{conf:.2f}"
            for cls, conf in zip(detections.class_id, detections.confidence)
        ]

        annotated = box_annotator.annotate(image.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections, labels)

        out_path = os.path.join(args.output_dir, fname)
        annotated.save(out_path)

        print(f"Saved → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Batch inference with RF-DETR")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="sample_outputs")
    parser.add_argument("--threshold", type=float, default=0.25)

    args = parser.parse_args()
    main(args)
