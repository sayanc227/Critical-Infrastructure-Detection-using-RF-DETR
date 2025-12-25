import argparse
from rfdetr import RFDETRBase


def main(args):
    model = RFDETRBase()

    # Evaluation-only load (safe)
    model.train(
        dataset_dir=args.dataset_dir,
        epochs=0,
        resume=args.checkpoint
    )

    results = model.evaluate(
        dataset_dir=args.dataset_dir,
        save_predictions=True
    )

    print("Evaluation results:\n", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained RF-DETR model")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()
    main(args)
