"""
infer.py
---------
Performs inference on test images using a trained YOLOv12 model.

The script runs detection on a folder of test images and saves
the output predictions (with bounding boxes) into `outputs/`.
"""

import os
import argparse
import subprocess


def run_inference(model_path: str, source_path: str, output_dir: str, name: str):
    """
    Run YOLOv12 inference using pretrained weights.

    Args:
        model_path (str): Path to the trained .pt weights.
        source_path (str): Directory containing test images.
        output_dir (str): Directory to save the output predictions.
        name (str): Name of the output folder (baseline/enhanced).
    """
    print(f"[INFO] Running inference using model: {model_path}")

    cmd = (
        f"yolo detect predict model={model_path} "
        f"source={source_path} "
        f"save=True project={output_dir} name={name}"
    )
    subprocess.run(cmd, shell=True, check=True)

    print(f"[INFO] Inference complete. Results saved in {os.path.join(output_dir, name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv12 inference on test images.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained .pt model file.")
    parser.add_argument("--source", type=str, required=True, help="Path to test image folder.")
    parser.add_argument("--output", type=str, default="../outputs", help="Output directory.")
    parser.add_argument("--name", type=str, default="baseline", help="Name of result subfolder.")
    args = parser.parse_args()

    run_inference(args.model, args.source, args.output, args.name)
