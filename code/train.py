"""
train.py
---------
Wrapper script for training YOLOv12 models on the PCB Defects dataset.

This script supports both the baseline and enhanced configurations defined
in `configs/` folder. It executes YOLOv12's built-in training command
via the command line, logging all results under the `runs/` directory.
"""

import os
import argparse
import subprocess


def train_model(config_path: str):
    """
    Execute YOLOv12 training with the given configuration YAML.

    Args:
        config_path (str): Path to the training configuration YAML file.
    """
    print(f"[INFO] Starting YOLOv12 training with config: {config_path}")

    # The command calls YOLOv12's CLI (ultralytics interface)
    cmd = f"yolo detect train cfg={config_path}"
    subprocess.run(cmd, shell=True, check=True)

    print("[INFO] Training complete. Results saved in /runs directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv12 on PCB Defect dataset.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training configuration YAML (baseline or enhanced).",
    )
    args = parser.parse_args()
    train_model(args.config)
