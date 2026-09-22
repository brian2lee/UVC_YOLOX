#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate every saved YOLOX checkpoint and write:

    /workspace/UVC_YOLOX/YOLOX_outputs/result.txt

Output columns:

    epoch    map50    map5095    cs    angle_map_50    angle_map_5095

Definitions
-----------
mAP50:
    Normal detection mAP at IoU = 0.50.

mAP50-95:
    Mean normal detection AP over IoU thresholds 0.50 ... 0.95.

CS:
    Mean cosine similarity of the matched detection angles:

        CS = mean(cos(abs(GT_angle - predicted_angle)))

    This follows the angle-similarity calculation in:

        yolox/evaluators/voc_eval.py

angle_map_50:
    The existing angle AP-like metric reported by the modified
    VOC evaluator at IoU = 0.50.

angle_map_5095:
    The mean of the existing angle AP-like metric over IoU
    thresholds 0.50 ... 0.95.

This script evaluates existing checkpoints.
It does NOT retrain the model.
"""

import contextlib
import io
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch


# ============================================================================
# Paths
# ============================================================================

OUT_ROOT = Path(__file__).resolve().parent

# /workspace/UVC_YOLOX
PROJECT_ROOT = OUT_ROOT.parent

# /workspace/UVC_YOLOX/YOLOX_outputs/yolo_MVTEC
CKPT_DIR = OUT_ROOT / "yolo_MVTEC"

# /workspace/UVC_YOLOX/YOLOX_outputs/result.txt
RESULT_FILE = OUT_ROOT / "result.txt"

# Your experiment file
EXP_FILE = (
    PROJECT_ROOT
    / "exps/example/yolox_voc/yolox_MVTEC_voc_s.py"
)


# ============================================================================
# Settings
# ============================================================================

# This is evaluation batch size, NOT training batch size.
EVAL_BATCH_SIZE = 8

# GPU / CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Whether to use FP16 during evaluation.
USE_FP16 = False


# ============================================================================
# Utility
# ============================================================================

def parse_metric(text, pattern):
    """
    Extract the last occurrence of a printed metric.
    """
    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    if not matches:
        return None

    return float(matches[-1])


def parse_metrics(text):
    """
    Extract the metrics printed by your modified VOC evaluator.
    """

    return {
        "map50": parse_metric(
            text,
            r"map_50:\s*([0-9.eE+-]+)"
        ),

        "map5095": parse_metric(
            text,
            r"map_5095:\s*([0-9.eE+-]+)"
        ),

        "angle_map50": parse_metric(
            text,
            r"angle_map_50:\s*([0-9.eE+-]+)"
        ),

        "angle_map5095": parse_metric(
            text,
            r"angle_map_5095:\s*([0-9.eE+-]+)"
        ),
    }


# ============================================================================
# Load checkpoint
# ============================================================================

def load_checkpoint(model, checkpoint):
    """
    Load one YOLOX checkpoint.
    """

    ckpt = torch.load(
        str(checkpoint),
        map_location="cpu"
    )

    if "model" not in ckpt:
        raise RuntimeError(
            f"No 'model' key found in {checkpoint}"
        )

    model.load_state_dict(
        ckpt["model"],
        strict=False
    )

    return ckpt


# ============================================================================
# Evaluate one checkpoint
# ============================================================================

def evaluate_checkpoint(
    exp,
    evaluator,
    checkpoint,
):
    """
    Load and evaluate one checkpoint.

    Returns:

        metrics
        raw evaluator output
    """

    print(
        f"      Loading {checkpoint.name}...",
        flush=True
    )

    model = exp.get_model()

    load_checkpoint(
        model,
        checkpoint
    )

    model.eval()

    if DEVICE == "cuda":
        model.cuda()

        if USE_FP16:
            model.half()

    # ------------------------------------------------------------------------
    # IMPORTANT:
    #
    # Your YOLOX version does NOT use:
    #
    # evaluator.evaluate(
    #     model,
    #     is_distributed=False,
    #     half=False,
    #     ...
    # )
    #
    # It uses positional arguments:
    #
    # evaluator.evaluate(
    #     model,
    #     False,
    #     False,
    #     return_outputs=False
    # )
    # ------------------------------------------------------------------------

    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):

        evaluator.evaluate(
            model,
            False,              # is_distributed
            USE_FP16,           # half
            return_outputs=False,
        )

    output = buffer.getvalue()

    metrics = parse_metrics(output)

    return metrics, output


# ============================================================================
# Discover checkpoints
# ============================================================================

def get_checkpoints():
    """
    Find all:

        epoch_xxx_ckpt.pth

    and sort them numerically.
    """

    checkpoints = []

    for path in CKPT_DIR.glob(
        "epoch_*_ckpt.pth"
    ):

        match = re.fullmatch(
            r"epoch_(\d+)_ckpt\.pth",
            path.name
        )

        if match is None:
            continue

        epoch = int(
            match.group(1)
        )

        checkpoints.append(
            (epoch, path)
        )

    checkpoints.sort(
        key=lambda x: x[0]
    )

    return checkpoints


# ============================================================================
# Write result.txt
# ============================================================================

def write_results(rows):
    """
    Write the final result.txt.
    """

    def fmt(value):
        return f"{value:.6f}"

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        # ------------------------------------------------------------
        # Header
        # ------------------------------------------------------------

        f.write(
            "epoch    map50    map5095    cs    "
            "angle_map_50    angle_map_5095\n"
        )

        f.write(
            "------   -------   --------   ------   "
            "------------   ---------------\n"
        )

        # ------------------------------------------------------------
        # Every epoch
        # ------------------------------------------------------------

        for row in rows:

            f.write(
                f"{row['epoch']:<8d}"
                f"{fmt(row['map50']):>9}"
                f"{fmt(row['map5095']):>11}"
                f"{fmt(row['cs']):>9}"
                f"{fmt(row['angle_map50']):>15}"
                f"{fmt(row['angle_map5095']):>17}"
                "\n"
            )

        # ------------------------------------------------------------
        # Best results
        # ------------------------------------------------------------

        best_map50 = max(
            rows,
            key=lambda x: x["map50"]
        )

        best_map5095 = max(
            rows,
            key=lambda x: x["map5095"]
        )

        best_cs = max(
            rows,
            key=lambda x: x["cs"]
        )

        best_angle_map50 = max(
            rows,
            key=lambda x: x["angle_map50"]
        )

        best_angle_map5095 = max(
            rows,
            key=lambda x: x["angle_map5095"]
        )

        f.write("\n")

        f.write(
            "Best results\n"
        )

        f.write(
            "------------\n"
        )

        f.write(
            f"Best mAP50:        "
            f"{fmt(best_map50['map50'])} "
            f"(epoch {best_map50['epoch']})\n"
        )

        f.write(
            f"Best mAP50-95:     "
            f"{fmt(best_map5095['map5095'])} "
            f"(epoch {best_map5095['epoch']})\n"
        )

        f.write(
            f"Best CS:           "
            f"{fmt(best_cs['cs'])} "
            f"(epoch {best_cs['epoch']})\n"
        )

        f.write(
            f"Best angle_map_50: "
            f"{fmt(best_angle_map50['angle_map50'])} "
            f"(epoch {best_angle_map50['epoch']})\n"
        )

        f.write(
            f"Best angle_map_5095:"
            f" {fmt(best_angle_map5095['angle_map5095'])} "
            f"(epoch {best_angle_map5095['epoch']})\n"
        )


# ============================================================================
# Main
# ============================================================================

def main():

    print("=" * 70)
    print("YOLOX checkpoint evaluation")
    print("=" * 70)

    print(
        f"Project:      {PROJECT_ROOT}"
    )

    print(
        f"Experiment:   {EXP_FILE}"
    )

    print(
        f"Checkpoint:   {CKPT_DIR}"
    )

    print(
        f"Result file:  {RESULT_FILE}"
    )

    print(
        f"Device:       {DEVICE}"
    )

    print()

    # ------------------------------------------------------------------------
    # Check paths
    # ------------------------------------------------------------------------

    if not EXP_FILE.exists():

        raise FileNotFoundError(
            "\nExperiment file does not exist:\n"
            f"  {EXP_FILE}\n"
        )

    if not CKPT_DIR.exists():

        raise FileNotFoundError(
            "\nCheckpoint directory does not exist:\n"
            f"  {CKPT_DIR}\n"
        )

    # ------------------------------------------------------------------------
    # Make project importable
    # ------------------------------------------------------------------------

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )

    # ------------------------------------------------------------------------
    # Import YOLOX
    # ------------------------------------------------------------------------

    from yolox.exp import get_exp

    # ------------------------------------------------------------------------
    # Load experiment
    # ------------------------------------------------------------------------

    exp = get_exp(
        str(EXP_FILE),
        None
    )

    # ------------------------------------------------------------------------
    # Find checkpoints
    # ------------------------------------------------------------------------

    checkpoints = get_checkpoints()

    if not checkpoints:

        raise RuntimeError(
            "\nNo epoch checkpoints found in:\n"
            f"  {CKPT_DIR}\n"
        )

    print(
        f"Found {len(checkpoints)} checkpoints."
    )

    print()

    # ------------------------------------------------------------------------
    # Create evaluator
    # ------------------------------------------------------------------------

    print(
        "Creating evaluator..."
    )

    evaluator = exp.get_evaluator(
        batch_size=EVAL_BATCH_SIZE,
        is_distributed=False,
        testdev=False,
        legacy=False,
    )

    print(
        "Evaluator created."
    )

    print()

    # ------------------------------------------------------------------------
    # Evaluate every checkpoint
    # ------------------------------------------------------------------------

    rows = []

    for index, (epoch, checkpoint) in enumerate(
        checkpoints,
        start=1
    ):

        print(
            f"[{index}/{len(checkpoints)}] "
            f"Epoch {epoch}"
        )

        try:

            metrics, raw_output = evaluate_checkpoint(
                exp,
                evaluator,
                checkpoint
            )

            # ------------------------------------------------------------
            # Check metrics
            # ------------------------------------------------------------

            required = [
                "map50",
                "map5095",
                "angle_map50",
                "angle_map5095",
            ]

            missing = [
                name
                for name in required
                if metrics[name] is None
            ]

            if missing:

                print(
                    "      WARNING: Missing metrics:"
                    f" {', '.join(missing)}"
                )

                print()
                continue

            # ------------------------------------------------------------
            # IMPORTANT:
            #
            # We DO NOT do:
            #
            #     cs = angle_map50
            #
            # Those are different quantities.
            #
            # The actual CS is:
            #
            #     cos(abs(GT_angle - predicted_angle))
            #
            # for matched detections.
            #
            # ------------------------------------------------------------
            #
            # The modified VOC evaluator currently prints angle_map_50
            # and angle_map_5095, but does not print the raw CS mean.
            #
            # Therefore this script asks the evaluator to expose the
            # actual CS through the attribute below if available.
            #
            # ------------------------------------------------------------

            cs = None

            if hasattr(
                evaluator,
                "last_cs"
            ):

                cs = evaluator.last_cs

            # Some versions place it on the dataset.
            if cs is None:

                dataset = getattr(
                    evaluator,
                    "dataset",
                    None
                )

                if dataset is not None:

                    if hasattr(
                        dataset,
                        "last_cs"
                    ):

                        cs = dataset.last_cs

            # ------------------------------------------------------------
            # If the evaluator does not expose raw CS, don't silently
            # substitute angle_map_50.
            # ------------------------------------------------------------

            if cs is None:

                raise RuntimeError(
                    "The evaluator returned angle_map metrics, "
                    "but it did not expose the actual CS value. "
                    "See the required voc_eval.py modification below."
                )

            cs = float(cs)

            # ------------------------------------------------------------
            # Store
            # ------------------------------------------------------------

            row = {
                "epoch": epoch,
                "map50": float(
                    metrics["map50"]
                ),
                "map5095": float(
                    metrics["map5095"]
                ),
                "cs": cs,
                "angle_map50": float(
                    metrics["angle_map50"]
                ),
                "angle_map5095": float(
                    metrics["angle_map5095"]
                ),
            }

            rows.append(row)

            print(
                "      "
                f"mAP50={row['map50']:.6f}  "
                f"mAP50-95={row['map5095']:.6f}  "
                f"CS={row['cs']:.6f}"
            )

            print(
                "      "
                f"angle_map_50={row['angle_map50']:.6f}  "
                f"angle_map_5095={row['angle_map5095']:.6f}"
            )

            print()

        except Exception as e:

            print(
                f"      ERROR: "
                f"{type(e).__name__}: {e}"
            )

            print()

    # ------------------------------------------------------------------------
    # Check results
    # ------------------------------------------------------------------------

    if not rows:

        raise RuntimeError(
            "\nNo checkpoints were successfully evaluated.\n"
        )

    # ------------------------------------------------------------------------
    # Write file
    # ------------------------------------------------------------------------

    write_results(
        rows
    )

    # ------------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------------

    best_map50 = max(
        rows,
        key=lambda x: x["map50"]
    )

    best_map5095 = max(
        rows,
        key=lambda x: x["map5095"]
    )

    best_cs = max(
        rows,
        key=lambda x: x["cs"]
    )

    print("=" * 70)
    print("Finished")
    print("=" * 70)

    print(
        f"Successfully evaluated: "
        f"{len(rows)}/{len(checkpoints)}"
    )

    print()

    print(
        f"Best mAP50: "
        f"{best_map50['map50']:.6f} "
        f"(epoch {best_map50['epoch']})"
    )

    print(
        f"Best mAP50-95: "
        f"{best_map5095['map5095']:.6f} "
        f"(epoch {best_map5095['epoch']})"
    )

    print(
        f"Best CS: "
        f"{best_cs['cs']:.6f} "
        f"(epoch {best_cs['epoch']})"
    )

    print()

    print(
        f"Result saved to:"
    )

    print(
        f"  {RESULT_FILE}"
    )


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    main()
