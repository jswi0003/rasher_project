"""
Run one complete, reproducible experiment from the command line.

Steps:

1. Check the dataset structure (abort if anything is wrong).
2. Train a YOLO detector (or reuse weights with ``--skip-train``).
3. Measure box-level detection metrics on the validation split.
4. Sweep the tray pass/fail threshold on the validation split, choose the
   operating threshold, and compare with the "always fail" baseline.
5. Only with ``--test``: apply the *fixed* validation threshold to the
   test split. Run this once per model, after all choices are final.

Everything is saved to ``results/<name>/`` (``summary.json`` and the
threshold sweep CSVs).

Examples (from the repository root, with the venv active)::

    # 1-epoch check that the pipeline runs end to end
    python -m rasher_qc.experiment --weights yolov8n.pt --name smoke --epochs 1 --fraction 0.1

    # full runs
    python -m rasher_qc.experiment --weights yolov8n.pt --name yolov8n
    python -m rasher_qc.experiment --weights yolov8s.pt --name yolov8s

    # final, once only: re-use trained weights and add the test split
    python -m rasher_qc.experiment --name yolov8n --skip-train --test
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .data import DATA_DIR, check_dataset, write_training_yaml
from .detector import (RESULTS_DIR, RUNS_DIR, device_name, evaluate_detector,
                       max_confidences, pick_device, train_detector)
from .tray import choose_threshold, threshold_sweep, tray_is_defective, tray_metrics

#: Dataset used by default (Roboflow version 4).
DEFAULT_DATASET = DATA_DIR / "pork-rasher-error-packaging-v4"


def split_images_and_truth(dataset_dir, split):
    """
    List a split's images with their ground-truth tray verdicts.

    Parameters
    ----------
    dataset_dir : pathlib.Path
        Dataset folder.
    split : {"train", "valid", "test"}
        Split folder name.

    Returns
    -------
    tuple of (list of pathlib.Path, numpy.ndarray of bool)
        Sorted image paths and, for each, True if its label file has a box.
    """
    image_dir = dataset_dir / split / "images"
    label_dir = dataset_dir / split / "labels"
    images = sorted(p for p in image_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    truth = np.array([tray_is_defective(label_dir / f"{p.stem}.txt") for p in images])
    return images, truth


def tray_evaluation(weights, dataset_dir, split, imgsz, threshold=None,
                    min_recall=0.95):
    """
    Evaluate the tray pass/fail decision on one split.

    If ``threshold`` is None it is chosen on this split (use for the
    validation split only); otherwise the given threshold is applied.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        Summary (threshold, model metrics, baseline metrics, timing) and
        the full threshold sweep.
    """
    images, truth = split_images_and_truth(dataset_dir, split)

    start = time.perf_counter()
    conf = max_confidences(weights, images, imgsz=imgsz)
    elapsed = time.perf_counter() - start

    sweep = threshold_sweep(truth, conf)
    if threshold is None:
        threshold = choose_threshold(sweep, min_recall=min_recall)

    model = tray_metrics(truth, conf >= threshold)
    # Baseline: fail every tray. Catches every defect but rejects every
    # good tray; any useful method must beat its false-reject rate.
    baseline = tray_metrics(truth, np.ones_like(truth))
    summary = {
        "n_images": int(truth.size), "n_defective": int(truth.sum()),
        "threshold": threshold, "model": model, "baseline_always_fail": baseline,
        "seconds_total": elapsed,
    }
    return summary, sweep


def main(argv=None):
    """Parse command-line arguments and run the experiment."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", default="yolov8n.pt",
                        help="starting weights (default: yolov8n.pt)")
    parser.add_argument("--name", required=True, help="run name, e.g. yolov8n")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="fraction of training data (for quick checks only)")
    parser.add_argument("--min-recall", type=float, default=0.95,
                        help="tray recall required when choosing the threshold")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--skip-train", action="store_true",
                        help="reuse results/runs/<name>/weights/best.pt")
    parser.add_argument("--test", action="store_true",
                        help="also evaluate on the test split (final run only)")
    args = parser.parse_args(argv)

    out_dir = RESULTS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset checks
    report = check_dataset(args.dataset)
    if report["problems"]:
        raise SystemExit("Dataset problems found:\n  " + "\n  ".join(report["problems"]))
    data_yaml = write_training_yaml(args.dataset)

    device = pick_device()
    summary = {"settings": {k: str(v) for k, v in vars(args).items()},
               "device": device_name(device)}
    print(f"Device: {summary['device']}")

    # 2. Training
    if args.skip_train:
        weights = RUNS_DIR / args.name / "weights" / "best.pt"
        if not weights.is_file():
            raise SystemExit(f"No trained weights at {weights}")
    else:
        start = time.perf_counter()
        weights = train_detector(data_yaml, weights=args.weights, name=args.name,
                                 epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                                 seed=args.seed, fraction=args.fraction, device=device)
        summary["train_seconds"] = time.perf_counter() - start
    summary["weights"] = str(weights)
    summary["weights_mb"] = weights.stat().st_size / 1e6

    # 3-4. Validation split: detection metrics, then tray threshold
    summary["val_detection"] = evaluate_detector(
        weights, data_yaml, "val", imgsz=args.imgsz, batch=args.batch,
        device=device, name=f"{args.name}_val")
    summary["val_tray"], sweep = tray_evaluation(
        weights, args.dataset, "valid", args.imgsz, min_recall=args.min_recall)
    sweep.to_csv(out_dir / "val_threshold_sweep.csv", index=False)

    # 5. Test split, with the threshold fixed from validation
    if args.test:
        summary["test_detection"] = evaluate_detector(
            weights, data_yaml, "test", imgsz=args.imgsz, batch=args.batch,
            device=device, name=f"{args.name}_test")
        summary["test_tray"], sweep = tray_evaluation(
            weights, args.dataset, "test", args.imgsz,
            threshold=summary["val_tray"]["threshold"])
        sweep.to_csv(out_dir / "test_threshold_sweep.csv", index=False)

    # NaN is written as null so the file is valid JSON.
    text = json.dumps(summary, indent=2, default=float).replace("NaN", "null")
    (out_dir / "summary.json").write_text(text, encoding="utf-8")

    val = summary["val_tray"]
    print(f"\nVal mAP50 = {summary['val_detection']['map50']:.3f}, "
          f"mAP50-95 = {summary['val_detection']['map50_95']:.3f}")
    print(f"Tray threshold = {val['threshold']:.2f}: recall = {val['model']['recall']:.3f}, "
          f"false reject = {val['model']['false_reject_rate']:.3f} "
          f"(baseline false reject = {val['baseline_always_fail']['false_reject_rate']:.3f})")
    print(f"Saved {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
