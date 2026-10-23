"""
Training, evaluation and inference with Ultralytics YOLO defect detectors.

This module wraps the Ultralytics API so that every experiment uses the
same data config, output folders and settings:

* :func:`train_detector` fine-tunes a COCO-pretrained YOLO model.
* :func:`evaluate_detector` computes box-level metrics (mAP, precision,
  recall) on one split.
* :func:`max_confidences` gets the highest defect confidence per image,
  the input to the tray-level decision in :mod:`rasher_qc.tray`.
* :func:`inspect_tray` is the end-user entry point: one image in, a
  pass/fail verdict and list of defects out.

Ultralytics and PyTorch are imported inside functions so that the rest of
the package (and its tests) can be used without loading them.
"""

from pathlib import Path

import numpy as np

from .data import REPO_ROOT
from .tray import load_image

#: Folder for all experiment outputs (training runs, metrics, plots).
RESULTS_DIR = REPO_ROOT / "results"

#: Ultralytics writes each training/validation run to RESULTS_DIR/runs/<name>.
RUNS_DIR = RESULTS_DIR / "runs"


def pick_device():
    """
    Choose the compute device: the first CUDA GPU if available, else CPU.

    Returns
    -------
    int or str
        ``0`` for GPU 0, or ``"cpu"``.
    """
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def device_name(device):
    """
    Describe a device for logging, e.g. ``"NVIDIA GeForce RTX 4050 ..."``.

    Parameters
    ----------
    device : int or str
        Value returned by :func:`pick_device`.

    Returns
    -------
    str
        GPU name, or ``"cpu"``.
    """
    import torch

    return torch.cuda.get_device_name(device) if device != "cpu" else "cpu"


def train_detector(data_yaml, weights="yolov8n.pt", name="yolov8n", epochs=50,
                   imgsz=640, batch=16, seed=0, patience=15, fraction=1.0,
                   workers=4, device=None):
    """
    Fine-tune a pretrained YOLO detector on the rasher dataset.

    Transfer learning from COCO weights is used because ~3,300 training
    images is too few to train a detector from scratch. Training stops
    early if validation mAP does not improve for ``patience`` epochs, and
    the weights from the best validation epoch are kept.

    Parameters
    ----------
    data_yaml : str or pathlib.Path
        Dataset config from :func:`rasher_qc.data.write_training_yaml`.
    weights : str, optional
        Starting weights, e.g. ``"yolov8n.pt"`` or ``"yolov8s.pt"``
        (downloaded automatically on first use).
    name : str, optional
        Run name; outputs go to ``results/runs/<name>``.
    epochs : int, optional
        Maximum number of passes over the training set.
    imgsz : int, optional
        Training image size in pixels (longest side after letterboxing).
    batch : int, optional
        Images per batch. 16 fits YOLOv8n/s at 640 px in 6 GB of GPU memory.
    seed : int, optional
        Random seed, fixed so runs can be repeated and compared fairly.
    patience : int, optional
        Early-stopping patience in epochs.
    fraction : float, optional
        Fraction of the training set to use. Values < 1 are only for
        quick pipeline checks.
    workers : int, optional
        Data-loading processes.
    device : int or str, optional
        Compute device; chosen automatically if None.

    Returns
    -------
    pathlib.Path
        Path to the best weights file (``weights/best.pt``).
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
        seed=seed, deterministic=True, patience=patience, fraction=fraction,
        workers=workers, device=pick_device() if device is None else device,
        project=str(RUNS_DIR), name=name, exist_ok=True, verbose=False,
    )
    return Path(model.trainer.best)


def evaluate_detector(weights, data_yaml, split="val", imgsz=640, batch=16,
                      device=None, name=None):
    """
    Compute box-level detection metrics on one dataset split.

    A predicted box counts as correct if it has the right class and an
    intersection-over-union (IoU) with a labelled box of at least 0.5
    (for mAP50), or averaged over IoU 0.50-0.95 (for mAP50-95).

    Parameters
    ----------
    weights : str or pathlib.Path
        Trained weights (``best.pt``).
    data_yaml : str or pathlib.Path
        Dataset config.
    split : {"val", "test"}, optional
        Split to evaluate. Use ``"test"`` only once, for the final model.
    imgsz, batch : int, optional
        Inference image size and batch size.
    device : int or str, optional
        Compute device; chosen automatically if None.
    name : str, optional
        Output folder name under ``results/runs`` for plots.

    Returns
    -------
    dict
        ``map50``, ``map50_95``, ``precision``, ``recall`` (means over
        classes present in the split), ``per_class`` (the same metrics
        for each class present) and ``speed_ms`` (preprocess, inference
        and postprocess time per image). Classes with no labels in the
        split are omitted from ``per_class`` because they cannot be scored.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml), split=split, imgsz=imgsz, batch=batch,
        device=pick_device() if device is None else device,
        project=str(RUNS_DIR), name=name or f"val_{split}", exist_ok=True,
        plots=True, verbose=False,
    )
    box = metrics.box
    per_class = {}
    for i, class_id in enumerate(box.ap_class_index):
        per_class[model.names[int(class_id)]] = {
            "ap50": float(box.ap50[i]), "ap50_95": float(box.ap[i]),
            "precision": float(box.p[i]), "recall": float(box.r[i]),
        }
    return {
        "map50": float(box.map50), "map50_95": float(box.map),
        "precision": float(box.mp), "recall": float(box.mr),
        "per_class": per_class,
        "speed_ms": {k: float(v) for k, v in metrics.speed.items()},
    }


def max_confidences(weights, image_paths, imgsz=640, batch=16, device=None):
    """
    Return the highest defect-detection confidence in each image.

    Predictions are made with a very low confidence cut-off (0.001) so the
    tray decision threshold can be swept afterwards without re-running
    the model.

    Parameters
    ----------
    weights : str or pathlib.Path
        Trained weights.
    image_paths : sequence of str or pathlib.Path
        Images to process.
    imgsz, batch : int, optional
        Inference image size and batch size.
    device : int or str, optional
        Compute device; chosen automatically if None.

    Returns
    -------
    numpy.ndarray of float
        One value per image, in the same order as ``image_paths``; 0.0
        where nothing was detected.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    paths = [str(p) for p in image_paths]
    best = {}
    # stream=True yields results one batch at a time instead of holding
    # every result in memory at once.
    for result in model.predict(paths, imgsz=imgsz, batch=batch, conf=0.001,
                                device=pick_device() if device is None else device,
                                stream=True, verbose=False):
        conf = result.boxes.conf
        best[str(Path(result.path))] = float(conf.max()) if len(conf) else 0.0
    # Re-order by input path rather than trusting the output order.
    return np.array([best[str(Path(p))] for p in paths])


def inspect_tray(weights, image_path, threshold, imgsz=640, device=None):
    """
    Inspect one tray image and return a pass/fail verdict.

    Parameters
    ----------
    weights : str or pathlib.Path
        Trained weights.
    image_path : str or pathlib.Path
        Tray image (validated by :func:`rasher_qc.tray.load_image`).
    threshold : float
        Decision threshold chosen on the validation set.
    imgsz : int, optional
        Inference image size; should match training.
    device : int or str, optional
        Compute device; chosen automatically if None.

    Returns
    -------
    dict
        ``verdict`` (``"PASS"`` or ``"FAIL"``), ``threshold``, and
        ``defects``: a list of ``{"class", "confidence", "box_xyxy"}``
        for each detection at or above the threshold, where
        ``box_xyxy`` is ``[x1, y1, x2, y2]`` in pixels of the input image.

    Raises
    ------
    FileNotFoundError, ValueError
        If the image is missing or unsupported (see ``load_image``).
    """
    from ultralytics import YOLO

    image = load_image(image_path)
    model = YOLO(str(weights))
    result = model.predict(image, imgsz=imgsz, conf=threshold, verbose=False,
                           device=pick_device() if device is None else device)[0]
    defects = [
        {"class": model.names[int(c)], "confidence": round(float(p), 3),
         "box_xyxy": [round(float(v), 1) for v in xyxy]}
        for c, p, xyxy in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy)
    ]
    return {"verdict": "FAIL" if defects else "PASS", "threshold": threshold,
            "defects": defects}
