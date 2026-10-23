"""
Tray-level pass/fail inspection built on top of defect detections.

The detector finds and localises individual defects (boxes), but the
decision a packaging line needs is per tray: **pass** it on, or **fail** it
for rework. This module turns detections into that decision and scores it:

* :func:`tray_is_defective` reads the ground-truth verdict from a label file.
* :func:`tray_verdicts` applies a confidence threshold to detections.
* :func:`tray_metrics` and :func:`threshold_sweep` score verdicts.
* :func:`choose_threshold` picks the operating threshold on validation data.
* :func:`load_image` validates an input image before it reaches the model.

Decision rule: a tray FAILS if the most confident defect detection in the
image has confidence ``>= threshold``; otherwise it PASSES.

Nothing here depends on PyTorch, so these functions are fast to unit-test.
"""

import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

#: Image shape (height, width, channels) of every image in the dataset.
#: Other sizes are accepted (YOLO resizes internally) but are outside the
#: conditions the model was validated on, so they trigger a warning.
EXPECTED_SHAPE = (540, 720, 3)

#: Image file extensions accepted by :func:`load_image` (lowercase).
SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


def tray_is_defective(label_path):
    """
    Read the ground-truth tray verdict from a YOLO label file.

    Parameters
    ----------
    label_path : str or pathlib.Path
        Label file for one image.

    Returns
    -------
    bool
        True if the file contains at least one box (the tray has a
        labelled defect), False if it is empty (a defect-free tray).
    """
    text = Path(label_path).read_text(encoding="utf-8")
    return any(line.strip() for line in text.splitlines())


def tray_verdicts(max_confidences, threshold):
    """
    Convert per-image maximum detection confidences into tray verdicts.

    Parameters
    ----------
    max_confidences : array_like of float
        Highest defect confidence found in each image, in ``[0, 1]``.
        Use 0.0 for images with no detections at all.
    threshold : float
        Confidence at or above which a detection fails the tray, in
        ``[0, 1]``.

    Returns
    -------
    numpy.ndarray of bool
        True = FAIL (defect detected), False = PASS.

    Raises
    ------
    ValueError
        If ``threshold`` is outside ``[0, 1]``.

    Examples
    --------
    >>> tray_verdicts([0.9, 0.1, 0.0], threshold=0.5).tolist()
    [True, False, False]
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    return np.asarray(max_confidences, dtype=float) >= threshold


def _ratio(numerator, denominator):
    """Return numerator / denominator, or NaN when the denominator is 0."""
    return numerator / denominator if denominator else float("nan")


def tray_metrics(is_defective, predicted_fail):
    """
    Score tray verdicts against ground truth.

    "Positive" means defective. In food packaging a missed defect (false
    negative) ships a bad product, while a false reject (false positive)
    only costs a manual re-check, so recall and miss rate are the
    primary measures and accuracy is reported for completeness only.

    Parameters
    ----------
    is_defective : array_like of bool
        Ground truth, True for trays with a labelled defect.
    predicted_fail : array_like of bool
        Verdicts, True for trays the method fails.

    Returns
    -------
    dict
        Counts ``tp``, ``fn``, ``fp``, ``tn`` and the rates
        ``recall`` (defective trays caught), ``miss_rate`` (1 - recall),
        ``false_reject_rate`` (good trays failed), ``precision``,
        ``accuracy`` and ``balanced_accuracy``. A rate whose denominator
        is zero is NaN rather than a misleading 0 or 1.

    Raises
    ------
    ValueError
        If the two inputs have different lengths.

    Examples
    --------
    >>> m = tray_metrics([True, True, False, False], [True, False, True, False])
    >>> m["recall"], m["false_reject_rate"]
    (0.5, 0.5)
    """
    truth = np.asarray(is_defective, dtype=bool)
    pred = np.asarray(predicted_fail, dtype=bool)
    if truth.shape != pred.shape:
        raise ValueError(f"length mismatch: {truth.size} labels, {pred.size} verdicts")

    tp = int(np.sum(truth & pred))
    fn = int(np.sum(truth & ~pred))
    fp = int(np.sum(~truth & pred))
    tn = int(np.sum(~truth & ~pred))

    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": recall,
        "miss_rate": _ratio(fn, tp + fn),
        "false_reject_rate": _ratio(fp, fp + tn),
        "precision": _ratio(tp, tp + fp),
        "accuracy": _ratio(tp + tn, truth.size),
        # Mean of the per-class rates, so a large defective majority
        # cannot hide poor performance on good trays (or vice versa).
        "balanced_accuracy": (recall + specificity) / 2,
    }


def threshold_sweep(is_defective, max_confidences, thresholds=None):
    """
    Evaluate tray metrics over a range of confidence thresholds.

    Parameters
    ----------
    is_defective : array_like of bool
        Ground truth per image.
    max_confidences : array_like of float
        Highest detection confidence per image (0.0 if none).
    thresholds : array_like of float, optional
        Thresholds to test. Defaults to 0.05, 0.10, ..., 0.95.

    Returns
    -------
    pandas.DataFrame
        One row per threshold, with a ``threshold`` column followed by the
        columns returned by :func:`tray_metrics`.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.951, 0.05), 2)

    rows = []
    for t in thresholds:
        verdicts = tray_verdicts(max_confidences, float(t))
        rows.append({"threshold": float(t), **tray_metrics(is_defective, verdicts)})
    return pd.DataFrame(rows)


def choose_threshold(sweep, min_recall=0.95):
    """
    Pick the operating threshold from a validation-set sweep.

    Rule: the **highest** threshold whose recall is at least
    ``min_recall``. Raising the threshold can only lower both recall and
    the false-reject rate, so this keeps the required share of defects
    caught while failing as few good trays as possible. If no threshold
    reaches ``min_recall``, the threshold with the highest recall is
    returned (lowest threshold on ties).

    Only validation data should be passed here; choosing the threshold on
    the test set would make the test result optimistic.

    Parameters
    ----------
    sweep : pandas.DataFrame
        Output of :func:`threshold_sweep`.
    min_recall : float, optional
        Required fraction of defective trays caught. Default 0.95.

    Returns
    -------
    float
        The chosen threshold.
    """
    ok = sweep[sweep["recall"] >= min_recall]
    if len(ok):
        return float(ok["threshold"].max())
    best = sweep[sweep["recall"] == sweep["recall"].max()]
    return float(best["threshold"].min())


def load_image(path):
    """
    Load and validate one tray image for inspection.

    Parameters
    ----------
    path : str or pathlib.Path
        Image file. Supported types: .jpg, .jpeg, .png, .bmp.

    Returns
    -------
    numpy.ndarray
        Image as an 8-bit BGR array of shape (height, width, 3), the
        layout OpenCV and Ultralytics expect. A PNG alpha channel is
        dropped.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the extension is unsupported, the file cannot be decoded as an
        image, the image is greyscale (the model was trained only on
        colour images), or it is not 8-bit.

    Warns
    -----
    UserWarning
        If the image is not 720 x 540, the size the model was validated on.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported file type '{path.suffix}': {path.name}")

    # np.fromfile + imdecode instead of cv2.imread: imread silently fails
    # on Windows paths containing non-ASCII characters.
    # IMREAD_UNCHANGED keeps greyscale as greyscale so it can be rejected,
    # rather than being silently copied into three identical channels.
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED) if raw.size else None
    if image is None:
        raise ValueError(f"file could not be decoded as an image: {path.name}")

    if image.dtype != np.uint8:
        raise ValueError(f"expected an 8-bit image, got {image.dtype}: {path.name}")
    if image.ndim == 2 or image.shape[2] == 1:
        raise ValueError(f"greyscale images are not supported: {path.name}")
    if image.shape[2] == 4:
        image = image[:, :, :3]

    if image.shape != EXPECTED_SHAPE:
        h, w = image.shape[:2]
        warnings.warn(
            f"{path.name} is {w} x {h}; the model was validated on 720 x 540 "
            "images, so results at this size are unverified.",
            stacklevel=2,
        )
    return image
