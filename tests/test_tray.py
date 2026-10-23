"""
Tests for the tray-level decision logic and input validation in rasher_qc.tray.

Expected values are worked out by hand from small examples, so these tests
check the arithmetic independently of the code under test.
"""

import math

import cv2
import numpy as np
import pytest

from rasher_qc.tray import (choose_threshold, load_image, threshold_sweep,
                            tray_is_defective, tray_metrics, tray_verdicts)


# --- ground truth and verdicts ---------------------------------------------

def test_tray_is_defective(tmp_path):
    (tmp_path / "bad.txt").write_text("1 0.5 0.5 0.2 0.1\n")
    (tmp_path / "good.txt").write_text("")
    (tmp_path / "blank_lines.txt").write_text("\n  \n")
    assert tray_is_defective(tmp_path / "bad.txt")
    assert not tray_is_defective(tmp_path / "good.txt")
    assert not tray_is_defective(tmp_path / "blank_lines.txt")


def test_verdict_threshold_is_inclusive():
    # A confidence exactly at the threshold fails the tray.
    assert tray_verdicts([0.5, 0.49999, 0.0], 0.5).tolist() == [True, False, False]


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_verdict_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        tray_verdicts([0.5], threshold)


# --- metrics -----------------------------------------------------------------

def test_metrics_hand_worked_example():
    # 4 defective trays (3 caught, 1 missed), 2 good trays (1 falsely rejected)
    truth = [True, True, True, True, False, False]
    fail = [True, True, True, False, True, False]
    m = tray_metrics(truth, fail)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 1)
    assert m["recall"] == pytest.approx(0.75)
    assert m["miss_rate"] == pytest.approx(0.25)
    assert m["false_reject_rate"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.75)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["balanced_accuracy"] == pytest.approx((0.75 + 0.5) / 2)


def test_always_fail_baseline_has_high_accuracy_but_rejects_every_good_tray():
    # Mirrors the validation split: 106 defective, 14 good trays.
    truth = np.array([True] * 106 + [False] * 14)
    m = tray_metrics(truth, np.ones_like(truth))
    assert m["recall"] == 1.0
    assert m["false_reject_rate"] == 1.0
    assert m["accuracy"] == pytest.approx(106 / 120)   # looks good, is useless
    assert m["balanced_accuracy"] == pytest.approx(0.5)


def test_undefined_rate_is_nan_not_zero():
    m = tray_metrics([False, False], [False, False])   # no defective trays
    assert math.isnan(m["recall"])
    assert m["false_reject_rate"] == 0.0


def test_metrics_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        tray_metrics([True, False], [True])


# --- threshold sweep and selection ------------------------------------------

def test_sweep_recall_never_increases_with_threshold():
    rng = np.random.default_rng(0)
    truth = rng.random(200) < 0.8
    conf = np.clip(truth * 0.5 + rng.random(200) * 0.5, 0, 1)
    sweep = threshold_sweep(truth, conf)
    assert np.all(np.diff(sweep["recall"]) <= 0)
    assert np.all(np.diff(sweep["false_reject_rate"]) <= 0)


def test_choose_threshold_takes_highest_meeting_recall():
    truth = [True, True, True, True, False]
    conf = [0.9, 0.8, 0.7, 0.3, 0.6]
    sweep = threshold_sweep(truth, conf, thresholds=[0.2, 0.5, 0.65, 0.75])
    # recall: 0.2 -> 1.0, 0.5 -> 0.75, 0.65 -> 0.75, 0.75 -> 0.5
    assert choose_threshold(sweep, min_recall=0.75) == 0.65
    assert choose_threshold(sweep, min_recall=1.0) == 0.2


def test_choose_threshold_falls_back_to_best_recall():
    sweep = threshold_sweep([True, True], [0.4, 0.1], thresholds=[0.3, 0.5])
    assert choose_threshold(sweep, min_recall=0.99) == 0.3


# --- input validation ---------------------------------------------------------

def write_image(path, array):
    """Save an array with OpenCV and return the path."""
    assert cv2.imwrite(str(path), array)
    return path


def test_load_valid_image(tmp_path):
    path = write_image(tmp_path / "tray.jpg", np.zeros((540, 720, 3), np.uint8))
    assert load_image(path).shape == (540, 720, 3)


def test_png_alpha_channel_is_dropped(tmp_path):
    path = write_image(tmp_path / "tray.png", np.zeros((540, 720, 4), np.uint8))
    assert load_image(path).shape == (540, 720, 3)


def test_other_size_is_accepted_with_warning(tmp_path):
    path = write_image(tmp_path / "small.jpg", np.zeros((100, 200, 3), np.uint8))
    with pytest.warns(UserWarning, match="200 x 100"):
        load_image(path)


def test_greyscale_is_rejected(tmp_path):
    path = write_image(tmp_path / "grey.png", np.zeros((540, 720), np.uint8))
    with pytest.raises(ValueError, match="greyscale"):
        load_image(path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.jpg")


def test_unsupported_extension_is_rejected(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    with pytest.raises(ValueError, match="unsupported file type"):
        load_image(tmp_path / "notes.txt")


def test_corrupt_image_is_rejected(tmp_path):
    (tmp_path / "broken.jpg").write_bytes(b"not really a jpeg")
    with pytest.raises(ValueError, match="could not be decoded"):
        load_image(tmp_path / "broken.jpg")
