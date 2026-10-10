"""
Tests for the dataset checks in rasher_qc.data.

Most tests build a tiny fake dataset in a temporary folder (pytest's
``tmp_path``), so they run in seconds and do not need the real download.
The final test checks the real dataset and is skipped if it is not present.
"""

import pytest
import yaml

from rasher_qc.data import (
    DATA_DIR,
    check_dataset,
    check_label_line,
    read_class_names,
    write_training_yaml,
)

CLASS_NAMES = ["loose-meat", "packaging-error", "twisted-meat", "unsealed", "wrinkle"]


def make_dataset(root, labels=None):
    """
    Create a minimal Roboflow-style YOLO dataset under ``root``.

    ``labels`` maps (split, stem) to label-file text. Each entry gets a
    matching empty .jpg (the checks only look at file names, not pixels).
    By default every split gets one valid image/label pair.
    """
    if labels is None:
        labels = {(s, "img0"): "1 0.5 0.5 0.2 0.1\n" for s in ("train", "valid", "test")}

    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)

    for (split, stem), text in labels.items():
        (root / split / "images" / f"{stem}.jpg").write_bytes(b"")
        (root / split / "labels" / f"{stem}.txt").write_text(text)

    config = {"names": CLASS_NAMES, "nc": len(CLASS_NAMES),
              "train": "../train/images", "val": "../valid/images",
              "test": "../test/images"}
    (root / "data.yaml").write_text(yaml.safe_dump(config))
    return root


# --- check_label_line -------------------------------------------------------

@pytest.mark.parametrize("line, expected", [
    ("0 0.5 0.5 0.2 0.1", None),
    ("4 0.0 1.0 1.0 1.0", None),                      # boundary values allowed
    ("5 0.5 0.5 0.2 0.1", "class id 5 is outside 0-4"),
    ("-1 0.5 0.5 0.2 0.1", "class id -1 is outside 0-4"),
    ("1 1.2 0.5 0.2 0.1", "coordinates outside [0, 1]"),
    ("1 0.5 0.5 0 0.1", "box has zero width or height"),
    ("1 0.5 0.5 0.2", "expected 5 values, found 4"),
    ("1 0.1 0.1 0.2 0.2 0.3 0.3", "expected 5 values, found 7"),  # polygon
    ("a 0.5 0.5 0.2 0.1", "values are not numbers"),
])
def test_check_label_line(line, expected):
    assert check_label_line(line, num_classes=5) == expected


# --- check_dataset ----------------------------------------------------------

def test_valid_dataset_has_no_problems(tmp_path):
    report = check_dataset(make_dataset(tmp_path))
    assert report["problems"] == []
    assert report["splits"]["train"] == {"images": 1, "labels": 1,
                                         "boxes": 1, "empty_labels": 0}
    assert report["class_counts"]["packaging-error"] == 3


def test_empty_label_file_is_allowed_and_counted(tmp_path):
    dataset = make_dataset(tmp_path, {("train", "good_tray"): ""})
    report = check_dataset(dataset)
    assert report["problems"] == []
    assert report["splits"]["train"]["empty_labels"] == 1


def test_image_without_label_is_reported(tmp_path):
    dataset = make_dataset(tmp_path)
    (dataset / "train" / "images" / "orphan.jpg").write_bytes(b"")
    assert check_dataset(dataset)["problems"] == ["train: image orphan has no label file"]


def test_label_without_image_is_reported(tmp_path):
    dataset = make_dataset(tmp_path)
    (dataset / "valid" / "labels" / "orphan.txt").write_text("")
    assert check_dataset(dataset)["problems"] == ["valid: label orphan.txt has no image"]


def test_bad_label_line_is_reported_with_location(tmp_path):
    dataset = make_dataset(tmp_path, {("test", "img0"): "1 0.5 0.5 0.2 0.1\n9 0.5 0.5 0.2 0.1\n"})
    report = check_dataset(dataset)
    assert report["problems"] == ["test/img0.txt line 2: class id 9 is outside 0-4"]
    assert report["splits"]["test"]["boxes"] == 1   # only the valid line counted


def test_missing_split_folder_is_reported(tmp_path):
    dataset = make_dataset(tmp_path)
    (dataset / "test" / "labels" / "img0.txt").unlink()
    (dataset / "test" / "labels").rmdir()
    assert "test: missing folder labels/" in check_dataset(dataset)["problems"]


def test_nc_mismatch_raises(tmp_path):
    dataset = make_dataset(tmp_path)
    (dataset / "data.yaml").write_text(yaml.safe_dump({"names": CLASS_NAMES, "nc": 4}))
    with pytest.raises(ValueError, match="nc = 4"):
        read_class_names(dataset)


# --- write_training_yaml ----------------------------------------------------

def test_training_yaml_points_at_split_folders(tmp_path):
    dataset = make_dataset(tmp_path)
    config = yaml.safe_load(write_training_yaml(dataset).read_text())

    root = tmp_path.resolve()
    assert config["path"] == str(root)
    for key in ("train", "val", "test"):
        assert (root / config[key]).is_dir()
    assert config["names"][4] == "wrinkle"


# --- real dataset -----------------------------------------------------------

REAL_DATASET = DATA_DIR / "pork-rasher-error-packaging-v4"


@pytest.mark.skipif(not REAL_DATASET.exists(), reason="dataset not downloaded")
def test_downloaded_dataset_is_complete():
    report = check_dataset(REAL_DATASET)
    assert report["problems"] == []
    # Split sizes recorded in data/README.md for version 4.
    assert {s: v["images"] for s, v in report["splits"].items()} == \
        {"train": 3349, "valid": 120, "test": 80}
