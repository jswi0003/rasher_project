"""
Dataset acquisition and structural checks for the pork rasher packaging dataset.

This module downloads Dataset 3 (Roboflow Universe, "pork-rasher-error-packaging")
and provides checks that the downloaded data is complete before any training.

Expected layout of a downloaded dataset (Roboflow "yolov8" export format)::

    data/pork-rasher-error-packaging-v4/
        data.yaml               class names and split locations
        train/images/*.jpg      training images
        train/labels/*.txt      one label file per image
        valid/images/ ...       validation split (same structure)
        test/images/ ...        test split (same structure)

Label format (YOLO): each line of a label file describes one bounding box as::

    class_id  x_centre  y_centre  width  height

where ``class_id`` is an integer index into the class names in ``data.yaml``,
and the four coordinates are normalised to [0, 1] by the image width and height.
An empty label file means the image contains no annotated defects.

The Roboflow API key is read from the ``ROBOFLOW_API_KEY`` environment
variable and is never stored in code or committed to the repository.
"""

import os
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Dataset identifiers (from the Roboflow Universe download dialog)
# ---------------------------------------------------------------------------

#: Roboflow workspace that publishes the dataset.
ROBOFLOW_WORKSPACE = "hello-2aqe0"

#: Roboflow project identifier for the pork rasher packaging dataset.
ROBOFLOW_PROJECT = "pork-rasher-error-packaging"

#: Dataset version specified in the MMA3001 project datasets brief.
DEFAULT_VERSION = 4

#: Roboflow export format. "yolov8" gives YOLO .txt labels plus a data.yaml
#: file, which all Ultralytics YOLO models can read.
DEFAULT_FORMAT = "yolov8"

#: Name of the environment variable that holds the Roboflow API key.
API_KEY_ENV_VAR = "ROBOFLOW_API_KEY"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Paths are built from this file's location rather than relative strings like
# "data/", because relative paths resolve from wherever Python is run.
# A notebook in examples/ runs from examples/, so "data/" would point to
# examples/data/ instead of the repository's data/ folder.

#: Repository root folder (this file is at <root>/rasher_qc/data.py).
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Folder that datasets are downloaded into. Its contents are git-ignored.
DATA_DIR = REPO_ROOT / "data"

# ---------------------------------------------------------------------------
# Dataset structure
# ---------------------------------------------------------------------------

#: Split folder names used by Roboflow exports, in order.
SPLITS = ("train", "valid", "test")

#: Image file extensions counted as dataset images (compared in lowercase).
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


def get_api_key(env_var=API_KEY_ENV_VAR):
    """
    Read the Roboflow API key from an environment variable.

    The key is deliberately kept out of code, notebooks and the repository.
    It must be set as a user environment variable before calling this
    function (see the README setup instructions).

    Parameters
    ----------
    env_var : str, optional
        Name of the environment variable holding the key.
        Defaults to ``"ROBOFLOW_API_KEY"``.

    Returns
    -------
    str
        The API key, with any leading or trailing whitespace removed.

    Raises
    ------
    RuntimeError
        If the environment variable is not set, or is empty or whitespace only.

    Examples
    --------
    >>> key = get_api_key()  # doctest: +SKIP
    """
    # Strip whitespace: a stray space or newline is easy to paste in by
    # accident and would make Roboflow reject an otherwise valid key.
    key = os.environ.get(env_var, "").strip()

    if not key:
        # The message explains how to fix the problem but never includes
        # the key itself, so it is safe to show in error output.
        raise RuntimeError(
            f"Environment variable {env_var} is not set or is empty. "
            "Add it under Windows 'Edit environment variables for your account', "
            "then fully close and reopen VS Code so new terminals can see it."
        )

    return key

def download_dataset(version=DEFAULT_VERSION, model_format=DEFAULT_FORMAT,
                     data_dir=DATA_DIR, overwrite=False):
    """
    Download the pork rasher dataset from Roboflow Universe.

    The dataset is saved to ``data_dir / "pork-rasher-error-packaging-v<version>"``.
    If that folder already exists and is not empty, nothing is downloaded
    unless ``overwrite`` is True, so this function is safe to re-run.

    Parameters
    ----------
    version : int, optional
        Roboflow dataset version. Defaults to 4 (the version in the brief).
    model_format : str, optional
        Roboflow export format. Defaults to ``"yolov8"``.
    data_dir : pathlib.Path, optional
        Parent folder for the download. Defaults to the repository's ``data/``.
    overwrite : bool, optional
        If True, download again even if the folder already has files.

    Returns
    -------
    pathlib.Path
        Folder containing ``data.yaml`` and the train/valid/test splits.

    Examples
    --------
    >>> dataset_dir = download_dataset()  # doctest: +SKIP
    """
    target = Path(data_dir) / f"{ROBOFLOW_PROJECT}-v{version}"

    # Skip the download if a previous run already filled the folder.
    if target.exists() and any(target.iterdir()) and not overwrite:
        print(f"Dataset already present at {target}; skipping download.")
        return target

    # Imported here so the rest of this module (paths, checks) works even
    # if the roboflow package is not installed.
    from roboflow import Roboflow

    rf = Roboflow(api_key=get_api_key())
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    project.version(version).download(model_format, location=str(target),
                                      overwrite=overwrite)

    return target


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

def read_class_names(dataset_dir):
    """
    Read the class names from a dataset's ``data.yaml``.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Folder containing ``data.yaml``.

    Returns
    -------
    list of str
        Class names, ordered so that ``names[class_id]`` is the name of
        that class id.

    Raises
    ------
    FileNotFoundError
        If ``data.yaml`` does not exist.
    ValueError
        If ``nc`` (the declared number of classes) does not match the
        number of names listed, which would make class ids ambiguous.

    Examples
    --------
    >>> read_class_names(DATA_DIR / "pork-rasher-error-packaging-v4")  # doctest: +SKIP
    ['loose-meat', 'packaging-error', 'twisted-meat', 'unsealed', 'wrinkle']
    """
    yaml_path = Path(dataset_dir) / "data.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names = config["names"]
    # YOLO allows names as a list or as a {id: name} mapping; normalise to a
    # list ordered by id so both forms are handled the same way.
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    nc = config.get("nc", len(names))
    if nc != len(names):
        raise ValueError(
            f"{yaml_path}: nc = {nc} but {len(names)} class names are listed"
        )

    return list(names)


def check_label_line(line, num_classes):
    """
    Check one line of a YOLO label file.

    A valid line has exactly five values,
    ``class_id x_centre y_centre width height``, where ``class_id`` is an
    integer in ``[0, num_classes - 1]``, every coordinate is in ``[0, 1]``
    and the box has non-zero width and height.

    Parameters
    ----------
    line : str
        One line of a label file (surrounding whitespace is ignored).
    num_classes : int
        Number of classes in the dataset.

    Returns
    -------
    str or None
        ``None`` if the line is valid, otherwise a short description of
        the first problem found.

    Examples
    --------
    >>> check_label_line("0 0.5 0.5 0.2 0.1", num_classes=5) is None
    True
    >>> check_label_line("5 0.5 0.5 0.2 0.1", num_classes=5)
    'class id 5 is outside 0-4'
    """
    parts = line.split()

    # Five values = a bounding box. More than five usually means a polygon
    # (segmentation) label, which this detection workflow does not support.
    if len(parts) != 5:
        return f"expected 5 values, found {len(parts)}"

    try:
        class_id = int(parts[0])
        coords = [float(p) for p in parts[1:]]
    except ValueError:
        return "values are not numbers"

    if not 0 <= class_id < num_classes:
        return f"class id {class_id} is outside 0-{num_classes - 1}"

    # Written as "not inside" so that NaN (which fails every comparison)
    # is also rejected.
    if not all(0.0 <= c <= 1.0 for c in coords):
        return "coordinates outside [0, 1]"

    _, _, width, height = coords
    if width <= 0 or height <= 0:
        return "box has zero width or height"

    return None


def check_dataset(dataset_dir):
    """
    Check that a downloaded YOLO dataset is structurally complete.

    For every split this checks that the ``images/`` and ``labels/``
    folders exist, that every image has a label file and vice versa, and
    that every label line is valid (see :func:`check_label_line`).
    Nothing is modified; problems are collected and returned so they can
    all be fixed in one pass.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Folder containing ``data.yaml`` and the split folders.

    Returns
    -------
    dict
        ``"splits"``
            ``{split: {"images", "labels", "boxes", "empty_labels"}}``,
            where ``boxes`` counts valid label lines only and
            ``empty_labels`` counts label files with no boxes
            (defect-free images).
        ``"class_counts"``
            ``{class_name: number of valid boxes across all splits}``.
        ``"problems"``
            List of human-readable problem descriptions. An empty list
            means the dataset passed every check.

    Examples
    --------
    >>> report = check_dataset(DATA_DIR / "pork-rasher-error-packaging-v4")  # doctest: +SKIP
    >>> report["problems"]  # doctest: +SKIP
    []
    """
    dataset_dir = Path(dataset_dir)
    names = read_class_names(dataset_dir)

    report = {
        "splits": {},
        "class_counts": {name: 0 for name in names},
        "problems": [],
    }
    problems = report["problems"]

    for split in SPLITS:
        image_dir = dataset_dir / split / "images"
        label_dir = dataset_dir / split / "labels"

        missing = [d.name for d in (image_dir, label_dir) if not d.is_dir()]
        for folder in missing:
            problems.append(f"{split}: missing folder {folder}/")

        # Map file stem -> path so images and labels can be paired by name.
        images = {}
        if image_dir.is_dir():
            images = {p.stem: p for p in image_dir.iterdir()
                      if p.suffix.lower() in IMAGE_EXTENSIONS}
        labels = {}
        if label_dir.is_dir():
            labels = {p.stem: p for p in label_dir.glob("*.txt")}

        stats = {"images": len(images), "labels": len(labels),
                 "boxes": 0, "empty_labels": 0}
        report["splits"][split] = stats

        # Only pair files when both folders exist; otherwise every image
        # would also be reported as unlabelled, burying the real problem.
        if not missing:
            for stem in sorted(images.keys() - labels.keys()):
                problems.append(f"{split}: image {stem} has no label file")
            for stem in sorted(labels.keys() - images.keys()):
                problems.append(f"{split}: label {stem}.txt has no image")

        for stem in sorted(labels):
            lines = labels[stem].read_text(encoding="utf-8").splitlines()
            # Blank lines are ignored; a file with no boxes is a valid
            # "no defects" label, so it is counted rather than reported.
            if not any(line.strip() for line in lines):
                stats["empty_labels"] += 1
                continue

            for line_no, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                error = check_label_line(line, len(names))
                if error:
                    problems.append(f"{split}/{stem}.txt line {line_no}: {error}")
                else:
                    stats["boxes"] += 1
                    report["class_counts"][names[int(line.split()[0])]] += 1

    return report


def write_training_yaml(dataset_dir, output_path=None):
    """
    Write a dataset config with absolute paths for Ultralytics training.

    Roboflow's ``data.yaml`` gives split locations relative to the yaml
    file (e.g. ``../train/images``), which Ultralytics can resolve
    against the wrong folder depending on where training is launched.
    This writes a config with an absolute ``path`` root and split folders
    relative to it, so training works from any working directory.

    Parameters
    ----------
    dataset_dir : str or pathlib.Path
        Folder containing ``data.yaml`` and the split folders.
    output_path : str or pathlib.Path, optional
        Where to write the config. Defaults to
        ``dataset_dir / "training.yaml"``.

    Returns
    -------
    pathlib.Path
        Path of the written config file.

    Examples
    --------
    >>> cfg = write_training_yaml(DATA_DIR / "pork-rasher-error-packaging-v4")  # doctest: +SKIP
    """
    dataset_dir = Path(dataset_dir).resolve()
    if output_path is None:
        output_path = dataset_dir / "training.yaml"
    output_path = Path(output_path)

    config = {
        "path": str(dataset_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": read_class_names(dataset_dir),
    }
    output_path.write_text(yaml.safe_dump(config, sort_keys=False),
                           encoding="utf-8")
    return output_path
