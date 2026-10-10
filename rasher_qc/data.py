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