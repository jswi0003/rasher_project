"""
Measure where inference time goes, for the Optimisation and Performance section.

Three measurements are made on a trained model:

1. **Stage timing** (:func:`time_inference`): for every image, the
   pre-processing, inference and post-processing times reported by
   Ultralytics (which synchronises the GPU before reading the clock), plus
   the end-to-end wall-clock time measured here with ``time.perf_counter``
   (this also includes reading the image from disk). Warm-up runs are
   discarded first, because the first GPU calls include one-off set-up.
2. **Function profile** (:func:`profile_inference`): Python's built-in
   profiler, ``cProfile``, records how long is spent inside every function
   called during inference, showing which parts of the pipeline dominate.
3. **Computational cost** (:func:`model_gflops`): floating-point operations
   per image, used with the measured time to estimate the achieved
   throughput in GFLOP/s.

Timing is repeated for several confidence thresholds and input sizes. A low
threshold (0.001, as used during evaluation) keeps thousands of candidate
boxes, so non-maximum suppression (post-processing) has much more work to
do than at the operating threshold.

Run from the repository root (with the venv active)::

    python -m rasher_qc.profiling

Results are saved to ``results/profiling/``.
"""

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

import pandas as pd

from .data import DATA_DIR, IMAGE_EXTENSIONS
from .detector import RESULTS_DIR, RUNS_DIR, device_name, pick_device

#: Trained weights profiled by default.
DEFAULT_WEIGHTS = RUNS_DIR / "yolov8n" / "weights" / "best.pt"

#: Images used for timing. Only speed is measured, so using the test split
#: here does not affect the accuracy results.
DEFAULT_IMAGES = DATA_DIR / "pork-rasher-error-packaging-v4" / "test" / "images"


def list_images(folder):
    """
    List the image files in a folder, sorted by name.

    Parameters
    ----------
    folder : str or pathlib.Path
        Folder to search.

    Returns
    -------
    list of pathlib.Path

    Raises
    ------
    FileNotFoundError
        If the folder contains no images.
    """
    images = sorted(p for p in Path(folder).iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"no images found in {folder}")
    return images


def model_gflops(model, imgsz=640):
    """
    Return the computational cost of one forward pass, in GFLOPs.

    Parameters
    ----------
    model : ultralytics.YOLO
        Loaded model.
    imgsz : int, optional
        Square input size the cost is computed for.

    Returns
    -------
    float or None
        Billions of floating-point operations per image, or None if
        Ultralytics cannot compute it. This is for a square ``imgsz x imgsz``
        input; during prediction 720 x 540 images are padded only to
        640 x 480, so the real cost is about 25 % lower.
    """
    from ultralytics.utils.torch_utils import get_flops

    try:
        return float(get_flops(model.model, imgsz)) or None
    except Exception:  # cost estimate is optional; never stop the profile
        return None


def time_inference(model, images, imgsz=640, conf=0.2, n_warmup=10, device=None):
    """
    Time each prediction stage for every image, one image at a time.

    One image per call mirrors an inspection station, which receives one
    frame at a time rather than a batch.

    Parameters
    ----------
    model : ultralytics.YOLO
        Loaded model.
    images : sequence of pathlib.Path
        Images to process.
    imgsz : int, optional
        Inference input size in pixels.
    conf : float, optional
        Confidence threshold; boxes below it are discarded before
        non-maximum suppression.
    n_warmup : int, optional
        Number of initial predictions run and discarded.
    device : int or str, optional
        Compute device; chosen automatically if None.

    Returns
    -------
    pandas.DataFrame
        One row per image with columns ``preprocess_ms``, ``inference_ms``,
        ``postprocess_ms`` (from Ultralytics), ``wall_ms`` (end to end,
        including reading the file) and ``n_boxes`` (detections kept).
    """
    device = pick_device() if device is None else device
    settings = dict(imgsz=imgsz, conf=conf, device=device, verbose=False)

    for path in images[:n_warmup]:
        model.predict(str(path), **settings)

    rows = []
    for path in images:
        start = time.perf_counter()
        result = model.predict(str(path), **settings)[0]
        wall_ms = (time.perf_counter() - start) * 1000
        rows.append({
            "image": path.name,
            "preprocess_ms": result.speed["preprocess"],
            "inference_ms": result.speed["inference"],
            "postprocess_ms": result.speed["postprocess"],
            "wall_ms": wall_ms,
            "n_boxes": len(result.boxes),
        })
    return pd.DataFrame(rows)


def summarise(timings, gflops=None):
    """
    Summarise per-image timings.

    Parameters
    ----------
    timings : pandas.DataFrame
        Output of :func:`time_inference`.
    gflops : float, optional
        Cost per image from :func:`model_gflops`, used to estimate the
        achieved throughput during the inference stage.

    Returns
    -------
    dict
        Mean of each stage in ms, median and 95th-percentile wall time,
        images per second (from the mean wall time), mean boxes per image,
        each stage's share of the stage total, and ``achieved_gflop_per_s``
        (None if ``gflops`` is not given).
    """
    stages = ["preprocess_ms", "inference_ms", "postprocess_ms"]
    means = {s: float(timings[s].mean()) for s in stages}
    stage_total = sum(means.values())
    wall = timings["wall_ms"]
    return {
        **means,
        "wall_ms_mean": float(wall.mean()),
        "wall_ms_median": float(wall.median()),
        "wall_ms_p95": float(wall.quantile(0.95)),
        "images_per_s": 1000.0 / float(wall.mean()),
        "boxes_per_image": float(timings["n_boxes"].mean()),
        **{f"{s.removesuffix('_ms')}_share": means[s] / stage_total for s in stages},
        "achieved_gflop_per_s": (gflops / (means["inference_ms"] / 1000)
                                 if gflops else None),
    }


def profile_inference(model, images, imgsz=640, conf=0.2, device=None, top=25):
    """
    Profile the prediction loop with cProfile.

    Parameters
    ----------
    model : ultralytics.YOLO
        Loaded model (warm it up first so set-up is not profiled).
    images : sequence of pathlib.Path
        Images to process.
    imgsz : int, optional
        Inference input size.
    conf : float, optional
        Confidence threshold.
    device : int or str, optional
        Compute device; chosen automatically if None.
    top : int, optional
        Number of functions to list.

    Returns
    -------
    tuple of (cProfile.Profile, str)
        The raw profile (can be saved and opened in a viewer such as
        ``snakeviz``) and a text table of the ``top`` functions sorted by
        cumulative time.

    Notes
    -----
    cProfile measures time spent in Python functions. GPU work runs
    asynchronously, so its time shows up wherever Python next waits for
    the GPU (usually when results are copied back), not inside the layer
    that launched it.
    """
    device = pick_device() if device is None else device
    profiler = cProfile.Profile()
    profiler.enable()
    for path in images:
        model.predict(str(path), imgsz=imgsz, conf=conf, device=device, verbose=False)
    profiler.disable()

    text = io.StringIO()
    pstats.Stats(profiler, stream=text).sort_stats("cumulative").print_stats(top)
    return profiler, text.getvalue()


def main(argv=None):
    """Run the timing sweep and the profile, and save the results."""
    parser = argparse.ArgumentParser(description="Profile inference speed.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--confs", type=float, nargs="+", default=[0.001, 0.2],
                        help="confidence thresholds to time")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[320, 480, 640],
                        help="input sizes to time")
    parser.add_argument("--profile-conf", type=float, default=0.2,
                        help="threshold used for the cProfile run")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    out_dir = RESULTS_DIR / "profiling"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    model = YOLO(str(args.weights))
    images = list_images(args.images)
    print(f"Device: {device_name(device)}; {len(images)} images; weights {args.weights}")

    # 1. Stage timing for every combination of input size and threshold
    rows, per_image = [], []
    for imgsz in args.imgsz:
        gflops = model_gflops(model, imgsz)
        for conf in args.confs:
            timings = time_inference(model, images, imgsz=imgsz, conf=conf, device=device)
            per_image.append(timings.assign(imgsz=imgsz, conf=conf))
            rows.append({"imgsz": imgsz, "conf": conf, "gflops": gflops,
                         **summarise(timings, gflops)})
            r = rows[-1]
            print(f"imgsz {imgsz:4d}, conf {conf:<5}: "
                  f"pre {r['preprocess_ms']:.2f} + inf {r['inference_ms']:.2f} + "
                  f"post {r['postprocess_ms']:.2f} ms; wall {r['wall_ms_mean']:.2f} ms "
                  f"({r['images_per_s']:.0f} img/s); {r['boxes_per_image']:.1f} boxes/img")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "timing_summary.csv", index=False)
    pd.concat(per_image).to_csv(out_dir / "timing_per_image.csv", index=False)

    # 2. Function-level profile at the operating settings
    profile_size = 640 if 640 in args.imgsz else args.imgsz[-1]
    profiler, text = profile_inference(model, images, imgsz=profile_size,
                                       conf=args.profile_conf, device=device)
    profiler.dump_stats(out_dir / "inference.prof")
    (out_dir / "cprofile_top.txt").write_text(text, encoding="utf-8")

    print(f"\nTop functions by cumulative time (imgsz {profile_size}, "
          f"conf {args.profile_conf}):")
    print("\n".join(text.splitlines()[:40]))
    print(f"\nSaved {out_dir}")


if __name__ == "__main__":
    main()
