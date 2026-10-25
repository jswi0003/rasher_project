# Pork rasher packaging defect detection (MMA3001)

Automated visual inspection of packaged pork rashers. A YOLOv8n object detector is
fine-tuned to find and localise five packaging defects (loose meat, packaging error,
twisted meat, unsealed, wrinkle) in overhead images of trays on a production line.
Each image then **passes or fails** inspection: it fails if any defect is detected
with a confidence at or above a threshold chosen on the validation set.

The model is compared with a baseline that fails every tray. Full method, validation
and discussion are in the project report.

## Results (test set, 80 images, threshold 0.20)

| | YOLOv8n | Baseline (fail every tray) |
|---|---|---|
| Defective images caught (recall) | 89.7% (61 / 68) | 100% |
| Good trays wrongly rejected | 25.0% (3 / 12) | 100% |
| Balanced accuracy | 0.82 | 0.50 |
| Detection mAP50 / mAP50-95 | 0.53 / 0.25 | — |
| Speed (RTX 4050 laptop GPU, one image at a time) | 12.5 ms, about 80 images/s | — |

The 95% recall target was met on validation (96.2%) but not on the test set.
Unsealed packs are detected reliably; small packaging-error defects are the main weakness.

## Repository structure

```
rasher_project/
├── rasher_qc/              Python package
│   ├── data.py             download the dataset, check its structure, write the YOLO config
│   ├── detector.py         train, evaluate and run the YOLOv8 model; inspect one image
│   ├── tray.py             pass/fail decision, metrics, threshold choice, input checks
│   ├── experiment.py       one command that runs a full experiment
│   └── profiling.py        timing per stage, cProfile, GFLOPs
├── tests/                  pytest tests (41)
├── docs/
│   ├── api/index.html      HTML documentation of the code (pdoc)
│   └── test_report.html    HTML test report (pytest-html)
├── results/
│   ├── runs/yolov8n/       training run: weights/best.pt, training curves, plots
│   ├── runs/yolov8n_val/   validation plots;  runs/yolov8n_test/  test plots
│   ├── yolov8n/            summary.json and threshold-sweep CSVs (all key numbers)
│   └── profiling/          timing tables and cProfile output
├── data/README.md          dataset source and licence (images are not stored in Git)
├── AI_USE_LOG.md           record of AI use
├── requirements.txt        pinned package versions
└── LICENSE                 MIT
```

## Setup (Windows, Python 3.12.4)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**GPU (recommended for training).** `requirements.txt` installs the CPU build of
PyTorch. For an NVIDIA GPU, replace it with a CUDA build: run `nvidia-smi` to see your
CUDA version, choose the matching command at <https://pytorch.org/get-started/locally/>,
and run it after uninstalling the CPU build, for example:

```
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
python -c "import torch; print(torch.cuda.is_available())"
```

Everything also runs on a CPU, but training is much slower.

## Data

Roboflow Universe, *Pork Rasher Error (Packaging)*, version 4, CC BY 4.0:
<https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4>
(3,349 training / 120 validation / 80 test images, 720 × 540 px, YOLOv8 format).

The images are not stored in this repository. To download them, create a free Roboflow
account, save your API key as a Windows user environment variable named
`ROBOFLOW_API_KEY`, restart VS Code, then run:

```
python -c "from rasher_qc.data import download_dataset; download_dataset()"
```

The data is saved to `data/pork-rasher-error-packaging-v4/`.

## Usage

All commands are run from the repository root with the venv active.

**Train and evaluate on the validation set** (about 30 minutes on an RTX 4050):
```
python -m rasher_qc.experiment --weights yolov8n.pt --name yolov8n
```
This checks the dataset, trains, measures detection metrics, chooses the pass/fail
threshold and saves `results/yolov8n/summary.json`.
Options: `--epochs`, `--imgsz`, `--batch`, `--seed`, `--min-recall`
(see `python -m rasher_qc.experiment --help`).

**Final test** (run once, after all choices are made):
```
python -m rasher_qc.experiment --name yolov8n --skip-train --test
```

**Inspect a single image:**
```python
from rasher_qc.detector import inspect_tray
print(inspect_tray("results/runs/yolov8n/weights/best.pt", "path/to/image.jpg", threshold=0.20))
```
Returns `PASS` or `FAIL` and, for each detected defect, its class, confidence and
pixel bounding box. Missing, corrupt, greyscale or unsupported images raise an error.

**Profile inference speed:**
```
python -m rasher_qc.profiling
```

## Tests and documentation

```
python -m pytest                                                  # run the 41 tests
python -m pytest --html=docs/test_report.html --self-contained-html   # HTML test report
pdoc --docformat numpy rasher_qc -o docs/api                      # HTML documentation
```

Open `docs/api/index.html` and `docs/test_report.html` in a web browser
(GitHub shows HTML files as source code, so download or clone the repository first).

## AI acknowledgement

This project used Claude (Anthropic, Claude Opus 5.5) for planning, code generation,
code explanation, data investigation and documentation support. All AI-generated
content was reviewed, tested and edited by the project owner, who remains responsible
for all code, results and decisions. Details are in `AI_USE_LOG.md` and in the report's
AI reflection. See [Monash University: Acknowledging the use of AI](https://www.monash.edu/student-academic-success/learning-with-ai/academic-integrity-and-ai/acknowledging-the-use-of-ai).

## Licence and credits

- Code: MIT licence (see `LICENSE`).
- Dataset: Hello, *Pork Rasher Error (Packaging)*, Roboflow Universe, CC BY 4.0.
- Model and training framework: Ultralytics YOLOv8 (AGPL-3.0),
  <https://github.com/ultralytics/ultralytics>.

Author: James Swiatlo (jswi0003@student.monash.edu)
