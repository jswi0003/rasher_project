# Tool used: Claude (Anthropic, Claude Opus 5.5) through the Claude desktop app, linked to my project folder

# Navigatiting github in order to set up file structure and vscode environment
# Steps for downloading and organising project data, checked license and key steps manually
# Generated data.py to download and structurally veryify the project data with claude
# Fixed data check functions
# Claude wrote the missing data check functions (read_class_names, check_label_line, check_dataset, write_training_yaml) to match my existing tests, verified by running pytest (18 passed) and checking the real dataset counts (3349/120/80)
# Claude found the venv had the CPU-only build of PyTorch, reinstalled the CUDA build myself and confirmed the RTX 4050 was detected
# Generated tray.py, detector.py, experiment.py and test_tray.py, ran pytest (36 passed) and a 1-epoch smoke test myself before trusting them
# Asked Claude to explain the whole workflow, every file and function, and every training setting in plain language so I could understand and defend the code
# AI error: Claude claimed 13 filenames shared between splits proved data leakage, a later image-by-image comparison showed they were different images
# Claude wrote leakage audit code which found no duplicate images across splits, decided myself to remove this code from the project
# Clarified the difference ebtween training a model from scratch, and using a pretrained YOLOv8 model with trasfer learning
# Ran the full training myself (27 minutes, best epoch 33), Claude explained the training outputs and helped analysed the results
# Claude wrote profiling.py and test_profiling.py, ran them myself (41 tests passed)
# AI error: the profiler disproved Claude's earlier claim that post-processing was slow because of the low confidence setting, removed the claim from the report
# Claude drafted report paragraphs, edited them myself for clarity, relevance, and accuracy be checking every number against my results
# Claude explained how to generate HTML documentation with pdoc and a test report with pytest-html, generated both myself
# Claude drafted README text, reviewed and edited all of it myself
# Claude drafted this expanded AI log from our conversations, edited it to only include key uses
