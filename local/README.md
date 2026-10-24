# Local run (Windows, NVIDIA GPU)

Same experiments as `notebooks/EfficientDet_Strawberry.ipynb`, run on a local NVIDIA GPU instead of Google Colab. Model, split, hyperparameters and metrics are identical; only the paths and the data loading differ.

## Folder layout

```
E:\cilek\
  EfficientDet_Strawberry_local.ipynb
  effdet_lib.py
  kur.bat              (one-time setup: virtual environment and libraries)
  baslat.bat           (starts Jupyter; caches and temp files stay in this folder)
  splits.json          (created on first run; same split as Colab)
                       (data: E:\Çilek Boy Numaralanmış)
```

## Setup (Command Prompt)

```
cd E:\cilek
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install effdet==0.4.1 albumentations==2.0.8 torchmetrics pycocotools pandas matplotlib seaborn scikit-learn notebook
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
jupyter notebook
```

The last check must print `True` and the GPU name. If it prints `False`, the CPU build of PyTorch was installed: take the install command for your CUDA version from https://pytorch.org/get-started/locally/.

## Why `effdet_lib.py`

On Windows, DataLoader worker processes are started with `spawn` and cannot use classes defined inside a notebook. The dataset and collate function therefore live in a module, so several CPU cores can decode the large original JPEGs in parallel. `read_image` uses `cv2.imdecode`, which also works with non-ASCII paths.

Results are written to `results\` (logs, curves, checkpoints, `experiment_log.csv`).

## No files outside the project folder

`kur.bat` and `baslat.bat` redirect temporary files, the pip cache (disabled), Jupyter/IPython settings, the Matplotlib cache and the PyTorch/Hugging Face caches into the project folder, so nothing is written to the system drive except the existing Python installation, which is only read.
