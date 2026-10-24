@echo off
cd /d "%~dp0"
if not exist tmp mkdir tmp
set "TEMP=%~dp0tmp"
set "TMP=%~dp0tmp"
set PIP_NO_CACHE_DIR=1
set PIP_DISABLE_PIP_VERSION_CHECK=1
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir effdet==0.4.1 albumentations==2.0.8 torchmetrics pycocotools pandas matplotlib seaborn scikit-learn notebook
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo.
echo Kurulum bitti.
pause
