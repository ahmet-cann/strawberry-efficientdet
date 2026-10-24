@echo off
cd /d "%~dp0"
if not exist tmp mkdir tmp
set "TEMP=%~dp0tmp"
set "TMP=%~dp0tmp"
set "JUPYTER_CONFIG_DIR=%~dp0.jupyter\config"
set "JUPYTER_DATA_DIR=%~dp0.jupyter\data"
set "JUPYTER_RUNTIME_DIR=%~dp0.jupyter\runtime"
set "IPYTHONDIR=%~dp0.ipython"
set "MPLCONFIGDIR=%~dp0.matplotlib"
set "TORCH_HOME=%~dp0.torch"
set "HF_HOME=%~dp0.hf"
set "XDG_CACHE_HOME=%~dp0.cache"
set "PYTHONPYCACHEPREFIX=%~dp0.pycache"
set PIP_NO_CACHE_DIR=1
call .venv\Scripts\activate.bat
jupyter notebook
