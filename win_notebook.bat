@echo off
start tensorboard --logdir="_:%~dp0_cache\tensorboard_logs" --host=localhost
start "" http://localhost:6006
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter notebook --notebook-dir="%~dp0."
taskkill /IM tensorboard.exe /F >nul
