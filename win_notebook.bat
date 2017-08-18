@echo off
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter notebook --notebook-dir="%~dp0."