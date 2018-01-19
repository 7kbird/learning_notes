@echo off
start tensorboard --logdir="_:%~dp0_cache\tensorboard_logs" --host=localhost
start "" http://localhost:6006
set THEANO_FLAGS=floatX=float32,device=cuda0,optimizer_including=cudnn,gpuarray.preallocate=0.8
set MKL_THREADING_LAYER=GNU
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter notebook --notebook-dir="%~dp0."
taskkill /IM tensorboard.exe /F >nul
