@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set DISTUTILS_USE_SDK=1
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
echo Installing ppisp into 4dgs env...
C:\Users\qinlei\AppData\Local\anaconda3\envs\4dgs\python.exe -m pip install "git+https://github.com/nv-tlabs/ppisp@v1.0.0" --no-build-isolation
echo Done. Exit code: %ERRORLEVEL%
