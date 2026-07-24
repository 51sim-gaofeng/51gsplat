@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set "DISTUTILS_USE_SDK=1"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"

echo Toolchain check...
where cl || goto :err
where nvcc || goto :err
where ninja || goto :err

echo Building and installing gsplat from source into test env...
@REM C:\Users\qinlei\AppData\Local\anaconda3\envs\test\python.exe -m pip install -e G:\ws\Libs\51-gsplat --no-build-isolation -v
C:\Users\qinlei\AppData\Local\anaconda3\envs\test\python.exe -m pip install G:\ws\Libs\51-gsplat --no-build-isolation
if errorlevel 1 goto :err

echo Build completed.
exit /b 0

:err
echo Build failed.
exit /b 1
