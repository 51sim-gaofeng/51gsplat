@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set "DISTUTILS_USE_SDK=1"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"

echo Activating conda environment: 4dgs
call conda activate 4dgs
if %ERRORLEVEL% NEQ 0 goto :err

echo Toolchain check...
where cl || goto :err
where nvcc || goto :err
where ninja || goto :err

rem Source = this script's own directory (the gsplat fork), no hardcoded path.
set "GSPLAT_SRC=%~dp0"
if "%GSPLAT_SRC:~-1%"=="\" set "GSPLAT_SRC=%GSPLAT_SRC:~0,-1%"

echo Building and installing gsplat from source into 4dgs env: "%GSPLAT_SRC%"
@REM python -m pip install -e "%GSPLAT_SRC%" --no-build-isolation -v
python -m pip install "%GSPLAT_SRC%" --no-build-isolation
if %ERRORLEVEL% NEQ 0 goto :err

echo Build completed.
exit /b 0

:err
echo Build failed with exit code %ERRORLEVEL%.
exit /b 1
