@echo off
setlocal

rem Locate vcvarsall.bat dynamically via vswhere (works for Community/Professional/Enterprise/BuildTools)
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo ERROR: vswhere.exe not found. Install Visual Studio or Build Tools first.
    exit /b 1
)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualCpp.Tools.HostX64.TargetX64 -property installationPath`) do set "VS_INSTALL=%%i"
if not defined VS_INSTALL (
    echo ERROR: No Visual Studio installation with MSVC x64 tools found.
    exit /b 1
)
set "VCVARSALL=%VS_INSTALL%\VC\Auxiliary\Build\vcvarsall.bat"
echo Using MSVC: "%VCVARSALL%"
call "%VCVARSALL%" x64
set "DISTUTILS_USE_SDK=1"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"

rem 加速编译：并行 nvcc 线程数、拆分编译单元数，以及限定实例化的通道数组合。
if not defined NVCC_THREADS set "NVCC_THREADS=5"
if not defined NVCC_SPLIT_COMPILE set "NVCC_SPLIT_COMPILE=8"
REM 32 为语义分割渲染补零后的通道数（cfg.render.semantic_raster_channels 默认 32）。
if not defined NUM_CHANNELS set "NUM_CHANNELS=1,2,3,4,32"

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
rem -v 打印完整编译日志；ninja 并行任务数默认取 CPU 核数，可用 MAX_JOBS 覆盖。
if not defined MAX_JOBS set "MAX_JOBS=%NUMBER_OF_PROCESSORS%"
@REM python -m pip install -e "%GSPLAT_SRC%" --no-build-isolation -v
python -m pip install "%GSPLAT_SRC%" --no-build-isolation -v
if %ERRORLEVEL% NEQ 0 goto :err

echo Build completed.
exit /b 0

:err
echo Build failed with exit code %ERRORLEVEL%.
exit /b 1
