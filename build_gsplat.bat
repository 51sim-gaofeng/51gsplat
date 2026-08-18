@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set "DISTUTILS_USE_SDK=1"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"

REM Target GPU architectures. Must cover the deploy machine's compute capability,
REM otherwise runtime raises "no kernel image is available for execution on the device".
REM Keep in sync with MnemoGS/Build/build-4dgs.bat.
if not defined TORCH_CUDA_ARCH_LIST set "TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0;12.0"
REM 32 = zero-padded channel count for semantic-segmentation render
REM (cfg.render.semantic_raster_channels default 32).
if not defined NUM_CHANNELS set "NUM_CHANNELS=1,2,3,4,32"
REM Compile acceleration: split each huge kernel across NVCC threads.
if not defined NVCC_THREADS set "NVCC_THREADS=5"
if not defined NVCC_FLAGS set "NVCC_FLAGS=--threads %NVCC_THREADS%"
REM Limit parallel ninja jobs. Each Projection*Fused.cu instantiates huge
REM templates (many channels x many arches) and can use several GB of host RAM;
REM too many parallel jobs exhaust memory ("catastrophic error: out of memory").
REM Lower this further (e.g. 1) if the build still OOMs.
if not defined MAX_JOBS set "MAX_JOBS=2"
set "FORCE_CUDA=1"

echo Toolchain check...
where cl || goto :err
where nvcc || goto :err
where ninja || goto :err

if not defined CONDA_PREFIX (
    echo CONDA_PREFIX not set. Run "conda activate 4dgs" before this script.
    goto :err
)

echo Build config: TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST% NUM_CHANNELS=%NUM_CHANNELS% NVCC_FLAGS=%NVCC_FLAGS%
echo Building and installing gsplat from source (current dir) into env: %CONDA_PREFIX%
powershell -NoProfile -Command "& '%CONDA_PREFIX%\python.exe' -m pip install '%~dp0.' --no-build-isolation -v 2>&1 | Tee-Object -FilePath '%~dp0build_gsplat.log'; exit $LASTEXITCODE"
if %ERRORLEVEL% NEQ 0 goto :err

echo Build completed.
exit /b 0

:err
echo Build failed with exit code %ERRORLEVEL%.
exit /b 1
