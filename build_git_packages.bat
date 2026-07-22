@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set DISTUTILS_USE_SDK=1
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"

echo ===== gsplat env =====
@REM set PYTHON=C:\Users\qinlei\AppData\Local\anaconda3\envs\gsplat\python.exe

@REM echo Installing fused-ssim (patched local) into gsplat env...
@REM %PYTHON% -m pip install "G:\ws\projects\fused-ssim" --no-build-isolation
@REM echo.

@REM echo Installing fused-bilagrid into gsplat env...
@REM %PYTHON% -m pip install "git+https://github.com/harry7557558/fused-bilagrid@49f0ef06c9f81810fb9b5dd9027cf1844950cc16" --no-build-isolation
@REM echo.

echo ===== 4dgs env =====
set PYTHON=C:\Users\qinlei\AppData\Local\anaconda3\envs\4dgs\python.exe

echo Installing fused-ssim into 4dgs env...
%PYTHON% -m pip install "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5" --no-build-isolation
echo.

echo Installing fused-bilagrid into 4dgs env...
%PYTHON% -m pip install "git+https://github.com/harry7557558/fused-bilagrid@49f0ef06c9f81810fb9b5dd9027cf1844950cc16" --no-build-isolation
echo.

echo Installing ppisp into 4dgs env...
%PYTHON% -m pip install "git+https://github.com/nv-tlabs/ppisp@v1.0.0" --no-build-isolation
echo.

echo Done.
