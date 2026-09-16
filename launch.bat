@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
title 视频人脸检测比对工具 - 一键启动

set "DIR=%~dp0"
set "VENV=%DIR%venv"
cd /d "%DIR%"
set "BATCH_LOG=%DIR%launch.log"

REM 清除可能指向系统 Python 的环境变量，避免 PyQt6/insightface 被系统残缺安装劫持
set "PYTHONPATH="
set "PYTHONHOME="

echo [%date% %time%] launch.bat 开始 > "%BATCH_LOG%"

echo ============================================
echo   视频人脸检测比对工具 - 一键启动
echo ============================================

REM ===== 快速通道：venv 已就绪则直接启动，跳过重复安装（更快更稳）=====
if exist "%VENV%\Scripts\python.exe" (
    "%VENV%\Scripts\python.exe" -c "import PyQt6, insightface, onnxruntime, PIL, cv2" >nul 2>&1
    if !errorlevel!==0 goto :launch
    echo [提示] venv 依赖不完整，进入修复/安装流程...>> "%BATCH_LOG%"
)

REM ===== 0) 寻找可用基础解释器（3.9~3.11）=====
set "PY_FOUND="
for %%P in (
    "C:\miniforge3\envs\3.11.9\python.exe"
    "C:\Program Files\Python311\python.exe"
    "C:\Program Files\Python310\python.exe"
    "C:\Python311\python.exe"
    "python"
) do (
    call :try_py "%%~P"
    if defined PY_FOUND goto :have_py
)
goto :no_py

:try_py
"%~1" -c "import sys; sys.exit(0 if (3,9)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1
if not errorlevel 1 set "PY_FOUND=%~1"
exit /b

:no_py
echo [错误] 未找到 Python 3.9~3.11。请安装 Python 3.11 并勾选 Add to PATH。>> "%BATCH_LOG%"
echo [错误] 未找到 Python 3.9~3.11。请安装 Python 3.11 并勾选 Add to PATH。
pause
exit /b 1

:have_py
echo 使用基础解释器：%PY_FOUND%>> "%BATCH_LOG%"
echo 使用基础解释器：%PY_FOUND%

REM ===== 1) 已有 venv 但缺 PyQt6 时不再删库重建（避免重下 CUDA 依赖），
REM         交给下面第 3 步的 pip 增量安装补齐即可 =====
if exist "%VENV%\Scripts\python.exe" (
    "%VENV%\Scripts\python.exe" -c "import PyQt6" >nul 2>&1
    if not !errorlevel!==0 echo 检测到 venv 缺少 PyQt6，将由依赖安装步骤补齐...>> "%BATCH_LOG%"
)

REM ===== 2) 创建 venv =====
if not exist "%VENV%\Scripts\python.exe" (
    echo 正在创建虚拟环境...>> "%BATCH_LOG%"
    "%PY_FOUND%" -m venv "%VENV%"
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败。>> "%BATCH_LOG%"
        echo [错误] 创建虚拟环境失败。
        pause
        exit /b 1
    )
)

REM ===== 3) 安装依赖（有 GPU 装 GPU 版，无则 CPU 版）=====
call "%VENV%\Scripts\activate.bat" >nul 2>&1
"%VENV%\Scripts\python.exe" -m pip install -U pip -q >> "%BATCH_LOG%" 2>&1
REM 注意：nvidia-smi 成功时 errorlevel 为 0，即「有 GPU」走第一个分支。
REM （历史版本这里写反了，会在有独显的机器上装成 CPU 版，导致 GPU 静默失效。）
nvidia-smi >nul 2>&1
if !errorlevel!==0 (
    echo [环境] 检测到 NVIDIA GPU，安装 GPU 加速版依赖（onnxruntime-gpu==1.22.0）...>> "%BATCH_LOG%"
    echo [环境] 检测到 NVIDIA GPU，安装 GPU 加速版依赖（onnxruntime-gpu==1.22.0）...
    "%VENV%\Scripts\python.exe" -m pip install -q insightface --no-deps >> "%BATCH_LOG%" 2>&1
    "%VENV%\Scripts\python.exe" -m pip install -q "onnxruntime-gpu==1.22.0" opencv-python-headless pillow numpy onnx easydict prettytable pyyaml tqdm requests pyqt6 >> "%BATCH_LOG%" 2>&1
) else (
    echo [环境] 未检测到 NVIDIA GPU，安装 CPU 版依赖...>> "%BATCH_LOG%"
    echo [环境] 未检测到 NVIDIA GPU，安装 CPU 版依赖...
    "%VENV%\Scripts\python.exe" -m pip install -q insightface --no-deps >> "%BATCH_LOG%" 2>&1
    "%VENV%\Scripts\python.exe" -m pip install -q onnxruntime opencv-python-headless pillow numpy onnx easydict prettytable pyyaml tqdm requests pyqt6 >> "%BATCH_LOG%" 2>&1
)

REM ===== 4) 健康检查 =====
"%VENV%\Scripts\python.exe" -c "import PyQt6, insightface, onnxruntime, PIL, cv2" >nul 2>&1
if not !errorlevel!==0 (
    echo [错误] 依赖未就绪，请查看 launch.log 末尾安装报错后重试。>> "%BATCH_LOG%"
    echo [错误] 依赖未就绪，请查看 launch.log 末尾安装报错后重试。
    pause
    exit /b 1
)

:launch
set "QT_QPA_PLATFORM_PLUGIN_PATH=%VENV%\Lib\site-packages\PyQt6\Qt6\plugins"
echo 启动主程序...>> "%BATCH_LOG%"
"%VENV%\Scripts\python.exe" face_clip_qt.py >> "%BATCH_LOG%" 2>&1
set "RC=%errorlevel%"
echo [%date% %time%] 主程序退出码=%RC% >> "%BATCH_LOG%"

if not "%RC%"=="0" (
    echo.
    echo ============================================================
    echo  [启动失败] 主程序异常退出（退出码 %RC%）
    echo  完整日志：%BATCH_LOG%
    echo  崩溃堆栈：%~dp0crash.log
    echo  ---- launch.log 末尾 30 行 ----
    powershell -NoProfile -Command "Get-Content -Tail 30 -Encoding UTF8 '%BATCH_LOG%'"
    echo.
    pause
) else (
    echo 主程序已正常结束。
)

endlocal
