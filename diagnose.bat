@echo off
chcp 65001 >nul
setlocal
title 人脸检测视频截取工具 - 诊断模式
cd /d "%~dp0"

echo 诊断模式：以 venv 运行主程序，所有输出写入 diagnose.log 并随后完整显示。
echo （如果主程序内部崩溃，还会额外生成 crash.log）
echo.

if not exist "%~dp0venv\Scripts\python.exe" (
    echo [错误] 未找到 venv，请先双击 launch.bat 完成初始化。
    pause
    exit /b 1
)

call "%~dp0venv\Scripts\activate.bat" 2>nul
set "QT_QPA_PLATFORM_PLUGIN_PATH=%~dp0venv\Lib\site-packages\PyQt6\Qt6\plugins"
"%~dp0venv\Scripts\python.exe" face_clip_qt.py > diagnose.log 2>&1
set "RC=%errorlevel%"

echo.
echo ============================================================
echo  主程序退出码 = %RC%
echo ============================================================
echo  ---- diagnose.log 内容 ----
type diagnose.log
echo  ---- 结束 ----
echo.

if exist "%~dp0crash.log" (
    echo 检测到 crash.log（运行时异常），内容如下：
    echo ------------------------------------------------------------
    type "%~dp0crash.log"
    echo ------------------------------------------------------------
)

echo.
echo  诊断结束。请按任意键关闭，并把 diagnose.log / crash.log 发我排查。
pause
endlocal
