#!/usr/bin/env bash
# 视频人脸检测比对工具 - 一键启动脚本 (Linux / macOS)
# 说明：本项目以 Windows 为主（launch.bat 是首选入口），本脚本供 Linux/macOS 上跑同一份代码。
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$DIR/venv"
cd "$DIR"

echo "============================================"
echo "  视频人脸检测比对工具 启动器"
echo "============================================"

# 0) 选择基础解释器（PyQt6 需要 Python >= 3.9）
PY=""
if [ -n "$PYTHON_EXE" ] && "$PYTHON_EXE" -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,9) else 1)" >/dev/null 2>&1; then
    PY="$PYTHON_EXE"
fi
if [ -z "$PY" ]; then
    for cand in python3 python; do
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,9) else 1)" >/dev/null 2>&1; then
            PY="$cand"
            break
        fi
    done
fi
if [ -z "$PY" ]; then
    echo "[错误] 未找到 Python >= 3.9。"
    exit 1
fi
echo "使用 Python: $PY"

# 1) 创建虚拟环境
if [ ! -x "$VENV/bin/python" ]; then
    echo "正在创建虚拟环境（基于 $PY）..."
    "$PY" -m venv "$VENV"
fi

# 2) 激活并升级 pip
source "$VENV/bin/activate"
python -m pip install -U pip -q

# 3) 自适应安装依赖：检测 NVIDIA GPU / CUDA
#    insightface 官方把 onnxruntime(CPU) 列为硬依赖，直接安装会和 onnxruntime-gpu
#    冲突并使 GPU 失效。故先 --no-deps 安装 insightface，再单独安装 GPU 版
#    onnxruntime。onnxruntime-gpu 锁定 1.22.0（CUDA 12.x 兼容）。
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[环境] 检测到 NVIDIA GPU，安装 GPU 加速版依赖（onnxruntime-gpu==1.22.0）..."
    python -m pip install -q insightface --no-deps
    python -m pip install -q "onnxruntime-gpu==1.22.0" opencv-python-headless pillow numpy onnx easydict prettytable pyyaml tqdm requests pyqt6
else
    echo "[环境] 未检测到 NVIDIA GPU，安装 CPU 版依赖（onnxruntime）..."
    python -m pip install -q insightface --no-deps
    python -m pip install -q onnxruntime opencv-python-headless pillow numpy onnx easydict prettytable pyyaml tqdm requests pyqt6
fi

# 4) 启动前健康检查（兜底，绝不静默闪退）
if ! python -c "import PyQt6, insightface, onnxruntime, PIL, cv2" >/dev/null 2>&1; then
    echo "[错误] 依赖不完整，请检查上面的安装输出后重试。"
    exit 1
fi

# 5) 启动 GUI
echo "启动主程序..."
python face_clip_qt.py
