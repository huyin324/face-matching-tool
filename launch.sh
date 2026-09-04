#!/usr/bin/env bash
# 人脸检测视频截取工具 - 一键启动脚本 (Linux / macOS)
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$DIR/venv"
cd "$DIR"

echo "============================================"
echo "  人脸检测视频截取工具 启动器"
echo "============================================"

# 0) 选择一个“带有 tkinter 的 Python”作为虚拟环境的基础解释器。
#    GUI 程序必须依赖 tkinter；部分精简/托管 Python 不含 tkinter，
#    用它创建 venv 运行时会闪退。这里逐个探测 tkinter，取第一个可用的。
TK_PY=""
if [ -n "$PYTHON_EXE" ] && "$PYTHON_EXE" -c "import tkinter" >/dev/null 2>&1; then
    TK_PY="$PYTHON_EXE"
fi
if [ -z "$TK_PY" ]; then
    for cand in python3 python; do
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import tkinter" >/dev/null 2>&1; then
            TK_PY="$cand"
            break
        fi
    done
fi
if [ -z "$TK_PY" ]; then
    echo "[错误] 未找到“带 tkinter 的 Python”（GUI 必需）。"
    echo "请安装完整版 Python 3.9~3.11（含 tcl/tk）。"
    exit 1
fi
echo "使用 Python（含 tkinter）: $TK_PY"

# 1) 若已存在 venv，但其中 python 缺少 tkinter，则重建
if [ -x "$VENV/bin/python" ] && ! "$VENV/bin/python" -c "import tkinter" >/dev/null 2>&1; then
    echo "检测到已有 venv 缺少 tkinter，正在重建..."
    rm -rf "$VENV"
fi

# 2) 创建虚拟环境
if [ ! -x "$VENV/bin/python" ]; then
    echo "正在创建虚拟环境（基于 $TK_PY）..."
    "$TK_PY" -m venv "$VENV"
fi

# 3) 激活并升级 pip
source "$VENV/bin/activate"
python -m pip install -U pip -q

# 4) 自适应安装依赖：检测 NVIDIA GPU / CUDA
#    insightface 官方把 onnxruntime(CPU) 列为硬依赖，直接安装会和 onnxruntime-gpu
#    冲突并使 GPU 失效。故先 --no-deps 安装 insightface，再单独安装 GPU 版
#    onnxruntime。onnxruntime-gpu 锁定 1.22.0（CUDA 12.x 兼容）。
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[环境] 检测到 NVIDIA GPU，安装 GPU 加速版依赖（onnxruntime-gpu==1.22.0）..."
    python -m pip install -q insightface --no-deps
    python -m pip install -q "onnxruntime-gpu==1.22.0" opencv-python-headless pillow numpy onnx scikit-image albumentations easydict prettytable pyyaml tqdm requests scipy
else
    echo "[环境] 未检测到 NVIDIA GPU，安装 CPU 版依赖（onnxruntime）..."
    python -m pip install -q insightface --no-deps
    python -m pip install -q onnxruntime opencv-python-headless pillow numpy onnx scikit-image albumentations easydict prettytable pyyaml tqdm requests scipy
fi

# 5) 启动前 tkinter 健康检查（兜底，绝不静默闪退）
if ! python -c "import tkinter" >/dev/null 2>&1; then
    echo "[错误] 虚拟环境中仍缺少 tkinter，无法启动 GUI。"
    echo "请改用“完整版 Python（含 tcl/tk）”重新创建 venv 后重试。"
    exit 1
fi

# 6) 启动 GUI
echo "启动主程序..."
python face_clip_tool.py
