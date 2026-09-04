# -*- mode: python ; coding: utf-8 -*-
# 打包为单 exe：集成 Python 运行时 + 全部第三方依赖 + CUDA 12.6 运行时
# + buffalo_l 人脸模型 + 工具图标。目标机只需有 NVIDIA 驱动（>=560）。
import os

HERE = os.path.dirname(os.path.abspath(SPEC))
VENV_SP = os.path.join(HERE, "venv", "Lib", "site-packages")
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
MODEL_DIR = os.path.expanduser(r"~\.insightface\models\buffalo_l")

# ---- 第三方包自身的二进制（onnxruntime 的 CUDA EP 等，由 hook 收集，这里兜底） ----
binaries = []
ort_capi = os.path.join(VENV_SP, "onnxruntime", "capi")
if os.path.isdir(ort_capi):
    for f in os.listdir(ort_capi):
        if f.lower().endswith((".dll", ".pyd")):
            binaries.append((os.path.join(ort_capi, f), "onnxruntime/capi"))

# ---- CUDA 12.6 运行时（与 onnxruntime CUDA EP 同目录，便于加载） ----
# onnxruntime_providers_cuda.dll 依赖 cublas / cublasLt / cudnn / cufft / cudart，
# 而 cudnn9 还拆成多个子 DLL，因此直接复制整个 CUDA bin 下的全部 .dll。
if os.path.isdir(CUDA_BIN):
    for f in os.listdir(CUDA_BIN):
        if f.lower().endswith(".dll"):
            binaries.append((os.path.join(CUDA_BIN, f), "onnxruntime/capi"))
else:
    print("WARN: 未找到 CUDA 运行时目录:", CUDA_BIN)

# ---- 数据文件 ----
datas = []
if os.path.isdir(MODEL_DIR):
    for f in os.listdir(MODEL_DIR):
        if f.lower().endswith(".onnx"):
            datas.append((os.path.join(MODEL_DIR, f),
                          os.path.join("models", "buffalo_l")))
else:
    print("WARN: 未找到 buffalo_l 模型目录:", MODEL_DIR)
datas.append((os.path.join(HERE, "icon.png"), "."))

a = Analysis(
    [os.path.join(HERE, "face_clip_qt.py")],
    pathex=[HERE, VENV_SP],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "cv2", "numpy", "PIL", "PIL.Image",
        "onnxruntime", "onnxruntime.capi",
        "insightface", "insightface.app", "insightface.app.face_analysis",
        "insightface.model_zoo", "insightface.utils", "insightface.data",
        "PyQt5", "PyQt5.QtCore", "PyQt5.QtGui", "PyQt5.QtWidgets",
        "PyQt5.QtSvg", "PyQt5.sip",
        "prettytable", "easydict", "tqdm", "requests",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PySide6", "PySide2", "matplotlib", "tkinter", "mxnet",
        "torch", "tensorflow", "keras", "notebook", "IPython",
        "pandas", "scipy",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="视频人脸检测比对工具.exe",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(HERE, "icon.ico"),
)
