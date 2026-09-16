#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
人脸检测视频截取工具（PyQt6 版）
================================
基于 insightface + onnxruntime 的 GUI 工具：上传目标人脸 + 视频/RTSP 流，
实时播放视频并在画面叠加人脸框，命中目标人脸时截取前后各 N 秒视频片段
并保存红框截图。支持多视频队列、GPU/CPU 自动切换、中文路径。

两种运行模式：
- 人脸识别比对模式：检测画面人脸并与目标人脸比对，相似度 >= 阈值即比中并截取。
- 人脸检测抓图模式：只要检测到的人脸质量 >= 质量阈值即截取（无需目标人脸）。

两种分析模式（V1.2 新增，仅本地视频可选；RTSP 仅支持实时分析）：
- 实时分析：按视频正常播放速度边播放边比对，预览即真实播放画面。
- 全速分析：不按播放速度推进，尽最大算力全速解码 + 逐帧比对，尽快跑完整个视频；
  预览显示分析进度画面，界面实时显示分析帧率（帧/s）。

界面布局（V1.3，PyQt6，扁平化卡片式）：
- 顶部：视频源（RTSP 实时流 / 本地视频 二选一）+ 输出目录  |  参数设置
- 中部左侧：目标人脸（竖版 3:4，占据左列全部纵向空间）+ 实时信息
- 中部右侧：实时视频（16:9）+ 分析模式（实时/全速）+ 分析帧率 + 状态/进度 + 操作按钮
- 底部：日志栏（通栏全宽）
- 操作：开始 / 暂停 / 结束

V1.3 变更：
1. 日志栏移至界面底部通栏；腾出的纵向空间全部给「目标人脸」栏。
2. 取消播放倍速功能（播放恒定 1X；提速请改用「全速分析」）。
3. 界面由 PyQt5 迁移到 PyQt6，改为扁平化现代化配色（卡片式面板、弱描边、统一间距）。
4. 性能优化：显示管线改用 BGR 直通（省去每帧 cvtColor）、去掉多余的图像副本、
   全速分析预览降频，降低 UI 线程占用。

本文件为自包含实现（引擎 + PyQt6 界面）。
"""

import os
import sys
import time
import queue
import threading
import traceback
from collections import deque

# ---------------- 强制使用本工具自带的 venv 依赖 ----------------
# 若用户机器上设置了 PYTHONPATH 指向“系统 Python 的 site-packages”，
# 会导致 PyQt6 / insightface 等被优先从系统那一份（可能残缺）加载，
# 典型报错：Could not find the Qt platform plugin "windows"。
# 因此在导入任何第三方库之前，把本 venv 的 site-packages 置顶，并强制
# Qt 插件目录为 venv 内的插件，彻底规避该问题。
_HERE = os.path.dirname(os.path.abspath(__file__))
_VENV_SP = os.path.normpath(os.path.join(_HERE, "venv", "Lib", "site-packages"))
if os.path.isdir(_VENV_SP):
    sys.path = [
        p for p in sys.path
        if not (os.path.normpath(p).endswith("site-packages")
                and not os.path.normpath(p).startswith(_VENV_SP))
    ]
    if _VENV_SP not in sys.path:
        sys.path.insert(0, _VENV_SP)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
        _VENV_SP, "PyQt6", "Qt6", "plugins")

# ---------------- 打包(frozen)环境适配 ----------------
# 打包成单 exe 后，所有依赖与模型均位于 PyInstaller 解压目录 sys._MEIPASS。
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _MEI = sys._MEIPASS
    # 强制使用打包内的 Qt 插件，避免继承环境中错误的 QT_QPA_PLATFORM_PLUGIN_PATH
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
        _MEI, "PyQt6", "Qt6", "plugins")
    # 模型根目录：打包后 buffalo_l 位于 <MEI>/models/buffalo_l
    _MODEL_ROOT = _MEI
else:
    _MODEL_ROOT = None

import cv2
import numpy as np

# 【性能】限制 OpenCV 内部线程数：
# OpenCV 默认会拉起与 CPU 核数相同的并行线程（本机 16），解码 / 缩放时会与
# onnxruntime 的推理线程、以及播放/检测/显示三条线程争抢 CPU，实测会造成
# 明显抖动（尤其全速分析时）。收到 4 之后总吞吐反而更稳。
try:
    cv2.setNumThreads(4)
except Exception:
    pass

# ---------------- 第三方库可用性 ----------------
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception as e:
    ORT_AVAILABLE = False
    _ORT_ERR = e

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except Exception as e:
    INSIGHTFACE_AVAILABLE = False
    _IF_ERR = e

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception as e:
    PIL_AVAILABLE = False
    _PIL_ERR = e

try:
    from PyQt6.QtCore import Qt, QTimer, QSize, QUrl, QRect, QRectF
    from PyQt6.QtGui import (QImage, QPixmap, QFont, QColor, QPainter, QPen,
                             QDesktopServices, QIcon, QFontDatabase,
                             QPainterPath)
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QLabel, QPushButton, QListWidget, QAbstractItemView,
        QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QProgressBar,
        QTextEdit, QGroupBox, QFileDialog, QMessageBox, QScrollArea,
        QSizePolicy, QFrame, QRadioButton, QButtonGroup,
    )
    PYQT_AVAILABLE = True
except Exception as e:
    PYQT_AVAILABLE = False
    _QT_ERR = e


# ---------------- 性能相关常量 ----------------
# 显示帧宽度上限：播放线程先把帧缩放到该宽度以内再送给 UI 线程，
# 使主线程开销与源分辨率解耦（4K/2K 视频也不会拖慢界面）。
DISPLAY_MAX_W = 1280


class _IOWorker:
    """后台串行 I/O 工作者：把耗时的磁盘/编码操作移出检测线程。

    关键问题：extract_segment 需要重新打开视频、seek、逐帧解码再编码，
    一段 20 秒片段往往耗时数百毫秒到数秒。若像以前那样在检测线程里同步调用，
    检测会整体停摆、画面上的目标框直接卡住，用户体验上是“又卡又飘”。
    截图写盘同理（PNG 编码 + 磁盘写入）。

    这里用一个后台守护线程串行消费任务，检测线程只负责投递，立刻继续下一帧。
    串行（而非并发）是刻意的：避免多个片段同时争抢解码/编码与磁盘 IO。
    """
    _q = queue.Queue()
    _lock = threading.Lock()
    _started = False

    @classmethod
    def submit(cls, fn, *args, **kwargs):
        cls._ensure_started()
        cls._q.put((fn, args, kwargs))

    @classmethod
    def _ensure_started(cls):
        with cls._lock:
            if cls._started:
                return
            t = threading.Thread(target=cls._loop, name="IOWorker", daemon=True)
            t.start()
            cls._started = True

    @classmethod
    def _loop(cls):
        while True:
            try:
                fn, args, kwargs = cls._q.get()
                try:
                    fn(*args, **kwargs)
                except Exception:
                    pass
                finally:
                    cls._q.task_done()
            except Exception:
                pass

    @classmethod
    def wait_idle(cls, timeout=60.0):
        """等待已排队的任务全部完成（收尾时调用，避免“分析完成”时文件还没落盘）。

        轮询 unfinished_tasks 而不是 queue.join()：后者无法设置超时，
        一旦某个任务卡住就会永久阻塞结束流程。
        """
        deadline = time.time() + max(0.0, timeout)
        while time.time() < deadline:
            try:
                if cls._q.unfinished_tasks <= 0:
                    return True
            except Exception:
                return True
            time.sleep(0.05)
        return False


# ---------------- 全局异常兜底：避免“一闪而过”看不到错误 ----------------
def _global_excepthook(exc_type, exc_val, exc_tb):
    msg = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
    try:
        with open(os.path.join(os.getcwd(), "crash.log"), "w", encoding="utf-8") as _f:
            _f.write(msg)
    except Exception:
        pass
    try:
        from PyQt6.QtWidgets import QMessageBox
        try:
            QMessageBox.critical(None, "程序异常（已写入 crash.log）", msg)
        except Exception:
            pass
    except Exception:
        pass


if PYQT_AVAILABLE:
    sys.excepthook = _global_excepthook


# ============================================================
# 引擎辅助函数
# ============================================================
def cosine(a, b):
    """余弦相似度（自动归一化）"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a / na, b / nb))


def sim_to_label(sim):
    """相似度(0~1) -> 标注文字，如 '98.50%'"""
    try:
        return f"{float(sim) * 100:.2f}%"
    except Exception:
        return "--"


def fmt_time(sec):
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def fourcc_for_ext(ext):
    ext = (ext or "").lower()
    return {
        ".mp4": "mp4v",
        ".mov": "mp4v",
        ".avi": "XVID",
        ".mkv": "mp4v",
        ".wmv": "wmv2",
    }.get(ext, "mp4v")


def face_quality_score(face, frame):
    """返回 0~80 的人脸质量分。

    综合：像素/尺寸、亮度（过暗/过曝）、模糊（拉普拉斯方差）、
    光比/对比度（灰度标准差）、角度（偏航，由 5 关键点估算）。
    光线暗 / 像素低 / 模糊 / 角度偏 / 光比大 均会拉低分数。
    低于设定阈值的人脸视为质量低，不做比对/抓图。
    """
    try:
        bbox = face.bbox
        x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
        h0, w0 = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w0, x2), min(h0, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hh, ww = gray.shape

        # 1) 像素/尺寸：人脸最小边长（>=80px 视为满）
        size_score = min(1.0, min(ww, hh) / 80.0)

        # 2) 亮度：过暗 / 过曝
        mean = float(gray.mean())
        if mean < 35:
            bright_score = max(0.0, mean / 35.0)
        elif mean > 220:
            bright_score = max(0.0, (255.0 - mean) / 35.0)
        else:
            bright_score = 1.0

        # 3) 模糊：拉普拉斯方差（越大越清晰）
        # 用 CV_32F 而非 CV_64F：8 位灰度下结果几乎一致，但耗时约减半。
        # 该函数在“逐帧检测”下会按人脸数×帧率调用，是检测线程的主要热点之一。
        lap = cv2.Laplacian(gray, cv2.CV_32F).var()
        blur_score = min(1.0, lap / 100.0)

        # 4) 光比 / 对比度：灰度标准差
        std = float(gray.std())
        if std < 18:
            contrast_score = max(0.0, std / 18.0)
        elif std > 95:
            contrast_score = max(0.0, (125.0 - std) / 30.0)
        else:
            contrast_score = 1.0

        # 5) 角度（偏航）：用 5 关键点估算
        kps = getattr(face, "kps", None)
        if kps is not None and len(kps) >= 5:
            kps = np.asarray(kps, dtype=np.float32)
            le, re, nose, lm, rm = kps[0], kps[1], kps[2], kps[3], kps[4]
            eye_mid_x = (le[0] + re[0]) / 2.0
            mouth_mid_x = (lm[0] + rm[0]) / 2.0
            ref_x = (eye_mid_x + mouth_mid_x) / 2.0
            eye_dist = max(1.0, abs(re[0] - le[0]))
            yaw = abs(nose[0] - ref_x) / eye_dist
            angle_score = max(0.0, 1.0 - yaw * 1.3)
        else:
            angle_score = 0.7

        score = 80.0 * (0.25 * size_score + 0.25 * bright_score +
                        0.20 * blur_score + 0.15 * contrast_score + 0.15 * angle_score)
        return float(max(0.0, min(80.0, score)))
    except Exception:
        return 0.0


def draw_box(frame, bbox, label=None, color=(0, 0, 255), thickness=None):
    """在 frame 上画框，返回副本。label 为框上方文字（None 则不画文字）。

    线宽默认 2（细），避免遮挡人脸；标注字体同步缩小。
    """
    out = frame.copy()
    draw_box_inplace(out, bbox, label=label, color=color, thickness=thickness)
    return out


def draw_box_inplace(frame, bbox, label=None, color=(0, 0, 255), thickness=None):
    """在 frame 上**原地**画框（不复制），直接修改传入的数组。

    【性能】同一帧上要画多个人脸框时（预览叠加、检测抓图模式），
    旧写法每画一个框就整帧 copy 一次（1080p 约 6MB/次），
    10 张脸 = 10 次全帧拷贝。本函数让调用方只拷一次、循环原地画。
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if thickness is None:
        thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        font_scale = 0.5
        (lw, lh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = max(0, y1 - lh - 4)
        cv2.rectangle(frame, (x1, ty), (x1 + lw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, max(lh, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def imread_utf8(path):
    """以 UTF-8 安全方式读取图像为 BGR numpy 数组（兼容中文路径）。"""
    try:
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return cv2.imread(path)


def imwrite_utf8(frame_bgr, path):
    """保存 BGR 图像到 path，自动创建中文输出目录。

    修复“写入图片失败”：旧实现不创建父目录，且依赖 os.replace（在部分环境被拦截）。
    改用 Pillow 直接保存（Pillow 经由 Python 文件 API，原生支持 Unicode/中文路径），
    并在保存前创建父目录。
    """
    try:
        path = os.path.abspath(path)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(path)
        return os.path.exists(path)
    except Exception:
        try:
            d = os.path.dirname(os.path.abspath(path))
            if d:
                os.makedirs(d, exist_ok=True)
            return bool(cv2.imwrite(path, frame_bgr))
        except Exception:
            return False


def extract_segment(src, start_t, end_t, out_path, logger):
    """截取 [start_t, end_t] 区间的视频片段（流复制式逐帧重编码）。"""
    cap = None
    writer = None
    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{src}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            raise RuntimeError("无法获取视频尺寸")

        start_frame = max(0, int(round(start_t * fps)))
        end_frame = int(round(end_t * fps))
        if total > 0:
            end_frame = min(end_frame, total - 1)

        ext = os.path.splitext(out_path)[1].lower()
        fourcc = cv2.VideoWriter_fourcc(*fourcc_for_ext(ext))
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError("无法创建视频写入器（编码器不可用）")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        f = start_frame
        written = 0
        while f <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            written += 1
            f += 1
        logger(f"  片段写入完成：{os.path.basename(out_path)}（{written} 帧，{fmt_time(end_t - start_t)}）")
        return True
    except Exception as e:
        logger(f"  片段截取失败：{e}")
        try:
            if writer is not None:
                writer.release()
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        return False
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()


def detect_runtime():
    """检测 NVIDIA GPU / CUDA，返回 (providers, use_gpu, gpu_name, note)。"""
    providers = ["CPUExecutionProvider"]
    use_gpu = False
    gpu_name = "CPU"
    note = "使用 CPU 推理（未检测到 GPU/CUDA）。"
    try:
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            # 显式指定 CUDA EP 选项：
            #  - cudnn_conv_algo_search=HEURISTIC：默认的 EXHAUSTIVE 会在推理期反复
            #    穷举卷积算法，实测明显更慢；HEURISTIC 略快且稳定。
            #  - arena_extend_strategy=kSameAsRequested：减少显存反复扩张带来的抖动。
            providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "arena_extend_strategy": "kSameAsRequested",
                }),
                "CPUExecutionProvider",
            ]
            use_gpu = True
            try:
                import subprocess
                out = subprocess.run(["nvidia-smi", "--query-gpu=name",
                                      "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=5)
                gpu_name = out.stdout.strip().splitlines()[0] if out.stdout.strip() else "GPU"
            except Exception:
                gpu_name = "GPU"
            note = f"检测到 GPU（{gpu_name}），启用 CUDA 加速。"
    except Exception:
        pass
    return providers, use_gpu, gpu_name, note


# ============================================================
# 检测引擎
# ============================================================
class Engine:
    """封装人脸检测 + 比对逻辑（在后台线程调用）"""

    def __init__(self, providers, use_gpu, logger, mode='compare'):
        self.providers = providers
        self.use_gpu = use_gpu
        self.logger = logger
        self.mode = mode
        self.app = None
        self.target_emb = None
        self.target_bbox = None

    def load_model(self):
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("insightface 未安装或导入失败，无法运行人脸检测。"
                               "请通过启动脚本安装依赖后重试。")
        self.logger("正在加载人脸检测模型 (buffalo_l) ...")
        ctx_id = 0 if self.use_gpu else -1
        _root = _MODEL_ROOT

        # 【性能关键】det_size 必须显式固定为 (640, 640)。
        # insightface 的 prepare 签名是 prepare(ctx_id, det_thresh=0.5, det_size=None)：
        # 若不传 det_size，会退化成动态尺寸列表 [(128,128),(640,640)]。而 det_10g 的
        # onnx 输入是动态的 [1,3,'?','?']，动态尺寸下 CUDA/cuDNN 无法缓存卷积算法，
        # 每帧都要重新搜索 —— 实测 266ms/帧（约 3.7fps），比固定尺寸慢 17 倍以上。
        # 固定 (640,640) 后实测 13.8ms/帧（约 72fps），检测即可逐帧运行。
        det_size = (640, 640)

        # 【性能关键】只加载真正用到的模型。
        # insightface 在 allowed_modules=None 时会加载并运行 buffalo_l 的全部 5 个模型，
        # 其中 genderage(96x96) 与 landmark_3d_68(192x192) 本工具完全用不到，却会
        # “每帧 × 每个人脸”各跑一次推理，是纯浪费。
        #   本工具只用到：detection(bbox) / landmark_2d_106(kps，质量分用) /
        #                 recognition(embedding，仅比对模式需要)
        allowed = (['detection', 'landmark_2d_106', 'recognition']
                   if self.mode == 'compare'
                   else ['detection', 'landmark_2d_106'])

        provs = self.providers
        for attempt in (0, 1, 2):
            try:
                kw = {"providers": provs}
                if attempt >= 1:
                    kw = {}  # 退回默认 providers
                if attempt <= 0:
                    kw["allowed_modules"] = allowed
                if _root is not None:
                    self.app = FaceAnalysis(name="buffalo_l", root=_root,
                                            det_size=det_size, **kw)
                else:
                    self.app = FaceAnalysis(name="buffalo_l", det_size=det_size, **kw)
                break
            except Exception:
                # providers 带选项的元组格式 / allowed_modules 不被支持时逐级退回
                provs = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.use_gpu \
                    else ["CPUExecutionProvider"]
                if attempt == 2:
                    raise

        try:
            self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        except TypeError:
            self.app.prepare(ctx_id=ctx_id)
        self.logger(f"模型加载完成（检测输入尺寸 {det_size}，"
                    f"{'GPU' if self.use_gpu else 'CPU'} 推理，"
                    f"模型 {len(self.app.models)} 个：{', '.join(self.app.models.keys())}）。")

    def set_target(self, img_path):
        img = imread_utf8(img_path)
        if img is None:
            raise RuntimeError(f"无法读取目标人脸图片：{img_path}")
        faces = self.app.get(img)
        if not faces:
            raise RuntimeError("目标图片中未检测到人脸，请换一张清晰正脸。")
        faces.sort(key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                   reverse=True)
        self.target_emb = faces[0].embedding
        self.target_bbox = faces[0].bbox
        self.logger(f"目标人脸已加载（检测到 {len(faces)} 张人脸，取最大的一张）。")


# ============================================================
# 线程：播放（按帧率读取并喂检测）+ 检测（含画面标注后回传显示）
# ============================================================
class RingBuffer:
    """线程安全的环形缓冲（RTSP 实时片段用）。

    内存优化：原始帧在 1080p 下约 6MB/帧，pad=10s、25fps 时需缓存约 285 帧，
    直接存原始像素会占用 1.7GB 并带来巨大的分配/GC 压力（表现为周期性卡顿）。
    这里对较宽的帧改用 JPEG 压缩存储（质量 92，肉眼几乎无损），
    内存降到约 1/20；小帧仍直接存，避免无谓的编解码开销。
    """

    _COMPRESS_MIN_W = 960     # 宽度达到该值才压缩
    _JPEG_Q = 92              # JPEG 质量：92 对人脸细节基本无损

    def __init__(self, maxlen=0):
        self.dq = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def append(self, f, t, frame):
        """存入一帧（大帧以 JPEG 字节存储以节省内存）。"""
        h, w = frame.shape[:2]
        if w >= self._COMPRESS_MIN_W:
            ok, buf = cv2.imencode('.jpg', frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self._JPEG_Q])
            if ok:
                payload = (buf, True)
            else:
                payload = (frame.copy(), False)
        else:
            payload = (frame.copy(), False)
        with self.lock:
            self.dq.append((f, t, payload))

    def snapshot_from(self, t0):
        """取出 t >= t0 的帧（压缩帧会被解码回图像）。"""
        with self.lock:
            items = [x for x in self.dq if x[1] >= t0]
        out = []
        for f, t, (data, is_jpeg) in items:
            if is_jpeg:
                fr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if fr is None:
                    continue
            else:
                fr = data
            out.append((f, t, fr))
        return out


class PlaybackThread(threading.Thread):
    """按视频帧率读取并“立即显示”每帧；检测在独立线程采样运行（二者解耦）。

    关键：显示与检测解耦 —— 每帧读取后立刻把原始帧推给 disp_q('raw')，
    主线程即时刷新画面，不再等待检测完成；检测线程只消费 det_q 的采样帧。
    因此视频播放速度只取决于解码与显示，不受 insightface 推理速度拖累，
    彻底解决 HEVC / RTSP 下的卡顿。目标框以检测节奏刷新（滞后仅数帧），
    既流畅又保持实时。

    detect_every：每隔多少帧送一帧给检测（由“实时性平衡”滑块控制）。
      - 滑块偏“流畅”：detect_every 大 -> 检测负载低 -> 视频最顺，但框更新慢。
      - 滑块偏“检测”：detect_every=1 -> 每帧检测 -> 框最实时（负载高）。
    tight_box：滑块拉到最右（“紧跟踪”）时为 True。此时显示帧不领先检测超过
      max_lead 帧，用“牺牲播放流畅度”换取目标框紧贴人脸，彻底消除检测滞后。
      仅在 tight_box 模式下生效；其余模式维持解耦（显示流畅优先）。
    暂停：pause_event 置位时挂起读取。

    分析模式（V1.2 新增，fullspeed）：
      - False（实时分析，默认）：按视频帧率节流（恒定 1X），边播放边比对。
      - True（全速分析）：不等待播放节奏，**全速解码 + 逐帧送检测**，把算力全给比对，
        尽量快地跑完全片；预览变成“分析进度画面”，帧率以「分析帧率」标签显示。
        实现要点：
          * 去掉 frame_interval 节流；
          * det_q 改为**阻塞投递**（队列满则等待），保证一帧不漏地送检；
          * 关闭紧跟踪门控（本就没有播放领先的概念）；
          * 预览帧降频发送（disp_every）且队列满即丢，避免 UI 侧开销反向拖慢检测。
        运行中可动态切换（每轮循环重新读取 self.fullspeed）。

    V1.3：取消播放倍速（speed）参数，实时分析恒定按视频原始帧率播放。
    """

    def __init__(self, kind, src, disp_q, det_q, stop_event, pause_event,
                 src_idx, total_src, pad, detect_every=2,
                 tight_box=False, det_progress=None, max_lead=2, fullspeed=False):
        super().__init__(daemon=True)
        self.kind = kind
        self.src = src
        self.disp_q = disp_q
        self.det_q = det_q
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.src_idx = src_idx
        self.total_src = total_src
        self.pad = pad
        self.detect_every = detect_every
        self.tight_box = tight_box
        self.det_progress = det_progress
        self.max_lead = max_lead
        self.fullspeed = fullspeed
        # 全速模式下预览帧发送间隔：每 N 帧才做一次缩放+投递，把 CPU 让给解码与检测。
        # 实测每次“缩放 + 建 QImage + 建 QPixmap”约 2~3ms，全速下每 3 帧发一帧
        # （约 15fps 预览）完全够看，且能显著减少 UI 线程反向拖慢检测的概率。
        self.disp_every = 3
        self.fps = 25.0
        self.total = 0
        self.ring = None  # RTSP 用，由 Runner 注入
        self.disp_size = None  # (w, h) 显示目标尺寸，由主线程持续更新

    def _to_display(self, frame):
        """把帧缩放到显示尺寸（保持比例，只缩不放），返回 (帧, 缩放系数)。

        缩放放在工作线程完成，UI 线程只处理小图，主线程开销与源分辨率解耦：
        4K 素材也能流畅播放。缩放系数用于把“原图坐标系下的检测框”换算到显示帧。
        """
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return frame, 1.0
        target = self.disp_size
        if target:
            tw, th = int(target[0]), int(target[1])
        else:
            tw, th = DISPLAY_MAX_W, DISPLAY_MAX_W
        if tw <= 0 or th <= 0:
            tw, th = DISPLAY_MAX_W, DISPLAY_MAX_W
        if w <= tw and h <= th:
            return frame.copy(), 1.0
        s = min(tw / float(w), th / float(h))
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA), s

    @staticmethod
    def _put(q, item, maxsize=4):
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except Exception:
                pass
            try:
                q.put_nowait(item)
            except Exception:
                pass

    def _put_blocking(self, q, item):
        """阻塞投递：队列满则等待检测消费（全速模式用，保证一帧不漏地送检）。

        以 0.1s 为粒度轮询 stop_event，保证点「结束」时能立即退出而不永久卡住。
        返回 False 表示期间收到了停止信号，调用方应结束循环。
        """
        while not self.stop_event.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.src)
            if self.kind == "rtsp":
                # 降低 RTSP 缓冲，减少延迟；优先 TCP 传输，降低抖动卡顿
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                try:
                    cap.set(cv2.CAP_PROP_RTSP_TRANSPORT, 1)
                except Exception:
                    pass
            if not cap.isOpened():
                self._put(self.disp_q, ('error', f"无法打开视频/流：{self.src}"))
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            if fps <= 0:
                fps = 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.fps = fps
            self.total = total

            if self.kind == "rtsp":
                self._put(self.disp_q, ('progress_mode', 'indeterminate'))

            ring_max = max(30, int((self.pad + 1.0) * fps) + 10)
            f = 0
            last_prog = -1
            frame_interval = 1.0 / fps
            while not self.stop_event.is_set():
                # 暂停：挂起播放与检测喂帧
                if self.pause_event.is_set():
                    time.sleep(0.1)
                    continue
                ret, frame = cap.read()
                if not ret:
                    if self.kind == "rtsp":
                        # 断流重连
                        time.sleep(2.0)
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = cv2.VideoCapture(self.src)
                        try:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass
                        try:
                            cap.set(cv2.CAP_PROP_RTSP_TRANSPORT, 1)
                        except Exception:
                            pass
                        continue
                    break
                t = f / fps
                if self.ring is not None:
                    self.ring.append(f, t, frame)
                    while len(self.ring.dq) > ring_max:
                        self.ring.dq.popleft()
                # 每轮重新读取，支持运行中在「实时分析 / 全速分析」之间切换
                fs = self.fullspeed
                # 检测喂帧：全速模式逐帧阻塞投递（不丢帧）；实时模式按 detect_every 采样
                if fs or (f % self.detect_every == 0):
                    item = ('det', self.src_idx, frame.copy(), f, t)
                    if fs:
                        if not self._put_blocking(self.det_q, item):
                            break  # 已被要求停止
                    else:
                        self._put(self.det_q, item, 4)
                # 紧跟踪模式（滑块最右）：显示帧不领先检测超过 max_lead 帧，
                # 用“降低播放流畅度”换取目标框紧贴人脸。代价是显示会出现卡顿，
                # 但检测滞后被限制在 max_lead 帧内，配合预测几乎无延迟。
                # 全速模式没有“播放领先”的概念，直接跳过该门控。
                if (not fs) and self.tight_box and self.det_progress is not None:
                    waited = 0.0
                    while not self.stop_event.is_set():
                        last = self.det_progress.get('f', -1)
                        if last < 0 or (f - last) <= self.max_lead:
                            break
                        time.sleep(0.004)
                        waited += 0.004
                        if waited > 3.0:  # 安全网：检测卡死也不永久阻塞播放
                            break
                # 显示：工作线程先缩放到显示尺寸，主线程只做“小帧→QPixmap”。
                # 全速模式下预览降频（每 disp_every 帧一次）且队列满即丢，
                # 确保 UI 侧的画面缩放开销不会反向拖慢解码与检测。
                if (not fs) or (f % self.disp_every == 0):
                    disp_frame, scale = self._to_display(frame)
                    self._put(self.disp_q,
                              ('raw', self.src_idx, disp_frame, f, t, scale), 3)
                # 进度（仅本地文件）
                if self.kind != "rtsp" and total > 0:
                    pct = int(((self.src_idx) + min(1.0, f / total)) / self.total_src * 100)
                    if pct != last_prog:
                        self._put(self.disp_q, ('progress', pct))
                        last_prog = pct
                # 节流：实时模式按视频原始帧率等待到下一帧；全速模式不等待，能跑多快跑多快
                if not fs:
                    time.sleep(frame_interval)
                f += 1
            self._put(self.disp_q, ('end', self.src_idx, None, f, t))
        except Exception as e:
            self._put(self.disp_q, ('error', f"播放线程异常：{e}"))
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass


class DetectThread(threading.Thread):
    """消费 det_q：检测 + 比对/质量过滤 + 画面标注 + 片段/截图保存。

    通过 disp_q 回传已标注画面：('frame', src_idx, annotated_frame, f, t)。
    通过 res_q 回传结果：('detect', dict)、('log', msg)、('episode', n)、
    ('framecount', n)、('error', msg)。

    由于“显示帧”与“人脸框”在同一帧上完成标注，二者严格同步，
    目标框跟踪不再有播放/检测错位造成的延时。

    mode='compare'：与目标人脸相似度 >= 阈值即比中并截取。
    mode='detect' ：人脸质量 >= 阈值即截取（无需目标人脸）；框上显示质量分。
    """

    def __init__(self, eng, sources, det_q, res_q, disp_q, stop_event, params, out, mode,
                 det_progress=None):
        super().__init__(daemon=True)
        self.eng = eng
        self.sources = sources
        self.det_q = det_q
        self.res_q = res_q
        self.disp_q = disp_q
        self.stop_event = stop_event
        self.params = params
        self.out = out
        self.mode = mode
        self.det_progress = det_progress  # 共享：检测线程写最新已完成帧号，供播放线程紧跟踪门控
        self._cur_src = -1
        self._active_seg = None
        self._last_frame_t = -999.0
        self._frame_idx = 0
        self._episode = 0
        self._last_boxes = []
        self.ring = None
        self._clip_writer = None
        self._clip_until = -1.0
        self._clip_path = None
        # 人脸优选缓冲：当前优选窗口起始时间 + 窗口内质量最高的帧
        self._prefer_t0 = None
        self._prefer_best = None
        # 分析帧率统计（每 0.5s 上报一次“实际完成检测的帧率”，供界面显示）
        self._fps_n = 0
        self._fps_t0 = time.time()

    def _put_res(self, item):
        try:
            self.res_q.put_nowait(item)
        except Exception:
            pass

    def _put_disp(self, item):
        try:
            self.disp_q.put_nowait(item)
        except queue.Full:
            try:
                self.disp_q.get_nowait()
            except Exception:
                pass
            try:
                self.disp_q.put_nowait(item)
            except Exception:
                pass

    def _close_clip(self):
        if self._clip_writer is not None:
            try:
                self._clip_writer.release()
            except Exception:
                pass
            self._clip_writer = None
            self._clip_until = -1.0
            self._put_res(('log', f"  片段写入完成：{os.path.basename(self._clip_path) if self._clip_path else ''}"))
            self._clip_path = None

    def _tick_fps(self):
        """统计并上报分析帧率（单位时间实际完成检测的帧数）。

        每 0.5s 汇总一次：太密的数字跳动无意义，太疏又看不出全速模式的提速效果。
        """
        self._fps_n += 1
        now = time.time()
        dt = now - self._fps_t0
        if dt >= 0.5:
            self._put_res(('detfps', self._fps_n / dt if dt > 0 else 0.0))
            self._fps_n = 0
            self._fps_t0 = now

    def run(self):
        while not self.stop_event.is_set() or not self.det_q.empty():
            try:
                item = self.det_q.get(timeout=0.3)
            except queue.Empty:
                continue
            if item[0] != 'det':
                continue
            _, src_idx, frame, f, t = item
            if src_idx != self._cur_src:
                self._cur_src = src_idx
                if self.det_progress is not None:
                    self.det_progress['f'] = -1  # 切换源：清零进度，避免新源被旧进度阻塞
                self._flush_prefer()  # 先落盘上一源残留的优选窗口
                self._active_seg = None
                self._last_frame_t = -999.0
                self._frame_idx = 0
                self._episode = 0
                self._close_clip()
                # 重置帧率统计，避免把上一源的速率混入新源
                self._fps_n = 0
                self._fps_t0 = time.time()
            self._process_frame(frame, f, t)
            self._tick_fps()
        self._flush_prefer()  # 结束前落盘最后一个优选窗口
        self._close_clip()
        # 等待所有异步 I/O（片段编码 / 截图写盘）完成，避免“分析完成”时文件还没落盘
        _IOWorker.wait_idle()

    def _process_frame(self, frame, f, t):
        try:
            faces = self.eng.app.get(frame)
        except Exception as e:
            self._put_res(('log', f"[警告] 第 {f} 帧检测异常：{e}"))
            return

        best_sim = -1.0
        boxes = []
        any_capture = False
        q_thr = self.params['quality_threshold']
        s_thr = self.params['sim_threshold']
        for face in faces:
            q = face_quality_score(face, frame)
            if self.mode == 'detect':
                # 检测抓图模式：所有检出人脸画目标框；质量达阈值即截取
                boxes.append({'bbox': face.bbox, 'q': q, 'kind': 'detect'})
                if q >= q_thr:
                    any_capture = True
            else:
                sim = cosine(face.embedding, self.eng.target_emb)
                if sim > best_sim:
                    best_sim = sim
                if q < q_thr:
                    # 质量低：检出但不比对
                    boxes.append({'bbox': face.bbox, 'sim': sim, 'q': q, 'kind': 'lowq'})
                    continue
                is_match = sim >= s_thr
                boxes.append({'bbox': face.bbox, 'sim': sim, 'q': q,
                              'kind': 'match' if is_match else 'ok'})
                if is_match:
                    any_capture = True

        self._last_boxes = boxes
        # 回传检测结果（含人脸框列表）-> 主线程据此在“当前显示帧”上实时叠加目标框。
        # 显示与检测解耦：播放线程已自行显示原始画面，这里不再回传标注帧，
        # 从而避免画面被检测速度拖慢（卡顿）。
        self._put_res(('detect', {'src': self._cur_src, 'boxes': boxes,
                                  'best_sim': best_sim, 'face_count': len(faces),
                                  'mode': self.mode, 'f': f, 't': t}))
        # 记录已完成检测的最新帧号，供播放线程“紧跟踪”门控使用
        if self.det_progress is not None:
            self.det_progress['f'] = f

        if any_capture:
            self._on_capture(frame, f, t)

        # RTSP 实时片段续写
        if self._clip_writer is not None:
            if t <= self._clip_until:
                try:
                    self._clip_writer.write(frame)
                except Exception:
                    pass
            else:
                self._close_clip()

    def _best_qualifying_q(self):
        """当前帧中触发截取的人脸里，质量最高的质量分。"""
        q_best = -1.0
        for b in self._last_boxes:
            if b['kind'] in ('match', 'detect'):
                if b['q'] > q_best:
                    q_best = b['q']
        return q_best

    def _flush_prefer(self):
        """把当前优选窗口内质量最高的帧落盘（窗口内无命中则跳过）。"""
        if self._prefer_best is not None:
            q_best, frame, boxes, cap_t = self._prefer_best
            self._save_screenshot(frame, boxes=boxes, t=cap_t)
            self._prefer_best = None

    def _on_capture(self, frame, f, t):
        pad = self.params['pad']
        kind, _src = self.sources[self._cur_src]
        enable_prefer = self.params.get('enable_prefer', True)
        prefer_dur = self.params['prefer_dur']
        q_best = self._best_qualifying_q()

        if self._active_seg is None or t > self._active_seg[1]:
            # 新一次出场：先把上一次出场残留的优选窗口落盘
            self._flush_prefer()
            self._episode += 1
            seg_start = max(0.0, t - pad)
            seg_end = t + pad
            self._active_seg = (seg_start, seg_end)
            self._put_res(('log', f"▶ 第 {self._episode} 次出场 @ {fmt_time(t)}，"
                                   f"截取 {fmt_time(seg_start)}~{fmt_time(seg_end)}"))
            if self.params.get('enable_segment', True):
                if kind == 'rtsp':
                    self._start_rtsp_clip(frame, t, pad)
                else:
                    name = f"片段_{self._episode:03d}_出场{fmt_time(t).replace(':', '-')}.mp4"
                    path = os.path.join(self.out, name)
                    # 异步截取：重新解码+编码整段视频很慢，绝不能阻塞检测线程
                    # （否则目标框会卡住不动）。
                    _IOWorker.submit(extract_segment, _src, seg_start, seg_end, path,
                                     lambda m: self._put_res(('log', m)))
                self._put_res(('episode', self._episode))
            else:
                self._put_res(('log', "  （已关闭视频截取，仅保存截图）"))
                self._put_res(('episode', self._episode))
            # 出场首帧直接保存，并开启新的优选窗口
            self._save_screenshot(frame, boxes=self._last_boxes, t=t)
            self._prefer_t0 = t
            self._prefer_best = None
            self._last_frame_t = t
            return

        # 同一出场内：按“人脸优选”或“时长节流”处理
        if enable_prefer:
            # 缓冲窗口内质量最高的帧，窗口结束（或出场/源结束）再落盘
            if self._prefer_best is None or q_best > self._prefer_best[0]:
                self._prefer_best = (q_best, frame, self._last_boxes, t)
            if self._prefer_t0 is None:
                self._prefer_t0 = t
            if t - self._prefer_t0 >= prefer_dur:
                self._flush_prefer()
                self._prefer_t0 = t
        else:
            # 未启用人脸优选：人脸质量达阈值即截，逐帧保存（无节流）
            self._save_screenshot(frame, boxes=self._last_boxes, t=t)

    def _start_rtsp_clip(self, frame, t, pad):
        if self._clip_writer is not None:
            return
        name = f"片段_{self._episode:03d}_出场{fmt_time(t).replace(':', '-')}.mp4"
        path = os.path.join(self.out, name)
        self._clip_path = path
        self._clip_until = t + pad
        h, w = frame.shape[:2]
        fps = 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._clip_writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if self._clip_writer.isOpened() and self.ring is not None:
            past = self.ring.snapshot_from(t - pad)
            for (_, tt, fr) in past:
                try:
                    self._clip_writer.write(fr)
                except Exception:
                    pass
            self._put_res(('log', f"  RTSP 实时录制中（{fmt_time(pad * 2)} 窗口）..."))
            self._put_res(('episode', self._episode))
        else:
            self._close_clip()

    def _save_screenshot(self, frame, boxes=None, t=None):
        if boxes is None:
            boxes = self._last_boxes
        if t is None:
            t = self._last_frame_t
        self._frame_idx += 1
        if self.mode == 'detect':
            # 检测模式：把当前帧所有检出人脸都画框（不显示相似度）
            # V1.3：只整帧拷贝一次，逐个原地画框（旧写法每画一个框就拷贝一次整帧）
            marked = frame.copy()
            for b in boxes:
                draw_box_inplace(marked, b['bbox'], label=None, color=(0, 170, 0))
            name = f"检测帧_{self._frame_idx:03d}_{fmt_time(t).replace(':', '-')}.png"
        else:
            best = None
            best_sim = -1.0
            for b in boxes:
                if b['kind'] == 'match' and b['sim'] > best_sim:
                    best_sim = b['sim']
                    best = b
            if best is None:
                return
            marked = draw_box(frame, best['bbox'], label=sim_to_label(best_sim),
                             color=(0, 0, 255))
            name = f"匹配帧_{self._frame_idx:03d}_{fmt_time(t).replace(':', '-')}.png"
        path = os.path.join(self.out, name)
        tag = "检测帧" if self.mode == 'detect' else "红框截图"
        idx = self._frame_idx

        # marked 已是独立副本（draw_box/copy 产生），可安全交给后台线程写盘；
        # PNG 编码与磁盘 IO 异步化，避免拖慢检测线程。
        def _write():
            ok = imwrite_utf8(marked, path)
            if ok:
                self._put_res(('log', f"  ✔ {tag}：{name}"))
                self._put_res(('framecount', idx))
            else:
                self._put_res(('log', f"  [警告] 截图失败：写入图片失败（{path}）"))

        _IOWorker.submit(_write)


class RunnerThread(threading.Thread):
    """编排：加载模型 -> 启动检测线程 -> 逐个视频源播放。"""

    def __init__(self, sources, face, out, params, providers, use_gpu,
                 stop_event, pause_event, disp_q, det_q, res_q, mode,
                 analyze_mode='realtime'):
        super().__init__(daemon=True)
        self.sources = sources
        self.face = face
        self.out = out
        self.params = params
        self.providers = providers
        self.use_gpu = use_gpu
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.disp_q = disp_q
        self.det_q = det_q
        self.res_q = res_q
        self.mode = mode
        self.analyze_mode = analyze_mode  # 'realtime' | 'fullspeed'
        self._active_pb = None

    def _put(self, item):
        try:
            self.res_q.put_nowait(item)
        except Exception:
            pass

    def run(self):
        eng = Engine(self.providers, self.use_gpu, lambda m: self._put(('log', m)),
                     mode=self.mode)
        try:
            eng.load_model()
            eng.threshold = self.params['sim_threshold']
            if self.mode == 'compare':
                eng.set_target(self.face)
        except Exception as e:
            self._put(('error', str(e)))
            return

        det_progress = {'f': -1}  # 检测线程写、播放线程读的共享进度（紧跟踪门控用）
        det = DetectThread(eng, self.sources, self.det_q, self.res_q, self.disp_q,
                           self.stop_event, self.params, self.out, self.mode,
                           det_progress=det_progress)
        det.start()

        total_src = len(self.sources)
        # 实时性平衡 -> 检测采样间隔：偏“流畅”检测稀疏，偏“检测”每帧检测
        balance = int(self.params.get('balance', 50))
        tight_box = balance >= 85  # 最右“紧跟踪”：牺牲流畅度换目标框紧贴
        detect_every = 1 if tight_box else max(1, int(round(1 + (100 - balance) / 16.67)))
        # 全速分析：不做播放节流、逐帧检测；RTSP 无法脱离实时流，强制走实时分析
        want_fullspeed = (self.analyze_mode == 'fullspeed')
        for si, (kind, src) in enumerate(self.sources, start=1):
            if self.stop_event.is_set():
                break
            self._put(('source', f"分析 {si}/{total_src}：{src if kind == 'file' else 'RTSP 视频流'}"))
            # 丢弃上一源的残留检测帧
            try:
                while not self.det_q.empty():
                    self.det_q.get_nowait()
            except Exception:
                pass
            det_progress['f'] = -1  # 新源：重置进度
            ring = None
            if kind == "rtsp":
                ring = RingBuffer(maxlen=max(30, int((self.params['pad'] + 1.0) * 25) + 10))
            det.ring = ring
            # 仅本地文件支持全速分析
            fs = want_fullspeed and kind == 'file'
            pb = PlaybackThread(kind, src, self.disp_q, self.det_q, self.stop_event,
                                self.pause_event, si - 1, total_src,
                                self.params['pad'],
                                detect_every=1 if fs else detect_every,
                                tight_box=False if fs else tight_box,
                                det_progress=det_progress,
                                max_lead=2, fullspeed=fs)
            pb.ring = ring
            self._active_pb = pb
            pb.start()
            pb.join()

        # 通知检测线程结束（排空后退出）
        self.stop_event.set()
        det.join(timeout=15)
        self._put(('done', {'total_src': total_src, 'out': self.out}))


# ============================================================
# PyQt6 主界面
# ============================================================
# ---------------- 扁平化 / 现代化配色（浅色主题） ----------------
# 设计约定：
#   * 页面底色略灰，内容区一律“白卡片 + 1px 极浅描边 + 10px 圆角”，不用重边框；
#   * 强调色统一蓝色 #2563eb；成功/警告/危险色仅用于按钮与关键数值；
#   * 控件高度、圆角、内边距统一，避免“高矮胖瘦混排”。
C_BG        = "#f4f6fa"   # 页面底色
C_CARD      = "#ffffff"   # 卡片底色
C_BORDER    = "#e5e9f0"   # 卡片描边
C_BORDER_2  = "#dfe4ec"   # 输入控件描边
C_TEXT      = "#1f2a37"   # 主文字
C_TEXT_2    = "#6b7280"   # 次要文字
C_TEXT_3    = "#94a3b8"   # 弱提示文字
C_ACCENT    = "#2563eb"   # 强调蓝
C_OK        = "#10b981"   # 成功绿
C_WARN      = "#f59e0b"   # 警告橙
C_DANGER    = "#ef4444"   # 危险红
C_VIDEO_BG  = "#0b0f14"   # 视频区底色（保持深色，突出画面）

APP_QSS = f"""
* {{ outline: none; }}
QMainWindow, QWidget#central {{ background: {C_BG}; }}

/* ---------- 卡片式分组 ---------- */
QGroupBox {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 10px;
    margin-top: 13px;
    padding: 14px 14px 12px 14px;
    font-size: 10.5pt;
    font-weight: 600;
    color: {C_TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 13px;
    padding: 0 6px;
    color: {C_ACCENT};
}}

/* ---------- 文字 ---------- */
QLabel {{ color: {C_TEXT}; background: transparent; }}
QLabel#hint  {{ color: {C_TEXT_3}; font-size: 9pt; }}
QLabel#sub   {{ color: {C_TEXT_2}; font-size: 9pt; }}
QLabel#infoV {{ color: {C_TEXT};   font-size: 10.5pt; }}
QLabel#fps   {{ color: {C_OK};     font-size: 10.5pt; font-weight: 700; }}
QLabel#status{{ color: {C_TEXT_2}; font-size: 10pt; }}
QLabel#faceThumb {{
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 10px;
    color: {C_TEXT_3};
    font-size: 9pt;
}}

/* ---------- 按钮：统一扁平外观 ---------- */
QPushButton {{
    background: {C_CARD};
    color: #334155;
    border: 1px solid {C_BORDER_2};
    border-radius: 8px;
    padding: 7px 16px;
    font-size: 10pt;
    min-height: 18px;
}}
QPushButton:hover    {{ background: #f1f5f9; border-color: #cbd5e1; }}
QPushButton:pressed  {{ background: #e8eef6; }}
QPushButton:disabled {{ color: #a8b3c1; background: #f7f9fc; border-color: #eaeef5; }}

QPushButton#primaryBtn, QPushButton#warnBtn, QPushButton#dangerBtn {{
    color: #ffffff; font-weight: 600; border: none; padding: 9px 26px;
}}
QPushButton#primaryBtn {{ background: {C_ACCENT}; }}
QPushButton#primaryBtn:hover {{ background: #1d4ed8; }}
QPushButton#primaryBtn:disabled {{ background: #dfe6f3; color: #9aa8bd; }}
QPushButton#warnBtn {{ background: {C_WARN}; }}
QPushButton#warnBtn:hover {{ background: #d97706; }}
QPushButton#warnBtn:disabled {{ background: #f6e8d0; color: #b3a184; }}
QPushButton#dangerBtn {{ background: {C_DANGER}; }}
QPushButton#dangerBtn:hover {{ background: #dc2626; }}
QPushButton#dangerBtn:disabled {{ background: #f6dede; color: #b99a9a; }}

/* ---------- 输入类控件 ---------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QListWidget {{
    background: {C_CARD};
    border: 1px solid {C_BORDER_2};
    border-radius: 8px;
    padding: 5px 9px;
    color: {C_TEXT};
    selection-background-color: {C_ACCENT};
    selection-color: #ffffff;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {{
    border: 1px solid {C_ACCENT};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QListWidget:disabled {{
    background: #f7f9fc; color: #a8b3c1; border-color: #eaeef5;
}}
QListWidget::item {{ padding: 3px 6px; border-radius: 5px; }}
QListWidget::item:selected {{ background: #e8f0fe; color: {C_ACCENT}; }}

/* ---------- 单选 / 复选 ---------- */
/* 注意：Qt 样式表里 ::indicator 的 width/height 指的是「内容盒」，边框会加在外面。
   所以圆角半径必须按 (内容尺寸 + 2×边框宽) 的一半来给，否则会渲染成圆角方块
   （单选按钮必须是正圆）。这里固定：单选 14+2×2=18 → r9。
   复选框（FlatCheckBox）不走 QSS 的 ::indicator：它要画 √，
   而 QSS 只能靠 image:url(...) 贴图（带资源文件 + 路径脆弱），故改由 paintEvent 自绘。 */
QRadioButton, QCheckBox {{ color: #334155; font-size: 10pt; spacing: 7px; }}
QRadioButton::indicator {{ width: 14px; height: 14px; }}
QRadioButton::indicator {{
    border: 2px solid #c3cbd8; border-radius: 9px; background: {C_CARD};
}}
QRadioButton::indicator:hover {{ border-color: #93a7c4; }}
QRadioButton::indicator:checked {{
    border: 2px solid {C_ACCENT};
    /* 用径向渐变画「蓝环 + 实心蓝点」，避免改变边框宽度导致控件尺寸跳动 */
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
                stop:0 {C_ACCENT}, stop:0.42 {C_ACCENT},
                stop:0.52 {C_CARD}, stop:1 {C_CARD});
}}
QRadioButton::indicator:disabled {{ border-color: #dfe4ec; }}

/* ---------- 滑块 ---------- */
QSlider::groove:horizontal {{
    height: 6px; background: #e8ecf3; border-radius: 3px;
}}
QSlider::sub-page:horizontal {{ background: {C_ACCENT}; border-radius: 3px; }}
QSlider::add-page:horizontal {{ background: #e8ecf3; border-radius: 3px; }}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -5px 0;
    border-radius: 9px; background: {C_CARD}; border: 2px solid {C_ACCENT};
}}
QSlider::handle:horizontal:hover {{ border-color: #1d4ed8; }}
QSlider::groove:horizontal:disabled {{ background: #eef1f6; }}
QSlider::sub-page:horizontal:disabled {{ background: #cdd6e4; }}
QSlider::handle:horizontal:disabled {{
    background: #f7f9fc; border-color: #c8d0dc;
}}

/* ---------- 进度条 ---------- */
/* 进度百分比文字由 Qt 用 palette 文本色画在整条进度条上，会同时压在「已填充」与
   「未填充」两段上。因此填充色取中浅蓝（#7ea6ee），保证深灰文字在两种底色上都清晰。 */
QProgressBar {{
    border: none; border-radius: 6px; background: #e8ecf3;
    height: 12px; text-align: center; font-size: 9pt; color: #334155;
}}
QProgressBar::chunk {{ background: #7ea6ee; border-radius: 6px; }}

/* ---------- 日志区 ---------- */
QTextEdit#logText {{
    background: #fbfcfe;
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    font-family: Consolas, "Cascadia Mono", "Microsoft YaHei UI", monospace;
    font-size: 9pt;
    color: #334155;
}}

/* ---------- 滚动条 ---------- */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{
    background: #cbd5e1; border-radius: 5px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: #a8b6c8; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{
    background: #cbd5e1; border-radius: 5px; min-width: 24px;
}}

/* ---------- 提示气泡 / 弹窗 ---------- */
QMessageBox {{ background: {C_CARD}; }}
QMessageBox QLabel {{ color: {C_TEXT}; }}
QToolTip {{
    background: #1f2a37; color: #f8fafc; border: none;
    border-radius: 6px; padding: 5px 8px;
}}
"""


class VideoLabel(QLabel):
    """保持 16:9 比例的视频显示区（深色底，圆角扁平）。"""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 270)
        self.setStyleSheet(
            f"QLabel{{background:{C_VIDEO_BG};color:#7f93a8;"
            "border:1px solid #1b2430;border-radius:10px;font-size:10pt;}}")
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return int(w * 9.0 / 16.0)


class FaceThumb(QLabel):
    """目标人脸预览（竖版 3:4 友好）：随容器自适应放大，保持原图比例不裁剪。

    V1.3：日志栏移到底部后，左列纵向空间全部给到本控件，因此不再用固定尺寸，
    而是保留原始 QPixmap 并在每次尺寸变化时平滑重缩放，充分吃满可用空间。
    """

    def __init__(self):
        super().__init__()
        self.setObjectName("faceThumb")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(210, 275)   # 3:4 下限
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setText("（未选择）")
        self._src = None                # 原图 QPixmap（未缩放）
        self._refitting = False

    def set_source(self, pix):
        """设置原图；传 None 表示清空。"""
        self._src = pix
        if pix is None or pix.isNull():
            self._src = None
            self.setPixmap(QPixmap())
            return
        self.setText("")
        self._refit()

    def clear_source(self):
        self._src = None
        self.setPixmap(QPixmap())

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._refit()

    def _refit(self):
        if self._src is None or self._refitting:
            return
        tw = max(1, self.width() - 10)
        th = max(1, self.height() - 10)
        if tw <= 2 or th <= 2:
            return
        self._refitting = True
        try:
            self.setPixmap(self._src.scaled(
                tw, th,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        finally:
            self._refitting = False


class FlatCheckBox(QCheckBox):
    """扁平复选框：自绘方框，选中时画白色对勾（√）。

    为什么不用 QSS 的 `QCheckBox::indicator { image: url(...) }`：
      * 需要额外提交一张 png 资源进仓库；且 QSS 里的相对路径是按**进程工作目录**解析的，
        「双击 launch.bat」与「从别处启动」结果不同，很脆；
      * 用纯色方块代替对勾虽然简单，但用户明确要求勾选态显示 √。
    自绘没有资源依赖，颜色随全局主题常量走，对勾的线宽/圆头/圆角都能精确控制。
    方框总尺寸与其它扁平控件统一：18x18 + 2px 边框 + 4px 圆角。
    """

    BOX = 18        # 方框外沿边长（含边框）
    BORDER = 2      # 边框线宽
    RADIUS = 4      # 圆角
    GAP = 8         # 方框与文字的间距

    # 未选中 / 悬停 / 选中 的（填充色, 边框色）
    _OFF = (C_CARD, "#c3cbd8")
    _HOVER = (C_CARD, "#93a7c4")
    _ON = (C_ACCENT, C_ACCENT)
    _DISABLED = ("#f7f9fc", "#dfe4ec")

    _warned = False   # 自绘异常只提示一次（见 paintEvent），避免刷屏

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        # 自绘时 QSS 的 :hover 不生效，需要开启 hover 属性才能收到重绘
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def sizeHint(self):
        fm = self.fontMetrics()
        w = self.BOX + self.GAP + fm.horizontalAdvance(self.text()) + 2
        h = max(self.BOX + 4, fm.height() + 4)
        return QSize(w, h)

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, event):
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            fm = self.fontMetrics()
            on = self.isChecked()
            en = self.isEnabled()

            if not en:
                fill, border = self._DISABLED
            elif on:
                fill, border = self._ON
            elif self.underMouse():
                fill, border = self._HOVER
            else:
                fill, border = self._OFF

            # ---- 方框（垂直居中） ----
            box = QRect(1, (self.height() - self.BOX) // 2, self.BOX, self.BOX)
            p.setPen(QPen(QColor(border), self.BORDER))
            p.setBrush(QColor(fill))
            # QPen 以线中线绘制，内缩半个线宽，方框外沿才正好落在 BOX 边界内。
            # 注意：必须用 QRectF —— PyQt6 的 QRect.adjusted() 只接受 int，
            # 传 float（BORDER/2.0）会抛 TypeError（PyQt5 宽容，迁移时踩过）。
            half = self.BORDER / 2.0
            p.drawRoundedRect(QRectF(box).adjusted(half, half, -half, -half),
                              float(self.RADIUS), float(self.RADIUS))

            # ---- 对勾 √（相对方框按比例取点，换尺寸也不会走形） ----
            if on and en:
                s = float(self.BOX)
                x0, y0 = float(box.left()), float(box.top())
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(QColor("#ffffff"), 2.2,
                              Qt.PenStyle.SolidLine,
                              Qt.PenCapStyle.RoundCap,
                              Qt.PenJoinStyle.RoundJoin))
                path = QPainterPath()
                path.moveTo(x0 + 0.26 * s, y0 + 0.52 * s)
                path.lineTo(x0 + 0.44 * s, y0 + 0.70 * s)
                path.lineTo(x0 + 0.76 * s, y0 + 0.30 * s)
                p.drawPath(path)

            # ---- 文字 ----
            tx = self.BOX + self.GAP
            p.setPen(QColor("#334155" if en else "#a8b3c1"))
            p.drawText(QRect(tx, 0, max(1, self.width() - tx), self.height()),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       self.text())
        except Exception:
            # 绘制异常不能冒泡：Qt 槽/事件内未捕获异常会走 qFatal 直接崩进程。
            # 但也绝不能静默 —— V1.3 首次实现时正因这里吞掉 TypeError，
            # 勾选框整块「消失」却毫无提示。首次失败打到 stderr，便于排查。
            if not FlatCheckBox._warned:
                FlatCheckBox._warned = True
                import traceback
                traceback.print_exc()
        finally:
            p.end()


def _widen_spin_by_50(spin):
    """把数值框的宽度按当前 sizeHint 加宽 50%。

    为什么需要：QSpinBox / QDoubleSpinBox 的默认宽度由 style 按「数字 + 上下箭头」
    算出来，带中文后缀（如「秒」）时放不下 —— 后缀会被右边缘裁掉，只剩半个字。
    加宽 50% 既够显示又不过分占位。

    必须在 setRange / setSuffix 之后再调用：sizeHint 依赖这些属性，
    提前调用拿到的是加后缀前的旧值。
    """
    spin.setMinimumWidth(int(spin.sizeHint().width() * 1.5))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频人脸检测比对工具V1.3_by_huyin")
        # 工具图标（优先使用打包进 exe 的 icon.png；未打包时回退 icon.svg）
        _icon_path = None
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            _icon_path = os.path.join(sys._MEIPASS, "icon.png")
        if not _icon_path or not os.path.exists(_icon_path):
            _icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
        if not os.path.exists(_icon_path):
            _icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.svg")
        if os.path.exists(_icon_path):
            self.setWindowIcon(QIcon(_icon_path))
        # 默认窗口大小：日志栏通栏置于底部（可显示约 8 行文字），中部左列纵向空间
        # 全部让给「目标人脸」栏。因为日志变高，整体高度比 V1.3 初版加高 60px，
        # 以保住目标人脸预览的尺寸（窗口拉高时多出的空间也归它）。
        self.resize(1340, 1080)
        self.setMinimumSize(1180, 900)

        # 运行状态
        self.face_path = ""
        self._face_pix = None          # 目标人脸原图 QPixmap（供自适应重缩放）
        self.mode = 'compare'          # 'compare' | 'detect'
        self.source_kind = 'file'      # 'file' | 'rtsp'
        self.analyze_mode = 'realtime'  # 'realtime' | 'fullspeed'（全速仅本地视频可用）
        self._an_guard = False         # 防止单选按钮联动导致递归
        self.file_list = []            # 本地视频路径列表
        self.running = False
        self.paused = False
        self.stop_event = threading.Event()
        self.pause_event = None
        self.disp_q = queue.Queue(maxsize=3)
        self.det_q = queue.Queue(maxsize=4)
        self.res_q = queue.Queue()
        self._frame = None
        self._boxes = []
        self._disp_f = 0           # 当前显示帧的序号（用于目标框运动补偿）
        self._disp_scale = 1.0     # 显示帧/原图 的缩放系数（画框坐标换算用）
        self._frame_dirty = False  # 有新帧/新检测才重绘，避免定时器空转重复重绘同一帧
        self._tracks = {}          # tid -> 轨迹字典（运动补偿跟踪）
        self._track_seq = 0
        self._cur_disp_src = -1
        self._episodes = 0
        self._framecount = 0
        self._last_fps = 0.0       # 最近一次上报的分析帧率（用于完成时汇总）
        self.runner = None
        self.out = ""

        self.providers, self.use_gpu, self.gpu_name, self.note = detect_runtime()

        self._build_ui()

        self._timer = QTimer()
        self._timer.timeout.connect(self._poll)
        self._timer.start(16)  # ~60fps 轮询：降低“读取→显示”延迟，播放与目标框更跟手

        self._append_log("视频人脸检测比对工具 V1.3 启动")
        self._append_log(f"运行环境：{'GPU (CUDA)' if self.use_gpu else 'CPU'}")
        self._append_log(self.note)

    # ---------------- UI 构建 ----------------
    @staticmethod
    def _group(title):
        """创建一张卡片式分组框（外观全部由全局 QSS 控制，保持视觉统一）。"""
        g = QGroupBox(title)
        lay = QVBoxLayout(g)
        lay.setSpacing(7)
        lay.setContentsMargins(0, 0, 0, 0)
        return g

    @staticmethod
    def _hint(text, tip=None, wrap=True):
        """弱提示文字（灰色小字）。"""
        lb = QLabel(text)
        lb.setObjectName("hint")
        lb.setWordWrap(wrap)
        if tip:
            lb.setToolTip(tip)
        return lb

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(12)
        root.setContentsMargins(14, 14, 14, 12)

        # ===== 顶部：视频源（RTSP / 本地视频 二选一）+ 输出目录  |  参数设置 =====
        top = QHBoxLayout()
        top.setSpacing(12)

        # --- 左：视频源 + 输出目录（合并为一张卡片：左列原本比右侧参数卡矮，
        #         合并后不增加顶部高度，却省下整张「输出目录」卡片的纵向占用，
        #         腾出的空间全部让给中部「目标人脸」栏） ---
        left_top = QVBoxLayout()
        left_top.setSpacing(12)

        g_src = self._group("视频源（RTSP 实时流 / 本地视频 二选一，不可同时分析）")
        self.src_bg = QButtonGroup(self)
        self.rb_file = QRadioButton("本地视频文件")
        self.rb_rtsp = QRadioButton("RTSP 实时流")
        self.rb_file.setChecked(True)
        self.src_bg.addButton(self.rb_file, 1)
        self.src_bg.addButton(self.rb_rtsp, 2)
        self.rb_file.toggled.connect(
            lambda c, k='file': self._on_source_toggle(k) if c else None)
        self.rb_rtsp.toggled.connect(
            lambda c, k='rtsp': self._on_source_toggle(k) if c else None)
        h_src = QHBoxLayout()
        h_src.setSpacing(18)
        h_src.addWidget(self.rb_file)
        h_src.addWidget(self.rb_rtsp)
        h_src.addStretch(1)
        g_src.layout().addLayout(h_src)
        # RTSP 地址
        self.rtsp_url = QLineEdit()
        self.rtsp_url.setPlaceholderText("rtsp://user:pass@ip:port/stream")
        g_src.layout().addWidget(self.rtsp_url)
        # 视频文件列表
        self.video_list = QListWidget()
        self.video_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.video_list.setMaximumHeight(80)
        g_src.layout().addWidget(self.video_list)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_add_video = QPushButton("添加视频")
        self.btn_remove_video = QPushButton("移除")
        self.btn_clear_video = QPushButton("清空")
        self.btn_add_video.clicked.connect(self._add_videos)
        self.btn_remove_video.clicked.connect(self._remove_video)
        self.btn_clear_video.clicked.connect(self._clear_videos)
        row.addWidget(self.btn_add_video)
        row.addWidget(self.btn_remove_video)
        row.addWidget(self.btn_clear_video)
        row.addStretch(1)
        g_src.layout().addLayout(row)
        # 输出目录（与视频源同卡片，压缩纵向行数）
        h_out = QHBoxLayout()
        h_out.setSpacing(8)
        h_out.addWidget(QLabel("输出目录："))
        self.out_dir = QLineEdit(os.path.join(os.getcwd(), "输出结果"))
        h_out.addWidget(self.out_dir, 1)
        self.btn_browse_out = QPushButton("浏览")
        self.btn_browse_out.clicked.connect(self._browse_out)
        self.btn_open_out = QPushButton("打开目录")
        self.btn_open_out.clicked.connect(self._open_out)
        h_out.addWidget(self.btn_browse_out)
        h_out.addWidget(self.btn_open_out)
        g_src.layout().addLayout(h_out)
        left_top.addWidget(g_src)
        left_top.addStretch(1)
        top.addLayout(left_top, 1)

        # --- 右：参数设置（运行模式为第一排） ---
        right_top = QVBoxLayout()
        right_top.setSpacing(0)
        g_param = self._group("参数设置")
        # 第一排：运行模式二选一
        h_mode = QHBoxLayout()
        h_mode.setSpacing(16)
        h_mode.addWidget(QLabel("运行模式："))
        self.mode_bg = QButtonGroup(self)
        self.rb_compare = QRadioButton("人脸识别比对模式")
        self.rb_detect = QRadioButton("人脸检测抓图模式")
        self.rb_compare.setChecked(True)
        self.mode_bg.addButton(self.rb_compare, 1)
        self.mode_bg.addButton(self.rb_detect, 2)
        self.rb_compare.toggled.connect(
            lambda c, m='compare': self._on_mode_toggle(m) if c else None)
        self.rb_detect.toggled.connect(
            lambda c, m='detect': self._on_mode_toggle(m) if c else None)
        h_mode.addWidget(self.rb_compare)
        h_mode.addWidget(self.rb_detect)
        h_mode.addStretch(1)
        g_param.layout().addLayout(h_mode)

        # 相似度阈值（仅比对模式可调）
        h1 = QHBoxLayout()
        h1.setSpacing(8)
        h1.addWidget(QLabel("相似度阈值(%)："))
        self.sim_slider = QSlider(Qt.Orientation.Horizontal)
        self.sim_slider.setRange(10, 100)
        self.sim_slider.setValue(70)
        self.sim_slider.valueChanged.connect(
            lambda v: self.sim_val_label.setText(f"{v}%"))
        h1.addWidget(self.sim_slider, 1)
        self.sim_val_label = QLabel("70%")
        self.sim_val_label.setFixedWidth(42)
        h1.addWidget(self.sim_val_label)
        g_param.layout().addLayout(h1)
        # 人脸质量阈值
        h2 = QHBoxLayout()
        h2.setSpacing(8)
        h2.addWidget(QLabel("人脸质量阈值："))
        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(10, 80)
        self.quality_slider.setValue(70)
        self.quality_slider.valueChanged.connect(
            lambda v: self.quality_val_label.setText(f"{v}"))
        h2.addWidget(self.quality_slider, 1)
        self.quality_val_label = QLabel("70")
        self.quality_val_label.setFixedWidth(42)
        h2.addWidget(self.quality_val_label)
        g_param.layout().addLayout(h2)
        g_param.layout().addWidget(self._hint(
            "低于该质量分的人脸不做比对/抓图（光线暗 / 像素低 / 模糊 / 角度偏 / 光比大）"))
        # 人脸优选（取代原“截图间隔秒数”）：开关与时长同排，压缩纵向行数，
        # 把腾出的高度留给「目标人脸」栏（V1.3 布局要求）。
        h_pref = QHBoxLayout()
        h_pref.setSpacing(10)
        self.chk_prefer = FlatCheckBox("人脸优选（窗口内仅保留质量最高的一帧）")
        self.chk_prefer.setChecked(True)
        self.chk_prefer.toggled.connect(self._on_prefer_toggle)
        h_pref.addWidget(self.chk_prefer)
        h_pref.addStretch(1)
        h_pref.addWidget(QLabel("优选时长："))
        self.prefer_spin = QDoubleSpinBox()
        self.prefer_spin.setRange(0.5, 30.0)
        self.prefer_spin.setSingleStep(0.5)
        self.prefer_spin.setValue(3.0)
        self.prefer_spin.setSuffix(" 秒")
        _widen_spin_by_50(self.prefer_spin)
        h_pref.addWidget(self.prefer_spin)
        g_param.layout().addLayout(h_pref)
        # 实时性平衡滑块：左=视频流畅优先，右=人脸检测实时优先
        h_bal = QHBoxLayout()
        h_bal.setSpacing(8)
        h_bal.addWidget(QLabel("流畅性 ◀"))
        self.balance_slider = QSlider(Qt.Orientation.Horizontal)
        self.balance_slider.setRange(0, 100)
        # 默认 75（逐帧检测）：检测提速后逐帧检测已不拖慢播放，
        # 直接给到“目标框紧贴人脸”的档位作为开箱默认值。
        self.balance_slider.setValue(75)
        self.balance_slider.setToolTip(
            "已优化检测速度，默认即可逐帧检测：视频流畅与目标框实时可同时满足；\n"
            "偏左可降低显卡占用（框靠运动补偿跟踪补齐）。")
        self.balance_slider.valueChanged.connect(self._on_balance)
        h_bal.addWidget(self.balance_slider, 1)
        h_bal.addWidget(QLabel("▶ 检测实时"))
        self.balance_val_label = QLabel("检测")
        self.balance_val_label.setFixedWidth(48)
        h_bal.addWidget(self.balance_val_label)
        g_param.layout().addLayout(h_bal)
        # 关联视频（默认关闭）：开关与前后时长同排
        h_seg = QHBoxLayout()
        h_seg.setSpacing(10)
        self.chk_segment = FlatCheckBox("关联视频")
        self.chk_segment.setToolTip(
            "勾选后，命中时刻向前/向后各「前后时长」秒会拼成一个完整出场片段，"
            "随截图一并保存为 mp4。")
        self.chk_segment.setChecked(False)
        self.chk_segment.toggled.connect(self._on_segment_toggle)
        h_seg.addWidget(self.chk_segment)
        h_seg.addStretch(1)
        h_seg.addWidget(QLabel("前后时长："))
        self.pad_spin = QSpinBox()
        self.pad_spin.setRange(0, 120)
        self.pad_spin.setValue(10)
        self.pad_spin.setSuffix(" 秒")
        _widen_spin_by_50(self.pad_spin)
        h_seg.addWidget(self.pad_spin)
        g_param.layout().addLayout(h_seg)
        # 全部说明合并成一行，避免每项各占一行把卡片撑高
        self._param_hint = self._hint(
            "说明：未启用人脸优选时，人脸质量达阈值即截；关联视频会保存命中时刻"
            "向前/向后各「前后时长」秒拼接成的完整出场片段。")
        g_param.layout().addWidget(self._param_hint)
        self._on_segment_toggle(False)  # 初始化：未启用 -> 禁用时长控件
        right_top.addWidget(g_param)
        right_top.addStretch(1)
        top.addLayout(right_top, 1)
        root.addLayout(top)

        # ===== 中部：左(1/4 目标人脸 + 实时信息) 右(3/4 实时视频) =====
        # V1.3：日志栏已移至界面最底部通栏，左列腾出的纵向空间全部给「目标人脸」栏
        # （人脸预览自适应放大，卡面随窗口拉高）。
        lower = QHBoxLayout()
        lower.setSpacing(12)

        # --- 左 1/4：目标人脸（吃满纵向空间）+ 实时信息 ---
        left_col = QVBoxLayout()
        left_col.setSpacing(12)
        g_face = self._group("目标人脸")
        self.btn_select_face = QPushButton("选择目标人脸图片")
        self.btn_select_face.clicked.connect(self._select_face)
        g_face.layout().addWidget(self.btn_select_face)
        # 竖版 3:4 预览：不再固定尺寸，随窗口自适应；原图比例不裁剪
        self.face_thumb = FaceThumb()
        g_face.layout().addWidget(self.face_thumb, 1)
        self.face_path_label = QLabel("")
        self.face_path_label.setObjectName("sub")
        # 只显示文件名、完整路径放 tooltip：长路径若折成两行会白白吃掉
        # 「目标人脸」栏的纵向空间（从而压缩人脸预览），这里强制单行。
        self.face_path_label.setWordWrap(False)
        g_face.layout().addWidget(self.face_path_label)
        left_col.addWidget(g_face, 1)

        g_info = self._group("实时信息")
        self.sim_val = QLabel("当前相似度：--")
        self.match_val = QLabel("比对结果：未匹配")
        self.facecount_val = QLabel("实时人脸数：0")
        self.episode_val = QLabel("出场片段：0")
        self.shot_val = QLabel("截图：0")
        for w in (self.sim_val, self.match_val, self.facecount_val,
                  self.episode_val, self.shot_val):
            w.setObjectName("infoV")
        # 两列三行排布（原为 5 行单列）：省下约 45px 纵向空间给上方「目标人脸」栏
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(5)
        grid.addWidget(self.sim_val, 0, 0)
        grid.addWidget(self.facecount_val, 0, 1)
        grid.addWidget(self.match_val, 1, 0)
        grid.addWidget(self.episode_val, 1, 1)
        grid.addWidget(self.shot_val, 2, 0)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        g_info.layout().addLayout(grid)
        left_col.addWidget(g_info, 0)
        lower.addLayout(left_col, 1)

        # --- 右 3/4：实时视频 + 分析模式 + 状态/进度 + 操作按钮 ---
        right_col = QVBoxLayout()
        right_col.setSpacing(0)
        g_video = self._group("实时视频")
        self.video_label = VideoLabel()
        self.video_label.setText("未开始分析\n点击「开始」后此处实时显示画面")
        g_video.layout().addWidget(self.video_label, 1)

        # 分析模式（V1.2）：本地视频可选「实时分析 / 全速分析」；RTSP 仅支持实时分析。
        # 同一行右侧显示当前分析帧率（帧/s）。V1.3 起本行不再有「播放倍速」滑块。
        h_an = QHBoxLayout()
        h_an.setSpacing(16)
        h_an.addWidget(QLabel("分析模式："))
        self.an_mode_bg = QButtonGroup(self)
        self.rb_realtime = QRadioButton("实时分析")
        self.rb_fullspeed = QRadioButton("全速分析")
        self.rb_realtime.setChecked(True)
        self.an_mode_bg.addButton(self.rb_realtime, 1)
        self.an_mode_bg.addButton(self.rb_fullspeed, 2)
        for rb in (self.rb_realtime, self.rb_fullspeed):
            h_an.addWidget(rb)
        self.an_mode_hint = QLabel("")
        self.an_mode_hint.setObjectName("hint")
        h_an.addWidget(self.an_mode_hint)
        h_an.addStretch(1)
        self.fps_label = QLabel("分析帧率：-- 帧/s")
        self.fps_label.setObjectName("fps")
        h_an.addWidget(self.fps_label)
        g_video.layout().addLayout(h_an)

        self.status_label = QLabel("状态：就绪")
        self.status_label.setObjectName("status")
        g_video.layout().addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        g_video.layout().addWidget(self.progress)

        # 操作按钮：开始 / 暂停 / 结束
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        self.btn_start = QPushButton("开始")
        self.btn_pause = QPushButton("暂停")
        self.btn_end = QPushButton("结束")
        self.btn_start.setObjectName("primaryBtn")
        self.btn_pause.setObjectName("warnBtn")
        self.btn_end.setObjectName("dangerBtn")
        self.btn_pause.setEnabled(False)
        self.btn_end.setEnabled(False)
        self.btn_start.clicked.connect(self._start)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_end.clicked.connect(self._stop)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(self.btn_end)
        btn_row.addStretch(1)
        g_video.layout().addLayout(btn_row)
        right_col.addWidget(g_video, 1)
        lower.addLayout(right_col, 3)
        root.addLayout(lower, 1)

        # ===== 底部：日志栏（V1.3 起改为通栏全宽，矮横条，不占用中部空间） =====
        g_log = self._group("日志")
        self.log_text = QTextEdit()
        self.log_text.setObjectName("logText")
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(168)
        self.log_text.setMaximumHeight(200)
        g_log.layout().addWidget(self.log_text)
        root.addWidget(g_log, 0)

        # 初始化控件可用状态
        # 两个单选按钮都要接：切到「全速」时是 rb_fullspeed 变选中，
        # 只接 rb_realtime 会漏掉该场景（它此时是取消选中，checked=False 被忽略）。
        self.rb_realtime.toggled.connect(self._on_analyze_mode)
        self.rb_fullspeed.toggled.connect(self._on_analyze_mode)
        self._on_source_toggle('file')
        self._on_mode_toggle('compare')
        self._refresh_ctrl_states()

    # ---------------- 模式 / 源切换 ----------------
    def _on_mode_toggle(self, mode):
        self.mode = mode
        if mode == 'detect':
            self.sim_slider.setEnabled(False)
            self.sim_val_label.setText("—")
            self.btn_select_face.setEnabled(False)
            self.face_thumb.set_source(None)
            self.face_thumb.setText("（检测抓图模式：无需目标人脸）")
            self.sim_val.setVisible(False)
            self.match_val.setVisible(False)
        else:
            self.sim_slider.setEnabled(True)
            self.sim_val_label.setText(f"{self.sim_slider.value()}%")
            self.btn_select_face.setEnabled(True)
            # 若之前已选过有效人脸图，切回比对模式时直接恢复预览，不必重新选择
            if self.face_path and os.path.isfile(self.face_path):
                self._apply_face_thumb()
            else:
                self.face_thumb.set_source(None)
                self.face_thumb.setText("（未选择）")
            self.sim_val.setVisible(True)
            self.match_val.setVisible(True)

    def _on_source_toggle(self, kind):
        self.source_kind = kind
        if kind == 'rtsp':
            self.rtsp_url.setEnabled(True)
            self.video_list.setEnabled(False)
            self.btn_add_video.setEnabled(False)
            self.btn_remove_video.setEnabled(False)
            self.btn_clear_video.setEnabled(False)
        else:
            self.rtsp_url.setEnabled(False)
            self.video_list.setEnabled(True)
            self.btn_add_video.setEnabled(True)
            self.btn_remove_video.setEnabled(True)
            self.btn_clear_video.setEnabled(True)
        self._refresh_ctrl_states()

    def _refresh_ctrl_states(self):
        """统一刷新「源类型 + 分析模式 + 运行状态」相关控件的可用性。

        规则：
          - 全速分析仅本地视频支持；切到 RTSP 时强制回到实时分析（并禁用全速选项）；
          - 全速分析不按播放速度推进，故「实时性平衡」滑块禁用。
        V1.3：播放倍速已取消，本函数不再管理倍速控件。
        """
        is_file = (self.source_kind == 'file')

        if not is_file and self.analyze_mode == 'fullspeed':
            # 强制回到实时分析：setChecked 会触发 _on_analyze_mode，
            # 用 _an_guard 挡住以免递归刷新
            self._an_guard = True
            self.rb_realtime.setChecked(True)
            self._an_guard = False
            self.analyze_mode = 'realtime'

        full = (self.analyze_mode == 'fullspeed')
        self.rb_fullspeed.setEnabled(is_file)
        self.balance_slider.setEnabled(not full)
        if not is_file:
            self.an_mode_hint.setText("（RTSP 仅支持实时分析）")
        elif full:
            self.an_mode_hint.setText("（不按播放速度，全算力逐帧分析）")
        else:
            self.an_mode_hint.setText("")

    def _on_analyze_mode(self, checked):
        """分析模式单选切换（运行中也可即时切换，播放线程每轮循环重读该标志）。"""
        if self._an_guard or not checked:
            return
        new_mode = 'fullspeed' if self.rb_fullspeed.isChecked() else 'realtime'
        if new_mode == self.analyze_mode:
            self._refresh_ctrl_states()
            return
        self.analyze_mode = new_mode
        self._refresh_ctrl_states()
        pb = getattr(self.runner, '_active_pb', None) if self.runner is not None else None
        if pb is not None:
            fs = (new_mode == 'fullspeed')
            pb.fullspeed = fs
            if fs:
                pb.detect_every = 1    # 全速：强制逐帧检测
                pb.tight_box = False   # 无“播放领先”概念，紧跟踪门控自动失效
        if new_mode == 'fullspeed':
            self._append_log("分析模式：全速分析（不按播放速度，全算力逐帧比对）")
            self._set_status("全速分析中...")
        else:
            self._append_log("分析模式：实时分析（按视频速度边播放边比对）")
            if self.running:
                self._set_status("分析中...")

    # ---------------- 槽函数 ----------------
    def _select_face(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择目标人脸图片", "", "图片 (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.face_path = path
            self.face_path_label.setText(os.path.basename(path))
            self.face_path_label.setToolTip(path)
            self._apply_face_thumb()

    def _apply_face_thumb(self):
        """读取 self.face_path 并刷新目标人脸预览（原图存起来，交给控件自适应缩放）。

        V1.3 性能：改用 QImage.Format_BGR888 直通，省掉一次全图 cvtColor。
        缩放不再在这里做一次（控件本身会在尺寸变化时平滑重缩放），
        这里只保留原始 QPixmap，避免重复插值导致的画质损失。
        """
        img = imread_utf8(self.face_path)
        if img is None:
            self.face_thumb.set_source(None)
            self.face_thumb.setText("（图片读取失败）")
            return
        img = np.ascontiguousarray(img)  # QImage 需要连续内存
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_BGR888).copy()
        self._face_pix = QPixmap.fromImage(qimg)
        self.face_thumb.set_source(self._face_pix)

    def _add_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "添加视频文件", "", "视频 (*.mp4 *.avi *.mov *.mkv *.wmv)")
        for p in paths:
            if p not in self.file_list:
                self.file_list.append(p)
                self.video_list.addItem(p)

    def _remove_video(self):
        for it in self.video_list.selectedItems():
            path = it.text()
            self.file_list = [s for s in self.file_list if s != path]
            self.video_list.takeItem(self.video_list.row(it))

    def _clear_videos(self):
        self.file_list = []
        self.video_list.clear()

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录", self.out_dir.text())
        if d:
            self.out_dir.setText(d)

    @staticmethod
    def _detect_every_from_balance(balance):
        """实时性平衡 -> 检测采样间隔（每几帧送一帧给检测）。

        检测提速后（GPU 上约 13~22ms/帧），逐帧检测已完全可行，因此档位重标定为：
          流畅(balance<34) -> 4：检测负载最低，框靠运动补偿跟踪补齐
          平衡(34~66)      -> 2：兼顾负载与实时
          检测实时(>=67)   -> 1：逐帧检测，目标框紧贴人脸
        """
        b = int(balance)
        if b >= 67:
            return 1
        if b >= 34:
            return 2
        return 4

    def _on_prefer_toggle(self, checked):
        # 未启用人脸优选时，时长仅作为截图节流间隔，禁用无意义
        self.prefer_spin.setEnabled(checked)

    def _on_segment_toggle(self, checked):
        # 仅勾选「关联视频」时才允许设置前后时长
        self.pad_spin.setEnabled(checked)

    def _open_out(self):
        """在系统资源管理器中打开当前输出目录。"""
        d = self.out_dir.text().strip() or os.path.join(os.getcwd(), "输出结果")
        try:
            os.makedirs(d, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(d)))
        except Exception as e:
            self._append_log(f"[警告] 无法打开目录：{e}")

    def _on_balance(self, v):
        if v < 34:
            self.balance_val_label.setText("流畅")
        elif v < 67:
            self.balance_val_label.setText("平衡")
        elif v < 85:
            self.balance_val_label.setText("检测")
        else:
            self.balance_val_label.setText("紧跟踪")
        # 全速分析强制逐帧检测、且不按播放速度推进，平衡档位在此模式下无意义
        if self.analyze_mode == 'fullspeed':
            return
        # 运行中实时调整检测密度 / 紧跟踪开关（本地视频与 RTSP 均支持）
        if self.runner is not None and self.runner._active_pb is not None:
            pb = self.runner._active_pb
            tight = v >= 85
            pb.tight_box = tight
            pb.detect_every = 1 if tight else self._detect_every_from_balance(v)

    def _set_status(self, text):
        self.status_label.setText(f"状态：{text}")

    def _append_log(self, text):
        self.log_text.append(text)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ---------------- 开始 / 暂停 / 结束 ----------------
    def _start(self):
        if self.running:
            return

        # 构建视频源（RTSP 与本地视频二选一，不可同时）
        if self.source_kind == 'rtsp':
            url = self.rtsp_url.text().strip()
            if not url:
                QMessageBox.warning(self, "提示", "请填写 RTSP 视频流地址。")
                return
            sources = [("rtsp", url)]
        else:
            if not self.file_list:
                QMessageBox.warning(self, "提示", "请先添加至少一个视频文件。")
                return
            sources = [("file", p) for p in self.file_list]

        # 比对模式必须选择目标人脸
        if self.mode == 'compare':
            if not self.face_path or not os.path.isfile(self.face_path):
                QMessageBox.warning(self, "提示", "请先选择有效的目标人脸图片。")
                return

        # 分析模式：RTSP 是实时流，无法脱离播放节奏全速跑，强制走实时分析
        analyze_mode = self.analyze_mode
        if self.source_kind == 'rtsp':
            analyze_mode = 'realtime'

        out = self.out_dir.text().strip() or os.path.join(os.getcwd(), "输出结果")
        os.makedirs(out, exist_ok=True)

        params = {
            'sim_threshold': (self.sim_slider.value() / 100.0) if self.mode == 'compare' else 0.0,
            'quality_threshold': float(self.quality_slider.value()),
            'pad': max(0.0, float(self.pad_spin.value())),
            'prefer_dur': max(0.5, float(self.prefer_spin.value())),
            'enable_prefer': self.chk_prefer.isChecked(),
            'balance': self.balance_slider.value(),
            'enable_segment': self.chk_segment.isChecked(),
            'analyze_mode': analyze_mode,
        }
        self.sources = sources
        self.out = out
        self.stop_event.clear()
        self.pause_event = threading.Event()
        self.paused = False
        self.running = True

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("暂停")
        self.btn_end.setEnabled(True)
        # 平衡滑块 / 全速选项的可用性由「源类型 + 分析模式 + 运行状态」统一决定
        self._refresh_ctrl_states()

        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._episodes = 0
        self._framecount = 0
        self._frame = None
        self._boxes = []
        self._tracks = {}
        self._track_seq = 0
        self._disp_f = 0
        self._cur_disp_src = -1
        self.episode_val.setText("出场片段：0")
        self.shot_val.setText("截图：0")
        self.fps_label.setText("分析帧率：-- 帧/s")
        if self.mode == 'compare':
            self.sim_val.setText("当前相似度：--")
            self.match_val.setText("比对结果：未匹配")
        else:
            self.sim_val.setText("（检测模式）")
            self.match_val.setText("（检测模式）")
        self._set_status("全速分析中..." if analyze_mode == 'fullspeed' else "分析中...")
        self._append_log("-" * 40)
        self._append_log(f"模式：{'人脸检测抓图' if self.mode == 'detect' else '人脸识别比对'}  "
                         f"源：{'RTSP' if self.source_kind == 'rtsp' else '本地视频'}  "
                         f"分析：{'全速（不按播放速度）' if analyze_mode == 'fullspeed' else '实时（按视频速度）'}  "
                         f"质量阈值：{params['quality_threshold']:.0f}" +
                         (f"  相似度阈值：{self.sim_slider.value()}%" if self.mode == 'compare' else "") +
                         (f"  关联视频：{'开' if params['enable_segment'] else '关'}"))
        self.runner = RunnerThread(sources, self.face_path, out, params,
                                   self.providers, self.use_gpu, self.stop_event,
                                   self.pause_event, self.disp_q, self.det_q,
                                   self.res_q, self.mode,
                                   analyze_mode=analyze_mode)
        self.runner.start()

    def _toggle_pause(self):
        if not self.running:
            return
        if not self.paused:
            if self.pause_event is not None:
                self.pause_event.set()
            self.paused = True
            self.btn_pause.setText("继续")
            self._set_status("已暂停")
        else:
            if self.pause_event is not None:
                self.pause_event.clear()
            self.paused = False
            self.btn_pause.setText("暂停")
            self._set_status("分析中...")

    def _stop(self):
        if not self.running:
            return
        self.stop_event.set()
        # 若处于暂停，先解除暂停以便后台线程顺利退出
        if self.paused and self.pause_event is not None:
            self.pause_event.clear()
        self.paused = False
        self.btn_pause.setText("暂停")
        self._set_status("正在结束...")

    # ---------------- 轮询队列（显示 + 结果） ----------------
    def _poll(self):
        try:
            while True:
                item = self.disp_q.get_nowait()
                k = item[0]
                if k == 'raw':
                    if len(item) >= 6:
                        _, src_idx, frame, f, t, scale = item[:6]
                        self._disp_scale = scale or 1.0
                    else:
                        _, src_idx, frame, f, t = item[:5]
                        self._disp_scale = 1.0
                    if src_idx != self._cur_disp_src:
                        # 切换视频源时清空旧轨迹，避免上一段的人脸框遗留到新视频
                        self._tracks = {}
                        self._track_seq = 0
                        self._cur_disp_src = src_idx
                    self._frame = frame
                    self._disp_f = f
                    self._frame_dirty = True
                elif k == 'progress':
                    self.progress.setRange(0, 100)
                    self.progress.setValue(item[1])
                elif k == 'progress_mode':
                    if item[1] == 'indeterminate':
                        self.progress.setRange(0, 0)
                    else:
                        self.progress.setRange(0, 100)
                elif k == 'error':
                    self._show_error(item[1])
        except queue.Empty:
            pass

        try:
            while True:
                self._handle_res(self.res_q.get_nowait())
        except queue.Empty:
            pass

        # 把当前显示区域尺寸同步给播放线程，让其按该尺寸预缩放帧
        # （缩放成本从 UI 线程挪到工作线程，主线程开销与源分辨率解耦）
        if self.runner is not None and getattr(self.runner, "_active_pb", None) is not None:
            try:
                lw, lh = self.video_label.width(), self.video_label.height()
                if lw > 0 and lh > 0:
                    self.runner._active_pb.disp_size = (lw, lh)
            except Exception:
                pass

        self._repaint()

    def _handle_res(self, item):
        k = item[0]
        if k == 'detect':
            d = item[1]
            # 用检测结果更新运动补偿轨迹（替代“直接覆盖 _boxes”），
            # 使目标框能向前预测到当前显示帧，消除“框在后面追”的滞后。
            self._update_tracks(d['boxes'], d['f'])
            self._frame_dirty = True
            if d.get('mode') == 'detect':
                self.sim_val.setText("（检测模式）")
                self.match_val.setText("（检测模式）")
            else:
                best = d['best_sim']
                if best < 0:
                    self.sim_val.setText("当前相似度：--")
                else:
                    self.sim_val.setText(f"当前相似度：{best * 100:.2f}%")
                any_match = any(b['kind'] == 'match' for b in d['boxes'])
                self.match_val.setText("比对结果：" + ("匹配 ✓" if any_match else "未匹配"))
            self.facecount_val.setText(f"实时人脸数：{d['face_count']}")
        elif k == 'log':
            self._append_log(item[1])
        elif k == 'source':
            self._append_log(item[1])
            self._set_status(item[1])
        elif k == 'episode':
            self._episodes = item[1]
            self.episode_val.setText(f"出场片段：{item[1]}")
        elif k == 'framecount':
            self._framecount = item[1]
            self.shot_val.setText(f"截图：{item[1]}")
        elif k == 'detfps':
            self._last_fps = item[1]
            self.fps_label.setText(f"分析帧率：{item[1]:.1f} 帧/s")
        elif k == 'done':
            self._on_done(item[1])
        elif k == 'error':
            self._show_error(item[1])

    def _update_tracks(self, boxes, f):
        """用一帧检测结果更新运动补偿轨迹。

        核心思路（解决“框在后面追”）：检测比显示慢几帧，若直接把“检测帧 f 的框”
        画到“当前显示帧 f_disp(>f)”上，框就会落在人脸后方。这里记录每张脸的速度
        （按帧），并在绘制时把框向前预测 (f_disp - f) 帧，使其贴住当下位置。
        """
        obs = []
        for b in boxes:
            x1, y1, x2, y2 = [float(v) for v in b['bbox'][:4]]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w, h = abs(x2 - x1), abs(y2 - y1)
            obs.append({'cx': cx, 'cy': cy, 'w': w, 'h': h,
                        'kind': b['kind'], 'sim': b.get('sim'), 'q': b.get('q')})
        used = set()
        for o in obs:
            best, bestd = None, 1e18
            for tid, tr in self._tracks.items():
                # 关键修正：把轨迹预测到当前检测帧 f 的位置再做关联。
                # 若直接用 tr['cx']/tr['cy']（上一次检测帧的位置），移动中的人脸
                # 会因距离过远而被判定为新目标，从而产生“拖影”/“一串框”。
                elapsed = max(0, f - tr['f']) if tr['f'] is not None else 0
                pred_cx = tr['cx'] + tr['vx'] * elapsed
                pred_cy = tr['cy'] + tr['vy'] * elapsed
                d = ((o['cx'] - pred_cx) ** 2 + (o['cy'] - pred_cy) ** 2) ** 0.5
                # 关联阈值：随框大小自适应，避免相邻人脸误关联
                thr = max(50.0, 0.8 * max(tr['w'], tr['h']))
                if d < thr and d < bestd:
                    best, bestd = tid, d
            if best is not None:
                tr = self._tracks[best]
                pf = tr['f']
                if pf is not None and f > pf:
                    nvx = (o['cx'] - tr['cx']) / (f - pf)
                    nvy = (o['cy'] - tr['cy']) / (f - pf)
                    nvx = max(-60.0, min(60.0, nvx))
                    nvy = max(-60.0, min(60.0, nvy))
                    # alpha=0.7 让速度对停步/转向更敏感，减少人停下后框继续前漂
                    tr['vx'] = 0.3 * tr['vx'] + 0.7 * nvx
                    tr['vy'] = 0.3 * tr['vy'] + 0.7 * nvy
                tr['cx'], tr['cy'] = o['cx'], o['cy']
                tr['w'], tr['h'] = o['w'], o['h']
                tr['kind'] = o['kind']
                tr['sim'] = o['sim']
                tr['q'] = o['q']
                tr['f'] = f
                tr['miss'] = 0
                used.add(best)
            else:
                self._track_seq += 1
                self._tracks[self._track_seq] = {
                    'cx': o['cx'], 'cy': o['cy'], 'w': o['w'], 'h': o['h'],
                    'kind': o['kind'], 'sim': o['sim'], 'q': o['q'],
                    'vx': 0.0, 'vy': 0.0, 'f': f, 'miss': 0}
        # 未被匹配到的轨迹记一次未命中；连续未命中过多则判定离开画面并移除
        for tid, tr in list(self._tracks.items()):
            if tid not in used:
                tr['miss'] += 1
        max_miss = 6   # 容忍约 6 次检测间隔的遮挡/漏检；拖影轨迹不会残留太久
        for tid in [t for t, tr in self._tracks.items() if tr['miss'] > max_miss]:
            del self._tracks[tid]

    def _repaint(self):
        if self._frame is None or not self._frame_dirty:
            return
        self._frame_dirty = False
        if self._tracks:
            # 在当前显示帧上叠加“向前预测”后的目标框：把检测框按速度推到
            # 当前显示帧位置，从而消除检测滞后导致的“框在后面追”。
            try:
                disp = self._frame.copy()
                f = self._disp_f
                sc = self._disp_scale or 1.0  # 原图坐标 -> 显示帧坐标
                for tr in self._tracks.values():
                    shift = max(0, f - tr['f']) if tr['f'] is not None else 0
                    # 预测上限 30 帧：防止检测极滞后时速度估计误差被放大
                    shift = min(shift, 30)
                    cx = (tr['cx'] + tr['vx'] * shift) * sc
                    cy = (tr['cy'] + tr['vy'] * shift) * sc
                    w, h = tr['w'] * sc, tr['h'] * sc
                    x1 = int(round(cx - w / 2.0))
                    y1 = int(round(cy - h / 2.0))
                    x2 = int(round(cx + w / 2.0))
                    y2 = int(round(cy + h / 2.0))
                    if tr['kind'] == 'detect':
                        label = f"质量 {tr['q']:.0f}"
                        color = (0, 170, 0)
                    elif tr['kind'] == 'match':
                        label = sim_to_label(tr['sim'])
                        color = (0, 0, 255)
                    elif tr['kind'] == 'ok':
                        label = sim_to_label(tr['sim'])
                        color = (0, 200, 0)
                    else:  # lowq
                        label = f"{sim_to_label(tr['sim'])} 低质"
                        color = (130, 130, 130)
                    # V1.3：整帧只拷贝一次，多个人脸框逐个原地画（旧写法每框拷一次整帧）
                    draw_box_inplace(disp, (x1, y1, x2, y2),
                                     label=label, color=color)
                self._set_pixmap(disp)
                return
            except Exception:
                pass
        self._set_pixmap(self._frame)

    def _set_pixmap(self, frame_bgr):
        """把 BGR 帧送到预览区（显示管线热路径，每次刷新都会走到）。

        V1.3 性能优化（实测每次刷新省下约 2.5~3ms，占全速分析单帧 UI 开销的大头）：
          1. QImage.Format_BGR888 直通：省掉一次全图 cv2.cvtColor(BGR->RGB)
             （1280x720 约 1.5ms）；
          2. 去掉 QImage(...).copy()：QPixmap.fromImage 本身就会把像素拷进
             Qt 的绘制后端，再 copy 一次纯属浪费（约 1MB 级 memcpy）；
             只要本函数执行期间 frame_bgr 存活即安全（它由调用方持有）。
        """
        try:
            arr = np.ascontiguousarray(frame_bgr)  # 已是连续内存时零成本返回自身
            h, w = arr.shape[:2]
            qimg = QImage(arr.data, w, h, 3 * w,
                          QImage.Format.Format_BGR888)
            pix = QPixmap.fromImage(qimg)
            # 播放线程已按显示尺寸预缩放，这里通常无需再缩放；
            # 仅当窗口尺寸变化导致不匹配时才兜底缩放一次。
            ls = self.video_label.size()
            if ls.width() > 0 and ls.height() > 0 and (
                    abs(ls.width() - w) > 2 or abs(ls.height() - h) > 2):
                pix = pix.scaled(ls, Qt.AspectRatioMode.KeepAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation)
            self.video_label.setPixmap(pix)
        except Exception:
            pass

    def _on_done(self, summary):
        self.running = False
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("暂停")
        self.btn_end.setEnabled(False)
        self._refresh_ctrl_states()  # 运行结束：收回平衡滑块的操作权
        self._tracks = {}
        self._track_seq = 0
        self._set_status("完成")
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self._append_log("=" * 40)
        self._append_log("分析完成！")
        self._append_log(f"视频源：{summary.get('total_src', 1)} 个")
        self._append_log(f"分析模式：{'全速分析' if self.analyze_mode == 'fullspeed' else '实时分析'}"
                         f"  末段分析帧率：约 {self._last_fps:.1f} 帧/s")
        self._append_log(f"出场片段：{self._episodes} 个")
        self._append_log(f"截图：{self._framecount} 张")
        self._append_log(f"结果保存于：{summary.get('out', '')}")
        self._append_log("=" * 40)

    def _show_error(self, msg):
        self._append_log("✗ 错误：" + msg)
        self._set_status("出错")
        self.running = False
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("暂停")
        self.btn_end.setEnabled(False)
        self._refresh_ctrl_states()  # 运行结束：收回平衡滑块的操作权
        QMessageBox.critical(self, "运行错误", msg)

    def closeEvent(self, event):
        if self.running:
            self.stop_event.set()
            if self.paused and self.pause_event is not None:
                self.pause_event.clear()
        event.accept()


# ============================================================
# 入口
# ============================================================
def _pick_ui_font():
    """挑选一个可用的中文界面字体（避免 Qt 默认字体在中文下显示为方框/过窄）。"""
    try:
        fams = set(QFontDatabase.families())
        for name in ("Microsoft YaHei UI", "Microsoft YaHei",
                     "HarmonyOS Sans SC", "PingFang SC", "Noto Sans CJK SC"):
            if name in fams:
                return QFont(name, 10)
    except Exception:
        pass
    return QFont("Microsoft YaHei", 10)


def main():
    if not PYQT_AVAILABLE:
        print("错误：未安装 PyQt6，无法启动界面。请运行 launch.bat 安装依赖。", file=sys.stderr)
        print(str(_QT_ERR), file=sys.stderr)
        sys.exit(1)
    if not (INSIGHTFACE_AVAILABLE and ORT_AVAILABLE and PIL_AVAILABLE):
        print("错误：缺少必要依赖 (insightface / onnxruntime / pillow)。", file=sys.stderr)
        sys.exit(1)

    # 高分屏适配：必须在 QApplication 创建前设置（Qt6 默认已启用高 DPI 缩放，
    # 这里只补上“按整数倍取整”的平滑策略，避免缩放过程出现半像素虚边）。
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except Exception:
        pass

    app = QApplication(sys.argv)
    try:
        app.setApplicationName("视频人脸检测比对工具")
        app.setFont(_pick_ui_font())
        app.setStyleSheet(APP_QSS)  # 全局扁平化主题
    except Exception:
        pass
    win = MainWindow()
    win.show()

    # 自检测：设置环境变量 FACE_CLIP_SELFTEST=1 时，加载模型并跑一次推理，
    # 用于验证打包后的 exe 能否正常初始化 GPU/CPU 推理（不影响正常使用）。
    if os.environ.get("FACE_CLIP_SELFTEST") == "1":
        _res = "SELFTEST_FAIL"
        try:
            import numpy as np
            prov, use_gpu, gpu_name, note = detect_runtime()
            win._append_log(f"SELFTEST: 推理后端={prov}，GPU={use_gpu}（{gpu_name}）")
            eng = Engine(prov, use_gpu, win._append_log)
            eng.load_model()
            dummy = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)
            faces = eng.app.get(dummy)
            win._append_log(f"SELFTEST OK：模型加载成功，空图检出人脸数={len(faces)}")
            _res = f"SELFTEST_OK providers={prov} use_gpu={use_gpu} gpu={gpu_name}"
        except Exception as e:
            _res = f"SELFTEST_FAIL {repr(e)}"
            traceback.print_exc()
        try:
            with open(os.path.join(os.getcwd(), "selftest_result.txt"), "w",
                      encoding="utf-8") as _f:
                _f.write(_res + "\n")
        except Exception:
            pass
        sys.exit(0)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
