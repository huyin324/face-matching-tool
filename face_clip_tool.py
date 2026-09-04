#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
人脸检测视频截取工具 (Face Clip Tool)
=====================================
GUI 工具：上传一张目标人脸图片 + 一段视频，自动在视频中检索该人脸出现的所有时刻，
每次出现截取比中帧 + 前后各 10 秒（共 20 秒）视频片段，并对目标人脸用红框标注截图。
自动检测 NVIDIA GPU / CUDA：可用时启用 GPU 加速，否则自动降级为 CPU 运行。

依赖：insightface, onnxruntime-gpu(或 onnxruntime), opencv-python-headless, pillow, numpy
"""

import os
import sys
import time
import math
import queue
import threading
import traceback
from collections import deque
from datetime import datetime

import cv2
import numpy as np

# ---------- 第三方库可用性检查（优雅降级提示） ----------
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    import tkinter.font as tkfont
    TK_AVAILABLE = True
except Exception as e:  # pragma: no cover
    TK_AVAILABLE = False
    _TK_ERR = e

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
    from PIL import ImageTk
    PILTK_AVAILABLE = True
except Exception:
    # ImageTk 需要 tkinter；无界面/精简环境可能缺失，仅影响 GUI 实时预览。
    PILTK_AVAILABLE = False


# ---------- 全局异常兜底：避免“一闪而过”看不到错误 ----------
import traceback as _tb

def _global_excepthook(exc_type, exc_val, exc_tb):
    """未捕获异常时：写入 crash.log 并弹窗提示，确保错误可见而非静默退出。"""
    msg = "".join(_tb.format_exception(exc_type, exc_val, exc_tb))
    try:
        with open(os.path.join(os.getcwd(), "crash.log"), "w", encoding="utf-8") as _f:
            _f.write(msg)
    except Exception:
        pass
    # 尽量用 GUI 弹窗展示；若 tkinter 不可用则忽略（此时 stderr 仍有 traceback）
    try:
        import tkinter.messagebox as _mb
        _mb.showerror("程序异常（已写入 crash.log）", msg)
    except Exception:
        pass

sys.excepthook = _global_excepthook


# ============================================================
# 环境与 GPU 检测
# ============================================================
def detect_runtime():
    """
    检测运行环境，返回 (providers, use_gpu, gpu_name, note)
    providers 用于传给 insightface / onnxruntime。
    """
    gpu_name = ""
    note = []
    use_gpu = False
    providers = ["CPUExecutionProvider"]

    # 1) 通过 onnxruntime 看是否注册了 CUDA 执行提供者
    if ORT_AVAILABLE:
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            use_gpu = True
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # 2) 通过 nvidia-smi 确认显卡型号（仅用于日志展示）
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8,
        )
        if out.returncode == 0 and out.stdout.strip():
            gpu_name = out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass

    if use_gpu:
        if gpu_name:
            note.append(f"检测到 NVIDIA 显卡：{gpu_name}，已启用 CUDA GPU 加速")
        else:
            note.append("检测到 CUDA 执行提供者，已启用 GPU 加速")
    else:
        if gpu_name:
            note.append(f"检测到显卡 {gpu_name}，但 CUDA 执行提供者不可用，已降级为 CPU 运行")
        else:
            note.append("未检测到可用的 CUDA 环境，已降级为 CPU 运行")

    return providers, use_gpu, gpu_name, "；".join(note)


# ============================================================
# 工具函数
# ============================================================
def fmt_time(sec):
    """秒 -> mm:ss.s 格式"""
    if sec is None or sec < 0:
        return "00:00.0"
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:04.1f}"


def cosine(a, b):
    """余弦相似度（自动归一化）"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a / na, b / nb))


def fourcc_for_ext(ext):
    ext = (ext or "").lower()
    return {
        ".mp4": "mp4v",
        ".mov": "mp4v",
        ".avi": "XVID",
        ".mkv": "mp4v",
        ".wmv": "wmv2",
    }.get(ext, "mp4v")


def sim_to_label(sim):
    """相似度(0~1) -> 标注文字，如 '98.50%'"""
    try:
        return f"{float(sim) * 100:.2f}%"
    except Exception:
        return "Target"


def frame_contrast_pct(frame):
    """计算帧的对比度（亮度标准差），归一化到 0~100%。

    用于“对比度阈值”门控：低于阈值的过暗/过曝/模糊帧可跳过检测以省算力。
    8-bit 灰度标准差范围约 0~127.5，按 /128*100 归一化。
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = frame
    if gray.ndim != 2:
        gray = np.mean(gray, axis=2).astype(np.uint8)
    std = float(np.std(gray))
    return min(100.0, std / 128.0 * 100.0)


def draw_red_box(frame, bbox, thickness=None, label=None):
    """在 frame 上为 bbox 画红框，返回副本（不修改原图）。

    label: 框上方文字；为 None 时回退显示 'Target'（兼容无相似度场景）。
           通常传入 sim_to_label(sim) 以显示相似度百分比。
    """
    out = frame.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
    h, w = out.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if thickness is None:
        thickness = max(2, int(min(w, h) / 220))
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), thickness)
    # 顶部标注：相似度百分比（优先）或默认 Target
    if label is None:
        label = "Target"
    (lw, lh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(out, (x1, max(0, y1 - lh - 6)), (x1 + lw + 6, y1), (0, 0, 255), -1)
    cv2.putText(out, label, (x1 + 3, max(lh + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def imread_utf8(path):
    """
    以 UTF-8 安全方式读取图像为 BGR numpy 数组。
    opencv 的 imread 在部分 Windows 构建下不支持含中文/非 ASCII 的路径，
    因此改用 Pillow 读取（Pillow 经临时文件或直接处理 Unicode 路径更可靠），
    再转换回 BGR。读取失败返回 None。
    """
    try:
        from PIL import Image
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        # 兜底：cv2 直接读（ASCII 路径场景）
        return cv2.imread(path)


def imwrite_utf8(frame_bgr, path):
    """
    保存 BGR 图像到 path，自动创建中文输出目录。

    修复“写入图片失败”：旧实现不创建父目录，且依赖 os.replace（部分环境被拦截）。
    改用 Pillow 直接保存（原生支持 Unicode/中文路径），并在保存前创建父目录。
    返回是否成功。
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
    """
    用 OpenCV 截取 [start_t, end_t] 区间的视频片段，保存为 out_path。
    采用流复制式逐帧重编码（无需外部 ffmpeg）。
    """
    cap = None
    writer = None
    try:
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
            # 回退到 mp4v
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
        logger(f"  片段写入完成：{os.path.basename(out_path)}（{written} 帧，{fmt_time(end_t-start_t)}）")
        return True
    except Exception as e:
        logger(f"  片段截取失败：{e}")
        # 清理半成品
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


# ============================================================
# 检测引擎
# ============================================================
class Engine:
    """封装人脸检测 + 比对逻辑（可在后台线程调用）"""

    def __init__(self, providers, use_gpu, logger):
        self.providers = providers
        self.use_gpu = use_gpu
        self.logger = logger
        self.app = None
        self.target_emb = None
        self.target_bbox = None

    def load_model(self):
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError(
                "insightface 未安装或导入失败，无法运行人脸检测。"
                "请通过启动脚本安装依赖后重试。"
            )
        self.logger("正在加载人脸检测模型 (buffalo_l) ...")
        ctx_id = 0 if self.use_gpu else -1
        # 兼容不同 insightface 版本对 providers 参数的位置
        try:
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=self.providers,
                det_size=(640, 640),
            )
        except TypeError:
            self.app = FaceAnalysis(name="buffalo_l", det_size=(640, 640))
        try:
            self.app.prepare(ctx_id=ctx_id, providers=self.providers)
        except TypeError:
            self.app.prepare(ctx_id=ctx_id)
        self.logger("模型加载完成。")

    def set_target(self, img_path):
        img = imread_utf8(img_path)
        if img is None:
            raise ValueError(f"无法读取目标人脸图片：{img_path}")
        faces = self.app.get(img)
        if not faces:
            raise ValueError("目标图片中未检测到人脸，请更换一张清晰的正脸照片。")
        # 选取最大的人脸作为目标
        faces.sort(
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        self.target_emb = faces[0].embedding
        self.target_bbox = faces[0].bbox
        self.logger(f"已锁定目标人脸（检测到 {len(faces)} 张脸，选用最大的一张）。")
        return faces[0]

    def detect_faces(self, frame):
        """返回 (faces, best_sim)；best_sim 为与目标的最大余弦相似度；faces 为匹配到的人脸列表"""
        faces = self.app.get(frame)
        best_sim = -1.0
        matched = []
        for f in faces:
            sim = cosine(f.embedding, self.target_emb)
            if sim > best_sim:
                best_sim = sim
            if sim >= self.threshold:
                matched.append((f, sim))
        return faces, best_sim, matched


# ============================================================
# GUI 应用
# ============================================================
class FaceClipGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("人脸检测视频截取工具")
        self.root.geometry("1080x760")
        try:
            self.root.tk.call("tk", "scaling", 1.0)
        except Exception:
            pass

        # 运行态
        self.engine = None
        self.thread = None
        self.stop_event = threading.Event()
        self.q = queue.Queue(maxsize=200)
        self.running = False
        self._photo = None          # 防止 PhotoImage 被 GC
        self._target_photo = None
        self.target_face_img = None  # PIL，用于左侧缩略图

        # 参数
        self.threshold = tk.DoubleVar(value=0.40)       # 相似度阈值
        self.sample_step = tk.IntVar(value=2)           # 每 N 帧处理一次
        self.pad_sec = tk.DoubleVar(value=10.0)         # 新增功能5：比中帧前后各多少秒
        self.frame_gap_sec = tk.DoubleVar(value=3.0)    # 新增功能6：红框截图最小间隔秒
        self.contrast_threshold = tk.DoubleVar(value=0.0)  # 新增功能3：对比度阈值(0~100%)
        self.use_rtsp = tk.BooleanVar(value=False)      # 新增功能1：是否使用 RTSP 流
        self.rtsp_url = tk.StringVar()                  # 新增功能1：RTSP 地址
        self.face_path = tk.StringVar()
        self.video_paths = []                           # 新增功能4：多视频队列（文件列表）
        self.out_dir = tk.StringVar(value=os.path.join(os.getcwd(), "输出结果"))

        self._build_ui()

        # 启动日志消费者
        self.root.after(60, self._poll)
        # 初始环境检测日志
        providers, use_gpu, gpu_name, note = detect_runtime()
        self.runtime_providers = providers
        self.runtime_use_gpu = use_gpu
        self.runtime_gpu_name = gpu_name
        self.log("=" * 46)
        self.log("人脸检测视频截取工具 启动")
        self.log(f"运行环境：{'GPU (CUDA)' if use_gpu else 'CPU'}")
        self.log(note)
        self.log("=" * 46)

    # ---------------- UI ----------------
    def _build_ui(self):
        # 顶部：上传区
        top = ttk.LabelFrame(self.root, text="① 上传区域", padding=8)
        top.pack(fill="x", padx=10, pady=(10, 4))

        ttk.Label(top, text="目标人脸：").grid(row=0, column=0, sticky="e")
        ttk.Entry(top, textvariable=self.face_path, width=60).grid(row=0, column=1, padx=4)
        ttk.Button(top, text="浏览...", command=self._pick_face).grid(row=0, column=2)

        ttk.Label(top, text="视频文件：").grid(row=1, column=0, sticky="ne")
        vid_frame = ttk.Frame(top)
        vid_frame.grid(row=1, column=1, columnspan=2, sticky="we", padx=4)
        self.video_listbox = tk.Listbox(vid_frame, height=3, width=62)
        self.video_listbox.pack(side="left", fill="x", expand=True)
        vid_btn = ttk.Frame(vid_frame)
        vid_btn.pack(side="left", padx=(4, 0))
        ttk.Button(vid_btn, text="添加...", command=self._add_videos).pack(fill="x")
        ttk.Button(vid_btn, text="移除", command=self._remove_video).pack(fill="x", pady=(2, 0))
        ttk.Button(vid_btn, text="清除", command=self._clear_videos).pack(fill="x", pady=(2, 0))

        # RTSP 视频流输入（新增功能1）
        rtsp_frame = ttk.Frame(top)
        rtsp_frame.grid(row=2, column=0, columnspan=3, sticky="we", pady=(2, 0))
        ttk.Checkbutton(rtsp_frame, text="使用 RTSP 视频流",
                        variable=self.use_rtsp,
                        command=self._on_rtsp_toggle).pack(side="left")
        ttk.Entry(rtsp_frame, textvariable=self.rtsp_url,
                  width=70).pack(side="left", padx=4, fill="x", expand=True)
        self._on_rtsp_toggle()  # 初始化控件可用状态

        ttk.Label(top, text="输出目录：").grid(row=3, column=0, sticky="e")
        ttk.Entry(top, textvariable=self.out_dir, width=60).grid(row=3, column=1, padx=4)
        ttk.Button(top, text="选择...", command=self._pick_out).grid(row=3, column=2)

        opt = ttk.Frame(top)
        opt.grid(row=4, column=0, columnspan=3, pady=(6, 0))
        ttk.Label(opt, text="相似度阈值：").pack(side="left")
        ttk.Scale(opt, from_=0.20, to_=0.75,
                  variable=self.threshold, length=140,
                  command=lambda v: self._th_label.config(text=f"{float(v):.2f}")).pack(side="left", padx=4)
        self._th_label = ttk.Label(opt, text=f"{self.threshold.get():.2f}")
        self._th_label.pack(side="left")
        ttk.Label(opt, text="  对比度阈值%：").pack(side="left")
        ttk.Spinbox(opt, from_=0, to=100, increment=5, width=4,
                    textvariable=self.contrast_threshold).pack(side="left", padx=2)
        ttk.Label(opt, text="  处理间隔(帧)：").pack(side="left")
        ttk.Spinbox(opt, from_=1, to=10, width=4, textvariable=self.sample_step).pack(side="left", padx=4)
        ttk.Label(opt, text="(越大越快)").pack(side="left")

        opt2 = ttk.Frame(top)
        opt2.grid(row=5, column=0, columnspan=3, pady=(4, 0))
        ttk.Label(opt2, text="片段前后秒数：").pack(side="left")
        ttk.Spinbox(opt2, from_=0, to=120, increment=1, width=5,
                    textvariable=self.pad_sec).pack(side="left", padx=2)
        ttk.Label(opt2, text="  截图冗余间隔(秒)：").pack(side="left")
        ttk.Spinbox(opt2, from_=0.5, to=30, increment=0.5, width=5,
                    textvariable=self.frame_gap_sec).pack(side="left", padx=2)

        # 中部：左状态 + 右实时画面
        mid = ttk.Frame(self.root)
        mid.pack(fill="both", expand=True, padx=10, pady=4)

        # 左侧
        left = ttk.LabelFrame(mid, text="③ 分析状态", padding=8)
        left.pack(side="left", fill="y", padx=(0, 6))

        self.target_canvas = tk.Canvas(left, width=200, height=200, bg="#f0f0f0",
                                       relief="sunken", borderwidth=1)
        self.target_canvas.pack(pady=(0, 6))
        ttk.Label(left, text="目标人脸缩略图").pack()

        self.status_var = tk.StringVar(value="待机")
        self.sim_var = tk.StringVar(value="当前相似度：--")
        self.match_var = tk.StringVar(value="比对结果：未匹配")
        self.episode_var = tk.StringVar(value="出场片段：0")
        self.frame_var = tk.StringVar(value="红框截图：0")

        ttk.Label(left, textvariable=self.status_var, foreground="#1565c0",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(6, 0))
        ttk.Label(left, textvariable=self.sim_var).pack(anchor="w")
        ttk.Label(left, textvariable=self.match_var).pack(anchor="w")
        ttk.Label(left, textvariable=self.episode_var).pack(anchor="w")
        ttk.Label(left, textvariable=self.frame_var).pack(anchor="w")

        # 进度条
        prog = ttk.LabelFrame(left, text="进度", padding=6)
        prog.pack(fill="x", pady=(8, 0))
        self.progress = ttk.Progressbar(prog, length=210, mode="determinate")
        self.progress.pack(fill="x")
        self.pct_var = tk.StringVar(value="0%")
        ttk.Label(prog, textvariable=self.pct_var).pack(anchor="e")

        # 右侧：实时画面
        right = ttk.LabelFrame(mid, text="② 实时逐帧分析画面", padding=8)
        right.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(right, width=560, height=315, bg="#101010",
                                relief="sunken", borderwidth=1)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.create_text(280, 157, text="等待开始分析...", fill="#888",
                                font=("Segoe UI", 12))

        # 底部：日志 + 控制
        bottom = ttk.Frame(self.root)
        bottom.pack(fill="x", padx=10, pady=(4, 10))

        ctrl = ttk.Frame(bottom)
        ctrl.pack(fill="x")
        self.btn_start = ttk.Button(ctrl, text="开始分析", command=self._start)
        self.btn_start.pack(side="left", padx=(0, 6))
        self.btn_stop = ttk.Button(ctrl, text="停止", command=self._stop, state="disabled")
        self.btn_stop.pack(side="left")
        self.btn_open = ttk.Button(ctrl, text="打开输出目录", command=self._open_out)
        self.btn_open.pack(side="right")

        logf = ttk.LabelFrame(bottom, text="④ 运行日志", padding=6)
        logf.pack(fill="both", expand=False, pady=(6, 0))
        self.log_text = scrolledtext.ScrolledText(
            logf, height=12, state="disabled",
            font=("Consolas", 9), wrap="word",
        )
        self.log_text.pack(fill="both", expand=True)

    # ---------------- 选择文件 ----------------
    def _pick_face(self):
        p = filedialog.askopenfilename(
            title="选择目标人脸图片",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.webp *.gif"), ("全部", "*.*")])
        if p:
            self.face_path.set(p)
            self._show_target_thumb(p)

    def _add_videos(self):
        """新增功能4：一次性选择多个视频文件加入队列"""
        ps = filedialog.askopenfilenames(
            title="选择视频文件（可多选）",
            filetypes=[("视频", "*.mp4 *.mov *.avi *.mkv *.wmv *.flv *.m4v"), ("全部", "*.*")])
        added = False
        for p in ps:
            if p and p not in self.video_paths:
                self.video_paths.append(p)
                added = True
        if added:
            self._refresh_video_list()

    def _remove_video(self):
        sel = self.video_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.video_paths):
            self.video_paths.pop(idx)
            self._refresh_video_list()

    def _clear_videos(self):
        self.video_paths = []
        self._refresh_video_list()

    def _refresh_video_list(self):
        try:
            self.video_listbox.delete(0, "end")
            for p in self.video_paths:
                self.video_listbox.insert("end", p)
        except Exception:
            pass

    def _on_rtsp_toggle(self):
        """RTSP 勾选状态切换：启用时禁用本地视频列表，反之相反"""
        use = bool(self.use_rtsp.get())
        vstate = "disabled" if use else "normal"
        try:
            self.video_listbox.configure(state=vstate)
            vid_frame = self.video_listbox.master
            for child in vid_frame.winfo_children():
                if isinstance(child, ttk.Frame):  # vid_btn 容器
                    for b in child.winfo_children():
                        if isinstance(b, ttk.Button):
                            b.configure(state=vstate)
        except Exception:
            pass

    def _pick_out(self):
        d = filedialog.askdirectory(title="选择输出目录")
        if d:
            self.out_dir.set(d)

    def _show_target_thumb(self, path):
        try:
            img = imread_utf8(path)
            if img is None:
                return
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img)
            pil.thumbnail((200, 200))
            self.target_face_img = pil
            if PILTK_AVAILABLE:
                self._target_photo = ImageTk.PhotoImage(pil)
                self.target_canvas.delete("all")
                self.target_canvas.create_image(100, 100, image=self._target_photo)
        except Exception as e:
            self.log(f"[警告] 目标缩略图预览失败：{e}")

    def _open_out(self):
        d = self.out_dir.get()
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        try:
            os.startfile(d) if os.name == "nt" else os.system(f'open "{d}"')
        except Exception as e:
            self.log(f"[警告] 打开输出目录失败：{e}")

    # ---------------- 日志/队列 ----------------
    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        try:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", line)
            self.log_text.configure(state="disabled")
            self.log_text.see("end")
        except Exception:
            pass

    def _enqueue(self, kind, payload):
        try:
            self.q.put_nowait((kind, payload))
        except queue.Full:
            pass  # 丢弃积压，保证不卡死

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "progress":
                    self.progress["value"] = payload
                    self.pct_var.set(f"{int(payload)}%")
                elif kind == "progress_mode":
                    # 文件：determinate（有总帧数）；RTSP：indeterminate（未知长度）
                    if payload == "indeterminate":
                        try:
                            self.progress.stop()
                            self.progress.config(mode="indeterminate")
                            self.progress.start(25)
                            self.pct_var.set("实时")
                        except Exception:
                            pass
                    else:
                        try:
                            self.progress.stop()
                            self.progress.config(mode="determinate")
                            self.progress["value"] = 0
                            self.pct_var.set("0%")
                        except Exception:
                            pass
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "source":
                    self.status_var.set(payload)
                elif kind == "sim":
                    self.sim_var.set(f"当前相似度：{payload:.3f}")
                elif kind == "match":
                    tag = "匹配 ✓" if payload else "未匹配"
                    self.match_var.set(f"比对结果：{tag}")
                elif kind == "frame_img":
                    self._show_frame(payload)
                elif kind == "episode":
                    self.episode_var.set(f"出场片段：{payload}")
                elif kind == "framecount":
                    self.frame_var.set(f"红框截图：{payload}")
                elif kind == "done":
                    self._on_done(payload)
                elif kind == "error":
                    self._on_error(payload)
        except queue.Empty:
            pass
        self.root.after(60, self._poll)

    def _show_frame(self, pil_img):
        try:
            if not PILTK_AVAILABLE:
                return
            self._photo = ImageTk.PhotoImage(pil_img)
            w = self.canvas.winfo_width() or 560
            h = self.canvas.winfo_height() or 315
            self.canvas.delete("all")
            self.canvas.create_image(w // 2, h // 2, image=self._photo)
        except Exception:
            pass

    # ---------------- 开始/停止 ----------------
    def _start(self):
        if self.running:
            return
        face = self.face_path.get().strip()
        out = self.out_dir.get().strip()
        if not face or not os.path.isfile(face):
            messagebox.showerror("错误", "请先选择有效的目标人脸图片。")
            return

        # 收集视频源：RTSP 流（新增功能1）或 本地视频队列（新增功能4）
        if self.use_rtsp.get():
            url = self.rtsp_url.get().strip()
            if not url:
                messagebox.showerror("错误", "请填写 RTSP 视频流地址。")
                return
            sources = [("rtsp", url)]
        else:
            if not self.video_paths:
                messagebox.showerror("错误", "请先添加至少一个视频文件。")
                return
            sources = [("file", p) for p in self.video_paths]

        os.makedirs(out, exist_ok=True)

        # 读取可调参数（新增功能2/3/5/6）
        threshold = self.threshold.get()
        sample_step = max(1, int(self.sample_step.get()))
        pad = max(0.0, float(self.pad_sec.get()))
        frame_gap = max(0.5, float(self.frame_gap_sec.get()))
        contrast = max(0.0, float(self.contrast_threshold.get()))

        self.stop_event.clear()
        self.running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.progress["value"] = 0
        self.pct_var.set("0%")

        self.thread = threading.Thread(
            target=self._worker,
            args=(face, sources, out, threshold, sample_step, pad, frame_gap,
                  contrast, self.runtime_providers, self.runtime_use_gpu),
            daemon=True,
        )
        self.thread.start()

    def _stop(self):
        if not self.running:
            return
        self.stop_event.set()
        self.log("已请求停止，正在安全退出当前分析...")

    def _overall_pct(self, si, total_src, frac):
        frac = max(0.0, min(1.0, frac))
        return min(100, int(((si - 1) + frac) / total_src * 100))

    def _on_done(self, summary):
        self.running = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status_var.set("完成")
        pad = summary.get("pad", 10.0)
        frame_gap = summary.get("frame_gap", 3.0)
        total_src = summary.get("total_src", 1)
        self.log("=" * 46)
        self.log("分析完成！")
        self.log(f"视频源：{total_src} 个")
        self.log(f"出场片段（前后各 {pad:.0f} 秒视频）：{summary.get('episodes', 0)} 个")
        self.log(f"红框截图（每 {frame_gap:.1f} 秒）：{summary.get('frames', 0)} 张")
        self.log(f"结果保存于：{summary.get('out', '')}")
        self.log("=" * 46)

    def _on_error(self, msg):
        self.running = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status_var.set("出错")
        self.log("✗ 发生错误：" + msg)
        messagebox.showerror("运行错误", msg)

    # ---------------- 工作线程（多视频/RTSP 调度） ----------------
    def _worker(self, face, sources, out, threshold, sample_step,
                pad, frame_gap, contrast, providers, use_gpu):
        eng = Engine(providers, use_gpu, lambda m: self._enqueue("log", m))
        all_episodes = 0
        all_frames = 0
        total_src = len(sources)
        try:
            eng.load_model()
            eng.threshold = threshold
            eng.set_target(face)

            for si, (kind, src) in enumerate(sources, start=1):
                if self.stop_event.is_set():
                    break
                label = src if kind == "file" else "RTSP 视频流"
                self._enqueue("source", f"分析 {si}/{total_src}：{label}")
                if kind == "rtsp":
                    eps, frs = self._process_rtsp(
                        eng, src, out, si, total_src, pad, frame_gap,
                        contrast, sample_step)
                else:
                    eps, frs = self._process_file(
                        eng, src, out, si, total_src, pad, frame_gap,
                        contrast, sample_step)
                all_episodes += eps
                all_frames += frs
                if self.stop_event.is_set():
                    break

            self._enqueue("done", {"episodes": all_episodes, "frames": all_frames,
                                   "out": out, "pad": pad, "frame_gap": frame_gap,
                                   "total_src": total_src})
        except Exception as e:
            tb = traceback.format_exc()
            self._enqueue("log", tb)
            self._enqueue("error", str(e))

    # ---------------- 本地视频文件处理（新增功能4 / 5 / 6） ----------------
    def _process_file(self, eng, video, out, si, total_src, pad, frame_gap,
                      contrast, sample_step):
        self._enqueue("progress_mode", "determinate")
        cap = None
        try:
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频：{video}（格式不支持或文件损坏）")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            if fps <= 0:
                fps = 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = (total / fps) if fps > 0 and total > 0 else 0
            if duration <= 0:
                # 回退：逐帧统计（带上限保护，避免卡死）
                cnt = 0
                while True:
                    if self.stop_event.is_set():
                        break
                    ret, _ = cap.read()
                    if not ret:
                        break
                    cnt += 1
                    if cnt > 200000:
                        break
                total = cnt
                duration = total / fps
                cap.release()
                cap = cv2.VideoCapture(video)

            self._enqueue("log", f"视频信息：时长 {fmt_time(duration)}，"
                                 f"帧率 {fps:.2f} fps，总帧数 {total}")
            self._enqueue("status", "分析中...")

            # 状态机
            active_seg = None   # (start, end) 当前出场片段窗口
            last_frame_t = -999.0
            episode_idx = 0
            frame_idx = 0
            processed = 0
            f = 0

            while True:
                if self.stop_event.is_set():
                    self._enqueue("log", "用户已停止，安全退出。")
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                f += 1
                if (f - 1) % sample_step != 0:
                    continue  # 跳帧以提速

                processed += 1
                t = f / fps

                # 对比度门控（新增功能3）
                if contrast > 0:
                    cp = frame_contrast_pct(frame)
                    if cp < contrast:
                        if processed % 50 == 0:
                            self._enqueue("log",
                                f"  低对比度({cp:.1f}% < {contrast:.0f}%)跳过检测")
                        if total > 0:
                            self._enqueue("progress", self._overall_pct(si, total_src, f / total))
                        continue

                # 检测 + 比对
                try:
                    faces, best_sim, matched = eng.detect_faces(frame)
                except Exception as e:
                    self._enqueue("log", f"[警告] 第 {f} 帧检测异常：{e}")
                    continue

                self._enqueue("sim", best_sim)
                is_match = len(matched) > 0
                self._enqueue("match", is_match)

                # 实时画面（降采样后推送，标注相似度百分比）
                try:
                    disp = frame
                    dh, dw = disp.shape[:2]
                    scale = min(1.0, 540.0 / dw)
                    if scale < 1.0:
                        disp = cv2.resize(disp, (int(dw * scale), int(dh * scale)))
                    if matched:
                        for mf, msim in matched:
                            disp = draw_red_box(disp, mf.bbox, label=sim_to_label(msim))
                    disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(disp)
                    self._enqueue("frame_img", pil)
                except Exception as e:
                    self._enqueue("log", f"[警告] 画面渲染异常：{e}")

                if is_match:
                    if active_seg is None or t > active_seg[1]:
                        seg_start = max(0.0, t - pad)
                        seg_end = min(duration, t + pad)
                        episode_idx += 1
                        seg_name = (f"片段_{episode_idx:03d}_"
                                    f"出场{fmt_time(t).replace(':', '-')}.mp4")
                        seg_path = os.path.join(out, seg_name)
                        self._enqueue("log",
                                      f"▶ 第 {episode_idx} 次出场 @ {fmt_time(t)}，"
                                      f"截取 {fmt_time(seg_start)}~{fmt_time(seg_end)}")
                        extract_segment(video, seg_start, seg_end, seg_path,
                                        lambda m: self._enqueue("log", m))
                        active_seg = (seg_start, seg_end)
                        self._enqueue("episode", episode_idx)
                        # 新片段起点必存一张红框截图（标注相似度）
                        self._save_frame_clip(frame, matched[0][0].bbox,
                                              matched[0][1], t, out, episode_idx, frame_idx)
                        frame_idx += 1
                        last_frame_t = t
                        self._enqueue("framecount", frame_idx)
                    else:
                        # 连续出场：每 N 秒再截一张红框图，避免冗余（新增功能6）
                        if t - last_frame_t >= frame_gap:
                            self._save_frame_clip(frame, matched[0][0].bbox,
                                                  matched[0][1], t, out, episode_idx, frame_idx)
                            frame_idx += 1
                            last_frame_t = t
                            self._enqueue("framecount", frame_idx)

                # 进度（整体 = (已完成视频 + 当前视频进度) / 总视频数）
                if total > 0:
                    self._enqueue("progress", self._overall_pct(si, total_src, f / total))

                # 防止单帧处理过慢时 UI 完全冻结：让出一点点
                if processed % 30 == 0:
                    time.sleep(0.001)

            self._enqueue("status", "完成")
            return episode_idx, frame_idx
        except Exception as e:
            tb = traceback.format_exc()
            self._enqueue("log", tb)
            raise
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    # ---------------- RTSP 实时流处理（新增功能1） ----------------
    @staticmethod
    def _ring_frame(frame):
        """环形缓冲用的帧副本：过宽则缩放到 960 以限制内存占用"""
        h, w = frame.shape[:2]
        if w > 960:
            s = 960.0 / w
            return cv2.resize(frame, (960, int(h * s)))
        return frame.copy()

    def _process_rtsp(self, eng, url, out, si, total_src, pad, frame_gap,
                      contrast, sample_step):
        self._enqueue("progress_mode", "indeterminate")
        self._enqueue("status", "RTSP 接收中...")
        cap = None
        active_writer = None
        active_path = None
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                raise RuntimeError("无法打开 RTSP 流，请检查地址、网络与认证。")

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            if fps <= 0:
                fps = 25.0
            self._enqueue("log", f"RTSP 流已连接，帧率约 {fps:.2f} fps，"
                                 f"片段前后各 {pad:.0f}s，截图间隔 {frame_gap:.1f}s")

            pad_frames = int(pad * fps)
            ring = deque(maxlen=pad_frames + 2)  # 缓存最近 pad 秒帧，用于前垫
            w = h = 0
            fourcc = "mp4v"
            active_end_f = -1
            last_frame_t = -999.0
            last_reconnect_log = 0.0
            episode_idx = 0
            frame_idx = 0
            processed = 0
            f = 0

            def ensure_size(frame):
                nonlocal w, h, fourcc
                if w <= 0 or h <= 0:
                    h, w = frame.shape[:2]
                    fourcc = fourcc_for_ext(".mp4")

            while True:
                if self.stop_event.is_set():
                    self._enqueue("log", "用户已停止，安全退出。")
                    break
                ret, frame = cap.read()
                if not ret:
                    # 断流重连（健壮性）
                    now = time.time()
                    if now - last_reconnect_log > 3:
                        self._enqueue("log", "RTSP 读取中断，尝试重连...")
                        last_reconnect_log = now
                    reconnected = False
                    for _ in range(5):
                        if self.stop_event.is_set():
                            break
                        time.sleep(2)
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                        if cap.isOpened() and cap.read()[0]:
                            reconnected = True
                            break
                    if not reconnected:
                        self._enqueue("log", "RTSP 重连失败，结束该流分析。")
                        break
                    continue

                f += 1
                if (f - 1) % sample_step != 0:
                    # 跳过的帧仍进缓冲（保证前垫完整），但不做检测
                    ring.append((f, self._ring_frame(frame)))
                    continue

                processed += 1
                t = f / fps
                ensure_size(frame)
                ring.append((f, self._ring_frame(frame)))

                # 对比度门控（新增功能3）
                if contrast > 0:
                    cp = frame_contrast_pct(frame)
                    if cp < contrast:
                        if processed % 50 == 0:
                            self._enqueue("log",
                                f"  低对比度({cp:.1f}% < {contrast:.0f}%)跳过检测")
                        continue

                # 检测 + 比对
                try:
                    faces, best_sim, matched = eng.detect_faces(frame)
                except Exception as e:
                    self._enqueue("log", f"[警告] 第 {f} 帧检测异常：{e}")
                    continue

                self._enqueue("sim", best_sim)
                is_match = len(matched) > 0
                self._enqueue("match", is_match)

                # 实时画面（标注相似度百分比）
                try:
                    disp = frame
                    dh, dw = disp.shape[:2]
                    scale = min(1.0, 540.0 / dw)
                    if scale < 1.0:
                        disp = cv2.resize(disp, (int(dw * scale), int(dh * scale)))
                    if matched:
                        for mf, msim in matched:
                            disp = draw_red_box(disp, mf.bbox, label=sim_to_label(msim))
                    disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(disp)
                    self._enqueue("frame_img", pil)
                except Exception as e:
                    self._enqueue("log", f"[警告] 画面渲染异常：{e}")

                # 片段录制：维护当前出场窗口的 VideoWriter（实时流不可 seek）
                if is_match:
                    if active_writer is None or f > active_end_f:
                        episode_idx += 1
                        seg_name = (f"片段_{episode_idx:03d}_"
                                    f"出场{fmt_time(t).replace(':', '-')}.mp4")
                        active_path = os.path.join(out, seg_name)
                        self._enqueue("log",
                                      f"▶ 第 {episode_idx} 次出场 @ {fmt_time(t)}，"
                                      f"录制前后各 {pad:.0f}s 片段")
                        try:
                            fourcc_vw = cv2.VideoWriter_fourcc(*fourcc)
                            active_writer = cv2.VideoWriter(active_path, fourcc_vw, fps, (w, h))
                            if not active_writer.isOpened():
                                fourcc_vw = cv2.VideoWriter_fourcc(*"mp4v")
                                active_writer = cv2.VideoWriter(active_path, fourcc_vw, fps, (w, h))
                            # 写入环形缓冲中的前 pad 秒帧（不含当前帧，当前帧随后写）
                            for bf, bframe in ring:
                                if bf >= f - pad_frames and bf < f:
                                    active_writer.write(bframe)
                        except Exception as e:
                            self._enqueue("log", f"  [警告] 片段写入器创建失败：{e}")
                            active_writer = None
                        active_end_f = f + pad_frames
                        self._enqueue("episode", episode_idx)
                        self._save_frame_clip(frame, matched[0][0].bbox,
                                              matched[0][1], t, out, episode_idx, frame_idx)
                        frame_idx += 1
                        last_frame_t = t
                        self._enqueue("framecount", frame_idx)
                    else:
                        if t - last_frame_t >= frame_gap:
                            self._save_frame_clip(frame, matched[0][0].bbox,
                                                  matched[0][1], t, out, episode_idx, frame_idx)
                            frame_idx += 1
                            last_frame_t = t
                            self._enqueue("framecount", frame_idx)

                # 当前片段持续推进写入实时帧（直到窗口结束）
                if active_writer is not None:
                    try:
                        active_writer.write(frame)
                    except Exception:
                        pass
                    if f >= active_end_f:
                        try:
                            active_writer.release()
                        except Exception:
                            pass
                        active_writer = None
                        self._enqueue("log", f"  片段写入完成：{os.path.basename(active_path)}")

                if processed % 30 == 0:
                    time.sleep(0.001)

            self._enqueue("status", "完成")
            return episode_idx, frame_idx
        except Exception:
            tb = traceback.format_exc()
            self._enqueue("log", tb)
            raise
        finally:
            if active_writer is not None:
                try:
                    active_writer.release()
                except Exception:
                    pass
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def _save_frame_clip(self, frame, bbox, sim, t, out, episode_idx, frame_idx):
        try:
            name = (f"匹配帧_{frame_idx+1:03d}_"
                    f"{fmt_time(t).replace(':', '-')}.png")
            path = os.path.join(out, name)
            # 修改功能1：红框上方标注相似度百分比（保留2位小数）
            marked = draw_red_box(frame, bbox, label=sim_to_label(sim))
            ok = imwrite_utf8(marked, path)
            if not ok:
                raise RuntimeError("写入图片失败")
            self._enqueue("log", f"  ✔ 红框截图：{name}（相似度 {sim_to_label(sim)}）")
        except Exception as e:
            self._enqueue("log", f"  [警告] 截图失败：{e}")


# ============================================================
# 入口
# ============================================================
def main():
    if not TK_AVAILABLE:
        print("错误：当前环境缺少 tkinter，无法启动 GUI。", file=sys.stderr)
        print(str(_TK_ERR), file=sys.stderr)
        sys.exit(1)
    if not (INSIGHTFACE_AVAILABLE and ORT_AVAILABLE and PIL_AVAILABLE):
        print("错误：缺少必要依赖 (insightface / onnxruntime / pillow)。", file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    # 关键：Tk 回调（按钮/after 计时器等）里抛出的异常不会走 sys.excepthook，
    # 默认只打印到 stderr 并继续。这里重定向到同一套“crash.log + 弹窗”逻辑，
    # 否则这类异常会让窗口表现成“一闪而过”却看不到任何错误。
    root.report_callback_exception = _global_excepthook
    try:
        # 尝试使用系统主题
        style = ttk.Style()
        try:
            style.theme_use("vista")  # Windows
        except Exception:
            pass
    except Exception:
        pass
    app = FaceClipGUI(root)
    try:
        root.mainloop()
    except Exception:
        # 兜底：mainloop 内部意外抛出的异常也记录到 crash.log + 弹窗
        _global_excepthook(*sys.exc_info())
        raise


if __name__ == "__main__":
    main()
