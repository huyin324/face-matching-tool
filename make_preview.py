# -*- coding: utf-8 -*-
"""离屏渲染主界面为 ui_preview.png（README 用图），无需真实显示器。

用法（在项目目录下）：
    set QT_QPA_PLATFORM=offscreen
    set QT_QPA_FONTDIR=C:/Windows/Fonts     # 关键：否则中文会渲染成方块
    .\\venv\\Scripts\\python.exe make_preview.py

说明：
  * 窗口内容用仿真数据填充（假视频路径 / 假日志 / 从 测试视频.mp4 取一帧当预览画面）；
  * 目标人脸的预览位置刻意使用**合成剪影占位图**，不加载 目标人脸.png ——
    预览图会提交进公开仓库，不能带上真实人像（仓库已按隐私策略排除该照片）；
  * 输出的 ui_preview.png 提交进仓库，README 的「三、界面速览」直接引用它。
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import face_clip_qt as F
from PyQt6 import QtWidgets

app = QtWidgets.QApplication(sys.argv)
app.setFont(F._pick_ui_font())
app.setStyleSheet(F.APP_QSS)

w = F.MainWindow()
w.resize(1340, 1080)
w.show()
for _ in range(6):
    app.processEvents()

# ---- 灌入仿真数据，让截图接近真实使用状态 ----
w.file_list = [r"D:\素材\监控_门口_20260916.mp4",
               r"D:\素材\监控_大堂_20260916.mp4"]
w.video_list.clear()
for p in w.file_list:
    w.video_list.addItem(p)
w.out_dir.setText(r"D:\素材\输出结果")

# 注意：预览图会进公开仓库，因此目标人脸用「合成剪影占位图」，
# 不能直接用 目标人脸.png（真实人像，仓库已按隐私策略排除）。
def make_placeholder_face(w=604, h=905):
    """生成一张中性的「人像剪影」占位图（灰底 + 深灰头肩轮廓），不含任何真人信息。"""
    img = np.full((h, w, 3), 238, dtype=np.uint8)
    img[:, :] = (240, 241, 244)
    cx, cy = w // 2, int(h * 0.36)
    cv2.circle(img, (cx, cy), int(w * 0.20), (198, 202, 210), -1)
    cv2.ellipse(img, (cx, int(h * 0.92)), (int(w * 0.36), int(h * 0.34)),
                0, 180, 360, (198, 202, 210), -1)
    return img


_ph = make_placeholder_face()
_ph = np.ascontiguousarray(_ph)
_qimg = F.QImage(_ph.data, _ph.shape[1], _ph.shape[0], 3 * _ph.shape[1],
                 F.QImage.Format.Format_BGR888).copy()
w.face_path = ""
w.face_path_label.setText("目标人脸_示例.png")
w.face_path_label.setToolTip("（预览图使用占位剪影，实际使用时会显示你选择的照片）")
w.face_thumb.set_source(F.QPixmap.fromImage(_qimg))

w.rb_fullspeed.setChecked(True)
app.processEvents()
w.progress.setValue(62)
w.fps_label.setText("分析帧率：46.3 帧/s")
w._set_status("全速分析中...")
w.sim_val.setText("当前相似度：96.83%")
w.match_val.setText("比对结果：匹配 ✓")
w.facecount_val.setText("实时人脸数：3")
w.episode_val.setText("出场片段：2")
w.shot_val.setText("截图：5")

for line in [
    "-" * 40,
    "模式：人脸识别比对  源：本地视频  分析：全速（不按播放速度）  "
    "质量阈值：70  相似度阈值：70%  关联视频：关",
    "正在加载人脸检测模型 (buffalo_l) ...",
    "模型加载完成（检测输入尺寸 (640, 640)，GPU 推理，模型 3 个：detection, "
    "landmark_2d_106, recognition）。",
    "目标人脸已加载（检测到 1 张人脸，取最大的一张）。",
    "分析 1/2：D:\\素材\\监控_门口_20260916.mp4",
    "分析模式：全速分析（不按播放速度，全算力逐帧比对）",
    "▶ 第 1 次出场 @ 00:12，截取 00:02~00:22",
    "  ✔ 红框截图：匹配帧_001_00-12.png",
    "▶ 第 2 次出场 @ 00:47，截取 00:37~00:57",
    "  ✔ 红框截图：匹配帧_002_00-47.png",
]:
    w._append_log(line)

# ---- 放一帧真实画面到预览区，并叠一个红框 ----
cap = cv2.VideoCapture(os.path.join(HERE, "测试视频.mp4"))
fr = None
for _ in range(30):
    ok, f = cap.read()
    if not ok:
        break
    fr = f
cap.release()
if fr is not None:
    tw, th = w.video_label.width(), w.video_label.height()
    h, w0 = fr.shape[:2]
    s = min(tw / float(w0), th / float(h))
    disp = cv2.resize(fr, (max(1, int(w0 * s)), max(1, int(h * s))),
                      interpolation=cv2.INTER_AREA)
    for (x1, y1, x2, y2), label, col in (
            ((int(0.30 * disp.shape[1]), int(0.18 * disp.shape[0]),
              int(0.52 * disp.shape[1]), int(0.72 * disp.shape[0])),
             "96.83%", (0, 0, 255)),
            ((int(0.62 * disp.shape[1]), int(0.25 * disp.shape[0]),
              int(0.78 * disp.shape[1]), int(0.68 * disp.shape[0])),
             "72.41%", (0, 200, 0))):
        F.draw_box_inplace(disp, (x1, y1, x2, y2), label=label, color=col)
    w._frame = None
    w._tracks = {}
    w._set_pixmap(disp)

for _ in range(8):
    app.processEvents()

out = os.path.join(HERE, "ui_preview.png")
w.grab().save(out)
print("saved:", out, "size:", w.width(), "x", w.height())
