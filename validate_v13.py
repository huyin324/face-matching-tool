# -*- coding: utf-8 -*-
"""V1.3 回归自测：PyQt6 迁移 + 布局重构（日志到底部）+ 取消倍速 + 性能优化。

用法（在项目目录下，off 屏运行不需要真实显示器）：
    set QT_QPA_PLATFORM=offscreen
    .\\venv\\Scripts\\python.exe validate_v13.py

A 部分：offscreen 构造 MainWindow，校验
        * 已全部运行在 PyQt6 上（不再依赖 PyQt5）；
        * 日志栏位于根布局末尾（界面底部）且为通栏全宽；
        * 顶部「视频源」与「参数设置」两张卡片严格等宽（各 50%）；
        * 日志栏高度可见约 8 行文字；
        * 扁平复选框（FlatCheckBox）**确实被画出来了**（方框 + 选中态对勾）——
          自绘控件的 paintEvent 异常会被吞掉，表现为控件「整块消失」且无任何报错，
          是 V1.3 真实踩过的坑，故此处用离屏渲染做像素断言；
        * 「目标人脸」栏吃到了日志移走后空出的纵向空间（预览高度显著增大）；
        * 播放倍速相关控件/属性已被彻底移除；
        * 分析模式（实时 / 全速）状态联动仍正确（RTSP 强制回实时）。
B 部分：真实 insightface + 从 测试视频.mp4 截出的 6 秒片段，跑实时 / 全速两种模式，
        对比墙钟耗时、实际分析帧数（是否 100% 覆盖）与截图产出。
C 部分：显示管线微基准 —— 对比 V1.2 旧写法（cvtColor + RGB888 + QImage.copy）
        与 V1.3 新写法（BGR888 直通 + 不 copy）的单帧耗时。

脚本自行清理临时产物；结果同时写入 _v13_result.txt 便于回溯。
"""
import os
import sys
import time
import queue
import shutil
import threading

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import face_clip_qt as F

LOG = []


def log(s):
    """即时落盘：即使随后崩溃也能看到执行到哪一步。"""
    print(s, flush=True)
    LOG.append(s)
    try:
        with open("_v13_result.txt", "a", encoding="utf-8") as f:
            f.write(s + "\n")
            f.flush()
    except Exception:
        pass


# ============================================================
# A 部分：UI 控件与状态联动
# ============================================================
def test_ui():
    from PyQt6 import QtWidgets, QtCore
    from PyQt6.QtGui import QPixmap
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(F.APP_QSS)
    # 复位“自绘异常只提示一次”的标记，让下面的断言能独立判定本次渲染
    F.FlatCheckBox._warned = False
    w = F.MainWindow()
    w.resize(1340, 1080)
    w.show()
    app.processEvents()
    app.processEvents()

    fails = []

    def chk(cond, msg):
        if not cond:
            fails.append(msg)

    # --- PyQt6 迁移 ---
    chk(sys.modules.get("PyQt6") is not None, "应已导入 PyQt6")
    chk("PyQt5" not in sys.modules, "不应再加载 PyQt5")
    chk(type(w).__mro__[0].__module__ == "face_clip_qt", "MainWindow 应来自本模块")

    # --- 取消播放倍速：控件与状态属性都必须不存在 ---
    for attr in ("speed_slider", "speed_label", "speed_val", "speed_steps", "_on_speed"):
        chk(not hasattr(w, attr), "倍速功能应已移除，但仍有属性：%s" % attr)

    # --- 日志栏移到底部通栏 ---
    g_log = w.log_text.parentWidget()
    central = w.centralWidget()
    rl = central.layout()
    chk(rl.indexOf(g_log) == rl.count() - 1,
        "日志栏应是根布局最后一个元素（界面底部）")
    chk(g_log.width() >= central.width() - 60,
        "日志栏应为通栏全宽（实测 %d / 容器 %d）" % (g_log.width(), central.width()))
    chk(w.log_text.height() >= 100, "日志文本区高度应仍可用（>=100px）")

    # --- 日志栏高度：目标可见约 8 行 ---
    # 两处口径必须用对，否则这个数字是假的：
    #   ① 用 viewport()（真正承载文本的区域），不是 log_text.height() —— 后者含 QSS padding；
    #   ② 用文档的**实际行高**（blockBoundingRect），不是 fontMetrics().lineSpacing() ——
    #      QSS 行高让每行实占 17px 而 lineSpacing 只有 14px，用它算会虚高 ~20%
    #      （曾算出 11.1 行，实际只放得下 9.2 行，与截图像素测量一致）。
    _t = w.log_text
    _doc = _t.document()
    _bh = _doc.documentLayout().blockBoundingRect(_doc.firstBlock()).height() or 1
    vh = _t.viewport().height()
    lines = vh / float(_bh)
    chk(lines >= 7.5, "日志栏应可见约 8 行文字（实测 %.1f 行，文本视口 %dpx / 实际行高 %.1fpx）"
        % (lines, vh, _bh))

    # --- 顶部「视频源」与「参数设置」等宽（各 50%）---
    g_src = w.rtsp_url.parentWidget()      # 视频源卡片
    g_param = w.chk_prefer.parentWidget()  # 参数设置卡片
    lw, rw = g_src.width(), g_param.width()
    chk(min(lw, rw) >= max(lw, rw) * 0.95,
        "顶部两张卡片应各占 50%%，实测 %d / %d（偏差 %.1f%%）"
        % (lw, rw, (1 - min(lw, rw) / float(max(lw, rw) or 1)) * 100))

    # --- 扁平复选框：必须真的画出方框（选中态还要有 √）---
    # 自绘控件若 paintEvent 抛异常，异常会被吞掉 -> 控件整块空白、且无处报错。
    # V1.3 实现时正是这样（PyQt6 的 QRect.adjusted 不收 float），故此处做像素断言。
    for nm, c in (("chk_prefer", w.chk_prefer), ("chk_segment", w.chk_segment)):
        chk(isinstance(c, F.FlatCheckBox), "%s 应是 FlatCheckBox 实例" % nm)
        chk(c.width() > 60 and c.height() > 0, "%s 应有有效尺寸" % nm)
        pm = QPixmap(c.size())
        pm.fill(QtCore.Qt.GlobalColor.white)
        c.render(pm)
        img = pm.toImage()
        seen = set()
        for yy in range(0, img.height(), 2):
            for xx in range(0, min(img.width(), 60), 2):
                seen.add(img.pixel(xx, yy))
        chk(len(seen) > 3,
            "%s 应真正绘制出方框（左 60px 内像素种类 %d，<=3 说明没画出来）"
            % (nm, len(seen)))
    chk(not F.FlatCheckBox._warned,
        "FlatCheckBox 的 paintEvent 不应抛异常（异常被吞掉会静默不绘制）")

    # --- 目标人脸栏吃到空出的纵向空间 ---
    chk(isinstance(w.face_thumb, F.FaceThumb), "目标人脸预览应是 FaceThumb 控件")
    fh = w.face_thumb.height()
    chk(fh >= 300, "目标人脸预览高度应 >=300px（日志移走后空间变高），实测 %d" % fh)
    g_face = w.face_thumb.parentWidget()
    chk(g_face.height() >= 400, "目标人脸卡片高度应 >=400px，实测 %d" % g_face.height())

    # --- 分析模式默认与联动 ---
    chk(w.analyze_mode == 'realtime', "默认分析模式应为 realtime")
    chk(w.rb_realtime.isChecked(), "默认应选中「实时分析」")
    chk(not w.rb_fullspeed.isChecked(), "默认不应选中「全速分析」")
    chk(w.rb_fullspeed.isEnabled(), "本地视频下「全速分析」应可用")
    chk(w.fps_label.text().startswith("分析帧率："), "应存在分析帧率标签")
    chk(w.balance_slider.isEnabled(), "实时模式下平衡滑块应可用")

    # 切到全速：平衡滑块禁用，提示文案更新
    w.rb_fullspeed.setChecked(True)
    app.processEvents()
    chk(w.analyze_mode == 'fullspeed', "选择全速后 analyze_mode 应为 fullspeed")
    chk(not w.balance_slider.isEnabled(), "全速模式下平衡滑块应禁用")
    chk("不按播放速度" in w.an_mode_hint.text(), "全速模式应有提示文案")

    # 切到 RTSP：必须强制回实时，且全速选项禁用
    w._on_source_toggle('rtsp')
    app.processEvents()
    chk(w.analyze_mode == 'realtime', "RTSP 下应强制回实时分析")
    chk(w.rb_realtime.isChecked(), "RTSP 下应选中「实时分析」")
    chk(not w.rb_fullspeed.isEnabled(), "RTSP 下「全速分析」应禁用")
    chk("RTSP 仅支持实时分析" in w.an_mode_hint.text(), "RTSP 应有提示文案")

    # 切回本地视频：全速恢复可用
    w._on_source_toggle('file')
    app.processEvents()
    chk(w.rb_fullspeed.isEnabled(), "切回本地视频后「全速分析」应恢复可用")
    chk(w.balance_slider.isEnabled(), "实时模式下平衡滑块应恢复可用")

    # --- 运行中动态切换：用真实 PlaybackThread 实例（不启动）验证标志是否被改写 ---
    class _StubRunner:
        def __init__(self, pb):
            self._active_pb = pb

    pb = F.PlaybackThread('file', "x.mp4", queue.Queue(maxsize=3), queue.Queue(maxsize=4),
                          threading.Event(), threading.Event(), 0, 1, 10.0,
                          detect_every=2, tight_box=True,
                          det_progress={'f': -1}, max_lead=2, fullspeed=False)
    chk(not hasattr(pb, "speed"), "PlaybackThread 不应再有 speed 属性（已取消倍速）")
    w.runner = _StubRunner(pb)
    w.running = True
    w.rb_fullspeed.setChecked(True)
    app.processEvents()
    chk(pb.fullspeed is True, "运行中切到全速应把 fullspeed 下发给播放线程")
    chk(pb.detect_every == 1, "全速模式应强制逐帧检测(detect_every=1)")
    chk(pb.tight_box is False, "全速模式应关闭紧跟踪门控")
    chk(pb.disp_every >= 2, "全速模式预览应降频发送（disp_every>=2）")
    w.rb_realtime.setChecked(True)
    app.processEvents()
    chk(pb.fullspeed is False, "运行中切回实时应把 fullspeed 置回 False")
    w.running = False
    w.runner = None

    if fails:
        log("[A] UI 状态联动：失败 %d 项" % len(fails))
        for m in fails:
            log("    ✗ " + m)
    else:
        log("[A] UI 状态联动 / PyQt6 迁移 / 布局重构：全部通过"
            "（顶部卡片 %d/%d 等宽 = %.1f%%/%.1f%%；日志卡 %dx%d 位于底部通栏，"
            "文本视口 %dpx 约容 %.1f 行；目标人脸预览 %dx%d；FlatCheckBox 自绘正常）"
            % (lw, rw, lw / float(lw + rw) * 100, rw / float(lw + rw) * 100,
               g_log.width(), g_log.height(), vh, lines,
               w.face_thumb.width(), fh))
    w.close()
    return not fails


# ============================================================
# B 部分：真实流水线吞吐对比
# ============================================================
def make_short_video(src, dst, seconds=6.0):
    """截取源视频前 N 秒，供两种模式做等量对比。"""
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(round(seconds * fps))
    vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cnt = 0
    while cnt < n:
        ok, fr = cap.read()
        if not ok:
            break
        vw.write(fr)
        cnt += 1
    vw.release()
    cap.release()
    return cnt, fps


def drain(q, acc):
    try:
        while True:
            item = q.get_nowait()
            k = item[0]
            if k == 'detect':
                acc['analyzed'] += 1
                now = time.time()
                if acc['t_first'] is None:
                    acc['t_first'] = now
                acc['t_last'] = now
            elif k == 'detfps':
                acc['fps_samples'].append(item[1])
            elif k == 'framecount':
                acc['shots'] = max(acc['shots'], item[1])
            elif k == 'episode':
                acc['episodes'] = max(acc['episodes'], item[1])
    except queue.Empty:
        pass


def run_mode(eng, src, out, mode, params, target_frames):
    os.makedirs(out, exist_ok=True)
    disp_q = queue.Queue(maxsize=3)
    det_q = queue.Queue(maxsize=4)
    res_q = queue.Queue()
    stop_event = threading.Event()
    pause_event = threading.Event()
    det_progress = {'f': -1}

    det = F.DetectThread(eng, [('file', src)], det_q, res_q, disp_q,
                         stop_event, params, out, 'compare',
                         det_progress=det_progress)
    det.start()

    pb = F.PlaybackThread('file', src, disp_q, det_q, stop_event, pause_event,
                          0, 1, params['pad'],
                          detect_every=1, tight_box=False,
                          det_progress=det_progress, max_lead=2,
                          fullspeed=(mode == 'fullspeed'))

    acc = {'analyzed': 0, 'fps_samples': [], 'shots': 0, 'episodes': 0,
           't_first': None, 't_last': None}
    t0 = time.time()
    pb.start()
    while pb.is_alive():
        drain(res_q, acc)
        time.sleep(0.01)
    pb.join()
    stop_event.set()
    det.join(timeout=30)
    wall = time.time() - t0
    drain(res_q, acc)

    fps_s = acc['fps_samples']
    peak = max(fps_s) if fps_s else 0.0
    tail = fps_s[len(fps_s) // 2:] if len(fps_s) >= 2 else fps_s
    steady = sum(tail) / len(tail) if tail else 0.0
    span = (acc['t_last'] - acc['t_first']) if acc['t_first'] else 0.0
    avg = (acc['analyzed'] - 1) / span if span > 0 else 0.0
    return {
        'mode': mode, 'wall': wall, 'analyzed': acc['analyzed'],
        'coverage': acc['analyzed'] / float(target_frames) if target_frames else 0,
        'peak': peak, 'steady': steady, 'avg': avg,
        'shots': acc['shots'], 'episodes': acc['episodes'],
    }


# ============================================================
# C 部分：显示管线微基准（V1.2 旧写法 vs V1.3 新写法）
# ============================================================
def bench_display(n=120):
    """对比单帧「BGR ndarray -> QPixmap」的耗时。

    旧（V1.2）：cvtColor(BGR->RGB) + QImage(Format_RGB888).copy() + QPixmap.fromImage
    新（V1.3）：QImage(Format_BGR888) 直通 + QPixmap.fromImage（不再 copy）
    """
    import numpy as np
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    frame = (np.random.rand(720, 1280, 3) * 255).astype("uint8")

    def old_way(fr):
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def new_way(fr):
        arr = np.ascontiguousarray(fr)
        h, w = arr.shape[:2]
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_BGR888)
        return QPixmap.fromImage(qimg)

    def run(fn):
        for _ in range(8):        # 预热
            fn(frame)
        t0 = time.time()
        for _ in range(n):
            fn(frame)
        return (time.time() - t0) / n * 1000.0

    t_old = run(old_way)
    t_new = run(new_way)

    # 正确性：两种写法必须产出完全一致的像素（BGR888 直通不能偏色）
    a = old_way(frame).toImage().convertToFormat(QImage.Format.Format_RGB888)
    b = new_way(frame).toImage().convertToFormat(QImage.Format.Format_RGB888)
    same = (bytes(a.constBits().asstring(a.sizeInBytes()))
            == bytes(b.constBits().asstring(b.sizeInBytes())))

    log("[C] 显示管线单帧（1280x720）：旧 %.3f ms -> 新 %.3f ms"
        "（省 %.3f ms，%.0f%%）  像素一致性：%s"
        % (t_old, t_new, t_old - t_new,
           (t_old - t_new) / t_old * 100.0 if t_old else 0.0,
           "一致" if same else "不一致"))
    return t_old, t_new, same


def main():
    if os.path.exists("_v13_result.txt"):
        os.remove("_v13_result.txt")
    log("=" * 72)
    ok_ui = test_ui()

    src_video = os.path.join(HERE, "测试视频.mp4")
    face_img = os.path.join(HERE, "目标人脸.png")
    short = os.path.join(HERE, "_v13_short.mp4")
    out = os.path.join(HERE, "_v13_out")
    if os.path.isdir(out):
        shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)

    t_old, t_new, same = bench_display()

    n_frames, fps = make_short_video(src_video, short, seconds=6.0)
    log("[B] 对比素材：%d 帧 @ %.1f fps（约 %.1f 秒）" % (n_frames, fps, n_frames / fps))

    providers, use_gpu, gpu, note = F.detect_runtime()
    log("[B] 运行环境：%s / %s" % ("GPU" if use_gpu else "CPU", gpu))
    eng = F.Engine(providers, use_gpu, lambda m: None, mode='compare')
    eng.load_model()
    eng.set_target(face_img)

    # 贴近真实使用：开人脸优选（默认）、相似度 0.70（默认）→ 截图稀疏，IO 不干扰
    params = {
        'sim_threshold': 0.70, 'quality_threshold': 70.0, 'pad': 10.0,
        'prefer_dur': 3.0, 'enable_prefer': True,
        'balance': 75, 'enable_segment': False, 'analyze_mode': 'realtime',
    }
    # 纯检测上限：质量阈值设为不可达 -> 完全不产截图，量化“解码+检测”的天花板
    params_nocap = dict(params, quality_threshold=999.0, sim_threshold=0.999)

    results = {}
    plan = [('realtime', params), ('fullspeed', params), ('fullspeed*', params_nocap)]
    for mode, pm in plan:
        real_mode = 'fullspeed' if mode.startswith('fullspeed') else mode
        r = run_mode(eng, short, os.path.join(out, mode.replace('*', 'x')),
                     real_mode, pm, n_frames)
        r['tag'] = mode
        results[mode] = r
        log("[B] %-10s 墙钟 %6.2fs | 分析 %4d 帧 | 覆盖率 %5.1f%% | "
            "实测平均 %5.1f fps | 界面峰值 %5.1f fps | 截图 %d 张"
            % (r['tag'], r['wall'], r['analyzed'], r['coverage'] * 100,
               r['avg'], r['peak'], r['shots']))

    rt, fs, fsx = results['realtime'], results['fullspeed'], results['fullspeed*']
    log("[B] 全速相对实时提速：墙钟 %.2f 倍 / 实测帧率 %.2f 倍"
        % ((rt['wall'] / fs['wall']) if fs['wall'] > 0 else 0,
           (fs['avg'] / rt['avg']) if rt['avg'] > 0 else 0))
    log("[B] 纯检测上限（不产截图）：%.1f fps -> 6 秒素材理论耗时 %.2fs"
        % (fsx['avg'], n_frames / fsx['avg'] if fsx['avg'] else 0))

    checks = [
        (rt['analyzed'] >= n_frames * 0.98, "实时模式应逐帧分析（覆盖率≥98%）"),
        (fs['analyzed'] >= n_frames * 0.98, "全速模式应逐帧不漏（覆盖率≥98%）"),
        (fs['analyzed'] == rt['analyzed'], "全速与实时分析帧数应一致（不丢帧）"),
        (fs['wall'] < rt['wall'] * 0.8, "全速模式墙钟应明显短于实时模式"),
        # 实测本机检测链约 22.5ms/帧（SCRFD 检测器本体）= 约 44fps 上限，
        # 因此全速实测帧率高于视频帧率(25) 30% 以上即视为达标
        (fs['avg'] > 32.5, "全速实测帧率应高于视频帧率(25)至少 30%"),
        (abs(rt['avg'] - 25.0) < 3.0, "实时模式帧率应贴合视频帧率 25fps（±3）"),
        (fs['shots'] == rt['shots'] and fs['shots'] > 0,
         "全速与实时应产出相同数量截图（结果一致）"),
        (fsx['avg'] >= fs['avg'] * 0.95, "纯检测上限应不低于含截图时的帧率"),
        (same, "新显示管线像素应与旧写法完全一致（BGR888 不可偏色）"),
        (t_new < t_old, "新显示管线单帧耗时应低于旧写法"),
    ]
    bad = [m for c, m in checks if not c]
    log("=" * 72)
    if ok_ui and not bad:
        log("RESULT: V13_OK")
    else:
        log("RESULT: V13_FAIL")
        for m in bad:
            log("    ✗ " + m)

    # 清理临时产物（_v13_result.txt 保留，供回溯）
    try:
        if os.path.exists(short):
            os.remove(short)
        shutil.rmtree(out, ignore_errors=True)
    except Exception as e:
        log("[清理] 跳过：%s" % e)
    log("[清理] 临时视频与输出目录已删除；明细见 _v13_result.txt")


if __name__ == "__main__":
    main()
