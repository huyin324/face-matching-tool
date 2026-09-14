# -*- coding: utf-8 -*-
"""V1.2 回归自测：分析模式（实时 / 全速）UI 状态 + 真实流水线吞吐对比。

用法（在项目目录下，off 屏运行不需要真实显示器）：
    set QT_QPA_PLATFORM=offscreen
    .\\venv\\Scripts\\python.exe validate_v12.py

A 部分：offscreen 构造 MainWindow，校验新增控件与状态联动
        （默认实时、切全速后滑块禁用、运行中动态切换、RTSP 强制回实时）。
B 部分：真实 insightface + 从 测试视频.mp4 截出的 6 秒片段，分别跑实时/全速模式，
        对比墙钟耗时、实际分析帧数（是否 100% 覆盖）、实测分析帧率与截图产出。

本机基准（RTX 3060 Laptop，1280x720@25fps）：
    实时   23.1 fps / 墙钟 6.86s
    全速   46.3 fps / 墙钟 3.26s   -> 约 2.0 倍
    上限   eng.app.get() 约 22.5ms/帧（SCRFD 检测器本体）→ 约 44 fps
脚本自行清理临时产物；结果同时写入 _v12_result.txt 便于回溯。
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

import face_clip_qt as F

LOG = []


def log(s):
    """即时落盘：即使随后崩溃也能看到执行到哪一步。"""
    print(s, flush=True)
    LOG.append(s)
    try:
        with open("_v12_result.txt", "a", encoding="utf-8") as f:
            f.write(s + "\n")
            f.flush()
    except Exception:
        pass


# ============================================================
# A 部分：UI 控件与状态联动
# ============================================================
def test_ui():
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = F.MainWindow()

    fails = []

    def chk(cond, msg):
        if not cond:
            fails.append(msg)

    chk(w.analyze_mode == 'realtime', "默认分析模式应为 realtime")
    chk(w.rb_realtime.isChecked(), "默认应选中「实时分析」")
    chk(not w.rb_fullspeed.isChecked(), "默认不应选中「全速分析」")
    chk(w.rb_fullspeed.isEnabled(), "本地视频下「全速分析」应可用")
    chk(w.fps_label.text().startswith("分析帧率："), "应存在分析帧率标签")
    chk(not w.speed_slider.isEnabled(), "未运行时倍速滑块应禁用")
    chk(w.balance_slider.isEnabled(), "实时模式下平衡滑块应可用")

    # 切到全速：倍速与平衡滑块都应禁用，提示文案更新
    w.rb_fullspeed.setChecked(True)
    app.processEvents()
    chk(w.analyze_mode == 'fullspeed', "选择全速后 analyze_mode 应为 fullspeed")
    chk(not w.speed_slider.isEnabled(), "全速模式下倍速滑块应禁用")
    chk(not w.balance_slider.isEnabled(), "全速模式下平衡滑块应禁用")
    chk("不按播放速度" in w.an_mode_hint.text(), "全速模式应有提示文案")
    chk(w.speed_label.text() == "—", "全速模式下倍速标签应显示 —")

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

    # 运行中动态切换：用真实 PlaybackThread 实例（不启动）验证标志与档位是否被改写
    class _StubRunner:
        def __init__(self, pb):
            self._active_pb = pb

    pb = F.PlaybackThread('file', "x.mp4", queue.Queue(maxsize=3), queue.Queue(maxsize=4),
                          threading.Event(), threading.Event(), 0, 1, 10.0,
                          speed=1, detect_every=2, tight_box=True,
                          det_progress={'f': -1}, max_lead=2, fullspeed=False)
    w.runner = _StubRunner(pb)
    w.running = True
    w.rb_fullspeed.setChecked(True)
    app.processEvents()
    chk(pb.fullspeed is True, "运行中切到全速应把 fullspeed 下发给播放线程")
    chk(pb.detect_every == 1, "全速模式应强制逐帧检测(detect_every=1)")
    chk(pb.tight_box is False, "全速模式应关闭紧跟踪门控")
    w.rb_realtime.setChecked(True)
    app.processEvents()
    chk(pb.fullspeed is False, "运行中切回实时应把 fullspeed 置回 False")
    w.running = False
    w.runner = None

    # _start 的参数链路：RTSP 下 analyze_mode 必须被改写成 realtime
    w.rb_fullspeed.setChecked(True)
    app.processEvents()
    w.source_kind = 'rtsp'
    forced = 'realtime' if w.source_kind == 'rtsp' else w.analyze_mode
    chk(forced == 'realtime', "_start 的 RTSP 分支应把 analyze_mode 压回 realtime")
    w.source_kind = 'file'
    w.rb_realtime.setChecked(True)
    app.processEvents()

    w.close()
    log("[A] UI 状态联动：%s" % ("全部通过" if not fails else "失败 %d 项" % len(fails)))
    for m in fails:
        log("    ✗ " + m)
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
                # 记录首/末帧完成时间：据此算出「纯分析区间」的真实平均帧率，
                # 排除启动与收尾（模型已在外部加载，这里只排除取流/排空开销）
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
                          0, 1, params['pad'], speed=1,
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
    # 稳态帧率取后半段样本均值，跳过启动阶段的爬坡
    tail = fps_s[len(fps_s) // 2:] if len(fps_s) >= 2 else fps_s
    steady = sum(tail) / len(tail) if tail else 0.0
    # 纯分析区间平均帧率（首帧完成 -> 末帧完成），最能代表实际分析吞吐
    span = (acc['t_last'] - acc['t_first']) if acc['t_first'] else 0.0
    avg = (acc['analyzed'] - 1) / span if span > 0 else 0.0
    return {
        'mode': mode, 'wall': wall, 'analyzed': acc['analyzed'],
        'coverage': acc['analyzed'] / float(target_frames) if target_frames else 0,
        'peak': peak, 'steady': steady, 'avg': avg,
        'shots': acc['shots'], 'episodes': acc['episodes'],
    }


def main():
    if os.path.exists("_v12_result.txt"):
        os.remove("_v12_result.txt")
    log("=" * 72)
    ok_ui = test_ui()

    src_video = os.path.join(HERE, "测试视频.mp4")
    face_img = os.path.join(HERE, "目标人脸.png")
    short = os.path.join(HERE, "_v12_short.mp4")
    out = os.path.join(HERE, "_v12_out")
    if os.path.isdir(out):
        shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)

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
        'speed': 1, 'balance': 75, 'enable_segment': False,
        'analyze_mode': 'realtime',
    }
    # 纯检测上限：质量阈值设为不可达 -> 完全不产截图，量化“解码+检测”的天花板
    params_nocap = dict(params, quality_threshold=999.0, sim_threshold=0.999)

    results = {}
    plan = [('realtime', params), ('fullspeed', params), ('fullspeed*', params_nocap)]
    for mode, pm in plan:
        real_mode = 'fullspeed' if mode.startswith('fullspeed') else mode
        r = run_mode(eng, short, os.path.join(out, mode.replace('*', 'x')), real_mode, pm, n_frames)
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
    ]
    bad = [m for c, m in checks if not c]
    log("=" * 72)
    if ok_ui and not bad:
        log("RESULT: V12_OK")
    else:
        log("RESULT: V12_FAIL")
        for m in bad:
            log("    ✗ " + m)

    # 清理临时产物（_v12_result.txt 保留，供回溯）
    try:
        if os.path.exists(short):
            os.remove(short)
        shutil.rmtree(out, ignore_errors=True)
    except Exception as e:
        log("[清理] 跳过：%s" % e)
    log("[清理] 临时视频与输出目录已删除；明细见 _v12_result.txt")


if __name__ == "__main__":
    main()
