#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无界面验证脚本：复用 face_clip_tool 的检测引擎，
在测试视频上跑前若干帧，确认 人脸检测 / 比对 / 片段截取 / 红框截图 全链路可用。
用法：
    venv/Scripts/python.exe validate_headless.py [--max-frames N] [--threshold 0.4]
"""
import os
import sys
import time
import argparse

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from face_clip_tool import (
    Engine, detect_runtime, extract_segment, draw_red_box, fmt_time, imwrite_utf8,
)

FACE = os.path.join(HERE, "目标人脸.png")
VIDEO = os.path.join(HERE, "测试视频.mp4")
OUT = os.path.join(HERE, "验证输出")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=2000,
                    help="最多处理的帧数（用于快速验证）")
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--sample-step", type=int, default=2)
    args = ap.parse_args()

    if not os.path.isfile(FACE) or not os.path.isfile(VIDEO):
        print("[错误] 缺少 目标人脸.png 或 测试视频.mp4")
        sys.exit(1)
    os.makedirs(OUT, exist_ok=True)

    providers, use_gpu, gpu_name, note = detect_runtime()
    print("[环境]", "GPU(CUDA)" if use_gpu else "CPU", "|", note)

    log = lambda m: print("[LOG]", m)
    eng = Engine(providers, use_gpu, log)
    t0 = time.time()
    eng.load_model()
    eng.threshold = args.threshold
    eng.set_target(FACE)
    print(f"[OK] 模型加载+目标锁定 用时 {time.time()-t0:.1f}s")

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("[错误] 无法打开视频")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total / fps) if fps > 0 and total > 0 else 0
    print(f"[视频] 时长 {fmt_time(duration)}，fps {fps:.2f}，总帧 {total}")

    pad = 10.0
    frame_gap = 3.0
    active_seg = None
    last_frame_t = -999.0
    episode_idx = 0
    frame_idx = 0
    f = 0
    max_sim_overall = -1.0
    matched_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        f += 1
        if (f - 1) % args.sample_step != 0:
            continue
        if f > args.max_frames:
            print(f"[信息] 已达到 --max-frames 上限 {args.max_frames}，停止。")
            break

        try:
            faces, best_sim, matched = eng.detect_faces(frame)
        except Exception as e:
            print(f"[警告] 第 {f} 帧异常: {e}")
            continue
        max_sim_overall = max(max_sim_overall, best_sim)
        is_match = len(matched) > 0
        if is_match:
            matched_frames += 1
            t = f / fps
            if active_seg is None or t > active_seg[1]:
                seg_start = max(0.0, t - pad)
                seg_end = min(duration, t + pad)
                episode_idx += 1
                seg_name = f"片段_{episode_idx:03d}_{fmt_time(t).replace(':','-')}.mp4"
                extract_segment(VIDEO, seg_start, seg_end, os.path.join(OUT, seg_name), log)
                active_seg = (seg_start, seg_end)
                # 起点必存一帧
                name = f"匹配帧_{frame_idx+1:03d}_{fmt_time(t).replace(':','-')}.png"
                imwrite_utf8(draw_red_box(frame, matched[0][0].bbox), os.path.join(OUT, name))
                frame_idx += 1
                last_frame_t = t
            else:
                if t - last_frame_t >= frame_gap:
                    name = f"匹配帧_{frame_idx+1:03d}_{fmt_time(t).replace(':','-')}.png"
                    imwrite_utf8(draw_red_box(frame, matched[0][0].bbox), os.path.join(OUT, name))
                    frame_idx += 1
                    last_frame_t = t

    cap.release()
    print("=" * 50)
    print(f"[结果] 处理帧数: {f}（采样步长 {args.sample_step}）")
    print(f"[结果] 最高相似度: {max_sim_overall:.3f}")
    print(f"[结果] 命中(匹配)帧数: {matched_frames}")
    print(f"[结果] 出场片段(20s视频): {episode_idx}")
    print(f"[结果] 红框截图: {frame_idx}")
    print(f"[结果] 输出目录: {OUT}")
    files = sorted(os.listdir(OUT))
    for x in files:
        p = os.path.join(OUT, x)
        print(f"   - {x}  ({os.path.getsize(p)/1024:.1f} KB)")
    print("=" * 50)
    if episode_idx == 0:
        print("[提示] 未发现匹配人脸。可尝试降低 --threshold（如 0.30）。")
    else:
        print("[OK] 全链路验证通过。")


if __name__ == "__main__":
    main()
