#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无界面验证脚本 v2：验证新版本的引擎与工具函数（不启动 GUI）。
覆盖：
  - 模块可导入（tkinter / insightface / onnxruntime 均可用）
  - 运行环境检测（GPU/CPU）
  - 目标人脸锁定 + 视频取样比对
  - 新增工具函数：sim_to_label（相似度百分比）、frame_contrast_pct（对比度）
  - draw_red_box 标注相似度百分比
  - extract_segment 片段截取 + 中文输出目录
不依赖显示器，可在任意终端运行。
"""
import os
import sys
import time

import numpy as np

import face_clip_tool as F

HERE = os.path.dirname(os.path.abspath(__file__))
FACE = os.path.join(HERE, "目标人脸.png")
VIDEO = os.path.join(HERE, "测试视频.mp4")
OUT = os.path.join(HERE, "验证输出_v2")


def main():
    print("== 模块导入检查 ==")
    print("TK_AVAILABLE   :", F.TK_AVAILABLE)
    print("INSIGHTFACE    :", F.INSIGHTFACE_AVAILABLE)
    print("ORT            :", F.ORT_AVAILABLE)
    print("PILTK          :", F.PILTK_AVAILABLE)
    if not F.TK_AVAILABLE:
        print("[警告] 当前解释器缺 tkinter（无头验证跳过了 GUI 启动检查）；"
              "GUI 能否运行由启动器的 tkinter 健康检查兜底。")
    assert F.INSIGHTFACE_AVAILABLE and F.ORT_AVAILABLE, "缺少核心依赖"

    # 工具函数单测
    print("\n== 工具函数检查 ==")
    assert F.sim_to_label(0.985) == "98.50%", F.sim_to_label(0.985)
    assert F.sim_to_label(None) == "Target"
    print("sim_to_label(0.985) =", F.sim_to_label(0.985))
    gray = np.full((100, 100), 128, dtype=np.uint8)
    cp = F.frame_contrast_pct(gray)
    print("frame_contrast_pct(纯色) =", round(cp, 2))
    assert cp < 1.0, "纯色帧对比度应接近 0"

    # 运行环境
    print("\n== 运行环境 ==")
    providers, use_gpu, gpu_name, note = F.detect_runtime()
    print("use_gpu:", use_gpu, "| gpu:", gpu_name)
    print("note:", note)

    os.makedirs(OUT, exist_ok=True)

    # 引擎
    print("\n== 引擎加载 + 目标锁定 ==")
    eng = F.Engine(providers, use_gpu, print)
    t0 = time.time()
    eng.load_model()
    print("模型加载耗时: %.1fs" % (time.time() - t0))
    eng.threshold = 0.40
    tgt = eng.set_target(FACE)
    print("目标相似度自比 =", round(float(np.dot(tgt.embedding, tgt.embedding) /
                                         (np.linalg.norm(tgt.embedding) ** 2)), 4))

    # 取样比对测试视频前若干帧
    print("\n== 视频比对（取样前 400 帧）==")
    cap = F.cv2.VideoCapture(VIDEO)
    fps = cap.get(F.cv2.CAP_PROP_FPS) or 25.0
    max_sim = -1.0
    match_frames = 0
    sample_n = 400
    n = 0
    sample_step = 3
    while n < sample_n:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1
        if (n - 1) % sample_step != 0:
            continue
        try:
            faces, best_sim, matched = eng.detect_faces(frame)
        except Exception as e:
            print("detect err:", e)
            continue
        max_sim = max(max_sim, best_sim)
        if matched:
            match_frames += 1
            # 标注相似度百分比
            marked = F.draw_red_box(frame, matched[0][0].bbox,
                                    label=F.sim_to_label(matched[0][1]))
            p = os.path.join(OUT, f"sample_match_{n:04d}.png")
            F.imwrite_utf8(marked, p)
            print(f"  帧{n}: 相似度 {F.sim_to_label(matched[0][1])} -> 已存 {os.path.basename(p)}")
    cap.release()
    print(f"取样 {n} 帧，最大相似度 = {max_sim:.3f}，命中帧 = {match_frames}")

    # 片段截取：若命中，则截取一段验证 extract_segment + 中文目录
    if match_frames > 0:
        print("\n== 片段截取验证 ==")
        seg_out = os.path.join(OUT, "片段截取测试")
        os.makedirs(seg_out, exist_ok=True)
        ok = F.extract_segment(VIDEO, 1.0, 5.0,
                               os.path.join(seg_out, "test_seg.mp4"), print)
        print("extract_segment 成功:", ok)

    print("\n== 验证完成 ==")
    print("结果目录:", OUT)


if __name__ == "__main__":
    main()
