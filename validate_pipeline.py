#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
仅验证“非 AI 管线”：视频片段截取(20秒) + 红框标注，不依赖人脸模型。
用于在没有下载到 insightface 模型时，也能证明 文件输出/视频写入/红框 逻辑正确。
"""
import os
import sys
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from face_clip_tool import extract_segment, draw_red_box, fmt_time, imwrite_utf8

VIDEO = os.path.join(HERE, "测试视频.mp4")
OUT = os.path.join(HERE, "管线验证输出")
os.makedirs(OUT, exist_ok=True)


def main():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("[FAIL] 无法打开测试视频"); sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = total / fps
    print(f"[视频] 时长 {fmt_time(dur)}，fps {fps:.2f}，总帧 {total}")

    # --- 测试1：截取 20 秒片段 [10s, 30s] ---
    seg = os.path.join(OUT, "test_segment_20s.mp4")
    ok = extract_segment(VIDEO, 10.0, 30.0, seg, print)
    if not ok or not os.path.exists(seg):
        print("[FAIL] 片段截取失败"); sys.exit(1)
    vc = cv2.VideoCapture(seg)
    sfps = vc.get(cv2.CAP_PROP_FPS) or fps
    sframes = int(vc.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sdur = sframes / sfps
    vc.release()
    print(f"[片段] 输出 {os.path.getsize(seg)/1024:.1f} KB，约 {sdur:.1f}s（预期~20s），帧数 {sframes}")
    assert 15 <= sdur <= 25, f"片段时长异常: {sdur}"

    # --- 测试2：红框标注一帧 ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(15 * fps))
    ret, frame = cap.read()
    assert ret, "读取测试帧失败"
    h, w = frame.shape[:2]
    bbox = [w*0.3, h*0.2, w*0.7, h*0.9]  # 伪造 bbox
    marked = draw_red_box(frame, bbox)
    img_path = os.path.join(OUT, "test_redbox.png")
    ok_write = imwrite_utf8(marked, img_path)
    # 验证红框像素存在（BGR 中 R 通道高 = 红）
    has_red = (marked[:, :, 2] > 200).any()
    print(f"[红框] 输出 {os.path.getsize(img_path)/1024:.1f} KB，imwrite成功: {ok_write}，存在红色像素: {bool(has_red)}")
    assert os.path.exists(img_path) and has_red and ok_write, "红框标注失败"

    cap.release()
    print("=" * 40)
    print("[PASS] 非 AI 管线验证通过：20秒片段截取 + 红框标注 均正常。")
    print(f"输出目录: {OUT}")


if __name__ == "__main__":
    main()
