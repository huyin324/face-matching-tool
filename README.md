# face-matching-tool · 视频人脸匹配截取工具

> 给它一张人脸照片 + 一段视频，它自动告诉你「这个人出现在第几秒」，并把他每次出场的片段和最佳画面截取出来。
>
> GUI 桌面工具 · Python + PyQt5 · insightface + ONNXRuntime(CUDA) · 支持本地视频与 RTSP 实时流

![Python](https://img.shields.io/badge/Python-3.9%20~%203.11-blue)
![PyQt5](https://img.shields.io/badge/UI-PyQt5-41cd52)
![insightface](https://img.shields.io/badge/Model-buffalo__l-orange)
![CUDA](https://img.shields.io/badge/Backend-CUDA%2012.x%20%2F%20CPU-76b900)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-V1.0-brightgreen)

---

## 一、它解决什么问题

监控排查、素材剪辑、活动录像归档最常见的需求是：**"把视频里所有出现某某人的地方找出来"**。
人工做法只能拖进度条一帧帧看，1 小时的素材要看一整天。

本工具把这件事全自动：

| 你提供 | 你得到 |
| --- | --- |
| 一张目标人脸照片 | 每次出场的时间点（精确到秒） |
| 一段视频 / 一路 RTSP 实时流 | 每次出场的视频片段 mp4（可省略） |
| （可选）连目标人脸都不用给 | 所有检测到人脸的高清抓图 |

典型场景：

- 监控录像中检索特定人员的所有出现时段
- 长视频（演唱会 / 会议 / 婚礼 / 课堂）中批量截取某人的镜头素材
- 无人值守 RTSP 摄像头实时流的自动抓拍与取证留档
- 从素材库中快速捞出「人脸最清晰的那一帧」用于建档

---

## 二、核心功能

### 1. 两种运行模式

| 模式 | 输入 | 触发条件 | 输出 |
| --- | --- | --- | --- |
| **人脸识别比对模式** | 目标人脸照片 + 视频 | 画面人脸与目标相似度 ≥ 阈值 | 红框标注截图（框上显示相似度，如 `98.50%`）+ 出场片段 |
| **人脸检测抓图模式** | 仅视频 | 检出人脸质量分 ≥ 阈值 | 所有人脸的绿框抓图（无需目标照片） |

### 2. 人脸优选（默认开启）

连续出场时逐帧截图会产生几百张几乎一样的图。开启优选后，在「优选时长」（默认 3 秒）窗口内**只保留人脸质量分最高的那一帧**，去重效果显著。

质量分综合了 6 个维度，不是单纯比大小：
`清晰度(Laplacian 方差) + 人脸尺寸 + 光照 + 对比度 + 正脸程度 + 居中程度`

低质量人脸（暗光 / 模糊 / 侧脸 / 像素过低 / 强光比）会被自动跳过，不参与比对，避免误判。

### 3. 出场片段截取

命中时刻向前 / 向后各保留 N 秒（默认各 10 秒，共 20 秒），自动拼接成完整出场片段。
本地视频走精确 seek 重编码；**RTSP 实时流走环形缓冲区**，实现「事后回捞」——即命中时刻之前的历史画面也能被保存下来。

### 4. 实时性与流畅性同时拉满

这是本项目重点打磨的部分。默认档位下（ RTX 3060 Laptop + 720p@25fps 实测）：

| 指标 | 优化前 | 优化后 |
| --- | --- | --- |
| 检测帧率 | 3.7 ~ 5.4 fps | **23.0 fps**（≈ 逐帧） |
| 播放帧率 | 23 fps | **23.2 fps** |
| 目标框滞后 | 4.6+ 帧 | **平均 0.47 帧，最大 1 帧** |

即：**视频流畅播放与目标框紧贴人脸不再互斥**。详见 [第七章 · 性能优化实录](#七性能优化实录实测驱动非拍脑袋)。

### 5. 其他

- 多视频批量排队，逐个自动处理，中途可暂停 / 结束
- 支持 RTSP 实时流，播放倍速可调
- 自动检测 NVIDIA GPU：有则启用 CUDA，无则静默降级 CPU，无需手动切换
- 所有图片读写走 Pillow 的 UTF-8 封装，**中文路径 / 中文文件名不会报错**（本机 OpenCV 构建不支持中文路径）
- 一键启动：`launch.bat` 自动建虚拟环境、装依赖、补模型、启程序

---

## 三、界面速览

```
┌────────────────────────────┬──────────────┬──────────────────────────────┐
│ 视频源列表                  │  参数设置     │                              │
│  [添加视频] [移除] [清空]   │  ◉ 比对模式   │                              │
│  rtsp://...（可选）         │  ○ 检测模式   │                              │
│                            │  相似度阈值 70%│        实时视频画面          │
│ 输出目录 [浏览][打开目录]   │  质量阈值 70  │      （红/绿框实时标注）     │
├────────────────────────────┤  人脸优选 ✔   │                              │
│ 目标人脸（竖版 3:4 预览）   │  流畅◀━●━▶实时│                              │
│  [选择目标人脸图片]         │  截取视频 □   │                              │
│  当前相似度：98.50%         ├──────────────┤                              │
│  比对结果：匹配             │  播放倍速 1X  │                              │
│  实时人脸数：3              ├──────────────┴──────────────────────────────┤
│  出场片段：2   截图：5      │  状态：运行中    [开始] [暂停] [结束]        │
│                            ├─────────────────────────────────────────────┤
│  运行日志                   │  日志实时滚动输出                            │
└────────────────────────────┴─────────────────────────────────────────────┘
```

---

## 四、快速开始

### 方式 A：Windows 一键启动（推荐）

```bat
双击 launch.bat
```

首次运行会自动完成：检查 Python → 创建 `venv/` → 安装全部依赖 → 下载 buffalo_l 模型 → 启动界面。
**首次约需 5~15 分钟（取决于网速）**，之后每次秒开（脚本会跳过已就绪的依赖检查）。

> 已装好依赖想跳过检查直接启动，双击 `run.bat`。
> 启动异常时先跑 `diagnose.bat`，它会输出环境自检报告。

### 方式 B：手动安装

```bash
# 1. 建虚拟环境
python -m venv venv
venv\Scripts\activate

# 2. 安装依赖（有 NVIDIA 显卡）
pip install insightface --no-deps
pip install "onnxruntime-gpu==1.22.0" opencv-python-headless pillow numpy onnx easydict prettytable pyyaml tqdm requests pyqt5

#    无 NVIDIA 显卡：把上一行的 onnxruntime-gpu==1.22.0 换成 onnxruntime

# 3. 启动
python face_clip_qt.py
```

### 方式 C：Linux / macOS

```bash
chmod +x launch.sh && ./launch.sh
```

---

## 五、参数详解

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| 运行模式 | 比对模式 | 比对模式需目标人脸；检测模式无需目标人脸 |
| 相似度阈值 | 70% | 余弦相似度。调高更严格（漏检↑ 误检↓），调低更宽松。同一个人建议 60~75；光线差或角度偏可降到 55 |
| 人脸质量阈值 | 70 | 低于该分的人脸不参与比对/抓图。素材模糊时可降到 50~60 |
| 人脸优选 | 开启 | 在优选时长内只保留质量最高的一帧 |
| 人脸优选时长 | 3.0 秒 | 去重窗口。想要更多画面就调小或关闭优选 |
| 流畅性 ◀ ▶ 检测实时 | 75（逐帧） | 见下表。默认已可逐帧检测，偏左可降低显卡占用 |
| 启用截取视频 | 关闭 | 关闭时只出截图（快）；开启时额外产出 20 秒片段 mp4 |
| 截取视频前后时长 | 各 10 秒 | 命中时刻前后各保留的秒数 |
| 播放倍速 | 1X | 仅影响预览速度，不影响检测精度 |

**平衡滑块档位**（`balance` → 每几帧检测一次）：

| 滑块值 | 检测间隔 | 适用 |
| --- | --- | --- |
| 85 ~ 100 | 每帧 + 紧跟踪模式（画面显示不领先检测） | 目标框必须严丝合缝，显卡有余量 |
| 67 ~ 84 | 每帧 | **默认 75**，流畅与实时兼得 |
| 34 ~ 66 | 每 2 帧 | 4K 素材 / 显卡吃紧 |
| 0 ~ 33 | 每 4 帧 | 纯 CPU 运行 |

---

## 六、输出说明

默认输出到程序目录下的 `输出结果/`：

```
输出结果/
├── 匹配帧_001_00-01-23.png     # 比对模式：红框 + 相似度标注
├── 匹配帧_002_00-04-56.png
├── 检测帧_001_00-00-12.png     # 检测模式：绿框
└── 片段_001_出场00-01-23.mp4   # 出场片段（启用截取视频时）
```

命名规则：`<类型>_<序号>_<时间点>.png` / `片段_<序号>_出场<时间点>.mp4`，时间点即该人在视频中的出现时刻（时-分-秒）。

**RTSP 实时流的片段**会在命中后继续录制完向后时长才落盘，因此文件名时间与内容区间略有差异，属正常现象。

---

## 七、性能优化实录（实测驱动，非拍脑袋）

初版的用户体验是「要么视频卡、要么框跟不上」。我们没有靠调参玄学，而是**先分阶段打点实测**，再对症下药。基线：`1280x720 @25fps`，每帧预算 40ms。

### 结论先行：检测是唯一真瓶颈

显示管线只占 3.08ms/帧，完全不是问题；而 `insightface app.get()` 单帧高达 **183~266ms**（3.7~5.4 fps），比视频帧率慢 5 倍。所以一切优化应围绕检测展开。

### 根因 1：`det_size` 未显式传入 —— 17~33 倍提速（最关键）

```python
# FaceAnalysis.prepare 的签名：prepare(ctx_id, det_thresh=0.5, det_size=None)
#                                                          ^^^^^^^^^ 默认是 None
```

`det_size=None` 会退化成动态尺寸列表 `[(128,128),(640,640)]`。而 buffalo_l 的 `det_10g.onnx` 输入是动态的 `[1,3,'?','?']`，**动态尺寸下 CUDA/cuDNN 无法缓存卷积算法，每帧都要重新搜索最优算法** → GPU 退化到 266ms/帧，比 CPU 还慢。

而早期的写法 `prepare(ctx_id, providers=...)` 会直接抛 `TypeError`，被 try/except 吞掉后回退到 `prepare(ctx_id=0)`，正好踩进这个坑，且**没有任何报错提示**。

```python
# 修复：显式固定检测尺寸
app.prepare(ctx_id=ctx_id, det_size=(640, 640))

# 附带：CUDA EP 选项
providers = [("CUDAExecutionProvider", {
    "device_id": 0,
    "cudnn_conv_algo_search": "HEURISTIC",     # 默认 EXHAUSTIVE 更慢
    "arena_extend_strategy": "kSameAsRequested",
}), "CPUExecutionProvider"]
```

**266ms → 13.8ms。**

### 根因 2：默认跑了全部 5 个模型

`allowed_modules=None` 时 buffalo_l 的 5 个模型会全部加载并逐个推理，其中 `genderage`、`landmark_3d_68` 本工具完全用不上，却要「每帧 × 每个人脸」各跑一次。

```python
# 按需裁剪
allowed_modules = ['detection', 'landmark_2d_106', 'recognition']  # 比对模式
allowed_modules = ['detection', 'landmark_2d_106']                 # 检测模式（不需要 embedding）
```

目标框滞后 1.35 → 0.69 帧，p95 6.9 → 2.35。

### 根因 3：片段截取与写盘阻塞检测线程

`extract_segment()` 会重新打开视频 + seek + 逐帧解码编码，一段 20 秒片段要几百毫秒到数秒，**期间检测完全停摆、目标框卡死**。截图写盘同理。

```python
class _IOWorker:
    """后台串行守护线程：检测线程只管投递任务，立即返回。"""
```

任务投递后即刻继续检测，写盘在后台串行完成。`wait_idle()` 用轮询 `unfinished_tasks` 而非 `queue.join()`，避免无超时死等。
验证：开启片段截取后仍是 23.12 fps / 滞后 0.48 帧，且正确产出 2 个 mp4 + 5 张 png。

### 其他优化

| 项 | 做法 | 效果 |
| --- | --- | --- |
| 显示管线 | 工作线程预缩放到显示尺寸（`DISPLAY_MAX_W=1280`），UI 线程只做小帧→QPixmap；`_disp_scale` 换算框坐标 | 主线程开销与源分辨率解耦，4K 也不卡界面 |
| 脏标记 | `_frame_dirty` 跳过无变化帧的重绘 | 减少无谓 repaint |
| RTSP 环形缓冲 | 宽 ≥960 的帧以 JPEG(q=92) 存储 | **788MB → 14MB（省 98.2%）**，画面平均误差仅 0.503 |
| 质量分计算 | `cv2.Laplacian` 由 `CV_64F` 改 `CV_32F` | 结果一致，耗时约减半 |
| 运动补偿追踪 | 速度 EMA 预测 + `max_miss=6` | 检测间隔内框位置平滑外推，不跳变 |

---

## 八、技术架构

### 线程模型

```
RunnerThread（编排）
├── PlaybackThread   读帧 → 环形缓冲 → 预缩放 → disp_q ──► UI 线程绘制
│                                       └─► det_q（按 balance 抽样）
└── DetectThread     det_q → insightface 推理 → 追踪/匹配 → 结果回传
                                   │
                                   └──► _IOWorker（后台串行：片段编码、PNG 写盘）
```

职责分离保证了**播放永不被检测阻塞**：检测再慢，画面依旧按帧率流畅播放，只是框更新频率下降。

### 关键模块

| 模块 | 职责 |
| --- | --- |
| `detect_runtime()` | 探测 CUDA，组装 EP 与 providers，失败自动降级 CPU |
| `Engine` | insightface 封装：按模式裁剪模型、显式 `det_size`、质量分计算 |
| `PlaybackThread` | 播放与取帧，预缩放，RTSP 环形缓冲，紧跟踪门控 |
| `DetectThread` | 推理、运动补偿追踪、命中判定、截图/片段调度 |
| `RingBuffer` | RTSP 历史帧缓冲（JPEG 压缩存储） |
| `_IOWorker` | 异步串行磁盘/编码任务队列 |
| `RunnerThread` | 模型加载 → 启动检测 → 逐个视频源播放 → 汇总 |

### 中文路径处理

本机 OpenCV 构建的 `imread`/`imwrite` 不支持中文路径。项目内统一封装：

```python
imread_utf8(path)   # np.fromfile + cv2.imdecode
imwrite_utf8(path)  # cv2.imencode + np.tofile
```

`VideoCapture` / `VideoWriter` 传路径正常，仅图片读写需要封装。

---

## 九、环境要求

| 项 | 要求 |
| --- | --- |
| 操作系统 | Windows 10/11（主力）· Linux / macOS（可用 `launch.sh`） |
| Python | 3.9 ~ 3.11（3.12+ 因 insightface 依赖链暂不推荐） |
| 显卡 | NVIDIA（GTX 10 系及以上，显存 ≥4GB）可选；无显卡自动 CPU 运行 |
| CUDA | 12.x（`onnxruntime-gpu==1.22.0` 对应 CUDA 12.4，前向兼容 12.6） |
| 磁盘 | 约 2GB（依赖）+ 输出空间 |

> **CPU 模式能跑，但慢**：720p 素材约 1~3 fps，建议把平衡滑块调到最左（每 4 帧检测）。

---

## 十、打包为单文件 exe

项目内已提供 `build.spec`（PyInstaller onefile，已配置好 CUDA 运行时 DLL、buffalo_l 模型与图标）：

```bash
pip install pyinstaller
pyinstaller build.spec
```

产物位于 `dist/视频人脸检测比对工具.exe`，可脱离 Python 环境运行。

> 打包前请确认 `~/.insightface/models/buffalo_l/` 下模型已就位。若 GitHub 下载慢，
> 可从 ModelScope `cunkai/ComfyUI_Notebook/buffalo_l/` 取 5 个 onnx 文件放入该目录。

---

## 十一、目录结构

```
face-matching-tool/
├── face_clip_qt.py        # 主程序（PyQt5 GUI，当前版本）
├── face_clip_tool.py      # 早期 tkinter 版本，保留供参考/无界面验证脚本调用
├── launch.bat             # Windows 一键启动（自动建环境装依赖）
├── run.bat                # 跳过检查，直接启动
├── diagnose.bat           # 环境自检
├── launch.sh              # Linux/macOS 启动
├── requirements.txt       # 依赖清单（附安装注意事项）
├── build.spec             # PyInstaller 打包配置
├── _make_icon.py          # 图标生成脚本
├── icon.svg / icon.ico / icon.png / icon_*.png
├── validate_*.py          # 流水线验证脚本（无界面）
├── .gitignore
├── LICENSE                # MIT
└── README.md
```

---

## 十二、常见问题

**Q：双击 `launch.bat` 窗口一闪就没了？**
A：通常是 `.bat` 文件被改成 LF 换行导致 `goto` 失效，或 Python 未加入 PATH。项目中的 `.bat` 均已使用 CRLF；若仍闪退，在 cmd 中运行查看报错，或看 `launch.log` / `crash.log`。

**Q：明明有 NVIDIA 显卡却跑在 CPU 上？**
A：① `onnxruntime-gpu` 装成了 latest（需 CUDA 13，会静默回退）—— 必须锁 `1.22.0`；② 直接 `pip install insightface` 会把 CPU 版 onnxruntime 一起装上覆盖 GPU 版 —— 必须用 `--no-deps`。启动日志首行会标明实际后端。

**Q：总漏检 / 误检？**
A：漏检 → 降低相似度阈值（70 → 60）与质量阈值（70 → 55）；误检 → 反向调高。目标人脸照片尽量选正脸、清晰、光线均匀的。

**Q：截图太多？**
A：开启「人脸优选」并适当调大优选时长（3 → 5~10 秒）。

**Q：处理 4K 素材很卡？**
A：把平衡滑块调到 34~66（每 2 帧检测）。4K 的瓶颈在解码与缩放，与检测引擎无关。

---

## 十三、免责声明

本工具为**本地离线**运行的视频处理辅助工具，不上传任何数据。请仅用于处理你拥有合法权利的视频素材，遵守所在地关于个人信息与肖像权、隐私保护的法律法规（如《个人信息保护法》）。因 misuse 造成的后果由使用者自行承担。

## 十四、License

[MIT](./LICENSE) © 2026 huyin324
