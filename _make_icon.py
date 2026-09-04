# 从 icon.svg 生成 icon.png（窗口图标）与 icon.ico（exe 图标）
import os
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtGui import QImage, QPainter, QColor
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
svg = os.path.join(HERE, "icon.svg")
r = QSvgRenderer(svg)
if not r.isValid():
    raise SystemExit("icon.svg 无效")

sizes = [256, 128, 64, 48, 32, 16]
pngs = []
for s in sizes:
    img = QImage(s, s, QImage.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))
    p = QPainter(img)
    r.render(p)
    p.end()
    fn = os.path.join(HERE, f"icon_{s}.png")
    img.save(fn)
    pngs.append((s, fn))

# 合并成多分辨率 .ico
im = Image.open(os.path.join(HERE, "icon_256.png")).convert("RGBA")
im.save(os.path.join(HERE, "icon.ico"),
        sizes=[(s, s) for s in sizes])
# 同时保留一份 256 的 png 供窗口图标使用
im.save(os.path.join(HERE, "icon.png"))
print("icon OK:", [f for _, f in pngs], "icon.ico", "icon.png")
