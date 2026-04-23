from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_PROMPTS = {
    "HIGH": {
        "text": "警告：发生危险",
        "fill": (210, 32, 32, 224),
        "outline": (255, 225, 225, 255),
    },
    "MEDIUM": {
        "text": "提示：请注意",
        "fill": (232, 132, 24, 216),
        "outline": (255, 239, 210, 255),
    },
}


def draw_chinese_risk_prompt(image: np.ndarray, decision_status: object) -> np.ndarray:
    """Draw a Chinese risk prompt on a BGR OpenCV frame."""

    if image is None:
        return image

    prompt = _PROMPTS.get(str(decision_status or "").upper())
    if prompt is None:
        return image

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return image

    font_size = max(28, min(58, int(height * 0.045)))
    font = load_chinese_font(font_size)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(rgb_image).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    text = prompt["text"]
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad_x = max(20, int(font_size * 0.65))
    pad_y = max(12, int(font_size * 0.35))
    panel_w = min(width - 24, text_w + pad_x * 2)
    panel_h = text_h + pad_y * 2
    x1 = max(12, (width - panel_w) // 2)
    y1 = max(12, int(height * 0.06))
    x2 = min(width - 12, x1 + panel_w)
    y2 = min(height - 12, y1 + panel_h)
    radius = max(10, int(panel_h * 0.18))

    draw.rounded_rectangle(
        (x1, y1, x2, y2),
        radius=radius,
        fill=prompt["fill"],
        outline=prompt["outline"],
        width=max(2, font_size // 12),
    )

    text_x = x1 + max(0, (x2 - x1 - text_w) // 2) - bbox[0]
    text_y = y1 + max(0, (y2 - y1 - text_h) // 2) - bbox[1]
    shadow_offset = max(1, font_size // 18)
    draw.text((text_x + shadow_offset, text_y + shadow_offset), text, font=font, fill=(0, 0, 0, 120))
    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))

    merged = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.asarray(merged), cv2.COLOR_RGB2BGR)


@lru_cache(maxsize=16)
def load_chinese_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in _font_candidates():
        try:
            if font_path and font_path.is_file():
                return ImageFont.truetype(str(font_path), font_size)
        except OSError:
            continue

    return ImageFont.load_default()


def _font_candidates() -> Iterable[Path]:
    env_font = os.environ.get("DRIVESAFE_CHINESE_FONT")
    if env_font:
        yield Path(env_font).expanduser()

    if sys.platform.startswith("win"):
        windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
        fonts = windir / "Fonts"
        names = [
            "msyhbd.ttc",
            "msyh.ttc",
            "simhei.ttf",
            "simsun.ttc",
            "Dengb.ttf",
            "Deng.ttf",
        ]
        for name in names:
            yield fonts / name
    elif sys.platform == "darwin":
        for path in [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Supplemental/Songti.ttc",
            "/System/Library/Fonts/Supplemental/Heiti SC.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]:
            yield Path(path)
    else:
        for path in [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf",
            "/usr/share/fonts/opentype/source-han-sans/SourceHanSansSC-Regular.otf",
            "/usr/share/fonts/opentype/adobe-source-han-sans/SourceHanSansSC-Regular.otf",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
        ]:
            yield Path(path)

    repo_root = Path(__file__).resolve().parents[2]
    yield repo_root / "deep_sort" / "DeepSORT_Monet_traffic" / "simsunttc" / "simsun.ttc"
