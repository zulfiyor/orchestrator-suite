# Program: Af Mapping
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge;
#   Xinjiang Institute of Ecology and Geography;
#   National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Translate UI coordinates to EVF pixels."""

from dataclasses import dataclass


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int


def _fit_letterbox(src_w, src_h, dst_w, dst_h):
    s = min(dst_w / src_w, dst_h / src_h)
    draw_w = int(src_w * s)
    draw_h = int(src_h * s)
    off_x = (dst_w - draw_w) // 2
    off_y = (dst_h - draw_h) // 2
    return draw_w, draw_h, off_x, off_y, s


def ui_to_evf_px(
    x_ui,
    y_ui,
    *,
    ui_w,
    ui_h,
    evf_w,
    evf_h,
    orientation=0,
    mirror_x=False,
    mirror_y=False,
    zoom_rect: Rect | None = None,
):
    draw_w, draw_h, off_x, off_y, scale = _fit_letterbox(evf_w, evf_h, ui_w, ui_h)
    # Clip letterbox bars.
    x = max(0, min(x_ui - off_x, draw_w))
    y = max(0, min(y_ui - off_y, draw_h))
    # Map back to EVF pixels.
    x = x / scale
    y = y / scale

    # Apply orientation.
    if orientation == 90:
        x, y = y, evf_w - 1 - x
        evf_w, evf_h = evf_h, evf_w
    elif orientation == 180:
        x, y = evf_w - 1 - x, evf_h - 1 - y
    elif orientation == 270:
        x, y = evf_h - 1 - y, x
        evf_w, evf_h = evf_h, evf_w

    if mirror_x:
        x = evf_w - 1 - x
    if mirror_y:
        y = evf_h - 1 - y

    # Account for zoom mode (when camera magnifies EVF).
    if zoom_rect is not None:
        x = zoom_rect.x + min(max(int(x), 0), zoom_rect.w - 1)
        y = zoom_rect.y + min(max(int(y), 0), zoom_rect.h - 1)
    else:
        x = min(max(int(x), 0), evf_w - 1)
        y = min(max(int(y), 0), evf_h - 1)

    return x, y


# Created by Dr. Z. Bakhtiyorov
