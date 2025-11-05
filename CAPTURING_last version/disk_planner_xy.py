#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Program: CaptuRing Disk Planner XY (GUI+CLI)
Version: 1.2
Author: Dr. Zulfiyor Bakhtiyorov
Institution: University of Cambridge, Xinjiang Institute of Ecology and Geography, National Academy of Sciences of Tajikistan
Year: 2025
License: MIT License

Overview
--------
Enhanced GUI/CLI for planning XY tiles inside a segmented sample polygon.
Key additions vs previous GUI:
- Brush cursor ring now persists reliably (no disappearance after first click)
- Mouse-wheel zoom on canvas
- Sample rotation (degrees) with re-segmentation & re-planning
- Area computation (mm²) alongside frames and grid step
- Movement mode (X-first or Y-first) + serpentine path ordering
- "Start capture" action that locks in ordering and saves rich plan metadata
- Manual mm/px scaling is honored (if no ArUco) for true mm units in GUI

Dependencies: opencv-contrib-python, numpy, pillow, tkinter
"""

import argparse
import json
import math
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

PROGRAM_NAME = "CaptuRing Disk Planner"
VERSION = "1.2"
AUTHOR = "Dr. Zulfiyor Bakhtiyorov"
INSTITUTION = (
    "University of Cambridge; Xinjiang Institute of Ecology and Geography; "
    "National Academy of Sciences of Tajikistan"
)
YEAR = "2025"
LICENSE_TEXT = "MIT License"

import cv2
import numpy as np
from PIL import Image


def load_preview_image(path: Path, max_dim: int = 2048) -> Tuple[Optional[np.ndarray], float]:
    """Decode a lightweight preview without fully loading the original frame."""
    # Read compressed bytes once; decoding happens with downscale hints.
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None, 1.0

    orig_w = orig_h = None
    try:
        with Image.open(path) as im:
            orig_w, orig_h = im.size
    except Exception:
        pass

    reduction = 1
    flag = cv2.IMREAD_COLOR
    if orig_w and orig_h:
        longest = max(orig_w, orig_h)
        while longest / reduction > max_dim and reduction < 8:
            reduction *= 2
        if reduction == 2:
            flag = cv2.IMREAD_REDUCED_COLOR_2
        elif reduction == 4:
            flag = cv2.IMREAD_REDUCED_COLOR_4
        elif reduction >= 8:
            flag = cv2.IMREAD_REDUCED_COLOR_8
            reduction = 8

    img = cv2.imdecode(data, flag)
    if img is None:
        return None, 1.0

    h, w = img.shape[:2]
    longest_decoded = max(h, w)
    scale_extra = 1.0
    if longest_decoded > max_dim and longest_decoded > 0:
        scale_extra = max_dim / float(longest_decoded)
        new_w = max(1, int(round(w * scale_extra)))
        new_h = max(1, int(round(h * scale_extra)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    if orig_w and orig_h and orig_w > 0 and orig_h > 0:
        scale = (1.0 / reduction) * scale_extra
    else:
        # Fallback: rely on decoded size (best effort)
        scale = 1.0
        if orig_w and orig_w > 0:
            scale = w / float(orig_w)
        elif orig_h and orig_h > 0:
            scale = h / float(orig_h)

    return img, scale

# --------------------------- ArUco helpers ---------------------------

def find_aruco_homography(image_bgr, marker_mm: float):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None, None, None
    areas = [cv2.contourArea(c.astype(np.float32)) for c in corners]
    idx = int(np.argmax(areas))
    c = corners[idx].reshape(-1, 2).astype(np.float32)
    src = np.array([c[0], c[1], c[2], c[3]], dtype=np.float32)
    dst = np.array([[0, 0], [marker_mm, 0], [marker_mm, marker_mm], [0, marker_mm]], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if H is None:
        return None, None, None
    img_side_px = (np.linalg.norm(c[1]-c[0]) + np.linalg.norm(c[2]-c[1])) / 2.0
    mm_per_pixel = float(marker_mm / img_side_px) if img_side_px > 0 else None
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, c.astype(np.int32), 255)
    return H, mm_per_pixel, mask

# --------------------------- Rectification ---------------------------

def warp_to_mm_plane(image_bgr, H, out_border_mm=20):
    h, w = image_bgr.shape[:2]
    pts = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32).reshape(-1,1,2)
    pts_mm = cv2.perspectiveTransform(pts, H).reshape(-1,2)
    min_xy = pts_mm.min(axis=0) - out_border_mm
    max_xy = pts_mm.max(axis=0) + out_border_mm
    tx, ty = -min_xy
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float32)
    H2 = T @ H
    out_w = int(math.ceil(max_xy[0]-min_xy[0]))
    out_h = int(math.ceil(max_xy[1]-min_xy[1]))
    rect = cv2.warpPerspective(image_bgr, H2, (out_w, out_h))
    return rect, H2, (tx, ty)

# --------------------------- Segmentation core ---------------------------

def mask_arucos(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None and len(ids) > 0:
        for c in corners:
            cv2.fillConvexPoly(img_bgr, c.astype(np.int32), (255, 255, 255))
    return img_bgr


def refine_with_grabcut(img_bgr, fg_mask_init, bg_mask_init=None, iters=3):
    """Robust GrabCut seeding with safe fallbacks."""
    h, w = img_bgr.shape[:2]
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)

    # Seed background
    if bg_mask_init is not None and np.any(bg_mask_init > 0):
        gc_mask[bg_mask_init > 0] = cv2.GC_BGD
    else:
        frame = 10
        gc_mask[:frame, :] = cv2.GC_BGD
        gc_mask[-frame:, :] = cv2.GC_BGD
        gc_mask[:, :frame] = cv2.GC_BGD
        gc_mask[:, -frame:] = cv2.GC_BGD

    # Seed foreground
    has_fg = fg_mask_init is not None and np.any(fg_mask_init > 0)
    if has_fg:
        gc_mask[fg_mask_init > 0] = cv2.GC_PR_FGD
        dist = cv2.distanceTransform((fg_mask_init > 0).astype(np.uint8), cv2.DIST_L2, 5)
        if np.any(dist > 0):
            thr = np.percentile(dist[dist > 0], 70)
            if thr > 0:
                gc_mask[dist > thr] = cv2.GC_FGD
    else:
        cx, cy = w // 2, h // 2
        r = max(3, min(h, w) // 20)
        cv2.circle(gc_mask, (cx, cy), r, cv2.GC_FGD, -1)

    # Ensure classes present
    if not np.any((gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD)):
        r = max(2, min(h, w) // 25)
        cv2.rectangle(gc_mask, (1, 1), (w - 2, h - 2), cv2.GC_BGD, thickness=2*r)
    if not np.any((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)):
        cx, cy = w // 2, h // 2
        cv2.circle(gc_mask, (cx, cy), max(3, min(h, w)//30), cv2.GC_FGD, -1)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, gc_mask, (1, 1, w - 2, h - 2), bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_MASK)
        out = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        out = (fg_mask_init > 0).astype(np.uint8) * 255 if fg_mask_init is not None else np.zeros((h, w), np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=1)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)
    return out


def compute_fg_hsv(rect_bgr, border_px=40, k_sigma=2.5, use_grabcut=True):
    """HSV border-based segmentation returning a FG mask."""
    img = rect_bgr.copy()
    img = mask_arucos(img)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S = hsv[:,:,1].astype(np.float32)
    V = hsv[:,:,2].astype(np.float32)

    b = max(10, int(border_px))
    border_mask = np.zeros((h,w), np.uint8)
    border_mask[:b,:] = 1; border_mask[-b:,:] = 1; border_mask[:, :b] = 1; border_mask[:, -b:] = 1

    Sb = S[border_mask==1]; Vb = V[border_mask==1]
    S_med, V_med = np.median(Sb), np.median(Vb)
    S_std = max(5.0, float(np.std(Sb)))
    V_std = max(5.0, float(np.std(Vb)))

    bg = (np.abs(S - S_med) <= k_sigma*S_std) & (np.abs(V - V_med) <= k_sigma*V_std)
    bg = (bg.astype(np.uint8) * 255)
    fg = cv2.bitwise_not(bg)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kern, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kern, iterations=2)

    if use_grabcut:
        fg = refine_with_grabcut(img, fg, (border_mask*255).astype(np.uint8), iters=3)
    return fg


def contour_from_mask(fg: np.ndarray):
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, 0.5, True)
    if cv2.contourArea(approx) < 500:
        return None
    return approx.reshape(-1,2).astype(np.float32)


def extract_disk_contour(rect_bgr, border_px=40, k_sigma=2.5, use_grabcut=True):
    fg = compute_fg_hsv(rect_bgr, border_px, k_sigma, use_grabcut)
    return contour_from_mask(fg)

# --------------------------- Tiles & exports ---------------------------

def generate_tiles_inside_polygon(poly_mm: np.ndarray, tile_mm: float, overlap: float):
    x_min, y_min = poly_mm.min(axis=0)
    x_max, y_max = poly_mm.max(axis=0)
    step = max(1.0, tile_mm * (1.0 - overlap))
    contour = poly_mm.reshape(-1,1,2).astype(np.float32)
    pts = []
    y = y_min + step/2
    while y <= y_max - step/2 + 1e-6:
        x = x_min + step/2
        while x <= x_max - step/2 + 1e-6:
            if cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0:
                pts.append((float(x), float(y)))
            x += step
        y += step
    return pts


def _infer_primary_step(tiles, axis_first: str):
    """Best-effort estimate of spacing between ordered rows/columns."""
    if len(tiles) < 2:
        return None
    axis = 1 if axis_first == 'Y-first' else 0
    coords = np.sort(np.array([float(p[axis]) for p in tiles], dtype=np.float64))
    diffs = np.diff(coords)
    diffs = diffs[np.abs(diffs) > 1e-3]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def group_ordered_tiles(tiles, axis_first: str, step_mm: Optional[float]):
    """Group already ordered tiles into primary-axis bands (rows/columns)."""
    rows = []
    if not tiles:
        return rows
    if step_mm is None or step_mm <= 0:
        step_mm = _infer_primary_step(tiles, axis_first)
    thr = max((step_mm or 0) * 0.5, 1e-3)
    current = [tiles[0]]
    ref = tiles[0][1] if axis_first == 'Y-first' else tiles[0][0]
    for (x, y) in tiles[1:]:
        key = y if axis_first == 'Y-first' else x
        if abs(key - ref) <= thr:
            current.append((x, y))
        else:
            rows.append(current)
            current = [(x, y)]
            ref = key
    rows.append(current)
    return rows


def order_tiles_raster(tiles, axis_first: str, serpentine: bool, step_mm: float):
    """Return raster-ordered tiles. axis_first in {"X-first","Y-first"}."""
    if not tiles:
        return []
    pts = np.array(tiles, dtype=np.float32)
    xs, ys = pts[:,0], pts[:,1]
    if axis_first == 'Y-first':
        bands = np.floor((ys - ys.min()) / max(step_mm, 1e-6)).astype(int)
        ordered = []
        for bi in np.unique(bands):
            idx = np.where(bands == bi)[0]
            # sort secondary axis (X)
            order = np.argsort(xs[idx])
            row = idx[order].tolist()
            if serpentine and (bi % 2 == 1):
                row = row[::-1]
            ordered.extend(row)
        return [tiles[i] for i in ordered]
    else:  # X-first
        bands = np.floor((xs - xs.min()) / max(step_mm, 1e-6)).astype(int)
        ordered = []
        for bi in np.unique(bands):
            idx = np.where(bands == bi)[0]
            order = np.argsort(ys[idx])
            col = idx[order].tolist()
            if serpentine and (bi % 2 == 1):
                col = col[::-1]
            ordered.extend(col)
        return [tiles[i] for i in ordered]


def rotate_points(pts, angle_deg, center):
    """Rotate list of (x,y) points by angle around center in image coords."""
    if not pts:
        return []
    ang = np.deg2rad(angle_deg)
    c, s = np.cos(ang), np.sin(ang)
    cx, cy = center
    out = []
    for (x,y) in pts:
        xr = c*(x-cx) - s*(y-cy) + cx
        yr = s*(x-cx) + c*(y-cy) + cy
        out.append((float(xr), float(yr)))
    return out


def save_overlay(rect_bgr, poly_mm, tiles, out_png):
    vis = rect_bgr.copy()
    cv2.polylines(vis, [poly_mm.reshape(-1,1,2).astype(np.int32)], True, (0,255,0), 2)
    for (x,y) in tiles:
        cv2.circle(vis, (int(round(x)), int(round(y))), 3, (255,0,0), -1)
    cv2.imwrite(str(out_png), vis)


def save_tiles_csv(tiles, out_csv):
    import csv
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["id","X_mm","Y_mm"]) 
        for i,(x,y) in enumerate(tiles):
            w.writerow([i, f"{x:.2f}", f"{y:.2f}"])


def save_polygon_json(poly_mm, out_json):
    with open(out_json, 'w') as f:
        json.dump({"polygon_mm": poly_mm.tolist()}, f, indent=2)


def save_gcode(tiles, safe_z, travel_f, z_f, settle_ms, out_gcode,
               manual_y=False, axis_first='Y-first', step_mm=None):
    lines = []
    lines.append("; route.gcode generated by disk_planner_xy.py")
    lines.append("G21 ; mm")
    if manual_y:
        lines.append("G90 ; absolute")
        lines.append("G28 X Z ; home only X and Z for safety (manual Y)")  # do not home Y
    else:
        lines.append("G90 ; absolute")
        lines.append("G28 ; home all")

    lines.append(f"G0 Z{safe_z:.2f} F{z_f}")

    if manual_y:
        rows = group_ordered_tiles(tiles, axis_first, step_mm)

        for ri, row in enumerate(rows):
            # Row header + operator prompt
            if axis_first == 'Y-first':
                y_med = np.median([p[1] for p in row])
                lines.append(f"; ---- Row {ri} (target Y≈{y_med:.2f} mm) ----")
                lines.append("M300 S880 P150 ; beep")
                lines.append(f"M0 Move Y to {y_med:.2f} mm, then press to continue")
            else:
                x_med = np.median([p[0] for p in row])
                lines.append(f"; ---- Column {ri} (target X≈{x_med:.2f} mm) ----")
                lines.append("M300 S880 P150 ; beep")
                lines.append(f"M0 Move X to {x_med:.2f} mm, then press to continue")

            # Traverse the row: X-only moves (avoid Y commands)
            for i, (x, y) in enumerate(row):
                lines.append(f"; tile row{ri}_{i}")
                lines.append(f"G0 X{float(x):.2f} F{travel_f}")
                lines.append(f"G4 P{int(settle_ms)} ; settle")
                lines.append("; trigger shutter here if needed")
                lines.append("G4 P200 ; dwell")
    else:
        # Normal XY mode: move both X and Y
        for i, (x, y) in enumerate(tiles):
            lines.append(f"; tile {i}")
            lines.append(f"G0 X{float(x):.2f} Y{float(y):.2f} F{travel_f}")
            lines.append(f"G4 P{int(settle_ms)} ; settle")
            lines.append("; trigger shutter here if needed")
            lines.append("G4 P200 ; dwell")

    lines.append(f"G0 Z{safe_z:.2f} F{z_f}")
    lines.append("M300 S440 P200 ; beep")
    Path(out_gcode).write_text("\n".join(lines), encoding='utf-8')

# --------------------------- GUI (Tk) ---------------------------
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk
    HAS_TK = True
except Exception:
    HAS_TK = False


class App:
    def __init__(self, root):
        self.root = root
        root.title(f"{PROGRAM_NAME} (GUI)")
        root.geometry("1480x940")

        # I/O & data
        self.img_path = None
        self.output_dir = Path("out_xy")
        self.img_bgr = None
        self.img_scale = 1.0
        self.rect_bgr_base = None  # un-rotated mm-plane image
        self.rect_bgr = None       # rotated working image
        self.fg_mask = None
        self.auto_fg = None
        self.poly = None
        self.tiles = []

        # Background refinement state
        self._refining = False
        self._last_error = None

        # Units & scaling
        self.mm_per_px = None  # if ArUco: derived; if manual: user
        self.mm_source = 'unknown'

        # Start picking
        self.await_start_pick = False
        self.start_xy = None

        # User edits
        self.user_add = None
        self.user_erase = None

        # Display state
        self._disp_scale = 1.0
        self._base_fit_scale = 1.0
        self._imgtk = None
        self._img_id = None
        self._cursor_id = None
        self._last_mouse = (None, None)
        self._hovering = False

        # Stats
        self.area_mm2 = None

        # Params (Tk variables)
        self.border_px = tk.IntVar(value=35)
        self.k_sigma = tk.DoubleVar(value=2.5)
        self.bg_shave = tk.IntVar(value=0)
        self.tile_mm = tk.IntVar(value=10)
        self.overlap_pct = tk.IntVar(value=20)
        self.show_mask = tk.BooleanVar(value=False)
        self.use_grabcut = tk.BooleanVar(value=False)
        self.brush_mode = tk.StringVar(value='Add')
        self.brush_size = tk.IntVar(value=10)
        self.aruco_mm = tk.DoubleVar(value=30.0)
        self.mmpp_manual = tk.DoubleVar(value=0.0)
        self.rot_deg = tk.DoubleVar(value=0.0)
        self.zoom = tk.DoubleVar(value=1.0)
        self.show_tiles = tk.BooleanVar(value=True)
        self.movement_axis = tk.StringVar(value='X-first')
        self.serpentine = tk.BooleanVar(value=True)
        self.manual_y = tk.BooleanVar(value=True)  # operator moves Y manually

        # Press-and-hold helpers
        self._repeat_job = None
        self._repeat_active = False
        # repaint/debounce helpers
        self._preview_job = None
        self._debounce_jobs = {}

        
        self.build_ui()

    # ---- UI ----
    def build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)
        
        # --- Scrollable LEFT panel (fixed width) ---
        left_container = ttk.Frame(main, width=360)
        left_container.pack(side=tk.LEFT, fill=tk.BOTH)
        left_container.pack_propagate(False)  # keep fixed width, do not shrink to children
        
        # Canvas + vertical scrollbar
        left_canvas = tk.Canvas(left_container, highlightthickness=0)
        left_vsb = ttk.Scrollbar(left_container, orient='vertical', command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_vsb.set)
        
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Inner frame that actually holds all left controls
        left_inner = ttk.Frame(left_canvas)
        left_window = left_canvas.create_window((0, 0), window=left_inner, anchor='nw')
        

        # --- after creating left_container, left_canvas, left_inner, left_window ---
        
        def _on_left_frame_configure(_):
            """Update scrollregion to match inner frame size."""
            left_canvas.configure(scrollregion=left_canvas.bbox('all'))
        left_inner.bind('<Configure>', _on_left_frame_configure)
        
        def _on_left_canvas_configure(e):
            """Ensure inner frame width tracks canvas visible width."""
            left_canvas.itemconfig(left_window, width=e.width)
        left_canvas.bind('<Configure>', _on_left_canvas_configure)
        
        # ROUTER: send wheel to left_canvas only if pointer is inside left_container
        def _wheel_router(e):
            """Global wheel router: scroll left panel if pointer is inside it."""
            x, y = self.root.winfo_pointerx(), self.root.winfo_pointery()
            lx, ly = left_container.winfo_rootx(), left_container.winfo_rooty()
            lw, lh = left_container.winfo_width(), left_container.winfo_height()
            inside_left = (lx <= x <= lx + lw) and (ly <= y <= ly + lh)
        
            if not inside_left:
                # Let other widgets (e.g., main image canvas) handle the wheel
                return
        
            # Normalize delta for Windows/macOS; handle Linux buttons too
            delta = getattr(e, 'delta', 0)
            if delta == 0:
                # X11: Button-4 (up), Button-5 (down)
                num = getattr(e, 'num', None)
                delta = 120 if num == 4 else -120
        
            left_canvas.yview_scroll(-1 if delta > 0 else 1, 'units')
            return 'break'
        
        # Bind globally so entries/combos don't swallow the wheel events
        self.root.bind_all('<MouseWheel>', _wheel_router)   # Windows/macOS
        self.root.bind_all('<Button-4>', _wheel_router)     # Linux up
        self.root.bind_all('<Button-5>', _wheel_router)     # Linux down
                
        # Mouse wheel scrolling for Windows/macOS/Linux
        def _on_left_mousewheel(e):
            """Scroll with wheel on Windows/macOS."""
            if e.delta:
                left_canvas.yview_scroll(-1 if e.delta > 0 else 1, 'units')
            return 'break'
        
        def _on_left_wheel_linux_up(e):
            left_canvas.yview_scroll(-1, 'units')
            return 'break'
        
        def _on_left_wheel_linux_down(e):
            left_canvas.yview_scroll(1, 'units')
            return 'break'
        
        # Bind wheel events to the container, canvas, and inner frame
        for w in (left_container, left_canvas, left_inner):
            w.bind('<MouseWheel>', _on_left_mousewheel)     # Windows / macOS
            w.bind('<Button-4>', _on_left_wheel_linux_up)   # Linux
            w.bind('<Button-5>', _on_left_wheel_linux_down) # Linux
        
        # IMPORTANT: keep the variable name 'left' to avoid changing the rest of build_ui()
        left = left_inner
        
        # --- RIGHT panel with the main image canvas ---
        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(right, bg="#111")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # Ensure initial scrollregion is correct even if window starts small
        self.root.after(0, lambda: left_canvas.configure(scrollregion=left_canvas.bbox('all')))

        # Zoom bar
        zb = ttk.Frame(right); zb.pack(fill=tk.X)
        ttk.Button(zb, text="Zoom -", command=lambda: self.set_zoom(max(0.1, self.zoom.get()*0.8))).pack(side=tk.LEFT)
        ttk.Button(zb, text="Zoom +", command=lambda: self.set_zoom(min(8.0, self.zoom.get()*1.25))).pack(side=tk.LEFT, padx=4)
        ttk.Button(zb, text="Fit", command=lambda: self.set_zoom(1.0, fit=True)).pack(side=tk.LEFT)

        # Mouse events
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_move)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canvas.bind('<Motion>', self.on_hover)
        self.canvas.bind('<Enter>', lambda e: setattr(self, '_hovering', True))
        self.canvas.bind('<Leave>', self.on_leave_canvas)
        # Wheel zoom (Windows/macOS)
        self.canvas.bind('<MouseWheel>', self.on_wheel)
        # Wheel zoom (X11/Linux)
        self.canvas.bind('<Button-4>', lambda e: self.on_wheel_linux(1))
        self.canvas.bind('<Button-5>', lambda e: self.on_wheel_linux(-1))

        # --- Left panel ---
        ttk.Label(left, text="1) Image & Output").pack(anchor='w', pady=(8,2))
        ttk.Button(left, text="Open Image…", command=self.open_image).pack(fill=tk.X)
        self.path_lbl = ttk.Label(left, text="—"); self.path_lbl.pack(anchor='w')
        ttk.Button(left, text="Choose Output Folder…", command=self.choose_out).pack(fill=tk.X, pady=(4,0))
        self.out_lbl = ttk.Label(left, text=str(self.output_dir)); self.out_lbl.pack(anchor='w')

        ttk.Separator(left).pack(fill=tk.X, pady=6)
        ttk.Label(left, text="2) Scale (mm)").pack(anchor='w')
        r = ttk.Frame(left); r.pack(fill=tk.X)
        ttk.Label(r, text="ArUco mm:").pack(side=tk.LEFT)
        tk.Entry(r, textvariable=self.aruco_mm, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(left, text="or mm/px (manual):").pack(anchor='w')
        tk.Entry(left, textvariable=self.mmpp_manual, width=10).pack(anchor='w')
        ttk.Button(left, text="Apply scale", command=self.prepare_rectified).pack(fill=tk.X, pady=(4,0))

        ttk.Separator(left).pack(fill=tk.X, pady=6)
        ttk.Label(left, text="3) Rotate").pack(anchor='w')
        self.mk_spin(left, "Rotate (deg)", self.rot_deg, -180.0, 180.0, step=0.5)

        ttk.Separator(left).pack(fill=tk.X, pady=6)
        ttk.Label(left, text="4) Segmentation").pack(anchor='w')
        self.mk_spin(left, "Border(px)", self.border_px, 10, 120, step=1)
        self.mk_spin(left, "k_sigma", self.k_sigma, 0.10, 10.0, step=0.01)
        self.mk_spin(left, "BG shave(px)", self.bg_shave, 0, 50, step=1)
        ttk.Checkbutton(left, text="Show mask", variable=self.show_mask, command=self.update_preview).pack(anchor='w')
        ttk.Checkbutton(left, text="Use GrabCut", variable=self.use_grabcut, command=self.segment).pack(anchor='w')
        tk.Label(
            left,
            text=(
                "⚠️ Continuous GrabCut is heavy and may freeze the app. "
                "Prefer the one-shot Refine button unless absolutely needed."
            ),
            fg="#aa5500",
            wraplength=220,
            justify="left",
        ).pack(anchor='w', padx=2, pady=(2, 6))
        ttk.Button(left, text="Auto Segment", command=self.segment).pack(fill=tk.X, pady=(6,0))

        ttk.Separator(left).pack(fill=tk.X, pady=6)
        ttk.Label(left, text="5) Brush (live)").pack(anchor='w')
        row = ttk.Frame(left); row.pack(fill=tk.X)
        ttk.Button(row, text="Brush: Add", command=lambda: self.brush_mode.set('Add')).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(row, text="Brush: Erase", command=lambda: self.brush_mode.set('Erase')).pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.mk_spin(left, "Brush size", self.brush_size, 1, 200, step=1)
        row2 = ttk.Frame(left); row2.pack(fill=tk.X)
        ttk.Button(row2, text="Apply edits", command=self.apply_edits).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(row2, text="Reset edits", command=self.reset_edits).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4,0))

        ttk.Separator(left).pack(fill=tk.X, pady=6)
        ttk.Label(left, text="6) Tiles").pack(anchor='w')
        self.mk_spin(left, "Tile(mm)", self.tile_mm, 5, 60, step=1)
        self.mk_spin(left, "Overlap(%)", self.overlap_pct, 0, 50, step=1)
        ttk.Checkbutton(left, text="Show tiles (blue dots)", variable=self.show_tiles, command=self.update_preview).pack(anchor='w')
        rowp = ttk.Frame(left); rowp.pack(fill=tk.X)
        ttk.Button(rowp, text="Pick start (click image)", command=self.begin_pick_start).pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.start_lbl = ttk.Label(left, text="Start: —"); self.start_lbl.pack(anchor='w')
        ttk.Button(left, text="Show stats", command=self.show_tiles_stats).pack(fill=tk.X)
        self.tiles_lbl = ttk.Label(left, text="Frames: 0"); self.tiles_lbl.pack(anchor='w', pady=(4,0))
        self.area_lbl = ttk.Label(left, text="Area: —"); self.area_lbl.pack(anchor='w')

        ttk.Separator(left).pack(fill=tk.X, pady=6)
        ttk.Label(left, text="7) Movement").pack(anchor='w')
        mv = ttk.Frame(left); mv.pack(fill=tk.X)
        ttk.Label(mv, text="Axis:").pack(side=tk.LEFT)
        axis_cb = ttk.Combobox(mv, values=['Y-first','X-first'], textvariable=self.movement_axis, state='readonly', width=8)
        axis_cb.pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(left, text="Serpentine", variable=self.serpentine, command=self.plan_tiles).pack(anchor='w')
        ttk.Checkbutton(left, text="Manual Y mode (pause per row)", variable=self.manual_y).pack(anchor='w')
        ttk.Button(left, text="Start capture (save plan)", command=self.start_capture).pack(fill=tk.X, pady=(6,0))

        ttk.Separator(left).pack(fill=tk.X, pady=6)
        ttk.Button(left, text="Export Overlay + CSV + G-code", command=self.export_all).pack(fill=tk.X)

        ttk.Button(left, text="Refine mask (GrabCut once)",
                   command=self.refine_mask_once
        ).pack(fill=tk.X)

        # Footer with developer info and About/Help actions
        footer = ttk.Frame(self.root)
        footer.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(6, 12))
        dev_label = ttk.Label(footer, text=f"Developed by {AUTHOR}", font=("TkDefaultFont", 10, "bold"))
        dev_label.pack(side=tk.LEFT)
        buttons = ttk.Frame(footer)
        buttons.pack(side=tk.RIGHT)
        ttk.Button(buttons, text="About", command=self.show_about).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(buttons, text="Help", command=self.show_help).pack(side=tk.LEFT)

    def show_about(self):
        about_text = (
            f"{PROGRAM_NAME} v{VERSION}\n"
            f"Author: {AUTHOR}\n"
            f"Institution: {INSTITUTION}\n"
            f"Year: {YEAR}\n"
            f"License: {LICENSE_TEXT}"
        )
        messagebox.showinfo("About", about_text)

    def show_help(self):
        help_text = (
            "CaptuRing Disk Planner — quick guide:\n\n"
            "1) Open Image… to choose the disk photograph.\n"
            "2) Apply scale via ArUco marker or manual mm/px.\n"
            "3) Rotate and segment the disk outline.\n"
            "4) Use the brush tools to refine the mask if needed.\n"
            "5) Configure tile size, overlap, and movement ordering.\n"
            "6) Pick the starting tile and export or start capture."
        )
        messagebox.showinfo("Help", help_text)

    def mk_spin(self, parent, text, var, a, b, step=1):
        frame = ttk.Frame(parent); frame.pack(fill=tk.X)
        ttk.Label(frame, text=text).pack(anchor='w')
        row = ttk.Frame(frame); row.pack(fill=tk.X)

        def inc():
            try:
                v = float(var.get()) + step
                if isinstance(var, tk.IntVar):
                    v = int(round(v))
                var.set(min(b, max(a, v)))
            except Exception:
                pass

        def dec():
            try:
                v = float(var.get()) - step
                if isinstance(var, tk.IntVar):
                    v = int(round(v))
                var.set(min(b, max(a, v)))
            except Exception:
                pass

        dec_btn = ttk.Button(row, text='−', width=3)
        dec_btn.pack(side=tk.LEFT)
        entry = tk.Entry(row, textvariable=var, width=10)
        entry.pack(side=tk.LEFT, padx=4)
        inc_btn = ttk.Button(row, text='+', width=3)
        inc_btn.pack(side=tk.LEFT)
        ttk.Label(row, text=f"range [{a}..{b}]").pack(side=tk.LEFT, padx=6)

        def on_return(_):
            self.on_spin(var)
        entry.bind('<Return>', on_return)
        entry.bind('<FocusOut>', on_return)

        var.trace_add('write', lambda *_: self._debounce_spin(var, wait=120))


        # press-and-hold
        def start_repeat(fn):
            fn()
            self._repeat_active = True
            def step_call():
                if not self._repeat_active:
                    return
                fn()
                self._repeat_job = self.root.after(60, step_call)
            self._repeat_job = self.root.after(300, step_call)
        def stop_repeat(event=None):
            self._repeat_active = False
            if self._repeat_job is not None:
                try:
                    self.root.after_cancel(self._repeat_job)
                except Exception:
                    pass
                self._repeat_job = None
        dec_btn.bind('<ButtonPress-1>', lambda e: start_repeat(dec))
        dec_btn.bind('<ButtonRelease-1>', stop_repeat)
        dec_btn.bind('<Leave>', stop_repeat)
        inc_btn.bind('<ButtonPress-1>', lambda e: start_repeat(inc))
        inc_btn.bind('<ButtonRelease-1>', stop_repeat)
        inc_btn.bind('<Leave>', stop_repeat)

    # ---- I/O ----
    def open_image(self):
        p = filedialog.askopenfilename(title="Open image", filetypes=[("Images","*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")])
        if not p: return
        self.img_path = Path(p)
        self.img_bgr, self.img_scale = load_preview_image(self.img_path)
        if self.img_bgr is None:
            messagebox.showerror("Error", "Cannot read image")
            return
        self.path_lbl.config(text=self.img_path.name)
        self.prepare_rectified()
        self.reset_edits()
        self.segment()

    def choose_out(self):
        p = filedialog.askdirectory(title="Choose output folder")
        if p:
            self.output_dir = Path(p)
            self.out_lbl.config(text=str(self.output_dir))

    def _rectify_with_aruco_or_manual(self):
        """Return mm-plane image, mm_per_px and source string."""
        if self.img_bgr is None:
            return None, None, 'unknown'
        # Prefer ArUco
        H, mmpp, _ = find_aruco_homography(self.img_bgr, float(self.aruco_mm.get()))
        if H is not None:
            rect, _, _ = warp_to_mm_plane(self.img_bgr, H, out_border_mm=20)
            return rect, 1.0, 'aruco'  # pixel coords == mm in rectified image
        # Manual mm/px
        mmpp_user = float(self.mmpp_manual.get())
        if mmpp_user > 0:
            scale = getattr(self, 'img_scale', 1.0) or 1.0
            mmpp_eff = mmpp_user / scale
            Hm = np.array([[mmpp_eff, 0, 0],[0, mmpp_eff, 0],[0, 0, 1]], dtype=np.float32)
            rect, _, _ = warp_to_mm_plane(self.img_bgr, Hm, out_border_mm=20)
            return rect, 1.0, 'manual'
        # Fallback: no scaling (units are pixels)
        return self.img_bgr.copy(), None, 'unknown'

    def prepare_rectified(self):
        self.rect_bgr_base, mmpp_eff, src = self._rectify_with_aruco_or_manual()
        self.mm_source = src
        self.mm_per_px = mmpp_eff  # 1.0 when in mm-plane; None otherwise
        # Reset rotation to 0 after re-rectification
        self.rot_deg.set(0.0)
        self.apply_rotation()

    # ---- Segmentation & tiles ----
    def ensure_masks(self):
        H, W = self.rect_bgr.shape[:2]
        self.user_add = np.zeros((H,W), np.uint8)
        self.user_erase = np.zeros((H,W), np.uint8)

    def segment(self):
        if self.rect_bgr is None:
            return
        if getattr(self, "_refining", False):
            return
        auto = compute_fg_hsv(
            self.rect_bgr,
            border_px=int(self.border_px.get()),
            k_sigma=float(self.k_sigma.get()),
            use_grabcut=bool(self.use_grabcut.get())
        )
        shave = int(self.bg_shave.get())
        if shave > 0:
            auto[:shave, :] = 0; auto[-shave:, :] = 0; auto[:, :shave] = 0; auto[:, -shave:] = 0
        self.auto_fg = auto
        self.fg_mask = self.compose_fg()
        self.poly = contour_from_mask(self.fg_mask)
        self._update_area_from_poly()
        self.plan_tiles()
        self._schedule_preview(33)  # ~30 FPS coalesced redraw while painting

    def _update_area_from_poly(self):
        if self.poly is None:
            self.area_mm2 = None
            return
        # In rectified image, 1 pixel == 1 mm when ArUco or manual scale applied
        if self.mm_source in ('aruco','manual'):
            self.area_mm2 = float(cv2.contourArea(self.poly))
        else:
            self.area_mm2 = None

    def _rows_from_ordered(self, tiles, step_mm):
        """Group the already ordered tiles into rows (bands) for manual-Y workflow."""
        return group_ordered_tiles(tiles, self.movement_axis.get(), step_mm)

    def plan_tiles(self):
        if self.poly is None:
            self.tiles = []
        else:
            step_mm = float(self.tile_mm.get()) * (1.0 - float(self.overlap_pct.get())/100.0)
            raw = generate_tiles_inside_polygon(self.poly, float(self.tile_mm.get()), float(self.overlap_pct.get())/100.0)
            # Movement raster ordering
            ordered = order_tiles_raster(
                raw,
                axis_first=self.movement_axis.get(),
                serpentine=bool(self.serpentine.get()),
                step_mm=step_mm
            )
            # If start chosen, rotate list so nearest is first (preserve raster path)
            if self.start_xy is not None and ordered:
                pts = np.array(ordered, dtype=np.float32)
                d2 = (pts[:,0]-self.start_xy[0])**2 + (pts[:,1]-self.start_xy[1])**2
                i0 = int(np.argmin(d2))
                ordered = ordered[i0:] + ordered[:i0]
                # sync start exactly to first tile
                self.start_xy = (ordered[0][0], ordered[0][1])
            self.tiles = ordered
        self.update_tiles_label()
        self.update_preview()

    def start_capture(self):
        """Lock-in ordering under current movement settings and save a rich plan."""
        if self.rect_bgr is None or self.poly is None or not self.tiles:
            messagebox.showwarning("Start capture", "Nothing planned yet.")
            return
        # Recompute to ensure up-to-date ordering
        self.plan_tiles()
        # Save plan metadata (same location as export_all)
        out_root = self.output_dir
        stem = self.img_path.stem if self.img_path else 'output'
        sub = stem[-6:] if len(stem) >= 6 else stem
        out = out_root / sub
        out.mkdir(parents=True, exist_ok=True)
        self._save_meta(out/"meta.json")
        messagebox.showinfo("Start capture", f"Plan locked and meta saved to:\n{out}")

    def _save_meta(self, path):
        if self.poly is None:
            bbox = None
        else:
            bbox = (
                float(self.poly[:,0].min()), float(self.poly[:,1].min()),
                float(self.poly[:,0].max()), float(self.poly[:,1].max())
            )
        step_mm = float(self.tile_mm.get()) * (1.0 - float(self.overlap_pct.get())/100.0)
        rows = self._rows_from_ordered(self.tiles, step_mm)
        row_info = []
        for i, r in enumerate(rows):
            # Use median Y (or X) as row value for robustness
            if self.movement_axis.get() == 'Y-first':
                y_med = float(np.median([p[1] for p in r]))
                row_info.append({'index': i, 'y_mm': y_med, 'count': len(r)})
            else:
                x_med = float(np.median([p[0] for p in r]))
                row_info.append({'index': i, 'x_mm': x_med, 'count': len(r)})
        meta = {
            'tiles': len(self.tiles),
            'tile_mm': float(self.tile_mm.get()),
            'overlap_pct': float(self.overlap_pct.get()),
            'grid_step_mm': step_mm,
            'area_mm2': (float(self.area_mm2) if self.area_mm2 is not None else None),
            'bbox_mm': bbox,
            'rotation_deg': float(self.rot_deg.get()),
            'movement_axis': self.movement_axis.get(),
            'serpentine': bool(self.serpentine.get()),
            'manual_y_mode': bool(self.manual_y.get()),
            'row_count': len(row_info),
            'rows': row_info,
            'mm_units_source': self.mm_source,
            'start_xy': self.start_xy,
            'output_root': str(self.output_dir),
            'image_stem': (self.img_path.stem if self.img_path else None),
            'path': [{'id': i, 'x': float(x), 'y': float(y)} for i,(x,y) in enumerate(self.tiles)]
        }
        Path(path).write_text(json.dumps(meta, indent=2), encoding='utf-8')

    def export_all(self):
        if self.rect_bgr is None or self.poly is None or not self.tiles:
            messagebox.showwarning("Export", "Nothing to export yet.")
            return
        out_root = self.output_dir
        stem = self.img_path.stem if self.img_path else 'output'
        sub = stem[-6:] if len(stem) >= 6 else stem
        out = out_root / sub
        out.mkdir(parents=True, exist_ok=True)
        save_overlay(self.rect_bgr, self.poly, self.tiles, out/"overlay.png")
        save_tiles_csv(self.tiles, out/"tiles.csv")
        save_polygon_json(self.poly, out/"polygon_mm.json")
        step_mm = float(self.tile_mm.get()) * (1.0 - float(self.overlap_pct.get())/100.0)
        save_gcode(self.tiles, 40, 9000, 1200, 250, out/"route.gcode",
                   manual_y=bool(self.manual_y.get()),
                   axis_first=self.movement_axis.get(),
                   step_mm=step_mm)
        self._save_meta(out/"meta.json")
        messagebox.showinfo("Export", f"Saved to\n{out}")

    # ---- Canvas, zoom, brush, rotation ----
    def begin_pick_start(self):
        if not self.tiles:
            messagebox.showinfo("Pick start", "Tiles are not planned yet.")
            return
        self.await_start_pick = True
        messagebox.showinfo("Pick start", "Click on the image near the camera position. We'll start from the nearest tile.")

    def set_start_from_xy(self, xi, yi):
        if not self.tiles:
            return
        pts = np.array(self.tiles, dtype=np.float32)
        d2 = (pts[:,0]-xi)**2 + (pts[:,1]-yi)**2
        i0 = int(np.argmin(d2))
        self.start_xy = (float(pts[i0,0]), float(pts[i0,1]))
        # rotate list instead of greedy NN to preserve rasterization
        self.tiles = self.tiles[i0:] + self.tiles[:i0]
        self.update_tiles_label()
        self.update_preview()

    def update_preview(self):
        if self.rect_bgr is None: return
        vis = self.rect_bgr.copy()
        if self.fg_mask is not None and self.show_mask.get():
            mask_rgb = cv2.cvtColor(self.fg_mask, cv2.COLOR_GRAY2BGR)
            vis = (0.6*vis + 0.4*mask_rgb).astype(np.uint8)
        if self.poly is not None:
            cv2.polylines(vis, [self.poly.reshape(-1,1,2).astype(np.int32)], True, (0,255,0), 2)
            if self.show_tiles.get():
                for (x,y) in self.tiles:
                    cv2.circle(vis, (int(round(x)), int(round(y))), 3, (255,0,0), -1)
        if self.start_xy is not None:
            x,y = self.start_xy
            cv2.circle(vis, (int(round(x)), int(round(y))), 7, (255,0,255), 2)
            cv2.putText(vis, 'START', (int(round(x))+8, int(round(y))-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1, cv2.LINE_AA)
        if self.user_add is not None:
            add_rgb = cv2.cvtColor(self.user_add, cv2.COLOR_GRAY2BGR)
            vis[add_rgb[:,:,0]>0] = (vis[add_rgb[:,:,0]>0] * 0.6 + np.array([0,255,0])*0.4).astype(np.uint8)
        if self.user_erase is not None:
            er_rgb = cv2.cvtColor(self.user_erase, cv2.COLOR_GRAY2BGR)
            vis[er_rgb[:,:,0]>0] = (vis[er_rgb[:,:,0]>0] * 0.6 + np.array([255,0,0])*0.4).astype(np.uint8)
        self.show_image_on_canvas(vis)

    def show_image_on_canvas(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
    
        cW = self.canvas.winfo_width() or 1
        cH = self.canvas.winfo_height() or 1
        self._base_fit_scale = min(cW / im.width, cH / im.height)
        self._base_fit_scale = max(0.1, min(self._base_fit_scale, 1.0))
        scale = self._base_fit_scale * float(self.zoom.get())
        self._disp_scale = scale
    
        # Use cheaper resample while painting to reduce CPU spikes
        resample_mode = Image.NEAREST if getattr(self, "_drawing", False) else Image.BILINEAR
        im_resized = im.resize((int(im.width * scale), int(im.height * scale)), resample_mode)
    
        self._imgtk = ImageTk.PhotoImage(im_resized)
    
        cX, cY = cW // 2, cH // 2
        if self._img_id is None:
            # Create once, then only update image/coords later
            self._img_id = self.canvas.create_image(cX, cY, image=self._imgtk, anchor='center')
        else:
            # No delete/create => no flicker
            try:
                self.canvas.itemconfig(self._img_id, image=self._imgtk)
                self.canvas.coords(self._img_id, cX, cY)
            except Exception:
                # Fallback: recreate only if item got lost
                self._img_id = self.canvas.create_image(cX, cY, image=self._imgtk, anchor='center')
    
        # Redraw cursor ring after image refresh
        if self._hovering and self._last_mouse[0] is not None:
            self.update_cursor_visual(self._last_mouse[0], self._last_mouse[1])
    

    def img_coords(self, x, y):
        cW = self.canvas.winfo_width(); cH = self.canvas.winfo_height()
        imH, imW = self.rect_bgr.shape[:2]
        dispW = int(imW*self._disp_scale); dispH = int(imH*self._disp_scale)
        offX = (cW - dispW)//2; offY = (cH - dispH)//2
        xi = int((x - offX)/self._disp_scale)
        yi = int((y - offY)/self._disp_scale)
        return xi, yi

    def on_mouse_down(self, e):
        if self.rect_bgr is None: return
        if self.await_start_pick:
            self.await_start_pick = False
            xi, yi = self.img_coords(e.x, e.y)
            self.set_start_from_xy(float(xi), float(yi))
            return
        self._drawing = True; self._paint(e.x, e.y)

    def on_mouse_move(self, e):
        self._last_mouse = (e.x, e.y)
        if getattr(self, '_drawing', False):
            self._paint(e.x, e.y)
        else:
            self.update_cursor_visual(e.x, e.y)

    def on_mouse_up(self, e):
        self._drawing = False
        self.root.after(0, self.plan_tiles)

    def _paint(self, cx, cy):
        if self.rect_bgr is None: return
        xi, yi = self.img_coords(cx, cy)
        if not (0 <= xi < self.rect_bgr.shape[1] and 0 <= yi < self.rect_bgr.shape[0]):
            return
        r = max(1, int(self.brush_size.get()))
        if self.brush_mode.get() == 'Add':
            cv2.circle(self.user_add, (xi, yi), r, 255, -1)
            cv2.circle(self.user_erase, (xi, yi), r+1, 0, -1)
        else:
            cv2.circle(self.user_erase, (xi, yi), r, 255, -1)
            cv2.circle(self.user_add, (xi, yi), r+1, 0, -1)
        self.fg_mask = self.compose_fg()
        self.poly = contour_from_mask(self.fg_mask)
        self._update_area_from_poly()
        self.update_preview()
        self.update_cursor_visual(cx, cy)

    def reset_edits(self):
        if self.rect_bgr is None: return
        H, W = self.rect_bgr.shape[:2]
        self.user_add = np.zeros((H,W), np.uint8)
        self.user_erase = np.zeros((H,W), np.uint8)
        self.fg_mask = self.compose_fg()
        self.poly = contour_from_mask(self.fg_mask)
        self._update_area_from_poly()
        self.plan_tiles()

    def apply_edits(self):
        self.fg_mask = self.compose_fg()
        self.poly = contour_from_mask(self.fg_mask)
        self._update_area_from_poly()
        self.plan_tiles()

    def on_spin(self, var):
        if self.rect_bgr_base is None and var is not self.rot_deg:
            return
        try:
            if var is self.k_sigma:
                self.k_sigma.set(min(10.0, max(0.10, float(self.k_sigma.get()))))
            elif var is self.border_px:
                self.border_px.set(min(120, max(10, int(self.border_px.get()))))
            elif var is self.bg_shave:
                self.bg_shave.set(min(50, max(0, int(self.bg_shave.get()))))
            elif var is self.tile_mm:
                self.tile_mm.set(min(60, max(5, int(self.tile_mm.get()))))
            elif var is self.overlap_pct:
                self.overlap_pct.set(min(50, max(0, int(self.overlap_pct.get()))))
            elif var is self.brush_size:
                self.brush_size.set(min(200, max(1, int(self.brush_size.get()))))
            elif var is self.rot_deg:
                pass
        except Exception:
            pass
        if var in (self.border_px, self.k_sigma, self.bg_shave):
            self.segment()
        elif var in (self.tile_mm, self.overlap_pct):
            self.plan_tiles()
        elif var in (self.brush_size,):
            self.update_preview()
        elif var is self.rot_deg:
            self.apply_rotation()

    def set_zoom(self, value, fit=False):
        if fit:
            self.zoom.set(1.0)
        else:
            self.zoom.set(float(value))
        self.update_preview()
        if self._hovering and self._last_mouse[0] is not None:
            self.update_cursor_visual(self._last_mouse[0], self._last_mouse[1])

    def on_wheel(self, e):
        """Mouse wheel zoom (Windows/macOS)."""
        delta = e.delta
        factor = 1.1 if delta > 0 else 1/1.1
        self.set_zoom(min(8.0, max(0.1, self.zoom.get()*factor)))

    def on_wheel_linux(self, direction):
        """Mouse wheel zoom (X11/Linux). direction: +1 up, -1 down."""
        factor = 1.1 if direction > 0 else 1/1.1
        self.set_zoom(min(8.0, max(0.1, self.zoom.get()*factor)))

    def compose_fg(self):
        if self.rect_bgr is None:
            return None
        base = np.zeros(self.rect_bgr.shape[:2], np.uint8) if self.auto_fg is None else ((self.auto_fg>0).astype(np.uint8)*255)
        if self.user_add is not None and np.any(self.user_add>0):
            base[self.user_add>0] = 255
        if self.user_erase is not None and np.any(self.user_erase>0):
            base[self.user_erase>0] = 0
        return base

    def update_cursor_visual(self, cx, cy):
        """Draw a ring around the mouse to preview brush size (in canvas coords)."""
        if self.rect_bgr is None:
            return
        r_disp = max(1, int(self.brush_size.get() * self._disp_scale))
        x0, y0, x1, y1 = cx - r_disp, cy - r_disp, cx + r_disp, cy + r_disp
        color = '#00ff00' if self.brush_mode.get() == 'Add' else '#ff4444'
        if self._cursor_id is None:
            self._cursor_id = self.canvas.create_oval(x0, y0, x1, y1, outline=color, width=2)
        else:
            try:
                self.canvas.coords(self._cursor_id, x0, y0, x1, y1)
                self.canvas.itemconfig(self._cursor_id, outline=color)
            except Exception:
                # Recreate if underlying item was removed for any reason
                self._cursor_id = self.canvas.create_oval(x0, y0, x1, y1, outline=color, width=2)

    def on_hover(self, e):
        self._hovering = True
        self._last_mouse = (e.x, e.y)
        self.update_cursor_visual(e.x, e.y)

    def on_leave_canvas(self, e):
        self._hovering = False
        if self._cursor_id is not None:
            try:
                self.canvas.delete(self._cursor_id)
            except Exception:
                pass
            self._cursor_id = None

    def update_tiles_label(self):
        if hasattr(self, 'tiles_lbl') and self.tiles is not None:
            self.tiles_lbl.config(text=f"Frames: {len(self.tiles)}")
        if hasattr(self, 'start_lbl'):
            if self.start_xy is None:
                self.start_lbl.config(text="Start: —")
            else:
                x, y = self.start_xy
                self.start_lbl.config(text=f"Start: ({x:.1f}, {y:.1f})")
        if hasattr(self, 'area_lbl'):
            if self.area_mm2 is None:
                self.area_lbl.config(text="Area: —")
            else:
                self.area_lbl.config(text=f"Area: {self.area_mm2:.1f} mm²")

    def show_tiles_stats(self):
        top = tk.Toplevel(self.root)
        top.title("Plan statistics")
        n = len(self.tiles) if self.tiles is not None else 0
        step = float(self.tile_mm.get()) * (1.0 - float(self.overlap_pct.get())/100.0)
        area_text = "unknown" if self.area_mm2 is None else f"{self.area_mm2:.1f} mm²"
        ttk.Label(top, text=f"Frames (tiles): {n}").pack(anchor='w', padx=10, pady=(10,4))
        ttk.Label(top, text=f"Grid step ≈ {step:.2f} mm").pack(anchor='w', padx=10)
        ttk.Label(top, text=f"Area: {area_text}").pack(anchor='w', padx=10, pady=(0,10))
        ttk.Button(top, text="Close", command=top.destroy).pack(pady=(0,10))

    # ---- Rotation ----
    def apply_rotation(self):
        """Apply current rot_deg to base rect image, reset masks and update pipeline."""
        if self.rect_bgr_base is None:
            return
        angle = float(self.rot_deg.get())
        h, w = self.rect_bgr_base.shape[:2]
        c = (w/2.0, h/2.0)
        M = cv2.getRotationMatrix2D(c, angle, 1.0)
        # compute bounds to avoid cropping
        cos = abs(M[0,0]); sin = abs(M[0,1])
        nw = int((h*sin) + (w*cos))
        nh = int((h*cos) + (w*sin))
        M[0,2] += (nw/2) - c[0]
        M[1,2] += (nh/2) - c[1]
        self.rect_bgr = cv2.warpAffine(self.rect_bgr_base, M, (nw, nh))
        # reset edits & start (since geometry changed)
        self.start_xy = None
        self.ensure_masks()
        if getattr(self, "_refining", False):
            self._schedule_preview(33)
        else:
            self.segment()

    def refine_mask_once(self):
        """Run GrabCut once asynchronously and apply result when done."""
        if self.rect_bgr is None or self.fg_mask is None:
            return
        if self._refining:
            return

        self._refining = True
        self._last_error = None
        self._set_busy_cursor(True)

        img = self.rect_bgr.copy()
        seed_mask_src = self.fg_mask.copy() if self.fg_mask is not None else self.compose_fg()
        if seed_mask_src is None:
            self._set_busy_cursor(False)
            self._refining = False
            return
        seed_mask = seed_mask_src.copy()

        def _worker() -> None:
            refined = None
            error_text = None
            try:
                refined = refine_with_grabcut(img, seed_mask)
            except Exception as exc:  # noqa: BLE001
                error_text = f"GrabCut error: {exc}"

            def _commit() -> None:
                try:
                    if refined is not None:
                        refined_mask = refined.copy()
                        self.auto_fg = refined_mask.copy()
                        if self.user_add is not None:
                            self.user_add.fill(0)
                        if self.user_erase is not None:
                            self.user_erase.fill(0)
                        self.fg_mask = refined_mask.copy()
                        self.poly = contour_from_mask(self.fg_mask)
                        self._update_area_from_poly()
                        try:
                            self.use_grabcut.set(False)
                        except Exception:  # pragma: no cover - Tk variable errors
                            pass
                        self.plan_tiles()
                        self.update_preview()
                    elif error_text:
                        self._last_error = error_text
                        try:
                            messagebox.showwarning("GrabCut", error_text)
                        except Exception:  # pragma: no cover - Tk fallback
                            pass
                finally:
                    self._set_busy_cursor(False)
                    self._refining = False

            self.root.after(0, _commit)

        threading.Thread(target=_worker, daemon=True).start()

    def _set_busy_cursor(self, busy: bool) -> None:
        """Toggle busy cursor on root and canvas widgets."""
        try:
            cursor = "watch" if busy else ""
            self.root.configure(cursor=cursor)
            if hasattr(self, "canvas") and self.canvas is not None:
                self.canvas.configure(cursor=cursor)
        except Exception:  # pragma: no cover - Tk fallback
            pass

    def _schedule_preview(self, delay_ms=33):
        """Coalesce heavy redraw into a timed call to avoid flicker and CPU spikes."""
        if self._preview_job is not None:
            try:
                self.root.after_cancel(self._preview_job)
            except Exception:
                pass
        self._preview_job = self.root.after(delay_ms, self._do_preview)
    
    def _do_preview(self):
        """Actual preview redraw."""
        self._preview_job = None
        self.update_preview()
    
    def _debounce_spin(self, var, wait=120):
        """Debounce on_spin to avoid recompute on each keystroke."""
        key = f"spin:{id(var)}"
        job = self._debounce_jobs.get(key)
        if job:
            try:
                self.root.after_cancel(job)
            except Exception:
                pass
        self._debounce_jobs[key] = self.root.after(wait, lambda: self.on_spin(var))
    
    
# --------------------------- CLI main ---------------------------

def run_cli(args):
    p = Path(args.out_dir); p.mkdir(parents=True, exist_ok=True)
    img_path = Path(args.image)
    img, scale = load_preview_image(img_path)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    H, mm_per_px, _ = find_aruco_homography(img, args.aruco_mm)
    if H is None:
        if args.mm_per_pixel is None:
            raise SystemExit("No ArUco found and --mm_per_pixel not provided. Provide either.")
        mmpp_eff = args.mm_per_pixel / (scale or 1.0)
        H = np.array([[mmpp_eff, 0, 0],[0, mmpp_eff, 0],[0, 0, 1]], dtype=np.float32)
    rect, _, _ = warp_to_mm_plane(img, H, out_border_mm=20)
    poly = extract_disk_contour(rect, border_px=args.border_px, k_sigma=args.k_sigma, use_grabcut=args.grabcut)
    if poly is None:
        raise SystemExit("Failed to extract disk contour. Try --grabcut or adjust border/k_sigma.")
    tiles_raw = generate_tiles_inside_polygon(poly, args.tile_mm, args.overlap)
    step_mm = args.tile_mm*(1.0-args.overlap)
    tiles = order_tiles_raster(tiles_raw, 'Y-first', True, step_mm)
    save_overlay(rect, poly, tiles, p/"overlay.png")
    save_tiles_csv(tiles, p/"tiles.csv")
    save_polygon_json(poly, p/"polygon_mm.json")
    save_gcode(tiles, args.safe_z, args.travel_f, args.z_f, args.settle_ms, p/"route.gcode",
               manual_y=False, axis_first='Y-first', step_mm=step_mm)
    rows = group_ordered_tiles(tiles, 'Y-first', step_mm)
    area_mm2 = cv2.contourArea(poly)
    bbox = (float(poly[:,0].min()), float(poly[:,1].min()), float(poly[:,0].max()), float(poly[:,1].max()))
    meta = {
        "tiles": len(tiles),
        "area_mm2": float(area_mm2),
        "bbox_mm": bbox,
        "tile_mm": args.tile_mm,
        "overlap_pct": args.overlap*100.0,
        "grid_step_mm": step_mm,
        "movement_axis": "Y-first",
        "serpentine": True,
        "manual_y_mode": False,
        "row_count": len(rows),
        "rows": [
            {
                "index": i,
                "y_mm": float(np.median([p[1] for p in r])),
                "count": len(r)
            }
            for i, r in enumerate(rows)
        ]
    }
    (p/"meta.json").write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print("Saved:", str(p.resolve()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gui', action='store_true', help='Launch GUI instead of CLI pipeline')
    ap.add_argument('--grabcut', action='store_true')
    ap.add_argument('--gc_iters', type=int, default=3)
    ap.add_argument('--border_px', type=int, default=40)
    ap.add_argument('--k_sigma', type=float, default=2.5)
    ap.add_argument('--image')
    ap.add_argument('--aruco_mm', type=float, default=30.0)
    ap.add_argument('--mm_per_pixel', type=float, default=None)
    ap.add_argument('--tile_mm', type=float, default=15.0)
    ap.add_argument('--overlap', type=float, default=0.2)
    ap.add_argument('--safe_z', type=float, default=40.0)
    ap.add_argument('--travel_f', type=int, default=9000)
    ap.add_argument('--z_f', type=int, default=1200)
    ap.add_argument('--settle_ms', type=int, default=250)
    ap.add_argument('--out_dir', default='out_xy')
    args = ap.parse_args()

    if args.gui:
        if not HAS_TK:
            raise SystemExit("Tkinter/Pillow not available. Install: pip install pillow")
        root = tk.Tk()
        try:
            style = ttk.Style(root)
            if 'vista' in style.theme_names(): style.theme_use('vista')
        except Exception:
            pass
        App(root)
        root.mainloop()
    else:
        if not args.image:
            raise SystemExit("--image is required in CLI mode. Or run with --gui.")
        run_cli(args)

if __name__ == '__main__':
    main()