# Program: Eos Utility
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge;
#   Xinjiang Institute of Ecology and Geography;
#   National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Mini EOS utility UI for controlling Canon cameras via EDSDK."""

import io
import os
import threading
import time
import tkinter as tk
from contextlib import suppress
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

from edsdk_runtime import Camera

W, H = 960, 640
HANDLE = 10
EVF_WARMUP_SECONDS = 1.5
MIN_EVF_GRID = 1000


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mini EOS Utility Pro (R6 Mark II)")
        self.geometry(f"{W+24}x{H+220}")
        self.minsize(640, 480)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self._outer = tk.Frame(self)
        self._outer.pack(fill="both", expand=True)
        self._outer.grid_rowconfigure(0, weight=1)
        self._outer.grid_columnconfigure(0, weight=1)

        scroll_area = tk.Frame(self._outer)
        scroll_area.grid(row=0, column=0, sticky="nsew")
        scroll_area.grid_rowconfigure(0, weight=1)
        scroll_area.grid_columnconfigure(0, weight=1)

        self._scroll_canvas = tk.Canvas(
            scroll_area, borderwidth=0, highlightthickness=0
        )
        self._scrollbar = tk.Scrollbar(
            scroll_area, orient="vertical", command=self._scroll_canvas.yview
        )
        self._scroll_canvas.configure(yscrollcommand=self._scrollbar.set)

        self._scroll_canvas.grid(row=0, column=0, sticky="nsew")
        self._scrollbar.grid(row=0, column=1, sticky="ns")

        self._content = tk.Frame(self._scroll_canvas)
        self._content.bind(
            "<Configure>",
            lambda e: self._scroll_canvas.configure(
                scrollregion=self._scroll_canvas.bbox("all")
            ),
        )
        self._content_window = self._scroll_canvas.create_window(
            (0, 0), window=self._content, anchor="nw"
        )

        self._scroll_canvas.bind(
            "<Configure>",
            lambda e: self._scroll_canvas.itemconfigure(
                self._content_window, width=e.width
            ),
        )

        self._bind_scroll_wheel()

        self.cam = Camera()
        self.download_dir = os.path.abspath("./shots")

        # --- Top button bar ---
        top = tk.Frame(self._content)
        top.pack(fill="x", padx=8, pady=6)

        # tk.Button(top, text="Connect", command=self.on_connect).pack(side="left", padx=4)
        tk.Button(top, text="AF (center)", command=lambda: self.af_at(0.5, 0.5)).pack(
            side="left", padx=4
        )
        tk.Button(top, text="Shoot", command=self.on_shoot).pack(side="left", padx=4)

        # --- FIX: moved here to keep the correct order ---
        self.af_mode = tk.StringVar(value="AF")
        tk.Button(
            top, textvariable=self.af_mode, width=6, command=self.toggle_af_mode
        ).pack(side="left", padx=4)

        # --- Calibration test button is now visible ---
        tk.Button(
            top, text="Run AF Test", bg="#FFDDC1", command=self.run_calibration_test
        ).pack(side="left", padx=10)

        # Zoom buttons on the right
        tk.Button(top, text="10x", command=lambda: self.set_zoom(10)).pack(
            side="right", padx=2
        )
        tk.Button(top, text="5x", command=lambda: self.set_zoom(5)).pack(
            side="right", padx=2
        )
        tk.Button(top, text="1x", command=lambda: self.set_zoom(1)).pack(
            side="right", padx=2
        )

        # --- Preview canvas ---
        self.canvas = tk.Canvas(
            self._content, width=W, height=H, bg="#111", highlightthickness=0
        )
        self.canvas.pack(padx=8, pady=(4, 0))
        self.img_id = self.canvas.create_image(0, 0, anchor="nw")

        self.canvas.focus_set()
        self.bind_all("<Left>", lambda e: self._nudge(-1, 0, e))
        self.bind_all("<Right>", lambda e: self._nudge(+1, 0, e))
        self.bind_all("<Up>", lambda e: self._nudge(0, -1, e))
        self.bind_all("<Down>", lambda e: self._nudge(0, +1, e))

        cx, cy, rw, rh = W // 2, H // 2, 160, 120
        self.box = [cx - rw // 2, cy - rh // 2, cx + rw // 2, cy + rh // 2]
        self.box_id = self.canvas.create_rectangle(
            *self.box, outline="#00FF88", width=2
        )
        self.handle_ids = []
        for _ in range(4):
            self.handle_ids.append(
                self.canvas.create_rectangle(
                    0, 0, 0, 0, outline="#00FF88", fill="#00FF88"
                )
            )
        self._place_handles()

        self.drag_mode = None
        self.drag_off = (0, 0)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)

        # --- Bottom bar ---
        bottom = tk.Frame(self._content)
        bottom.pack(fill="x", padx=8, pady=6)
        self.dir_var = tk.StringVar(value=self.download_dir)
        tk.Label(bottom, text="Folder:").pack(side="left")
        tk.Entry(bottom, textvariable=self.dir_var, width=54).pack(
            side="left", fill="x", expand=True, padx=6
        )
        tk.Button(bottom, text="Browse...", command=self.choose_dir).pack(
            side="left", padx=4
        )

        self.status = tk.StringVar(value="Ready")
        tk.Label(self._content, textvariable=self.status, anchor="w").pack(
            fill="x", padx=8, pady=(0, 8)
        )

        footer = tk.Frame(self._outer)
        footer.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        footer.grid_columnconfigure(0, weight=1)

        tk.Label(
            footer,
            text="Developed by Dr. Z. Bakhtiyorov",
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        footer_buttons = tk.Frame(footer)
        footer_buttons.grid(row=0, column=1, sticky="e", padx=(8, 0))
        tk.Button(footer_buttons, text="Help", command=self.show_help).pack(
            side="left", padx=4
        )
        tk.Button(footer_buttons, text="About", command=self.show_about).pack(
            side="left", padx=4
        )

        self._imgtk = None
        self._lv_running = False
        self._after_id = None
        self._schedule_update()

        self.can_shoot = True
        self.center_norm = (0.5, 0.5)
        # --- Autoconnect on startup (after UI initialization) ---
        self.after(250, self.auto_connect_on_start)

    def set_status(self, s):
        self.status.set(s)

    def _bind_scroll_wheel(self) -> None:
        system = self.tk.call("tk", "windowingsystem")
        if os.name == "nt":
            self._scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel_windows)
        elif system == "aqua":
            self._scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel_macos)
        else:
            self._scroll_canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
            self._scroll_canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _on_mousewheel_windows(self, event):
        self._scroll_canvas.yview_scroll(-1 * int(event.delta / 120), "units")

    def _on_mousewheel_macos(self, event):
        self._scroll_canvas.yview_scroll(-1 * int(event.delta), "units")

    def _on_mousewheel_linux(self, event):
        direction = -1 if event.num == 4 else 1
        self._scroll_canvas.yview_scroll(direction, "units")

    def show_help(self):
        messagebox.showinfo(
            "Help",
            (
                "Use the mouse wheel or scrollbar to navigate the interface."
                " Focus the live view and drag the green rectangle to reposition autofocus."
            ),
        )

    def show_about(self):
        messagebox.showinfo(
            "About",
            (
                "Mini EOS Utility Pro interface with autofocus helpers.\n"
                "Developed by Dr. Z. Bakhtiyorov\n"
                "Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography;"
                " National Academy of Sciences of Tajikistan\nLicense: MIT"
            ),
        )

    def choose_dir(self):
        d = filedialog.askdirectory(initialdir=self.download_dir)
        if d:
            self.download_dir = d
            self.dir_var.set(d)

    def _schedule_update(self):
        self._after_id = self.after(33, self.update_preview)

    def update_preview(self):
        if self._lv_running:
            jpg = self.cam.get_last_jpeg()
            if jpg:
                try:
                    img = Image.open(io.BytesIO(jpg)).resize((W, H))
                    self._imgtk = ImageTk.PhotoImage(img)
                    self.canvas.itemconfig(self.img_id, image=self._imgtk)
                    # Report actual UI canvas size to runtime for proper click->EVF mapping.
                    try:
                        ui_w = self.canvas.winfo_width() or W
                        ui_h = self.canvas.winfo_height() or H
                        self.cam._ui_w, self.cam._ui_h = int(ui_w), int(ui_h)
                    except Exception:
                        self.cam._ui_w, self.cam._ui_h = W, H
                except Exception:
                    pass
        self._schedule_update()

    def _place_handles(self):
        x0, y0, x1, y1 = self.box
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        half = HANDLE // 2
        for i, (x, y) in enumerate(pts):
            self.canvas.coords(
                self.handle_ids[i], x - half, y - half, x + half, y + half
            )

    def _hit_handle(self, x, y):
        for i, h in enumerate(self.handle_ids):
            hx0, hy0, hx1, hy1 = self.canvas.coords(h)
            if hx0 <= x <= hx1 and hy0 <= y <= hy1:
                return i
        return None

    def on_press(self, ev):
        i = self._hit_handle(ev.x, ev.y)
        if i is not None:
            self.drag_mode = i
            return
        x0, y0, x1, y1 = self.box
        if x0 <= ev.x <= x1 and y0 <= ev.y <= y1:
            self.drag_mode = "move"
            self.drag_off = (ev.x - x0, ev.y - y0)
        else:
            self.drag_mode = None

    def on_drag(self, ev):
        if self.drag_mode is None:
            return
        x, y = max(0, min(W, ev.x)), max(0, min(H, ev.y))
        x0, y0, x1, y1 = self.box
        if self.drag_mode == "move":
            dx = x - self.drag_off[0]
            dy = y - self.drag_off[1]
            w = x1 - x0
            h = y1 - y0
            x0, y0 = max(0, min(W - w, dx)), max(0, min(H - h, dy))
            self.box = [x0, y0, x0 + w, y0 + h]
        else:
            i = self.drag_mode
            if i == 0:
                x0, y0 = x, y
            elif i == 1:
                x1, y0 = x, y
            elif i == 2:
                x1, y1 = x, y
            elif i == 3:
                x0, y1 = x, y
            x0, x1 = sorted((max(0, x0), min(W, x1)))
            y0, y1 = sorted((max(0, y0), min(H, y1)))
            self.box = [x0, y0, x1, y1]
        self.canvas.coords(self.box_id, *self.box)
        self._place_handles()
        # Update center in EVF-normalized coords using runtime mapping.
        x0, y0, x1, y1 = self.box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.center_norm = self.cam._ui_to_norm_image(int(cx), int(cy))

    def on_release(self, ev):
        if self.drag_mode is None:
            return
        self.drag_mode = None
        x0, y0, x1, y1 = self.box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        nx, ny = self.cam._ui_to_norm_image(int(cx), int(cy))
        self.af_at(nx, ny)
        self._place_handles()
        x0, y0, x1, y1 = self.box
        self.center_norm = (nx, ny)

    def on_double_click(self, ev):
        x0, y0, x1, y1 = self.box
        w, h = x1 - x0, y1 - y0
        cx, cy = max(w / 2, min(W - w / 2, ev.x)), max(h / 2, min(H - h / 2, ev.y))
        self.box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        self.canvas.coords(self.box_id, *self.box)
        self._place_handles()
        nx, ny = self.cam._ui_to_norm_image(int(cx), int(cy))
        self.af_at(nx, ny)
        x0, y0, x1, y1 = self.box
        self.center_norm = (nx, ny)

    def on_connect(self, silent: bool = False):
        try:
            # Canon session -> AF mode -> EVF to PC+TFT -> start background LV
            self.cam.open(0)
            self.cam.try_set_one_shot_single_point()
            self.cam.enable_liveview_pc(also_tft=True)
            self.cam.start_liveview()  # alias to start_live_view; runs EVF thread
            self._lv_running = True

            # Warm up: wait for first EVF JPEG so VisibleRect/ZoomRect caches are valid
            for _ in range(100):
                if self.cam.get_last_jpeg():
                    break
                try:
                    self.update_idletasks()
                    self.after(20)
                except Exception:
                    pass

            # No manual AF offset here - keep calibration off by default
            if hasattr(self.cam, "_ofs_ready"):
                self.cam._ofs_ready = False

            self.cam.debug_dump_evf()
            self.set_status("Connected. Live View -> PC")
        except Exception as e:
            if silent:
                # Avoid blocking startup with modal dialog; update status only
                self.set_status(f"Autoconnect: {e}")
            else:
                messagebox.showerror("Error", f"Connection: {e}")

    def auto_connect_on_start(self):
        """Attempt to connect automatically once at startup."""
        if not getattr(self.cam, "_session", False):
            self.on_connect(silent=True)

    def af_at(self, nx, ny):
        def work():
            try:
                self.set_status(f"AF ({nx:.2f},{ny:.2f})...")
                ok = self.cam.set_touch_af_or_zoompos(nx, ny)
                self._set_box_color("#00FF88" if ok else "#FF4444")
                self.set_status("AF completed" if ok else "AF not confirmed")
            except Exception as e:
                self._set_box_color("#FF4444")
                self.set_status(f"AF error: {e}")

        threading.Thread(target=work, daemon=True).start()

    def set_zoom(self, z):
        def work():
            try:
                cx, cy = self.center_norm
                raw = self.cam.set_evf_zoom_keep_center(int(z), cx, cy)
                self.set_status(f"Zoom={z}x (raw={raw}) @ ({cx:.2f},{cy:.2f})")
            except Exception as e:
                self.set_status(f"Zoom error: {e}")

        threading.Thread(target=work, daemon=True).start()

    def on_shoot(self):
        if self.af_mode.get() == "AF" and not self.can_shoot:
            messagebox.showwarning(
                "Focus",
                "Autofocus not confirmed (red frame). " "Capture blocked.",
            )
            return

        def work():
            try:
                self.set_status("Capturing...")
                p = self.cam.shoot_and_download(self.download_dir, timeout_s=20)
                self.set_status(f"Saved: {p}")
            except Exception as e:
                self.set_status(f"Capture error: {e}")

        threading.Thread(target=work, daemon=True).start()

    def run_calibration_test(self):  # noqa: PLR0912,PLR0915
        """Interactive AF offset calibration helper."""
        if not self.cam._session:
            messagebox.showerror(
                "Error",
                "Connect to the camera first (Connect button).",
            )
            return

        def test_worker():
            # Warm up EVF caches so coordinate mapping is stable.
            t0 = time.time()
            while time.time() - t0 < EVF_WARMUP_SECONDS:
                if self.cam.get_last_jpeg() and (
                    getattr(self.cam, "_vr", None) or getattr(self.cam, "_cs_w", 0)
                ):
                    break
                time.sleep(0.05)
            print("\n" + "=" * 70)
            print(" STARTING CALIBRATION TEST | User: zulfiyor")
            print(f" UTC time: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}")
            print("=" * 70)

            test_points = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9), (0.5, 0.5)]

            # --- STEP 1: ANALYSIS ---
            print("\n--- STEP 1: Offset analysis (no correction) ---")
            cs_w, cs_h = self.cam.get_evf_coords()
            print(f"[CHECK] EVF grid = {cs_w} x {cs_h}")
            if cs_w < MIN_EVF_GRID or cs_h < MIN_EVF_GRID:
                print("[WARN] EVF grid too small; aborting.")
                self.after(
                    0, lambda: self.set_status("Test cancelled: EVF grid invalid")
                )
                return
            diffs = []

            # Ensure offset is disabled for baseline measurement
            if hasattr(self.cam, "_ofs_ready"):
                self.cam._ofs_ready = False

            for nx, ny in test_points:
                print(f"-> Testing point ({nx:.1f}, {ny:.1f})")
                # Synchronous AF: no background thread, no half-press fallback.
                # Also wait until focus point converges (tolerate Y grid ~70 px).
                ok = self.cam.set_touch_af_or_zoompos(
                    nx, ny, dwell=0.25, allow_halfpress=False
                )
                try:
                    # Map desired center to EVF px and wait for convergence
                    pt = self.cam._map_norm_to_evf(nx, ny)
                    self.cam._wait_center_converged(
                        int(pt.x),
                        int(pt.y),
                        tol_x=4,
                        tol_y=70,
                        timeout_s=0.9,
                        prefer_taf=True,
                    )
                except Exception:
                    pass

                try:
                    self.cam.debug_probe_af(nx, ny)
                    # Reuse mapped EVF coordinates for the desired point
                    desired_px = (int(pt.x), int(pt.y))
                    actual_px = None
                    if getattr(self.cam, "_taf", None):
                        actual_px = self.cam._taf
                    elif getattr(self.cam, "_zr", None):
                        zx, zy, zw, zh = self.cam._zr
                        actual_px = (zx + zw // 2, zy + zh // 2)

                    if actual_px:
                        diff = (
                            actual_px[0] - desired_px[0],
                            actual_px[1] - desired_px[1],
                        )
                        diffs.append(diff)
                except Exception as e:
                    print(f"   Error reading sample: {e}")

            if not diffs:
                print("\n[ERROR] Failed to gather data at Step 1. Test aborted.")
                # Tk is not thread-safe: update status via main loop.
                self.after(0, lambda: self.set_status("Test failed: no data"))
                return

            from statistics import mean, median

            avg_offset_x = mean([d[0] for d in diffs])
            avg_offset_y = mean([d[1] for d in diffs])
            med_offset_x = median([d[0] for d in diffs])
            med_offset_y = median([d[1] for d in diffs])
            print("\n--- STEP 1 RESULT ---")
            print(f"Mean offset: x={avg_offset_x:.1f}, y={avg_offset_y:.1f}")
            print(f"Median offset: x={med_offset_x:.1f}, y={med_offset_y:.1f}")
            # Use median as robust estimator (AF grid quantization may create outliers)
            use_offset_x = med_offset_x
            use_offset_y = med_offset_y

            # Robust median
            from statistics import median

            med_offset_x = median([d[0] for d in diffs])
            med_offset_y = median([d[1] for d in diffs])
            print(f"Median offset: x={med_offset_x:.1f}, y={med_offset_y:.1f}")

            # Grid-aware recommendation (Canon Y step ~63 px on many bodies)
            def _round_with_grid(v: float, grid: int = 63, dead: float = 0.35) -> int:
                if abs(v) < dead * grid:
                    return 0
                return int(round(v / grid) * grid)

            rec_x = int(round(med_offset_x))
            rec_y = _round_with_grid(med_offset_y, grid=63, dead=0.35)

            # --- STEP 2: VERIFICATION ---
            print("\n--- STEP 2: Verification with correction ---")

            # Temporarily apply offset in the camera object
            if hasattr(self.cam, "set_af_offset"):
                self.cam.set_af_offset(rec_x, rec_y)
            else:  # If method missing, set fields directly
                self.cam._ofs_x = rec_x
                self.cam._ofs_y = rec_y
                self.cam._ofs_ready = True
                print(
                    f"[AF OFFSET] Temporary offset applied: x={self.cam._ofs_x}, y={self.cam._ofs_y}"
                )

            for nx, ny in test_points:
                print(f"-> Testing point ({nx:.1f}, {ny:.1f}) with correction")
                ok = self.cam.set_touch_af_or_zoompos(
                    nx, ny, dwell=0.25, allow_halfpress=False
                )
                try:
                    pt = self.cam._map_norm_to_evf(nx, ny)
                    self.cam._wait_center_converged(
                        int(pt.x),
                        int(pt.y),
                        tol_x=4,
                        tol_y=70,
                        timeout_s=0.9,
                        prefer_taf=True,
                    )
                except Exception:
                    pass
                try:
                    self.cam.debug_probe_af(nx, ny)
                except Exception as e:
                    print(f"   Error reading sample: {e}")

            # Restore offset to original state
            if hasattr(self.cam, "_ofs_ready"):
                self.cam._ofs_ready = False

            print("\n" + "=" * 70)
            print(" TEST COMPLETE")
            print("=" * 70)
            print(
                "Check console output. If Step 2 diff_px is near (0, 0), "
                "calibration is correct."
            )
            print(f"Recommended offset: ofs_x={rec_x}, ofs_y={rec_y}")
            self.after(
                0,
                lambda: self.set_status(f"Test finished. Offset: x={rec_x}, y={rec_y}"),
            )

        threading.Thread(target=test_worker, daemon=True).start()

    def on_close(self):
        """Tear down Live View loop and close camera session."""
        try:
            if hasattr(self, "_after_id") and self._after_id:
                with suppress(Exception):
                    self.after_cancel(self._after_id)
            self._lv_running = False
            with suppress(Exception):
                self.cam.close()
        finally:
            self.destroy()

    def _focus_metric(self) -> float:
        jpg = self.cam.get_last_jpeg()
        if not jpg:
            return 0.0
        try:
            img = Image.open(io.BytesIO(jpg)).resize((W, H))
            x0, y0, x1, y1 = map(int, self.box)
            x0 = max(0, min(W - 1, x0))
            x1 = max(1, min(W, x1))
            y0 = max(0, min(H - 1, y0))
            y1 = max(1, min(H, y1))
            roi = img.crop((x0, y0, x1, y1)).convert("L")
            arr = np.asarray(roi, dtype=np.float32)
            k = np.array([[0, 1, 0], [1, -4, 1]], dtype=np.float32)
            h, w = arr.shape
            pad = np.pad(arr, 1, mode="edge")
            s = (
                pad[:-2, 1:-1] * 0
                + pad[1:-1, :-2]
                + pad[1:-1, 2:]
                + pad[2:, 1:-1]
                + pad[:-2, 1:-1]
                - 4 * pad[1:-1, 1:-1]
            )
            s = s if s.shape == (h, w) else s[:h, :w]
            return float(np.var(s))
        except Exception:
            return 0.0

    def toggle_af_mode(self):
        self.af_mode.set("MF" if self.af_mode.get() == "AF" else "AF")
        if self.af_mode.get() == "MF":
            self._set_box_color("#00FF88")
        else:
            self._set_box_color("#00FF88")

    def _set_box_color(self, color: str):
        self.canvas.itemconfig(self.box_id, outline=color)
        for hid in self.handle_ids:
            self.canvas.itemconfig(hid, outline=color, fill=color)
        self.can_shoot = color != "#FF4444"

    def _nudge(self, sx, sy, event=None):
        step = 0.01 if not (getattr(event, "state", 0) & 0x0001) else 0.05
        try:
            self.cam.nudge_zoom_norm(sx * step, sy * step)
            x0, y0, x1, y1 = self.box
            cx = (x0 + x1) / 2 + sx * step * W
            cy = (y0 + y1) / 2 + sy * step * H
            w = x1 - x0
            h = y1 - y0
            cx = max(w / 2, min(W - w / 2, cx))
            cy = max(h / 2, min(H - h / 2, cy))
            self.box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
            self.canvas.coords(self.box_id, *self.box)
            self._place_handles()
            # Keep center in EVF-normalized space via runtime mapping
            self.center_norm = self.cam._ui_to_norm_image(int(cx), int(cy))
        except Exception as e:
            self.set_status(f"Pan error: {e}")


if __name__ == "__main__":
    App().mainloop()

# Created by Dr. Z. Bakhtiyorov