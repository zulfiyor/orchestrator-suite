# Program: Capturing Disk — GUI Planner
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Institutions: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, Button, Entry, Frame, Label, StringVar, Tk, filedialog, messagebox

from .disk_planner_xy import plan_from_image, save_plan_json


def main() -> int:
    """Launch a minimal Tkinter GUI to set disk parameters and export plan."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="", help="Path to disk image")
    parser.add_argument("--out-json", type=str, default="out/plan.json", help="Output JSON plan path")
    args = parser.parse_args()

    root = Tk()
    root.title("Capturing Disk — Planner")

    state = {
        "image": StringVar(value=args.image),
        "pixel": StringVar(value="0.05"),  # mm per pixel (assumption)
        "outer": StringVar(value=""),
        "inner": StringVar(value=""),
        "cx": StringVar(value="0.0"),
        "cy": StringVar(value="0.0"),
        "pattern": StringVar(value="raster"),
        "step": StringVar(value="0.5"),
        "dwell": StringVar(value="250"),
        "feed": StringVar(value="1200.0"),
        "out": StringVar(value=args.out_json),
    }

    def browse_image() -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if path:
            state["image"].set(path)

    def export_plan() -> None:
        try:
            plan = plan_from_image(
                image_path=state["image"].get(),
                pixel_size_mm=float(state["pixel"].get()),
                outer_diameter_mm=float(state["outer"].get()) if state["outer"].get() else None,
                inner_diameter_mm=float(state["inner"].get()) if state["inner"].get() else None,
                center_xy_mm=(float(state["cx"].get()), float(state["cy"].get())),
                pattern=state["pattern"].get(),
                step_mm=float(state["step"].get()),
                dwell_ms=int(state["dwell"].get()),
                feed_mm_min=float(state["feed"].get()),
            )
            save_plan_json(plan, state["out"].get())
            messagebox.showinfo("Planner", f"Plan saved to {state['out'].get()}")
            root.quit()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Planner error", str(e))

    frm = Frame(root)
    frm.pack(fill=BOTH, expand=True, padx=12, pady=12)

    def row(r: int, text: str, key: str, width: int = 18, browse: bool = False) -> None:
        Label(frm, text=text, width=20, anchor="w").grid(row=r, column=0, pady=4)
        e = Entry(frm, textvariable=state[key], width=width)
        e.grid(row=r, column=1, pady=4)
        if browse:
            Button(frm, text="Browse", command=browse_image).grid(row=r, column=2, padx=4)

    row(0, "Image", "image", width=40, browse=True)
    row(1, "Pixel size [mm/pix]", "pixel")
    row(2, "Outer diameter [mm] (opt)", "outer")
    row(3, "Inner diameter [mm] (opt)", "inner")
    row(4, "Center X [mm]", "cx")
    row(5, "Center Y [mm]", "cy")
    row(6, "Pattern [raster|spiral]", "pattern")
    row(7, "Step [mm]", "step")
    row(8, "Dwell [ms]", "dwell")
    row(9, "Feed [mm/min]", "feed")
    row(10, "Out JSON", "out", width=40)

    Button(frm, text="Generate plan", command=export_plan).grid(row=11, column=0, pady=12)
    Button(frm, text="Close", command=root.quit).grid(row=11, column=1, pady=12)

    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Created by Dr. Z. Bakhtiyorov
