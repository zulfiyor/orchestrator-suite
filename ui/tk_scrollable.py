# Program: Tkinter Scrollable App Wrapper with Help/About
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Utilities to add vertical scrolling and Help/About UI to Tkinter apps.

Features:
- Scrollable frame with mouse wheel support (Windows/macOS/Linux).
- Optional bottom bar with Help/About buttons.
- Menu bar entries (Help -> About) for desktop-like UX.

Usage:
    from ui.tk_scrollable import ScrollableApp

    app = ScrollableApp(title="My App")
    frame = app.content  # put your widgets into this frame
    # ... pack/place/grid your content into frame ...
    app.run()
"""

from __future__ import annotations

import platform
import tkinter as tk
from tkinter import messagebox


class ScrollableApp:
    """Tkinter window with a scrollable content area and Help/About controls."""

    def __init__(self, title: str = "Application", width: int = 900, height: int = 600) -> None:
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")

        # Menu bar with Help/About
        menubar = tk.Menu(self.root)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Help", command=self._on_help)
        help_menu.add_command(label="About", command=self._on_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menubar)

        # Scrollable canvas + interior frame
        self._canvas = tk.Canvas(self.root, borderwidth=0, highlightthickness=0)
        self._v_scroll = tk.Scrollbar(self.root, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._v_scroll.set)

        self._container = tk.Frame(self._canvas)
        self._container.bind(
            "<Configure>", lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        )
        self._window = self._canvas.create_window((0, 0), window=self._container, anchor="nw")

        # Bottom bar with Help/About buttons
        bottom = tk.Frame(self.root)
        btn_help = tk.Button(bottom, text="Help", command=self._on_help)
        btn_about = tk.Button(bottom, text="About", command=self._on_about)
        btn_help.pack(side="left", padx=4)
        btn_about.pack(side="left", padx=4)

        # Layout
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._v_scroll.grid(row=0, column=1, sticky="ns")
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Mouse wheel binding
        self._bind_mousewheel(self._canvas)

        # Expose content frame for user widgets
        self.content = self._container

    # ---- public API -----------------------------------------------------
    def run(self) -> None:
        """Start Tkinter main loop."""
        self.root.mainloop()

    # ---- internals ------------------------------------------------------
    @staticmethod
    def _on_help() -> None:
        messagebox.showinfo(
            title="Help",
            message=(
                "Use the mouse wheel or scrollbar to navigate. "
                "Resize the window; content adapts with scrolling."
            ),
        )

    @staticmethod
    def _on_about() -> None:
        messagebox.showinfo(
            title="About",
            message=(
                "Developed by Dr. Zulfiyor Bakhtiyorov\n"
                "Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; "
                "National Academy of Sciences of Tajikistan\n"
                "License: MIT"
            ),
        )

    def _bind_mousewheel(self, widget: tk.Widget) -> None:
        osname = platform.system()
        if osname == "Windows":
            widget.bind_all("<MouseWheel>", self._on_mousewheel_windows)
        elif osname == "Darwin":  # macOS
            widget.bind_all("<MouseWheel>", self._on_mousewheel_macos)
        else:  # Linux/X11
            widget.bind_all("<Button-4>", self._on_mousewheel_linux)
            widget.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _on_mousewheel_windows(self, event: tk.Event) -> None:
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_macos(self, event: tk.Event) -> None:
        self._canvas.yview_scroll(int(-1 * (event.delta)), "units")

    def _on_mousewheel_linux(self, event: tk.Event) -> None:
        direction = -1 if event.num == 4 else 1
        self._canvas.yview_scroll(direction, "units")


# Created by Dr. Z. Bakhtiyorov
