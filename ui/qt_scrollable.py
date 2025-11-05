# Program: PyQt Scrollable App Wrapper with Help/About
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Utilities to add a scrollable area and Help/About to PyQt6 applications."""

from __future__ import annotations

from PyQt6 import QtWidgets


class ScrollableWindow(QtWidgets.QMainWindow):
    def __init__(self, title: str = "Application") -> None:
        super().__init__()
        self.setWindowTitle(title)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.content = QtWidgets.QWidget()
        self.scroll_area.setWidget(self.content)

        layout.addWidget(self.scroll_area)
        self.setCentralWidget(central)

        # Menu bar with Help/About
        bar = self.menuBar()
        help_menu = bar.addMenu("Help")
        act_help = help_menu.addAction("Help")
        act_about = help_menu.addAction("About")
        act_help.triggered.connect(self._on_help)
        act_about.triggered.connect(self._on_about)

    def _on_help(self) -> None:
        QtWidgets.QMessageBox.information(self, "Help", "Use mouse wheel or scrollbar to navigate.")

    def _on_about(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "About",
            (
                "Developed by Dr. Zulfiyor Bakhtiyorov\n"
                "Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; "
                "National Academy of Sciences of Tajikistan\n"
                "License: MIT"
            ),
        )


# Created by Dr. Z. Bakhtiyorov
