import sys
import time
import math
import serial
import os
import subprocess
from pathlib import Path

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QFileDialog, QPushButton, QLineEdit, QLabel, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton, QSpinBox, QDoubleSpinBox, QMessageBox

import serial.tools.list_ports

# Program metadata constants
PROGRAM_NAME = "Dendrochronology Scanning Interface"
VERSION = "1.0"
AUTHOR = "Dr. Zulfiyor Bakhtiyorov"
INSTITUTION = "Department of Geography, University of Cambridge"
YEAR = "2025"
LICENSE_TEXT = "MIT License"

class PrinterControl(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.serial_conn = None

        # Flags
        self._stopMotion = False
        # self._stopCanonSync = False  # if you don't need Canon Sync, it can be removed

        # Flag to prevent repeated start of scanning
        self.scanningInProgress = False

        # Logic for continuous movement when holding manual move buttons
        self._moveDirection = 0
        self.moveTimer = QtCore.QTimer()
        self.moveTimer.timeout.connect(self.moveStep)

        # Persistent settings storage (QSettings)
        self.settings = QSettings("University of Cambridge", PROGRAM_NAME)

        # Current X position
        self.currentX = 0.0
        # Current Y pass (current row for disk scanning)
        self.currentYPass = 0

        # Set to track seen image files (name, modification time) to avoid duplicates
        self.seen_files = set()

        # Disk planner process + monitoring of generated routes
        self.diskPlannerProcess = None
        self.diskPlannerLaunched = False
        self.planner_launch_time = 0.0
        self.handled_routes = {}
        self.planPollTimer = QtCore.QTimer()
        self.planPollTimer.setInterval(2000)
        self.planPollTimer.timeout.connect(self.checkForPlannerOutput)

        self.initUI()
        self.setWindowTitle(PROGRAM_NAME)

        # Connect signals and slots
        self.leftButton.clicked.connect(self.moveLeftOnce)
        self.rightButton.clicked.connect(self.moveRightOnce)
        self.leftHoldButton.pressed.connect(self.startMoveLeft)
        self.leftHoldButton.released.connect(self.stopMove)
        self.rightHoldButton.pressed.connect(self.startMoveRight)
        self.rightHoldButton.released.connect(self.stopMove)
        self.connectButton.clicked.connect(self.connectSerial)
        self.startButton.clicked.connect(self.goToStart)
        self.motionButton.clicked.connect(self.startMotionDependingOnSampleType)

        self.homeButton.clicked.connect(self.goHome)
        self.stopMotionButton.clicked.connect(self.stopMotion)

        # Radio buttons
        self.coreRadio.toggled.connect(self.updateSampleTypeUI)

        # Auto-calculation
        self.autoCalcButton.clicked.connect(self.autoCalcShots)

        # Load saved settings
        self.loadSettings()
        self.updateAvailablePorts()
        self.updateSampleTypeUI()

        # Ensure polling timer only runs when planner started at least once
        self.planPollTimer.stop()

    def initUI(self):
        # 1) Create a scroll area (for main content)
        scrollArea = QtWidgets.QScrollArea(self)
        scrollArea.setWidgetResizable(True)

        # 2) Create a container widget inside the scroll area
        mainWidget = QtWidgets.QWidget()
        scrollArea.setWidget(mainWidget)  # Important: set the scroll area widget

        # 3) Create main layout for the container widget
        mainLayout = QtWidgets.QVBoxLayout(mainWidget)

        # 4) Add scroll area to the main window's layout
        rootLayout = QtWidgets.QVBoxLayout(self)
        rootLayout.addWidget(scrollArea)

        # --- Serial ---
        portLayout = QtWidgets.QHBoxLayout()
        portLabel = QtWidgets.QLabel("Serial Port:")
        self.portCombo = QtWidgets.QComboBox()
        self.portCombo.setEditable(True)
        portLayout.addWidget(portLabel)
        portLayout.addWidget(self.portCombo)
        mainLayout.addLayout(portLayout)

        baudLayout = QtWidgets.QHBoxLayout()
        baudLabel = QtWidgets.QLabel("Baud Rate:")
        self.baudEdit = QtWidgets.QLineEdit()
        baudLayout.addWidget(baudLabel)
        baudLayout.addWidget(self.baudEdit)
        mainLayout.addLayout(baudLayout)

        # Speeds
        initSpeedLayout = QtWidgets.QHBoxLayout()
        initSpeedLabel = QtWidgets.QLabel("Initial Speed (F):")
        self.initSpeedSpin = QtWidgets.QSpinBox()
        self.initSpeedSpin.setRange(1, 200000)
        initSpeedLayout.addWidget(initSpeedLabel)
        initSpeedLayout.addWidget(self.initSpeedSpin)
        mainLayout.addLayout(initSpeedLayout)

        motionSpeedLayout = QtWidgets.QHBoxLayout()
        motionSpeedLabel = QtWidgets.QLabel("Motion Speed (F):")
        self.motionSpeedSpin = QtWidgets.QSpinBox()
        self.motionSpeedSpin.setRange(1, 200000)
        motionSpeedLayout.addWidget(motionSpeedLabel)
        motionSpeedLayout.addWidget(self.motionSpeedSpin)
        mainLayout.addLayout(motionSpeedLayout)

        # Home offsets
        spindleLayout = QtWidgets.QHBoxLayout()
        spindleLabel = QtWidgets.QLabel("Spindle Size (mm):")
        self.spindleSpin = QtWidgets.QSpinBox()
        self.spindleSpin.setRange(0, 100000)
        spindleLayout.addWidget(spindleLabel)
        spindleLayout.addWidget(self.spindleSpin)
        mainLayout.addLayout(spindleLayout)

        platformLayout = QtWidgets.QHBoxLayout()
        platformLabel = QtWidgets.QLabel("Platform (mm):")
        self.platformSpin = QtWidgets.QSpinBox()
        self.platformSpin.setRange(0, 100000)
        platformLayout.addWidget(platformLabel)
        platformLayout.addWidget(self.platformSpin)
        mainLayout.addLayout(platformLayout)

        # Sample Type
        stGroup = QtWidgets.QGroupBox("Sample Type")
        stLayout = QtWidgets.QHBoxLayout()
        self.coreRadio = QtWidgets.QRadioButton("Core")
        self.diskRadio = QtWidgets.QRadioButton("Disk")
        self.coreRadio.setChecked(True)
        stLayout.addWidget(self.coreRadio)
        stLayout.addWidget(self.diskRadio)
        stGroup.setLayout(stLayout)
        mainLayout.addWidget(stGroup)

        # --- Core group ---
        self.coreGroup = QtWidgets.QGroupBox("Core Parameters")
        cLayout = QtWidgets.QVBoxLayout()

        cLenLayout = QtWidgets.QHBoxLayout()
        cLenLabel = QtWidgets.QLabel("Core Length (mm):")
        self.coreLengthSpin = QtWidgets.QSpinBox()
        self.coreLengthSpin.setRange(0, 100000)
        cLenLayout.addWidget(cLenLabel)
        cLenLayout.addWidget(self.coreLengthSpin)
        cLayout.addLayout(cLenLayout)

        shotsCoreLayout = QtWidgets.QHBoxLayout()
        shotsCoreLabel = QtWidgets.QLabel("Shots Count (Core):")
        self.shotsCountCoreSpin = QtWidgets.QSpinBox()
        self.shotsCountCoreSpin.setRange(1, 99999)
        shotsCoreLayout.addWidget(shotsCoreLabel)
        shotsCoreLayout.addWidget(self.shotsCountCoreSpin)
        cLayout.addLayout(shotsCoreLayout)

        self.coreGroup.setLayout(cLayout)
        mainLayout.addWidget(self.coreGroup)

        # --- Disk planner integration ---
        self.diskGroup = QtWidgets.QGroupBox("Disk Planner Integration")
        dLayout = QtWidgets.QVBoxLayout()

        diskInfo = QtWidgets.QLabel(
            "All disk scanning parameters are configured in the CaptuRing disk planner.\n"
            "Select Disk mode to launch the planner, configure the scan there, and export the route."
        )
        diskInfo.setWordWrap(True)
        dLayout.addWidget(diskInfo)

        self.launchPlannerButton = QPushButton("Open Disk Planner")
        self.launchPlannerButton.clicked.connect(self.launchDiskPlanner)
        dLayout.addWidget(self.launchPlannerButton)

        self.diskGroup.setLayout(dLayout)
        mainLayout.addWidget(self.diskGroup)

        # --- X-axis params ---
        xAxisGroup = QtWidgets.QGroupBox("X-axis Parameters")
        xaLayout = QtWidgets.QVBoxLayout()

        offLayout = QtWidgets.QHBoxLayout()
        offLabel = QtWidgets.QLabel("Start Offset (mm):")
        self.offsetStartSpin = QtWidgets.QSpinBox()
        self.offsetStartSpin.setRange(0, 100000)
        offLayout.addWidget(offLabel)
        offLayout.addWidget(self.offsetStartSpin)
        xaLayout.addLayout(offLayout)

        stepLayout = QtWidgets.QHBoxLayout()
        stepLabel = QtWidgets.QLabel("Step Size (mm):")
        self.stepSpin = QtWidgets.QSpinBox()
        self.stepSpin.setRange(1, 10000)
        stepLayout.addWidget(stepLabel)
        stepLayout.addWidget(self.stepSpin)
        xaLayout.addLayout(stepLayout)

        dwellLayout = QtWidgets.QHBoxLayout()
        dwellLabel = QtWidgets.QLabel("Dwell Time (sec):")
        self.dwellSpin = QtWidgets.QDoubleSpinBox()
        self.dwellSpin.setRange(0.1, 999.9)
        self.dwellSpin.setSingleStep(0.1)
        dwellLayout.addWidget(dwellLabel)
        dwellLayout.addWidget(self.dwellSpin)
        xaLayout.addLayout(dwellLayout)

        xAxisGroup.setLayout(xaLayout)
        mainLayout.addWidget(xAxisGroup)

        # --- Manual move (one-shot) ---
        manualGroup = QtWidgets.QGroupBox("Manual Move (X, one-shot)")
        manLayout = QtWidgets.QHBoxLayout()
        self.manualDistanceSpin = QtWidgets.QSpinBox()
        self.manualDistanceSpin.setRange(1, 100000)
        self.manualDistanceSpin.setValue(10)
        manLayout.addWidget(self.manualDistanceSpin)

        self.leftButton = QtWidgets.QPushButton("Left (X-)")
        manLayout.addWidget(self.leftButton)
        self.rightButton = QtWidgets.QPushButton("Right (X+)")
        manLayout.addWidget(self.rightButton)
        manualGroup.setLayout(manLayout)
        mainLayout.addWidget(manualGroup)

        # --- Manual move (hold) ---
        holdGroup = QtWidgets.QGroupBox("Manual Move (X, hold)")
        holdLayout = QtWidgets.QHBoxLayout()
        self.continuousStepSpin = QtWidgets.QSpinBox()
        self.continuousStepSpin.setRange(1, 10000)
        self.continuousStepSpin.setValue(1)
        holdLayout.addWidget(self.continuousStepSpin)

        self.leftHoldButton = QtWidgets.QPushButton("Hold Left")
        holdLayout.addWidget(self.leftHoldButton)
        self.rightHoldButton = QtWidgets.QPushButton("Hold Right")
        holdLayout.addWidget(self.rightHoldButton)
        holdGroup.setLayout(holdLayout)
        mainLayout.addWidget(holdGroup)

        # --- Canon Folder & Format ---
        canonGroup = QtWidgets.QGroupBox("Canon Folder & Format")
        canonLayout = QtWidgets.QVBoxLayout()

        folderLayout = QtWidgets.QHBoxLayout()
        folderLabel = QtWidgets.QLabel("Canon Folder:")
        self.canonFolderEdit = QLineEdit()
        
        # Нормализация при ручном вводе (когда поле теряет фокус)  # NEW
        self.canonFolderEdit.editingFinished.connect(self._normalizeCanonEdit)  # NEW
        
        self.browseFolderBtn = QPushButton("📁")
        self.browseFolderBtn.setToolTip("Choose Folder")
        self.browseFolderBtn.clicked.connect(self.chooseCanonFolder)
        

        # Кнопка "Новая папка"                             # NEW
        self.newFolderBtn = QPushButton("New Folder")      # NEW
        self.newFolderBtn.setToolTip("Create folder at path/name")  # NEW
        self.newFolderBtn.clicked.connect(self.createCanonFolder)    # NEW
        
        folderLayout.addWidget(folderLabel)
        folderLayout.addWidget(self.canonFolderEdit)
        folderLayout.addWidget(self.browseFolderBtn)
        folderLayout.addWidget(self.newFolderBtn)  # NEW
        canonLayout.addLayout(folderLayout)
        

        formatLayout = QtWidgets.QHBoxLayout()
        formatLabel = QtWidgets.QLabel("Image Format (CR3, JPG, etc.):")
        self.fileFormatEdit = QLineEdit("CR3")
        formatLayout.addWidget(formatLabel)
        formatLayout.addWidget(self.fileFormatEdit)
        canonLayout.addLayout(formatLayout)

        # Auto-calc Shots
        autoCalcLayout = QtWidgets.QHBoxLayout()
        self.autoCalcButton = QPushButton("Auto-calc Shots")
        autoCalcLayout.addWidget(self.autoCalcButton)
        canonLayout.addLayout(autoCalcLayout)

        canonGroup.setLayout(canonLayout)
        mainLayout.addWidget(canonGroup)

        # Main Buttons
        btnLayout = QtWidgets.QHBoxLayout()
        self.connectButton = QPushButton("Connect")
        btnLayout.addWidget(self.connectButton)

        self.startButton = QPushButton("Go to Start (X=0)")
        btnLayout.addWidget(self.startButton)

        self.motionButton = QPushButton("Start Scanning")
        btnLayout.addWidget(self.motionButton)

        self.stopMotionButton = QPushButton("Stop Motion")
        btnLayout.addWidget(self.stopMotionButton)

        self.homeButton = QPushButton("Home (Center)")
        btnLayout.addWidget(self.homeButton)
        mainLayout.addLayout(btnLayout)

        # Status
        self.positionLabel = QLabel("X=0.00, Y-pass=0")
        mainLayout.addWidget(self.positionLabel)

        # Additional controls: reverse scanning and zero X
        self.reverseScanButton = QPushButton("Start Reverse Scan")
        btnLayout.addWidget(self.reverseScanButton)
        self.reverseScanButton.clicked.connect(self.startReverseScan)

        self.zeroXButton = QPushButton("Zero X (G92 X0)")
        btnLayout.addWidget(self.zeroXButton)
        self.zeroXButton.clicked.connect(self.setXZero)

        self.resize(720, 1000)      # Initial window size
        self.setMinimumSize(420, 600)  # Minimum window size

        # --- Manual X Set ---
        xSetLayout = QHBoxLayout()
        self.setXSpin = QDoubleSpinBox()
        self.setXSpin.setRange(-10000, 10000)
        self.setXSpin.setDecimals(2)
        self.setXSpin.setValue(0.00)
        xSetLayout.addWidget(QLabel("Set X ="))
        xSetLayout.addWidget(self.setXSpin)

        self.setXButton = QPushButton("Set X Position")
        self.setXButton.clicked.connect(self.setCustomXPosition)
        xSetLayout.addWidget(self.setXButton)

        mainLayout.addLayout(xSetLayout)

        # Footer with developer info and About/Help buttons
        footerLayout = QHBoxLayout()
        devLabel = QLabel("Developed by Dr. Zulfiyor Bakhtiyorov")
        devLabel.setStyleSheet("font-weight: bold;")
        footerLayout.addWidget(devLabel)
        footerLayout.addStretch(1)
        aboutButton = QPushButton("About")
        helpButton = QPushButton("Help")
        aboutButton.clicked.connect(self.showAbout)
        helpButton.clicked.connect(self.showHelp)
        footerLayout.addWidget(aboutButton)
        footerLayout.addWidget(helpButton)
        rootLayout.addLayout(footerLayout)
        self.setLayout(rootLayout)

    # --- Path helpers ---  # NEW
    def _normalize_separators(self, s: str) -> str:
        """
        Приводит разделители к системному (Windows: '\\', macOS/Linux: '/'),
        удаляет повторные разделители и лишние кавычки/пробелы.
        """
        if not s:
            return s
        s = s.strip().strip('"').strip("'")
        s = s.replace('/', os.sep).replace('\\', os.sep)
        # убрать двойные слэши
        while os.sep*2 in s:
            s = s.replace(os.sep*2, os.sep)
        return s
    
    def _normalizeCanonEdit(self):  # NEW
        txt = self.canonFolderEdit.text()
        self.canonFolderEdit.setText(self._normalize_separators(txt))
    
    def createCanonFolder(self):  # NEW
        """
        Создаёт папку по строке:
        - Если введён абсолютный путь или путь с подкаталогами — создаёт его.
        - Если введено только имя (без разделителей) — создаёт внутри текущего корня:
            • если в поле указан существующий путь — внутри него,
            • иначе — в домашней директории пользователя.
        """
        raw = self.canonFolderEdit.text().strip()
        name = self._normalize_separators(raw) or "new_folder"
    
        p = Path(name)
        if not p.is_absolute() and os.sep not in name:
            # только имя без разделителей → внутрь текущего корня (если существует), иначе в HOME
            current = Path(self._normalize_separators(self.canonFolderEdit.text().strip()))
            root = current if current.exists() else current.parent
            if not root or not root.exists():
                root = Path.home()
            p = root / name
    
        try:
            p.mkdir(parents=True, exist_ok=True)
            self.canonFolderEdit.setText(self._normalize_separators(str(p)))
            QMessageBox.information(self, "Folder", f"Folder ensured:\n{p}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Cannot create folder:\n{e}")
    
    def setXZero(self):
        # Reset X coordinate to zero
        if not self.serial_conn or not self.serial_conn.isOpen():
            QMessageBox.warning(self, "Warning", "Not connected to printer.")
            return

        self.serial_conn.write(b"G92 X0\n")
        self.waitForOk(timeout=5)
        self.updateCurrentX()
        QMessageBox.information(self, "Zero X", "X position has been set to 0.")

    def setCustomXPosition(self):
        if not self.serial_conn or not self.serial_conn.isOpen():
            QMessageBox.warning(self, "Warning", "Not connected to printer.")
            return

        x_value = self.setXSpin.value()
        cmd = f"G92 X{x_value:.2f}\n"
        self.serial_conn.write(cmd.encode())
        self.waitForOk(timeout=5)
        self.updateCurrentX()
        QMessageBox.information(self, "X Set", f"X position set to {x_value:.2f} mm.")

    def updateSampleTypeUI(self):
        if self.coreRadio.isChecked():
            self.coreGroup.setEnabled(True)
            self.diskGroup.setEnabled(False)
            self.launchPlannerButton.setEnabled(False)
        else:
            self.coreGroup.setEnabled(False)
            self.diskGroup.setEnabled(True)
            self.launchPlannerButton.setEnabled(True)
            if (
                not self.diskPlannerLaunched
                or self.diskPlannerProcess is None
                or self.diskPlannerProcess.poll() is not None
            ):
                self.launchDiskPlanner()

    def autoCalcShots(self):
        stepX = self.stepSpin.value()
        if stepX <= 0:
            QMessageBox.warning(self, "Warning", "Step Size (X) cannot be <= 0.")
            return

        if self.coreRadio.isChecked():
            length_core = self.coreLengthSpin.value()
            shots_core = math.ceil(length_core / stepX)
            self.shotsCountCoreSpin.setValue(max(shots_core, 1))
        else:
            QMessageBox.information(
                self,
                "Disk planner",
                "Disk scan shot counts are configured in the CaptuRing disk planner."
                " Export a route from the planner to begin scanning.",
            )

    def updateAvailablePorts(self):
        self.portCombo.clear()
        ports = serial.tools.list_ports.comports()
        port_names = [p.device for p in ports]
        self.portCombo.addItems(port_names)

        saved_port = self.settings.value("port", "")
        if saved_port:
            idx = self.portCombo.findText(saved_port)
            if idx >= 0:
                self.portCombo.setCurrentIndex(idx)
            else:
                self.portCombo.addItem(saved_port)
                self.portCombo.setCurrentIndex(self.portCombo.count() - 1)

    def launchDiskPlanner(self):
        script_path = Path(__file__).resolve().parent / "disk_planner_xy.py"
        if not script_path.exists():
            QMessageBox.critical(self, "Disk planner", f"Planner script not found:\n{script_path}")
            return

        if self.diskPlannerProcess and self.diskPlannerProcess.poll() is None:
            self.diskPlannerLaunched = True
            QMessageBox.information(self, "Disk planner", "Planner already running.")
            return

        self.planner_launch_time = time.time()
        self.handled_routes.clear()

        try:
            self.diskPlannerProcess = subprocess.Popen([sys.executable, str(script_path), "--gui"])
            self.diskPlannerLaunched = True
            self.planPollTimer.start()
            QMessageBox.information(self, "Disk planner", "Disk planner GUI launched.")
        except Exception as e:
            QMessageBox.critical(self, "Disk planner", f"Failed to start planner:\n{e}")

    def checkForPlannerOutput(self):
        if self.diskPlannerProcess and self.diskPlannerProcess.poll() is not None:
            self.diskPlannerProcess = None

        if self.scanningInProgress:
            return

        out_dir = Path("out_xy")
        if not out_dir.exists():
            return

        try:
            candidates = sorted(out_dir.rglob("route.gcode"), key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            return

        for route in candidates:
            try:
                mtime = route.stat().st_mtime
            except FileNotFoundError:
                continue
            if mtime <= self.planner_launch_time:
                continue
            last_seen = self.handled_routes.get(route)
            if last_seen is not None and mtime <= last_seen:
                continue
            self.handled_routes[route] = mtime
            self.startRouteFromPlanner(route)
            break

    def startRouteFromPlanner(self, route_path: Path):
        if not self.serial_conn or not self.serial_conn.isOpen():
            QMessageBox.warning(self, "Disk planner", "Connect to the printer before running the generated route.")
            return

        if self.scanningInProgress:
            QMessageBox.warning(self, "Disk planner", "Another scan is already in progress.")
            return

        self.runGcodeFile(route_path)

    def runGcodeFile(self, route_path: Path):
        try:
            lines = route_path.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            QMessageBox.critical(self, "Route", f"Cannot read G-code file:\n{e}")
            return

        if not lines:
            QMessageBox.warning(self, "Route", "G-code file is empty.")
            return

        self._stopMotion = False
        self.scanningInProgress = True
        try:
            for raw in lines:
                if self._stopMotion:
                    QMessageBox.information(self, "Route", "Scan interrupted by operator.")
                    break
                stripped = raw.strip()
                if not stripped or stripped.startswith(";"):
                    continue
                cmd = stripped + "\r\n"
                try:
                    self.serial_conn.write(cmd.encode("ascii", errors="ignore"))
                except Exception as exc:
                    QMessageBox.critical(self, "Route", f"Failed to send command:\n{exc}")
                    break
                if not self.waitForOk(timeout=60):
                    QMessageBox.warning(self, "Route", f"No OK response for command:\n{stripped}")
                    break
            else:
                QMessageBox.information(self, "Route", f"Completed route from:\n{route_path}")
        finally:
            self.scanningInProgress = False

    def chooseCanonFolder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder for Canon Images")
        if folder:
            self.canonFolderEdit.setText(self._normalize_separators(folder))  # NEW
    
    def loadSettings(self):
        self.portCombo.setEditText(self.settings.value("port", "COM4"))
        self.baudEdit.setText(self.settings.value("baud", "250000"))
        self.initSpeedSpin.setValue(int(self.settings.value("init_speed", 3000)))
        self.motionSpeedSpin.setValue(int(self.settings.value("motion_speed", 50)))
        self.spindleSpin.setValue(int(self.settings.value("spindle", 420)))
        self.platformSpin.setValue(int(self.settings.value("platform", 0)))

        self.coreLengthSpin.setValue(int(self.settings.value("core_length", 200)))
        self.shotsCountCoreSpin.setValue(int(self.settings.value("shots_count_core", 5)))

        self.offsetStartSpin.setValue(int(self.settings.value("start_offset", 0)))
        self.stepSpin.setValue(int(self.settings.value("step_size", 5)))
        self.dwellSpin.setValue(float(self.settings.value("dwell", 1.0)))

        canon_path = self.settings.value("canon_folder", "F:/sample")
        canon_path = self._normalize_separators(canon_path)                 # NEW
        
        if not Path(canon_path).exists():
            folder = QFileDialog.getExistingDirectory(self, "Select Folder for Canon Images")
            if folder:
                canon_path = self._normalize_separators(folder)             # NEW
        
        self.canonFolderEdit.setText(self._normalize_separators(canon_path))  # NEW
        

    def saveSettings(self):
        self.settings.setValue("port", self.portCombo.currentText().strip())
        self.settings.setValue("baud", self.baudEdit.text().strip())
        self.settings.setValue("init_speed", self.initSpeedSpin.value())
        self.settings.setValue("motion_speed", self.motionSpeedSpin.value())
        self.settings.setValue("spindle", self.spindleSpin.value())
        self.settings.setValue("platform", self.platformSpin.value())

        self.settings.setValue("core_length", self.coreLengthSpin.value())
        self.settings.setValue("shots_count_core", self.shotsCountCoreSpin.value())

        self.settings.setValue("start_offset", self.offsetStartSpin.value())
        self.settings.setValue("step_size", self.stepSpin.value())
        self.settings.setValue("dwell", self.dwellSpin.value())

        self.settings.setValue("canon_folder", self._normalize_separators(self.canonFolderEdit.text().strip()))  # NEW

        self.settings.setValue("file_format", self.fileFormatEdit.text().strip())

        if self.coreRadio.isChecked():
            self.settings.setValue("sample_type", "core")
        else:
            self.settings.setValue("sample_type", "disk")

    def closeEvent(self, event):
        self._stopMotion = True
        self.saveSettings()
        if self.diskPlannerProcess and self.diskPlannerProcess.poll() is None:
            try:
                self.diskPlannerProcess.terminate()
            except Exception:
                pass
        self.planPollTimer.stop()
        super().closeEvent(event)

    def connectSerial(self):
        port = self.portCombo.currentText().strip()
        try:
            baud = int(self.baudEdit.text().strip())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid baud rate")
            return

        if self.serial_conn and self.serial_conn.isOpen():
            self.serial_conn.close()
            self.serial_conn = None

        try:
            self.serial_conn = serial.Serial(port, baud, timeout=1)
            time.sleep(2)
            self.serial_conn.reset_input_buffer()
            QMessageBox.information(self, "Success", f"Connected to {port}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to connect: {str(e)}")

    def waitForOk(self, timeout=10):
        if not self.serial_conn:
            return False
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print("Received:", line)
            if "ok" in line.lower():
                return True
        return False

    def updateCurrentX(self):
        if not self.serial_conn:
            return
        self.serial_conn.write(b"M114\r\n")
        start_time = time.time()
        while (time.time() - start_time) < 1.2:
            line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
            print("📡 M114 Line:", line)  # Debug output for M114 response
            if line and "X:" in line:
                parts = line.split()
                for p in parts:
                    if p.startswith("X:") and "Count" not in p:
                        try:
                            self.currentX = float(p.replace("X:", ""))
                            break  # found X value, exit loop
                        except:
                            pass
                break
        self.positionLabel.setText(f"X={self.currentX:.2f}, Y-pass={self.currentYPass}")

    def goToStart(self):
        if not self.serial_conn or not self.serial_conn.isOpen():
            QMessageBox.warning(self, "Warning", "Not connected!")
            return
        if self.scanningInProgress:
            QMessageBox.warning(self, "Warning", "Scanning in progress!")
            return

        self.serial_conn.write(b"G28 X\r\n")
        if not self.waitForOk(timeout=120):
            QMessageBox.warning(self, "Timeout", "No response after G28 X.")
            return

        self.serial_conn.write(b"G92 X0\r\n")
        self.waitForOk(timeout=10)

        init_speed = self.initSpeedSpin.value()
        start_offset = self.offsetStartSpin.value()

        cmd = f"G0 X{start_offset} F{init_speed}\r\n"
        self.serial_conn.write(cmd.encode())
        self.waitForOk(timeout=300)

        self.serial_conn.write(b"M400\r\n")
        self.waitForOk(timeout=300)
        self.updateCurrentX()

        QMessageBox.information(self, "Start", "Homing complete. Table at X=0 + offset")

    def stopMotion(self):
        self._stopMotion = True
        self.scanningInProgress = False  # Ensure flag is reset
        if self.serial_conn:
            self.serial_conn.write(b"M0\n")  # M0 = stop command
        QMessageBox.information(self, "Stopped", "Motion was stopped.")

    def startMotionDependingOnSampleType(self):
        if not self.serial_conn or not self.serial_conn.isOpen():
            QMessageBox.warning(self, "Warning", "Not connected!")
            return
        if self.scanningInProgress:
            QMessageBox.warning(self, "Warning", "Scan already in progress!")
            return

        if self.coreRadio.isChecked():
            # Reset stop flag and enable scanning flag
            self._stopMotion = False
            self.scanningInProgress = True
            self.currentYPass = 0

            # Clear seen_files set
            self.seen_files.clear()

            try:
                self.goCoreMotion()
            finally:
                self.scanningInProgress = False
        else:
            QMessageBox.information(
                self,
                "Disk planner",
                "Disk scanning is launched from the CaptuRing disk planner."
                " Export a route there to have it run automatically.",
            )

    def goCoreMotion(self):
        folder_path = self._normalize_separators(self.canonFolderEdit.text().strip())  # NEW
        folder = Path(folder_path)

        ext = self.fileFormatEdit.text().strip().lower()
        self.seen_files = set((f.name.lower(), f.stat().st_mtime) for f in folder.glob("*." + ext))
        if not folder.exists():
            QMessageBox.warning(self, "Error", f"Folder not found: {folder_path}")
            return

        offsetStart = self.offsetStartSpin.value()
        step_mm = self.stepSpin.value()
        speed = self.motionSpeedSpin.value()
        dwell = self.dwellSpin.value()

        core_length = self.coreLengthSpin.value()
        final_pos = offsetStart + core_length
        shotsCore = self.shotsCountCoreSpin.value()

        # Move to offsetStart (initial position) - if needed, ensure position is already at offsetStart
        self.updateCurrentX()
        current_pos = self.currentX  # Now start from current X position

        # current_pos = offsetStart
        shotsSoFar = 0

        while True:
            if self._stopMotion:
                break
            if shotsSoFar >= shotsCore:
                break
            if current_pos >= final_pos:
                break

            # Wait for a new image (photo)
            new_file = self.waitForNewImage(folder)
            if not new_file:
                break
            shotsSoFar += 1

            # Move the X axis to the next position
            next_pos = current_pos + step_mm
            if next_pos > final_pos:
                next_pos = final_pos

            cmd = f"G0 X{next_pos} F{speed}\r\n"
            self.serial_conn.write(cmd.encode())
            if not self.waitForOk(timeout=10):
                break
            self.serial_conn.write(b"M400\r\n")
            self.waitForOk(timeout=10)

            current_pos = next_pos
            self.updateCurrentX()

            if dwell > 0:
                time.sleep(dwell)

        # Return to X=0 (if needed)
        # self.serial_conn.write(b"G0 X0 F3000\r\n")
        # if self.waitForOk(timeout=30):
        #     self.serial_conn.write(b"M400\r\n")
        #     self.waitForOk(timeout=10)
        self.updateCurrentX()

        QMessageBox.information(self, "Core Scan",
                                f"Core scan finished. Shots: {shotsSoFar}. Table at X=0.")
        return  # Important: end to prevent any unintended restart

    # Reverse scanning movement
    def startReverseScan(self):
        if not self.serial_conn or not self.serial_conn.isOpen():
            QMessageBox.warning(self, "Warning", "Not connected!")
            return
        if self.scanningInProgress:
            QMessageBox.warning(self, "Warning", "Scan already in progress!")
            return

        self._stopMotion = False
        self.scanningInProgress = True
        self.seen_files.clear()

        try:
            if self.coreRadio.isChecked():
                self.goCoreMotionReverse()
            else:
                QMessageBox.information(self, "Info", "Reverse scan only available for core mode.")
        finally:
            self.scanningInProgress = False

    def goCoreMotionReverse(self):
        folder_path = self.canonFolderEdit.text().strip()
        folder = Path(folder_path)
        ext = self.fileFormatEdit.text().strip().lower()
        self.seen_files = set((f.name.lower(), f.stat().st_mtime) for f in folder.glob("*." + ext))
        if not folder.exists():
            QMessageBox.warning(self, "Error", f"Folder not found: {folder_path}")
            return

        self.updateCurrentX()
        current_pos = self.currentX  # start from current position

        step_mm = self.stepSpin.value()
        speed = self.motionSpeedSpin.value()
        dwell = self.dwellSpin.value()
        shotsCore = self.shotsCountCoreSpin.value()

        shotsSoFar = 0

        while True:
            if self._stopMotion:
                break
            if shotsSoFar >= shotsCore:
                break
            if current_pos <= 0:
                break

            # Wait for a new image
            new_file = self.waitForNewImage(folder)
            if not new_file:
                break
            shotsSoFar += 1

            # Next step to the left
            next_pos = current_pos - step_mm
            if next_pos < 0:
                next_pos = 0

            cmd = f"G0 X{next_pos} F{speed}\r\n"
            self.serial_conn.write(cmd.encode())
            if not self.waitForOk(timeout=10):
                break
            self.serial_conn.write(b"M400\r\n")
            self.waitForOk(timeout=10)

            current_pos = next_pos
            self.updateCurrentX()

            if dwell > 0:
                time.sleep(dwell)

        QMessageBox.information(self, "Reverse Scan", f"Reverse scan complete. Shots: {shotsSoFar}.")

    def goHome(self):
        if not self.serial_conn or not self.serial_conn.isOpen():
            QMessageBox.warning(self, "Warning", "Not connected!")
            return
        if self.scanningInProgress:
            QMessageBox.warning(self, "Warning", "Scanning in progress!")
            return

        speed = self.initSpeedSpin.value()
        spindle = self.spindleSpin.value()
        platform = self.platformSpin.value()
        home_pos = (spindle / 2) + platform

        cmd_home = f"G0 X{home_pos} F{speed}\r\n"
        self.serial_conn.write(cmd_home.encode())
        if not self.waitForOk(timeout=100):
            QMessageBox.warning(self, "Timeout", f"No response after G0 X{home_pos}.")
            return
        self.serial_conn.write(b"M400\r\n")
        self.waitForOk(timeout=1000)

        time.sleep(2.0)
        self.updateCurrentX()
        QMessageBox.information(self, "Home", f"Table moved to X={home_pos}.")

    def moveLeftOnce(self):
        if not self.serial_conn or not self.serial_conn.isOpen():
            return
        if self.scanningInProgress:
            return

        dist = self.manualDistanceSpin.value()
        speed = self.initSpeedSpin.value()
        cmd = f"G91\r\nG0 X-{dist} F{speed}\r\nG90\r\n"
        self.serial_conn.write(cmd.encode())
        self.waitForOk(timeout=100)
        self.updateCurrentX()

    def moveRightOnce(self):
        if not self.serial_conn or not self.serial_conn.isOpen():
            return
        if self.scanningInProgress:
            return

        dist = self.manualDistanceSpin.value()
        speed = self.initSpeedSpin.value()
        cmd = f"G91\nG0 X{dist} F{speed}\nG90\n"
        self.serial_conn.write(cmd.encode())
        self.waitForOk(timeout=10)
        self.updateCurrentX()

    # Hold-based movement
    def startMoveLeft(self):
        if not self.serial_conn or not self.serial_conn.isOpen():
            return
        if self.scanningInProgress:
            return
        self._moveDirection = -1
        self.moveTimer.start(200)

    def startMoveRight(self):
        if not self.serial_conn or not self.serial_conn.isOpen():
            return
        if self.scanningInProgress:
            return
        self._moveDirection = 1
        self.moveTimer.start(200)

    def stopMove(self):
        self.moveTimer.stop()
        self._moveDirection = 0

    def moveStep(self):
        if not self.serial_conn or not self.serial_conn.isOpen():
            return
        if self._moveDirection == 0:
            return
        step_val = self.continuousStepSpin.value()
        speed = self.initSpeedSpin.value()
        cmd = f"G21\r\n G91\r\nG0 X{step_val * self._moveDirection} F{speed}\r\nG90\r\n"
        self.serial_conn.write(cmd.encode())
        self.updateCurrentX()

    # If Canon Sync is not needed, this section can be removed
    # def startCanonSync(self):
    #     pass

    def waitForNewImage(self, folder: Path):
        """
        Consider not only the file name, but also the modification time, so that
        an old file isn't treated as "new" even if it happens to have the same name.
        """
        ext = self.fileFormatEdit.text().strip().lower()
        print(f"⏳ Wait for new image with extension: {ext} in {folder}")

        while True:
            if self._stopMotion:
                return None

            all_files = folder.glob("*." + ext)
            new_candidates = []
            for f in all_files:
                # key: (file_name, last_modification_time)
                modtime = f.stat().st_mtime
                key = (f.name.lower(), modtime)
                if key not in self.seen_files:
                    new_candidates.append((f, key))

            if new_candidates:
                # take the most recent one
                newest = max(new_candidates, key=lambda x: x[1][1])
                self.seen_files.add(newest[1])  # add (name, time) to seen_files set
                print(f"📷 Found new image: {newest[0].name}")
                return newest[0]

            QtCore.QThread.msleep(500)
            QtWidgets.QApplication.processEvents()

    def showAbout(self):
        about_text = (f"{PROGRAM_NAME} v{VERSION}\n"
                      f"Author: {AUTHOR}\n"
                      f"Institution: {INSTITUTION}\n"
                      f"Year: {YEAR}\n"
                      f"License: {LICENSE_TEXT}")
        QMessageBox.about(self, "About", about_text)

    def showHelp(self):
        help_text = ("<h3>Serial Connection</h3>"
                     "<p><b>Serial Port:</b> Select the serial port that connects to the motion controller (e.g., COM port for Arduino).</p>"
                     "<p><b>Baud Rate:</b> Enter the baud rate for serial communication (e.g., 250000 for Marlin firmware).</p>"
                     "<h3>Speed Settings</h3>"
                     "<p><b>Initial Speed (F):</b> The initial feed rate (speed) for rapid moves (e.g., homing or initial positioning).</p>"
                     "<p><b>Motion Speed (F):</b> The feed rate used during scanning motion between image captures.</p>"
                     "<h3>Home Position Offsets</h3>"
                     "<p><b>Spindle Size (mm):</b> The diameter (or length) of the spindle holding the sample. Used to calculate the center (home) position.</p>"
                     "<p><b>Platform (mm):</b> The offset of the platform or base from the zero position. Home position X is calculated as spindle/2 + platform.</p>"
                     "<h3>Sample Type</h3>"
                     "<p>Select <b>Core</b> for a core sample (linear scan along a single axis), or <b>Disk</b> to work with the CaptuRing disk planner integration.</p>"
                     "<h3>Core Parameters</h3>"
                     "<p><b>Core Length (mm):</b> The length of the core sample to scan along the X axis.</p>"
                     "<p><b>Shots Count (Core):</b> The number of images (shots) to capture along the core. This can be auto-calculated based on core length and step size.</p>"
                     "<h3>Disk Planner Integration</h3>"
                     "<p>All disk scanning parameters are configured inside the separate CaptuRing disk planner GUI. Selecting <b>Disk</b> launches the planner so you can adjust dimensions, segmentation, and capture order there.</p>"
                     "<p>When you export a route from the planner, this application automatically detects the generated <code>route.gcode</code> file and streams it to the connected printer.</p>"
                     "<h3>X-axis Scan Parameters</h3>"
                     "<p><b>Start Offset (mm):</b> The X position offset where scanning will start (distance from home position to the first image position).</p>"
                     "<p><b>Step Size (mm):</b> The distance to move in X between consecutive shots.</p>"
                     "<p><b>Dwell Time (sec):</b> The delay after each movement (if any) to allow settling or to control capture rate.</p>"
                     "<h3>Manual Move (X, one-shot)</h3>"
                     "<p>The spin box sets the distance for a single manual move. Click <b>Left (X-)</b> or <b>Right (X+)</b> to move the stage by that distance in the respective direction.</p>"
                     "<h3>Manual Move (X, hold)</h3>"
                     "<p>The spin box sets the step size for continuous movement. Press and hold <b>Hold Left</b> or <b>Hold Right</b> to continuously move the stage in that direction, releasing the button stops the movement.</p>"
                     "<h3>Canon Folder & Format</h3>"
                     "<p><b>Canon Folder:</b> The directory where images from the camera are saved. Use the folder button to browse and select.</p>"
                     "<p><b>Image Format:</b> The file extension of the image files (e.g., CR3, JPG) to watch for new images.</p>"
                     "<p><b>Auto-calc Shots:</b> Automatically calculate the number of shots based on the sample dimensions and step size (updates shots count fields).</p>"
                     "<h3>Main Controls</h3>"
                     "<p><b>Connect:</b> Open the serial connection to the motion controller using the selected port and baud rate.</p>"
                     "<p><b>Go to Start (X=0):</b> Home the X-axis (move to the endstop at X=0) and then move to the start offset position to begin scanning.</p>"
                     "<p><b>Start Scanning:</b> Begin the automated scanning process. For a core sample, the stage will move along X capturing images. For disk samples, trigger scanning by exporting a route from the disk planner.</p>"
                     "<p><b>Stop Motion:</b> Stop the scanning process as soon as possible.</p>"
                     "<p><b>Home (Center):</b> Move the stage to the home position (center of spindle plus platform offset).</p>"
                     "<p><b>Start Reverse Scan:</b> (Core only) Perform a reverse scan from the current position back towards X=0, capturing images in reverse order.</p>"
                     "<p><b>Zero X (G92 X0):</b> Set the current X position as 0 (reset the X-axis coordinate to zero reference).</p>"
                     "<h3>Status Display</h3>"
                     "<p>The status label at the bottom shows the current X position and the current Y pass number (for legacy disk scans).</p>"
                     "<h3>Manual X Set</h3>"
                     "<p>The <b>Set X =</b> field and <b>Set X Position</b> button allow you to manually redefine the current X coordinate. Enter a value and press the button to set the stage's current position to that X value (using G92).</p>"
                     "<h3>Footer</h3>"
                     "<p><b>Developed by Dr. Zulfiyor Bakhtiyorov:</b> Credits to the developer of this interface.</p>"
                     "<p><b>About:</b> Show program information (name, version, author, etc.).</p>"
                     "<p><b>Help:</b> Show this help information dialog.</p>")
        help_dialog = QtWidgets.QDialog(self)
        help_dialog.setWindowTitle("Help")
        help_dialog.setModal(True)
        layout = QVBoxLayout(help_dialog)
        text_edit = QtWidgets.QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml(help_text)
        text_edit.setMinimumSize(600, 400)
        layout.addWidget(text_edit)
        help_dialog.setLayout(layout)
        help_dialog.exec_()

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = PrinterControl()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()