# Program: Edsdk Runtime
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge;
#   Xinjiang Institute of Ecology and Geography;
#   National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Runtime bindings and helpers for interacting with Canon EDSDK cameras."""

import ctypes as C
import os
import pathlib
import queue
import threading
import time
from ctypes import POINTER, Structure, byref, c_int32, c_uint32, c_uint64, c_void_p

# --- Import only the required items from constants.py ---
# ---- Bind kEds* names to numbers from enum classes/ALL_CONSTANTS ----
# ---- constants import + safe defaults (must stay before _bind and its calls) ----
ALL_CONSTANTS = {}
EdsPropertyID = None
EdsCameraCommand = None
EdsObjectEvent = None
EdsStateEvent = None
EdsPropertyEvent = None
EdsSaveTo = None

# First attempt a relative import (package mode), then direct import.
try:
    from .constants import ALL_CONSTANTS as _AC

    ALL_CONSTANTS = _AC
    try:
        from .constants import EdsCameraCommand as _CMD
        from .constants import EdsObjectEvent as _OE
        from .constants import EdsPropertyEvent as _PE
        from .constants import EdsPropertyID as _PID
        from .constants import EdsStateEvent as _SE

        (
            EdsPropertyID,
            EdsCameraCommand,
            EdsObjectEvent,
            EdsStateEvent,
            EdsPropertyEvent,
        ) = (_PID, _CMD, _OE, _SE, _PE)
    except Exception:
        pass
    try:
        from .constants import EdsSaveTo as _ST

        EdsSaveTo = _ST
    except Exception:
        pass
except Exception:
    try:
        from constants import ALL_CONSTANTS as _AC

        ALL_CONSTANTS = _AC
        try:
            from constants import EdsCameraCommand as _CMD
            from constants import EdsObjectEvent as _OE
            from constants import EdsPropertyEvent as _PE
            from constants import EdsPropertyID as _PID
            from constants import EdsStateEvent as _SE

            (
                EdsPropertyID,
                EdsCameraCommand,
                EdsObjectEvent,
                EdsStateEvent,
                EdsPropertyEvent,
            ) = (_PID, _CMD, _OE, _SE, _PE)
        except Exception:
            pass
        try:
            from constants import EdsSaveTo as _ST

            EdsSaveTo = _ST
        except Exception:
            pass
    except Exception:
        pass


def _bind(name: str, default=None):
    """Assign a number to the global variable `name`.

    Prefers enum members, then ALL_CONSTANTS[name], finally the provided default.
    """
    val = None
    for enum_cls in (
        EdsSaveTo,
        EdsPropertyID,
        EdsCameraCommand,
        EdsObjectEvent,
        EdsStateEvent,
        EdsPropertyEvent,
    ):
        if enum_cls is not None and hasattr(enum_cls, name):
            val = getattr(enum_cls, name)
            break
    if val is None:
        val = ALL_CONSTANTS.get(name)
    if val is None:
        val = default
    if val is not None:
        globals()[name] = int(val)


# ---- Bind frequently used kEds* ----
# SaveTo
_bind("kEdsSaveTo_Camera", 1)
_bind("kEdsSaveTo_Host", 2)
_bind("kEdsSaveTo_Both", 3)
_bind("kEdsPropID_SaveTo")

# UI Lock/Unlock
_bind("kEdsCameraStatusCommand_UILock")
_bind("kEdsCameraStatusCommand_UIUnLock")

# EVF and coordinate helpers
_bind("kEdsPropID_Evf_OutputDevice")
_bind("kEdsEvfOutputDevice_PC")
_bind("kEdsEvfOutputDevice_TFT")
_bind("kEdsPropID_Evf_CoordinateSystem")
_bind("kEdsPropID_Evf_Zoom")
_bind("kEdsPropID_Evf_ZoomRect")
_bind("kEdsPropID_Evf_ZoomPosition")
_bind("kEdsPropID_Evf_TouchCoordinates")
_bind("kEdsPropID_Evf_VisibleRect")
_bind("kEdsPropID_Evf_TouchAFPosition")


# Commands/events (when code references kEds*)
_bind("kEdsCameraCommand_DoEvfAf")
_bind("kEdsCameraCommand_PressShutterButton")
_bind("kEdsCameraCommand_ShutterButton_Halfway")
_bind("kEdsCameraCommand_ShutterButton_Completely")
_bind("kEdsCameraCommand_ShutterButton_OFF")

_bind("kEdsObjectEvent_All")
_bind("kEdsObjectEvent_DirItemRequestTransfer")
_bind("kEdsObjectEvent_DirItemCreated")
_bind("kEdsPropID_Evf_VisibleRect")

# ------ Types/structures ------
EdsError = c_int32
EdsInt32 = c_int32
EdsUInt32 = c_uint32
EdsUInt64 = c_uint64
EdsBool = c_uint32

EdsBaseRef = c_void_p
EdsCameraListRef = c_void_p
EdsCameraRef = c_void_p
EdsStreamRef = c_void_p
EdsEvfImageRef = c_void_p
EdsDirectoryItemRef = c_void_p


class EdsCapacity(Structure):
    _fields_ = [
        ("NumberOfFreeClusters", EdsUInt32),
        ("BytesPerSector", EdsUInt32),
        ("Reset", EdsUInt32),
    ]


# --- Canon EDSDK geometry types (official) ---
class EdsPoint(Structure):
    _fields_ = [
        ("x", c_int32),
        ("y", c_int32),
    ]


class EdsSize(Structure):
    _fields_ = [
        ("width", c_uint32),
        ("height", c_uint32),
    ]


class EdsRect(Structure):
    _fields_ = [
        ("point", EdsPoint),
        ("size", EdsSize),
    ]


class EdsDirectoryItemInfo(Structure):
    _fields_ = [
        ("size", EdsUInt64),
        ("isFolder", EdsBool),
        ("groupID", EdsUInt32),
        ("option", EdsUInt32),
        ("szFileName", C.c_char * 256),
        ("format", EdsUInt32),
        ("dateTime", EdsUInt32),
        ("reserved", EdsUInt32),
    ]


class EdsPropertyDesc(Structure):
    _fields_ = [
        ("form", EdsUInt32),
        ("access", c_int32),
        ("numElements", c_int32),
        ("propDesc", c_int32 * 128),
    ]


# ------ DLL loading ------
def _load_lib():
    if os.name == "nt":
        try:
            os.add_dll_directory(os.getcwd())
            os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)))
        except Exception:
            pass
        return C.WinDLL("EDSDK.dll")
    for n in ("libEDSDK.so", "libEDSDK.dylib"):
        try:
            return C.CDLL(n)
        except OSError:
            pass
    raise OSError("Failed to load the EDSDK library")


_lib = _load_lib()


def _proto(fn, restype, *argtypes):
    f = getattr(_lib, fn)
    f.restype = restype
    f.argtypes = list(argtypes)
    return f


# ------ SDK functions ------
EdsInitializeSDK = _proto("EdsInitializeSDK", EdsError)
EdsTerminateSDK = _proto("EdsTerminateSDK", EdsError)
EdsGetCameraList = _proto("EdsGetCameraList", EdsError, POINTER(EdsCameraListRef))
EdsGetChildCount = _proto("EdsGetChildCount", EdsError, EdsBaseRef, POINTER(EdsInt32))
EdsGetChildAtIndex = _proto(
    "EdsGetChildAtIndex", EdsError, EdsBaseRef, EdsInt32, POINTER(EdsBaseRef)
)
EdsOpenSession = _proto("EdsOpenSession", EdsError, EdsCameraRef)
EdsCloseSession = _proto("EdsCloseSession", EdsError, EdsCameraRef)

EdsGetPropertyData = _proto(
    "EdsGetPropertyData", EdsError, EdsBaseRef, EdsUInt32, EdsInt32, EdsUInt32, c_void_p
)
EdsSetPropertyData = _proto(
    "EdsSetPropertyData", EdsError, EdsBaseRef, EdsUInt32, EdsInt32, EdsUInt32, c_void_p
)

EdsSendCommand = _proto("EdsSendCommand", EdsError, EdsCameraRef, EdsUInt32, EdsUInt32)
EdsSendStatusCommand = _proto(
    "EdsSendStatusCommand", EdsError, EdsCameraRef, EdsUInt32, EdsUInt32
)
EdsGetPropertyDesc = _proto(
    "EdsGetPropertyDesc", EdsError, EdsBaseRef, EdsUInt32, POINTER(EdsPropertyDesc)
)


if os.name == "nt":
    CALLBACK = C.WINFUNCTYPE
else:
    CALLBACK = C.CFUNCTYPE
OBJECT_HANDLER = CALLBACK(EdsError, EdsUInt32, EdsBaseRef, c_void_p)
EdsSetObjectEventHandler = _proto(
    "EdsSetObjectEventHandler",
    EdsError,
    EdsCameraRef,
    EdsUInt32,
    OBJECT_HANDLER,
    c_void_p,
)
EdsRetain = _proto("EdsRetain", EdsError, EdsBaseRef)
try:
    EdsGetEvent = _proto("EdsGetEvent", EdsError)
except AttributeError:
    EdsGetEvent = None

EdsGetDirectoryItemInfo = _proto(
    "EdsGetDirectoryItemInfo",
    EdsError,
    EdsDirectoryItemRef,
    POINTER(EdsDirectoryItemInfo),
)
EdsDownload = _proto(
    "EdsDownload", EdsError, EdsDirectoryItemRef, EdsUInt64, EdsStreamRef
)
EdsDownloadComplete = _proto("EdsDownloadComplete", EdsError, EdsDirectoryItemRef)
EdsRelease = _proto("EdsRelease", EdsError, EdsBaseRef)

EdsCreateMemoryStream = _proto(
    "EdsCreateMemoryStream", EdsError, EdsUInt32, POINTER(EdsStreamRef)
)
EdsGetPointer = _proto("EdsGetPointer", EdsError, EdsStreamRef, POINTER(c_void_p))
EdsGetLength = _proto("EdsGetLength", EdsError, EdsStreamRef, POINTER(EdsUInt64))
EdsCreateEvfImageRef = _proto(
    "EdsCreateEvfImageRef", EdsError, EdsStreamRef, POINTER(EdsEvfImageRef)
)
EdsDownloadEvfImage = _proto(
    "EdsDownloadEvfImage", EdsError, EdsCameraRef, EdsEvfImageRef
)

EdsSetCapacity = _proto("EdsSetCapacity", EdsError, EdsCameraRef, POINTER(EdsCapacity))


def _ok(rc, where=""):
    if rc != 0:
        raise RuntimeError(f"EDSDK error {rc} at {where}")
    return rc


# ------ Camera class ------
class Camera:
    def __init__(self):
        self.cam = EdsCameraRef()
        self._session = False
        self._obj_cb = None
        self._queue = queue.Queue()
        self._download_dir = None
        self._run_events = False
        self._ev_thread = None
        # Live View
        self._lv_run = False
        self._lv_lock = threading.Lock()
        self._last_jpeg = None
        self._lv_pause = False  # pause Live View during critical operations
        # Mapping flags (quick axis swap/mirroring)
        self._map_swap_xy = False
        self._map_mirror_x = False
        self._map_mirror_y = False

        # --- Mapping flags: quickly correct swapped or mirrored axes ---
        self._map_swap_xy = False
        self._map_mirror_x = False
        self._map_mirror_y = False
        self._cs_w = 0
        self._cs_h = 0
        self._vr = None  # VisibleRect cache: (vx, vy, vw, vh)
        self._zr = None  # ZoomRect cache: (zx, zy, zw, zh)

        # --- EVF coordinate system (for AF/zoom) ---
        self._cs_w = 3120  # default AF grid: 3:2
        self._cs_h = 2080

        # --- EVF image size (JPEG preview), used only for UI/debug
        self._img_w = 0
        self._img_h = 0

        # --- Rectangle caches (as used previously)
        self._vr = None  # (vx,vy,vw,vh)
        self._zr = None  # (zx,zy,zw,zh)

        self._ofs_x = 0
        self._ofs_y = 0
        self._ofs_ready = False  # enable compensation only after calibration

    def _ensure_output_pc_tft(
        self, *, pc: bool = True, tft: bool = True, quiet: bool = True
    ) -> None:
        """Re-apply Evf_OutputDevice bitmask. Call around AF/zoom operations to keep TFT alive."""
        DEV_PC = globals().get("kEdsEvfOutputDevice_PC", 0x02)
        DEV_TFT = globals().get("kEdsEvfOutputDevice_TFT", 0x01)
        mask_val = (DEV_PC if pc else 0) | (DEV_TFT if tft else 0)
        try:
            mask = C.c_uint32(mask_val)
            _ok(
                EdsSetPropertyData(
                    self.cam,
                    kEdsPropID_Evf_OutputDevice,
                    0,
                    C.sizeof(mask),
                    byref(mask),
                ),
                f"Set Evf_OutputDevice=0x{mask_val:02X}",
            )
            if not quiet:
                cur = C.c_uint32()
                try:
                    EdsGetPropertyData(
                        self.cam,
                        kEdsPropID_Evf_OutputDevice,
                        0,
                        C.sizeof(cur),
                        byref(cur),
                    )
                    print(f"[EVF] OutputDevice now 0x{cur.value:02X}")
                except Exception:
                    pass
        except Exception:
            # some models briefly fail between states; ignore it
            pass

    def set_af_offset(self, ofs_x: float, ofs_y: float):
        """Explicitly set AF offset in EVF pixels."""
        self._ofs_x = float(ofs_x)
        self._ofs_y = float(ofs_y)
        self._ofs_ready = True
        print(f"[AF OFFSET] Offset set: x={self._ofs_x:.1f}px, y={self._ofs_y:.1f}px")

    def set_map_flags(self, *, swap_xy=None, mirror_x=None, mirror_y=None):
        """Enable or disable axis swap and mirroring for click mapping."""
        if swap_xy is not None:
            self._map_swap_xy = bool(swap_xy)
        if mirror_x is not None:
            self._map_mirror_x = bool(mirror_x)
        if mirror_y is not None:
            self._map_mirror_y = bool(mirror_y)
        print(
            f"[MAP FLAGS] swap_xy={self._map_swap_xy} mirror_x={self._map_mirror_x} mirror_y={self._map_mirror_y}"
        )

    # basic operations
    def open(self, index=0):
        _ok(EdsInitializeSDK(), "EdsInitializeSDK")
        cam_list = EdsCameraListRef()
        _ok(EdsGetCameraList(byref(cam_list)), "EdsGetCameraList")
        count = EdsInt32(0)
        _ok(EdsGetChildCount(cam_list, byref(count)), "EdsGetChildCount(list)")
        if count.value <= index:
            raise RuntimeError(
                f"Camera {index} not found; total detected: {count.value}"
            )
        cam_base = EdsBaseRef()
        _ok(
            EdsGetChildAtIndex(cam_list, EdsInt32(index), byref(cam_base)),
            "EdsGetChildAtIndex",
        )
        self.cam = C.cast(cam_base, EdsCameraRef)
        _ok(EdsOpenSession(self.cam), "EdsOpenSession")
        self._session = True

        # event pump
        self._run_events = True
        self._ev_thread = threading.Thread(target=self._event_pump, daemon=True)
        self._ev_thread.start()
        return True

    def _event_pump(self):
        while self._run_events:
            try:
                if EdsGetEvent is not None:
                    EdsGetEvent()
            except Exception:
                pass
            time.sleep(0.03)

    # SaveTo / download
    def set_save_to(self, mode: str = "both"):
        mode_map = {
            "camera": kEdsSaveTo_Camera,
            "host": kEdsSaveTo_Host,
            "both": kEdsSaveTo_Both,
        }
        val = mode_map.get(mode, kEdsSaveTo_Both)

        locked = self._try_uilock("SaveTo")
        try:
            v = EdsUInt32(val)
            _ok(
                EdsSetPropertyData(
                    self.cam, kEdsPropID_SaveTo, 0, C.sizeof(v), byref(v)
                ),
                f"Set SaveTo={mode}",
            )
            # report large capacity so the camera does not buffer to card
            cap = EdsCapacity(EdsUInt32(0x7FFFFFFF), EdsUInt32(512), EdsUInt32(1))
            _ok(EdsSetCapacity(self.cam, byref(cap)), "EdsSetCapacity")
        finally:
            if locked:
                try:
                    _ok(
                        EdsSendStatusCommand(
                            self.cam, kEdsCameraStatusCommand_UIUnLock, 0
                        ),
                        "UIUnLock SaveTo",
                    )
                except Exception:
                    pass
        return True

    def get_save_to(self):
        v = EdsUInt32(0)
        _ok(
            EdsGetPropertyData(self.cam, kEdsPropID_SaveTo, 0, C.sizeof(v), byref(v)),
            "Get SaveTo",
        )
        return v.value

    def _on_object(self, inEvent, inRef, inContext):
        try:
            # print("EVENT:", hex(int(inEvent)))  # handy for debugging
            try:
                from constants import EdsObjectEvent

                match_events = (
                    EdsObjectEvent.kEdsObjectEvent_DirItemRequestTransfer,
                    EdsObjectEvent.kEdsObjectEvent_DirItemCreated,
                )
            except Exception:
                match_events = (
                    kEdsObjectEvent_DirItemRequestTransfer,
                    kEdsObjectEvent_DirItemCreated,
                )

            if inEvent in match_events:
                EdsRetain(inRef)
                self._queue.put(inRef)
                return 0
        except Exception as e:
            print("ObjectEvent error:", e)
        return 0

    def _ensure_object_handler(self):
        if self._obj_cb is not None:
            return

        @OBJECT_HANDLER
        def cb(inEvent, inRef, inContext):
            return self._on_object(inEvent, inRef, inContext)

        self._obj_cb = cb
        try:
            from constants import EdsObjectEvent

            event_mask = EdsObjectEvent.kEdsObjectEvent_All
        except Exception:
            # if enum generation failed, fall back to kEdsObjectEvent_All
            event_mask = kEdsObjectEvent_All

        _ok(
            EdsSetObjectEventHandler(self.cam, event_mask, self._obj_cb, None),
            "EdsSetObjectEventHandler",
        )

    def set_download_dir(self, path: str):
        p = pathlib.Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._download_dir = p

    def _download_diritem_to(
        self, dir_item: EdsDirectoryItemRef, folder: pathlib.Path
    ) -> pathlib.Path:
        info = EdsDirectoryItemInfo()
        _ok(EdsGetDirectoryItemInfo(dir_item, byref(info)), "EdsGetDirectoryItemInfo")
        name = (info.szFileName.split(b"\x00", 1)[0]).decode(
            errors="ignore"
        ) or "image.jpg"
        dst = folder / name
        stream = EdsStreamRef()
        _ok(EdsCreateMemoryStream(0, byref(stream)), "EdsCreateMemoryStream")
        try:
            _ok(EdsDownload(dir_item, EdsUInt64(info.size), stream), "EdsDownload")
            _ok(EdsDownloadComplete(dir_item), "EdsDownloadComplete")
            p_ptr = c_void_p()
            ln = EdsUInt64(0)
            _ok(EdsGetPointer(stream, byref(p_ptr)), "EdsGetPointer")
            _ok(EdsGetLength(stream, byref(ln)), "EdsGetLength")
            data = (C.c_ubyte * ln.value).from_address(p_ptr.value)
            with open(dst, "wb") as f:
                f.write(bytes(data))
        finally:
            try:
                _ok(EdsRelease(stream), "EdsRelease(stream)")
            except Exception:
                pass
            try:
                _ok(EdsRelease(dir_item), "EdsRelease(item)")
            except Exception:
                pass
        return dst

    def take_picture(self):
        if "kEdsCameraCommand_TakePicture" in globals():
            _ok(
                EdsSendCommand(self.cam, kEdsCameraCommand_TakePicture, 0),
                "TakePicture",
            )
            return True
        # fallback: emulate shutter button
        _ok(
            EdsSendCommand(
                self.cam,
                kEdsCameraCommand_PressShutterButton,
                kEdsCameraCommand_ShutterButton_Halfway,
            ),
            "Shutter Half",
        )
        time.sleep(0.1)
        _ok(
            EdsSendCommand(
                self.cam,
                kEdsCameraCommand_PressShutterButton,
                kEdsCameraCommand_ShutterButton_Completely,
            ),
            "Shutter Full",
        )
        time.sleep(0.2)
        _ok(
            EdsSendCommand(
                self.cam,
                kEdsCameraCommand_PressShutterButton,
                kEdsCameraCommand_ShutterButton_OFF,
            ),
            "Shutter Off",
        )
        return True

    def shoot_and_download(
        self, dest_dir: str, timeout_s: float = 20.0
    ) -> pathlib.Path:
        if self._download_dir is None:
            self.set_download_dir(dest_dir)
        self._ensure_object_handler()
        self.set_save_to("both")
        self.take_picture()
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                item = self._queue.get(timeout=0.2)
                return self._download_diritem_to(item, self._download_dir)
            except queue.Empty:
                continue
        raise TimeoutError("Timed out waiting for file from camera (ObjectEvent)")

    def _sleep_pump(self, seconds: float):
        """Pause while pumping EDSDK events so the camera can finish busy work."""
        end = time.time() + seconds
        while time.time() < end:
            try:
                if EdsGetEvent is not None:
                    EdsGetEvent()
            except Exception:
                pass
            time.sleep(0.02)

    # ---- settling/wait helpers -------------------------------------------------
    def _wait_center_converged(
        self,
        target_x: int,
        target_y: int,
        *,
        tol_x: int = 4,
        tol_y: int = 70,
        timeout_s: float = 0.8,
        prefer_taf: bool = True,
    ) -> bool:
        """Poll EVF center (TouchAF position or ZoomRect center) until it converges to target.
        Tolerances are asymmetric on purpose: many Canon bodies quantize Y in ~60-70px steps.
        Returns True if converged within timeout.
        """
        t_end = time.time() + max(0.05, float(timeout_s))
        got_x = got_y = None
        while time.time() < t_end:
            # Prefer TouchAF position when available (faster to update), else ZoomRect center.
            if prefer_taf and getattr(self, "_taf", None):
                got_x, got_y = self._taf
            elif self._zr:
                zx, zy, zw, zh = self._zr
                got_x, got_y = zx + zw // 2, zy + zh // 2
            else:
                self._sleep_pump(0.04)
                continue
            if (
                abs(int(got_x) - int(target_x)) <= tol_x
                and abs(int(got_y) - int(target_y)) <= tol_y
            ):
                return True
            self._sleep_pump(0.04)
        return False

    def _try_uilock(self, tag: str, retries: int = 8, backoff: float = 0.06) -> bool:
        """Reliable UILock helper with retries for busy cameras (error 129).
        Returns True when the lock succeeded; otherwise False to continue without it.
        """
        for i in range(retries):
            try:
                _ok(
                    EdsSendStatusCommand(self.cam, kEdsCameraStatusCommand_UILock, 0),
                    f"UILock {tag}",
                )
                return True
            except Exception as e:
                if "error 129" in str(e):
                    # camera is busy - wait and pump events
                    self._sleep_pump(backoff + i * 0.02)
                    continue
                # other errors break the loop
                return False
        return False

    def get_evf_coords(self) -> tuple[int, int]:
        """Return EVF coordinate system (w,h) for AF/Zoom.
        This is NOT JPEG size. Ground truth is kEdsPropID_Evf_CoordinateSystem.
        We DO NOT upscale to JPEG dimensions when rects are unknown.
        """
        w = int(self._cs_w or 0)
        h = int(self._cs_h or 0)
        if not (w and h):
            # Safe default until first Evf_CoordinateSystem frame arrives
            w, h = 3120, 2080
        # Only extend by EVF rectangles (never by JPEG image size)
        if self._zr:
            zx, zy, zw, zh = self._zr
            w = max(w, int(zx + zw))
            h = max(h, int(zy + zh))
        if self._vr:
            vx, vy, vw, vh = self._vr
            w = max(w, int(vx + vw))
            h = max(h, int(vy + vh))
        self._cs_w, self._cs_h = int(w), int(h)
        return int(w), int(h)

    def _jpeg_size(self) -> tuple[int, int]:
        """Try to read (w, h) from self._last_jpeg (SOF0/SOF2). Returns (0, 0) on failure."""
        d = self._last_jpeg
        if not d:
            return 0, 0
        # fast SOF0/SOF2 parser
        data = memoryview(d)
        i = 0
        n = len(data)
        try:
            if data[0] != 0xFF or data[1] != 0xD8:
                return 0, 0
            i = 2
            while i + 3 < n:
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                # SOF0=0xC0, SOF2=0xC2
                if marker in (0xC0, 0xC2) and i + 8 < n:
                    # segment length
                    # len_hi,len_lo = data[i+2], data[i+3]
                    # next bytes: precision(1), height(2), width(2)
                    h = (data[i + 5] << 8) | data[i + 6]
                    w = (data[i + 7] << 8) | data[i + 8]
                    return int(w), int(h)
                # length of the current segment
                seglen = (data[i + 2] << 8) | data[i + 3]
                i += 2 + seglen
        except Exception:
            return 0, 0
        return 0, 0

    def _map_norm_to_evf(self, nx: float, ny: float) -> EdsPoint:
        """Map normalized (nx, ny) in [0..1] to the EVF coordinate system as a *desired center*.
        VisibleRect offset (vx,vy) is applied; optional AF offset calibration is subtracted.
        Returns: EdsPoint(center_x, center_y) in EVF coordinates.
        """
        w, h = self.get_evf_coords()
        nx = min(max(float(nx), 0.0), 1.0)
        ny = min(max(float(ny), 0.0), 1.0)

        # Optional axis transforms
        if getattr(self, "_map_swap_xy", False):
            nx, ny = ny, nx
        if getattr(self, "_map_mirror_x", False):
            nx = 1.0 - nx
        if getattr(self, "_map_mirror_y", False):
            ny = 1.0 - ny

        # VisibleRect (vx,vy,vw,vh) -> center inside EVF coordinate system
        vx, vy, vw, vh = 0, 0, w, h
        if self._vr:
            vx, vy, vw, vh = self._vr
        cx = vx + int(round(nx * (max(1, vw) - 1)))
        cy = vy + int(round(ny * (max(1, vh) - 1)))

        # Apply calibrated AF offset (if any)
        if getattr(self, "_ofs_ready", False):
            cx -= int(round(self._ofs_x))
            cy -= int(round(self._ofs_y))

        # Clip to EVF coordinate system
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))
        return EdsPoint(cx, cy)

    def _ui_to_norm_image(self, x_ui: int, y_ui: int):
        """Convert preview-widget click coordinates (with letterbox bars) to normalized
        image coordinates [0..1] without the bars.
        Requires widget sizes self._ui_w/self._ui_h and EVF frame size self._cs_w/self._cs_h.
        """
        ui_w, ui_h = int(self._ui_w), int(self._ui_h)
        evf_w, evf_h = int(self._cs_w), int(self._cs_h)

        # fit EVF frame into widget while preserving aspect ratio
        s = min(ui_w / evf_w, ui_h / evf_h)
        draw_w = int(round(evf_w * s))
        draw_h = int(round(evf_h * s))
        off_x = (ui_w - draw_w) // 2
        off_y = (ui_h - draw_h) // 2

        # clamp to the drawn area (without bars)
        x = max(0, min(x_ui - off_x, draw_w))
        y = max(0, min(y_ui - off_y, draw_h))

        # normalize to the EVF image
        nx = x / max(1, draw_w - 1)
        ny = y / max(1, draw_h - 1)
        # defensive clamp
        if nx < 0:
            nx = 0.0
        if ny < 0:
            ny = 0.0
        if nx > 1:
            nx = 1.0
        if ny > 1:
            ny = 1.0
        return nx, ny

    def debug_dump_evf(self):
        w, h = self.get_evf_coords()
        print(f"[DBG] Evf_CoordinateSystem: {w} x {h}")
        if self._vr:
            vx, vy, vw, vh = self._vr
            print(f"[DBG] VisibleRect: ({vx},{vy}) {vw}x{vh}")
        else:
            print("[DBG] VisibleRect: <no cache yet>")
        if self._zr:
            zx, zy, zw, zh = self._zr
            print(f"[DBG] ZoomRect:    ({zx},{zy}) {zw}x{zh}")
        else:
            print("[DBG] ZoomRect:    <no cache yet>")

    def set_zoom_pos_norm(
        self, nx: float, ny: float, retries: int = 2, tol_px: int = 2, **_ignore
    ) -> bool:
        """Move the zoom window to normalized EVF coordinates (nx, ny).
        tol_px is kept for compatibility with the UI; currently it only affects logging.
        Extra keyword arguments from the UI end up in **_ignore and are ignored.
        """
        # (nx,ny) -> desired center in EVF pixels (accounts for VisibleRect and calibration)
        center = self._map_norm_to_evf(float(nx), float(ny))

        # Convert desired center to top-left corner required by Evf_ZoomPosition
        w, h = self.get_evf_coords()
        if self._zr:
            _, _, zw, zh = self._zr
        else:
            # Conservative fallback if ZoomRect not yet cached (1x ~ approx 1/5 of CS size)
            zw, zh = max(1, w // 5), max(1, h // 5)
        tl_x = int(round(center.x - zw // 2))
        tl_y = int(round(center.y - zh // 2))
        # Keep the whole rect inside EVF
        tl_x = max(0, min(tl_x, max(0, w - zw)))
        tl_y = max(0, min(tl_y, max(0, h - zh)))
        pt = EdsPoint(tl_x, tl_y)

        locked = self._try_uilock("ZOOM_POS")
        ok = False
        try:
            # Keep the previous ZoomRect cache to avoid logging stale data
            zr_before = self._zr
            _ok(
                EdsSetPropertyData(
                    self.cam, kEdsPropID_Evf_ZoomPosition, 0, C.sizeof(pt), byref(pt)
                ),
                "Set Evf_ZoomPosition",
            )
            ok = True
            # Wait a little while Live View updates ZoomRect to avoid stale logs
            for _ in range(3):
                self._sleep_pump(0.04)
                if self._zr != zr_before:
                    break
            # Wait for ZoomRect readback to converge
            self._wait_center_converged(
                int(center.x),
                int(center.y),
                tol_x=max(2, tol_px),
                tol_y=70,
                timeout_s=0.9,
                prefer_taf=False,
            )
            # Log only if the final position is still far from the target
            if self._zr:
                zx, zy, zw, zh = self._zr
                cx = zx + zw // 2
                cy = zy + zh // 2
                dx = int(center.x) - cx
                dy = int(center.y) - cy
                if abs(dx) > max(2, tol_px) or abs(dy) > 70:
                    print(
                        f"[ZOOM] center_target=({int(center.x)},{int(center.y)}) "
                        f"center_actual=({cx},{cy}) diff=({dx},{dy}) tol=({max(2,tol_px)},70)px"
                    )
        except Exception:
            ok = False
        finally:
            if locked:
                try:
                    _ok(
                        EdsSendStatusCommand(
                            self.cam, kEdsCameraStatusCommand_UIUnLock, 0
                        ),
                        "UIUnLock ZOOM_POS",
                    )
                except Exception:
                    pass
        return bool(ok)

    def set_evf_zoom_keep_center(
        self, level: int, nx: float, ny: float, readback: bool = True
    ):
        mapping = {1: [1, 0], 5: [5, 1], 10: [10, 2]}
        self._lv_pause = True
        locked = self._try_uilock("ZOOM")
        try:
            # 0) ensure 1x zoom and pin the center
            try:
                vv = EdsUInt32(mapping[1][0])
                _ok(
                    EdsSetPropertyData(
                        self.cam, kEdsPropID_Evf_Zoom, 0, C.sizeof(vv), byref(vv)
                    ),
                    "Pre-set Evf_Zoom=1x",
                )
                self._sleep_pump(0.08)
            except Exception:
                pass
            self.set_zoom_pos_norm(nx, ny, retries=8)

            last_exc = None
            for raw in mapping.get(level, [level]):
                try:
                    # 1) set zoom level
                    vv = EdsUInt32(raw)
                    _ok(
                        EdsSetPropertyData(
                            self.cam, kEdsPropID_Evf_Zoom, 0, C.sizeof(vv), byref(vv)
                        ),
                        f"Set Evf_Zoom target={level}x raw={raw}",
                    )
                    self._sleep_pump(0.10)
                    # 2) re-apply the desired center
                    self.set_zoom_pos_norm(nx, ny, retries=10, tol_px=2)

                    if readback:
                        cur = EdsUInt32(0)
                        try:
                            _ok(
                                EdsGetPropertyData(
                                    self.cam,
                                    kEdsPropID_Evf_Zoom,
                                    0,
                                    C.sizeof(cur),
                                    byref(cur),
                                ),
                                "Get Evf_Zoom",
                            )
                            return int(cur.value)
                        except Exception:
                            return raw
                    return raw
                except Exception as e:
                    last_exc = e
                    self._sleep_pump(0.06)
            if last_exc:
                raise last_exc
        finally:
            if locked:
                try:
                    _ok(
                        EdsSendStatusCommand(
                            self.cam, kEdsCameraStatusCommand_UIUnLock, 0
                        ),
                        "UIUnLock ZOOM",
                    )
                except Exception:
                    pass
            self._lv_pause = False

    def nudge_zoom_norm(self, dnx: float, dny: float):
        """Shift the zoom center by a relative delta while keeping zoom mode."""
        w, h = self.get_evf_coords()
        # center based on ZoomRect cache, else full frame center
        if self._zr:
            zx, zy, zw, zh = self._zr
            cx = zx + zw // 2
            cy = zy + zh // 2
        else:
            cx, cy = w // 2, h // 2

        cx = int(min(max(0, cx + int(dnx * w)), w - 1))
        cy = int(min(max(0, cy + int(dny * h)), h - 1))
        pt = EdsPoint(cx, cy)
        _ok(
            EdsSetPropertyData(
                self.cam, kEdsPropID_Evf_ZoomPosition, 0, C.sizeof(pt), byref(pt)
            ),
            "Nudge Evf_ZoomPosition",
        )
        self._sleep_pump(0.03)
        return True

    def _read_zoom_center_norm(self):
        w, h = self.get_evf_coords()
        if not (self._zr and self._vr):
            return 0.5, 0.5
        zx, zy, zw, zh = self._zr
        vx, vy, vw, vh = self._vr
        cx = zx + zw // 2
        cy = zy + zh // 2
        nx = (cx - vx) / max(1, vw - 1)
        ny = (cy - vy) / max(1, vh - 1)
        return float(nx), float(ny)

    def force_zoom_center(
        self, nx: float, ny: float, attempts: int = 6, tol_px: int = 2
    ):
        """Repeatedly set Evf_ZoomPosition and read Evf_ZoomRect until the center
        approaches the target within tol_px.
        """
        w, h = self.get_evf_coords()
        for i in range(max(1, attempts)):
            # push desired position
            self.set_zoom_pos_norm(nx, ny)
            self._sleep_pump(0.05)
            # read the actual position
            rx, ry = self._read_zoom_center_norm()
            print(f"[EVF] target=({nx:.3f},{ny:.3f}) actual=({rx:.3f},{ry:.3f})")
            if abs(rx - nx) * w <= tol_px and abs(ry - ny) * h <= tol_px:
                return True
        return False

    AF_NG_CODES = {36097}  # 0x8D01

    def _is_af_ng(self, exc: Exception) -> bool:
        s = str(exc).lower()
        return "36097" in s or "0x8d01" in s

    def _safe_halfpress(self, dwell: float = 0.8) -> bool:
        """Half-press helper that suppresses AF_NG errors."""
        ok = True
        try:
            _ok(
                EdsSendCommand(
                    self.cam,
                    kEdsCameraCommand_PressShutterButton,
                    kEdsCameraCommand_ShutterButton_Halfway,
                ),
                "Half-press ON",
            )
            self._sleep_pump(max(0.6, dwell))
        except Exception as e:
            if self._is_af_ng(e):
                ok = False  # AF did not confirm - normal situation
            else:
                ok = False  # treat other errors as AF failure
        finally:
            try:
                _ok(
                    EdsSendCommand(
                        self.cam,
                        kEdsCameraCommand_PressShutterButton,
                        kEdsCameraCommand_ShutterButton_OFF,
                    ),
                    "Half-press OFF",
                )
            except Exception:
                pass
        return ok

    def _get_u32(self, prop):
        v = EdsUInt32(0)
        _ok(EdsGetPropertyData(self.cam, prop, 0, C.sizeof(v), byref(v)), f"Get {prop}")
        return int(v.value)

    def _set_u32(self, prop, val, tag="SetProp"):
        locked = self._try_uilock(tag)
        try:
            v = EdsUInt32(int(val))
            _ok(
                EdsSetPropertyData(self.cam, prop, 0, C.sizeof(v), byref(v)),
                f"{tag} {prop}={val}",
            )
            return True
        finally:
            if locked:
                try:
                    _ok(
                        EdsSendStatusCommand(
                            self.cam, kEdsCameraStatusCommand_UIUnLock, 0
                        ),
                        f"UIUnLock {tag}",
                    )
                except Exception:
                    pass

    def _get_desc(self, prop):
        desc = EdsPropertyDesc()
        _ok(EdsGetPropertyDesc(self.cam, prop, byref(desc)), f"Desc {prop}")
        return [int(desc.propDesc[i]) for i in range(max(0, desc.numElements))]

    def try_set_one_shot_single_point(self):
        """Try to force One-Shot AF with a single/spot EVF AF method."""
        import constants as K

        names = dir(K)
        cand_ops = [
            n
            for n in names
            if n.startswith("kEdsPropID_") and ("AFMode" in n or "AFOperation" in n)
        ]
        cand_evf = [
            n
            for n in names
            if n.startswith("kEdsPropID_")
            and ("Evf_AF" in n or "AFMethod" in n or "AFModeSelect" in n)
        ]
        val_one = [getattr(K, n) for n in names if "OneShot" in n or "One_Shot" in n]
        val_single = [getattr(K, n) for n in names if "Single" in n or "Spot" in n]

        print("[AF] Candidates ops:", cand_ops)
        print("[AF] Candidates evf:", cand_evf)

        for n in cand_ops:
            try:
                pid = getattr(K, n)
                allowed = self._get_desc(pid)
                target = next((v for v in val_one if v in allowed), None)
                if target is not None:
                    print(f"[AF] Set {n} = OneShot({target})")
                    self._set_u32(pid, target, "AFMode")
                    break
            except Exception as e:
                print(f"[AF] skip {n}:", e)

        for n in cand_evf:
            try:
                pid = getattr(K, n)
                allowed = self._get_desc(pid)
                target = next((v for v in val_single if v in allowed), None)
                if target is not None:
                    print(f"[AF] Set {n} = Single/Spot({target})")
                    self._set_u32(pid, target, "EvfAFMode")
                    break
            except Exception as e:
                print(f"[AF] skip {n}:", e)

    def _rect_to_tuple(self, r, w_fallback=1000, h_fallback=1000):
        """Return (x, y, w, h) for EdsRect. Prefer official fields, then flat compatibility names."""
        try:
            return int(r.point.x), int(r.point.y), int(r.size.width), int(r.size.height)
        except Exception:
            x = int(getattr(r, "point_x", 0))
            y = int(getattr(r, "point_y", 0))
            w = int(getattr(r, "size_x", w_fallback))
            h = int(getattr(r, "size_y", h_fallback))
            return x, y, w, h

    def _get_evf_coord_system(self):
        """Return (w, h) of the EVF coordinate system.
        1) try PID kEdsPropID_Evf_CoordinateSystem;
        2) else read from the latest EVF JPEG;
        3) else use a safe 1000x1000 fallback.
        """
        # (1) PID
        try:
            r = EdsRect()
            _ok(
                EdsGetPropertyData(
                    self.cam, kEdsPropID_Evf_CoordinateSystem, 0, C.sizeof(r), byref(r)
                ),
                "Get Evf_CoordinateSystem",
            )
            _, _, w, h = self._rect_to_tuple(r, 0, 0)
            if w and h:
                return int(w), int(h)
        except Exception:
            pass

        # (2) from the latest frame
        w = int(getattr(self, "_cs_w", 0) or 0)
        h = int(getattr(self, "_cs_h", 0) or 0)
        if not (w and h):
            w, h = self._jpeg_size()
            if w and h:
                self._cs_w, self._cs_h = w, h
                return int(w), int(h)

        # (3) safe fallback
        return 1000, 1000

    def focus_at_evf_pixel(self, x_evf: int, y_evf: int):
        """Set focus point via EVF pixel coordinates and trigger autofocus.
        Try Evf_TouchCoordinates first; fall back to Evf_ZoomRect when unsupported.
        """
        # --- 1) try Evf_TouchCoordinates ---
        if EdsPropertyID is not None:
            touch_pid = EdsPropertyID.kEdsPropID_Evf_TouchCoordinates
        else:
            touch_pid = kEdsPropID_Evf_TouchCoordinates  # fallback to kEds* constant

        try:
            pt = (C.c_int * 2)(int(x_evf), int(y_evf))
            _ok(
                EdsSetPropertyData(self.cam, touch_pid, 0, C.sizeof(pt), byref(pt)),
                "EdsSetPropertyData(Evf_TouchCoordinates)",
            )
            # Trigger AF touch: 1 = start, 0 = finish
            cmd = (
                EdsCameraCommand.kEdsCameraCommand_DoEvfAf
                if EdsCameraCommand is not None
                else kEdsCameraCommand_DoEvfAf
            )
            _ok(EdsSendCommand(self.cam, cmd, 1), "DoEvfAf start")
            time.sleep(0.15)
            _ok(EdsSendCommand(self.cam, cmd, 0), "DoEvfAf stop")
            return
        except Exception:
            # failed - fall back to alternate path
            pass

        # --- 2) alternative: center ZoomRect and focus ---
        if EdsPropertyID is not None:
            zoomrect_pid = EdsPropertyID.kEdsPropID_Evf_ZoomRect
        else:
            zoomrect_pid = kEdsPropID_Evf_ZoomRect

        # choose window size based on current zoom/EVF resolution
        half = 50  # 100x100 window - adjust as needed
        x0 = max(0, int(x_evf) - half)
        y0 = max(0, int(y_evf) - half)
        rect = (C.c_int * 4)(x0, y0, half * 2, half * 2)
        _ok(
            EdsSetPropertyData(self.cam, zoomrect_pid, 0, C.sizeof(rect), byref(rect)),
            "EdsSetPropertyData(Evf_ZoomRect)",
        )

        cmd = (
            EdsCameraCommand.kEdsCameraCommand_DoEvfAf
            if EdsCameraCommand is not None
            else kEdsCameraCommand_DoEvfAf
        )
        _ok(EdsSendCommand(self.cam, cmd, 1), "DoEvfAf start")
        time.sleep(0.15)
        _ok(EdsSendCommand(self.cam, cmd, 0), "DoEvfAf stop")

    def _ensure_evf_pc(self, keep_tft: bool = True):
        """Ensure EVF OutputDevice keeps PC and (optionally) TFT. Old behavior killed TFT - do not."""
        try:
            cur = EdsUInt32(0)
            _ok(
                EdsGetPropertyData(
                    self.cam, kEdsPropID_Evf_OutputDevice, 0, C.sizeof(cur), byref(cur)
                ),
                "Get EvfOut",
            )
            if keep_tft:
                want = cur.value | kEdsEvfOutputDevice_PC | kEdsEvfOutputDevice_TFT
                label = "Evf->PC|TFT"
            else:
                want = (cur.value | kEdsEvfOutputDevice_PC) & ~kEdsEvfOutputDevice_TFT
                label = "Evf->PC"
            if want != cur.value:
                vv = EdsUInt32(want)
                _ok(
                    EdsSetPropertyData(
                        self.cam,
                        kEdsPropID_Evf_OutputDevice,
                        0,
                        C.sizeof(vv),
                        byref(vv),
                    ),
                    label,
                )
        except Exception:
            pass

    def _touch_focus_at_px(self, x_evf: int, y_evf: int, dwell: float = 0.15) -> bool:
        """Set touch coordinates via kEdsPropID_Evf_TouchCoordinates and briefly trigger DoEvfAf.
        Return True/False depending on success.
        """
        locked = self._try_uilock("TouchAF")
        try:
            # keep LCD alive before sending touch
            self._ensure_evf_pc(keep_tft=True)
            # Canon expects EdsPoint (int32, int32), not a plain int pair
            pt = EdsPoint()
            pt.x = C.c_int32(int(x_evf))
            pt.y = C.c_int32(int(y_evf))
            _ok(
                EdsSetPropertyData(
                    self.cam,
                    kEdsPropID_Evf_TouchCoordinates,
                    0,
                    C.sizeof(pt),
                    byref(pt),
                ),
                "Set Evf_TouchCoordinates",
            )

            # short autofocus pulse
            cmd = (
                EdsCameraCommand.kEdsCameraCommand_DoEvfAf
                if EdsCameraCommand is not None
                else kEdsCameraCommand_DoEvfAf
            )
            _ok(EdsSendCommand(self.cam, cmd, 1), "DoEvfAf ON")
            self._sleep_pump(dwell)  # maintain legacy timing
            try:
                EdsSendCommand(self.cam, cmd, 0)
            except Exception:
                pass
            # some bodies drop TFT on AF - re-assert PC|TFT
            self._ensure_evf_pc(keep_tft=True)
            self._sleep_pump(0.05)
            # Some bodies drop OutputDevice after AF - restore PC|TFT
            self._ensure_output_pc_tft(pc=True, tft=True, quiet=True)
            self._sleep_pump(0.05)
            # Short delay so _lv_loop can read the updated _taf
            self._sleep_pump(0.05)

            return True
        except Exception:
            return False

    # ------ Live View ------
    def enable_liveview_pc(self, also_tft: bool = True) -> None:
        """Turn on Live View and route EVF to PC and (optionally) to camera TFT simultaneously.
        Many bodies require: Evf_Mode=On -> OutputDevice=TFT (kick LCD) -> OutputDevice=PC|TFT.
        """
        # 1) Turn EVF mode ON
        try:
            _ok(
                EdsSetPropertyData(
                    self.cam,
                    kEdsPropID_Evf_Mode,
                    0,
                    C.sizeof(C.c_uint32(1)),
                    byref(C.c_uint32(1)),
                ),
                "Set Evf_Mode=ON",
            )
        except Exception:
            pass
        self._sleep_pump(0.05)

        # Resolve constants, with safe fallbacks
        DEV_PC = globals().get("kEdsEvfOutputDevice_PC", 0x02)
        DEV_TFT = globals().get("kEdsEvfOutputDevice_TFT", 0x01)

        # 2) Kick LCD first (some bodies need this)
        if also_tft:
            try:
                mask = C.c_uint32(DEV_TFT)
                _ok(
                    EdsSetPropertyData(
                        self.cam,
                        kEdsPropID_Evf_OutputDevice,
                        0,
                        C.sizeof(mask),
                        byref(mask),
                    ),
                    "Set Evf_OutputDevice=TFT",
                )
                self._sleep_pump(0.08)
            except Exception:
                pass

        # 3) Finally set PC | (TFT)
        self._ensure_output_pc_tft(pc=True, tft=also_tft, quiet=True)

        self._sleep_pump(0.08)

    def start_liveview(self):
        self._lv_run = True
        # keep a handle to join on shutdown
        self._lv_thread = threading.Thread(
            target=self._lv_loop, daemon=True, name="EVFLoop"
        )
        self._lv_thread.start()

    def stop_liveview(self):
        self._lv_run = False
        # wait a bit for loop to exit
        t = getattr(self, "_lv_thread", None)
        if t and t.is_alive():
            t.join(timeout=0.7)
        self._lv_thread = None

    def _lv_loop(self):
        while self._lv_run:
            # Pause LiveView if requested
            if self._lv_pause:
                time.sleep(0.03)
                continue

            stream = EdsStreamRef()
            img = EdsEvfImageRef()
            try:
                # Create EVF stream/image and download the frame
                _ok(EdsCreateMemoryStream(0, byref(stream)), "CreateMemStream(EVF)")
                _ok(EdsCreateEvfImageRef(stream, byref(img)), "CreateEvfImageRef")
                _ok(EdsDownloadEvfImage(self.cam, img), "DownloadEvfImage")

                # Access buffer and length
                p_ptr = c_void_p()
                ln = EdsUInt64(0)
                _ok(EdsGetPointer(stream, byref(p_ptr)), "GetPointer(EVF)")
                _ok(EdsGetLength(stream, byref(ln)), "GetLength(EVF)")

                data = (C.c_ubyte * ln.value).from_address(p_ptr.value)

                # 1) Save last JPEG (thread-safe)
                with self._lv_lock:
                    self._last_jpeg = bytes(data)
                # Count frames for diagnostics
                try:
                    self._lv_frames = getattr(self, "_lv_frames", 0) + 1
                except Exception:
                    self._lv_frames = 1

                # 2) Fill EVF image size from JPEG if available
                try:
                    w, h = self._jpeg_size()
                    if w and h:
                        self._img_w, self._img_h = int(w), int(h)
                except Exception:
                    pass

                # 3) Read EVF coordinate system and rectangles from this frame
                try:
                    if "kEdsPropID_Evf_CoordinateSystem" in globals():
                        cs = EdsRect()
                        _ok(
                            EdsGetPropertyData(
                                img,
                                kEdsPropID_Evf_CoordinateSystem,
                                0,
                                C.sizeof(cs),
                                byref(cs),
                            ),
                            "Get Evf_CoordinateSystem(img)",
                        )
                        _, _, wcs, hcs = self._rect_to_tuple(cs, 0, 0)
                        if wcs and hcs:
                            self._cs_w, self._cs_h = int(wcs), int(hcs)
                except Exception:
                    pass

                try:
                    if "kEdsPropID_Evf_VisibleRect" in globals():
                        vr = EdsRect()
                        _ok(
                            EdsGetPropertyData(
                                img,
                                kEdsPropID_Evf_VisibleRect,
                                0,
                                C.sizeof(vr),
                                byref(vr),
                            ),
                            "Get Evf_VisibleRect(img)",
                        )
                        self._vr = self._rect_to_tuple(
                            vr, self._cs_w or 0, self._cs_h or 0
                        )
                except Exception:
                    pass

                try:
                    if "kEdsPropID_Evf_ZoomRect" in globals():
                        zr = EdsRect()
                        _ok(
                            EdsGetPropertyData(
                                img, kEdsPropID_Evf_ZoomRect, 0, C.sizeof(zr), byref(zr)
                            ),
                            "Get Evf_ZoomRect(img)",
                        )
                        self._zr = self._rect_to_tuple(
                            zr, self._cs_w or 0, self._cs_h or 0
                        )
                except Exception:
                    pass

                # 4) Touch-AF position (if body provides it)
                try:
                    pt = EdsPoint()
                    _ok(
                        EdsGetPropertyData(
                            img,
                            kEdsPropID_Evf_TouchAFPosition,
                            0,
                            C.sizeof(pt),
                            byref(pt),
                        ),
                        "Get Evf_TouchAFPosition(img)",
                    )
                    self._taf = (int(pt.x), int(pt.y))
                except Exception:
                    self._taf = None

            except Exception:
                # swallow per-frame errors to keep LV running
                pass
            finally:
                # Always release EVF objects
                try:
                    if img:
                        EdsRelease(img)
                except Exception:
                    pass
                try:
                    if stream:
                        EdsRelease(stream)
                except Exception:
                    pass

            # avoid hammering camera/UI
            time.sleep(0.02)

    def get_last_jpeg(self) -> bytes | None:
        with self._lv_lock:
            return self._last_jpeg

    # ---------- UI Lock helpers ----------
    def _try_uilock(self, label: str = "UILock") -> bool:
        """Try to acquire camera UI lock. Returns True if locked."""
        try:
            cmd = globals().get("kEdsCameraStatusCommand_UILock")
            if cmd is None:
                return False
            _ok(EdsSendStatusCommand(self.cam, cmd, 0), label)
            return True
        except Exception:
            return False

    def _uiunlock(self, label: str = "UIUnLock") -> None:
        """Release camera UI lock if supported."""
        try:
            cmd = globals().get("kEdsCameraStatusCommand_UIUnLock")
            if cmd is None:
                return
            _ok(EdsSendStatusCommand(self.cam, cmd, 0), label)
        except Exception:
            pass

    # ---------- Generic call-with-timeout helper ----------
    def _call_with_timeout(self, label: str, func, timeout_s: float = 0.6) -> bool:
        """Run `func()` inside a short-lived daemon thread and wait up to timeout_s.
        Returns True if finished, False if timed out. Exceptions inside func are swallowed.
        """
        done = {"ok": False}

        def worker():
            try:
                func()
                done["ok"] = True
            except Exception:
                done["ok"] = False

        th = threading.Thread(target=worker, name=f"call:{label}", daemon=True)
        th.start()
        th.join(timeout=max(0.1, float(timeout_s)))
        return bool(done["ok"])

    # ---------- Non-blocking property setter (with UI lock & timeout) ----------
    def _set_prop_u32_with_lock(
        self, prop_id, value: int, label: str, timeout_s: float = 0.4
    ) -> bool:
        """Set a U32 property under UI lock in a short-lived thread.
        If the SDK call stalls, we time out and return False (and proceed with shutdown).
        """
        if not self.cam or prop_id is None:
            return False
        done = {"ok": False}

        def worker():
            locked = self._try_uilock(f"UILock:{label}")
            try:
                vv = EdsUInt32(int(value))
                try:
                    EdsSetPropertyData(self.cam, prop_id, 0, C.sizeof(vv), byref(vv))
                    done["ok"] = True
                except Exception:
                    done["ok"] = False
            finally:
                if locked:
                    self._uiunlock(f"UIUnLock:{label}")

        th = threading.Thread(target=worker, name=f"setprop:{label}", daemon=True)
        th.start()
        th.join(timeout=max(0.05, float(timeout_s)))
        return bool(done["ok"])

    # ========== Diagnostics for clean shutdown ==========
    def _get_u32(self, prop_id) -> int | None:
        """Get U32 property from camera safely; return None on error."""
        try:
            cur = C.c_uint32(0)
            _ok(
                EdsGetPropertyData(self.cam, prop_id, 0, C.sizeof(cur), byref(cur)),
                f"GetU32({prop_id})",
            )
            return int(cur.value)
        except Exception:
            return None

    def _probe_evf_download_ok(self) -> bool:
        """Try a single EdsDownloadEvfImage in a short-lived thread.
        Returns False on error or timeout to avoid hangs during shutdown.
        """
        if not self.cam:
            return False
        result = {"ok": False}

        def worker():
            stream = EdsStreamRef()
            img = EdsEvfImageRef()
            try:
                EdsCreateMemoryStream(0, byref(stream))
                EdsCreateEvfImageRef(stream, byref(img))
                EdsDownloadEvfImage(self.cam, img)
                result["ok"] = True
            except Exception:
                result["ok"] = False
            finally:
                try:
                    if img:
                        EdsRelease(img)
                except Exception:
                    pass
                try:
                    if stream:
                        EdsRelease(stream)
                except Exception:
                    pass

        th = threading.Thread(target=worker, name="probe:evf_dl", daemon=True)
        th.start()
        th.join(timeout=0.4)
        return bool(result["ok"])

    def debug_snapshot(self, title: str = "SNAP") -> None:
        """Print a compact snapshot of camera/runtime state: threads, EVF mode, output mask, rects, frames."""
        try:
            out = (
                self._get_u32(kEdsPropID_Evf_OutputDevice)
                if "kEdsPropID_Evf_OutputDevice" in globals()
                else None
            )
        except Exception:
            out = None
        try:
            evf_mode = (
                self._get_u32(kEdsPropID_Evf_Mode)
                if "kEdsPropID_Evf_Mode" in globals()
                else None
            )
        except Exception:
            evf_mode = None
        lv_alive = bool(
            getattr(self, "_lv_thread", None) and self._lv_thread.is_alive()
        )
        ev_alive = bool(
            getattr(self, "_ev_thread", None) and self._ev_thread.is_alive()
        )
        frames = int(getattr(self, "_lv_frames", 0) or 0)
        vr = getattr(self, "_vr", None)
        zr = getattr(self, "_zr", None)
        taf = getattr(self, "_taf", None)
        print(
            f"[{title}] lv_run={self._lv_run} lv_alive={lv_alive} frames={frames} "
            f"ev_alive={ev_alive} out=0x{(out if out is not None else -1):02X} "
            f"evf_mode={evf_mode} vr={vr} zr={zr} taf={taf}"
        )

    def debug_close_probe(self, *, wait_first_frame: float = 3.0) -> None:
        """End-to-end close self-test. Shows what keeps the camera busy.
        1) Connect + EVF to PC|TFT + start LV + wait first frame.
        2) Snapshot.
        3) Stop LV (thread join) + snapshot.
        4) Disable LV on camera (OutputDevice->TFT, Evf_Mode=OFF) + snapshot.
        5) Try single EVF download to see if EVF is still serviceable.
        6) Unregister events, close session, release, terminate.
        """
        self.debug_snapshot("A:pre")
        # Warmup wait (if needed)
        t0 = time.time()
        while (time.time() - t0) < max(0.1, float(wait_first_frame)):
            if getattr(self, "_last_jpeg", None):
                break
            time.sleep(0.05)
        self.debug_snapshot("B:warm")
        # Stop LV and wait thread
        self.stop_liveview()
        self.debug_snapshot("C:stopped")
        # Stop event pump BEFORE touching any EVF props
        self._run_events = False
        t = getattr(self, "_ev_thread", None)
        if t and t.is_alive():
            t.join(timeout=0.7)
        self._ev_thread = None
        self.debug_snapshot("C2:events-off")
        # Disable output to PC, turn EVF mode OFF (if available) non-blocking
        self.disable_liveview_pc()
        self.debug_snapshot("D:output->TFT")
        # Probe EVF download possibility (skip when LV never produced frames)
        if int(getattr(self, "_lv_frames", 0) or 0) == 0 and not getattr(
            self, "_last_jpeg", None
        ):
            print("[D:probe] skipped (no LV frames ever)")
        else:
            can_dl = self._probe_evf_download_ok()
            print(f"[D:probe] EdsDownloadEvfImage still OK? {can_dl}")

        # Unhook events and close
        try:
            if self.cam:
                if "EdsSetObjectEventHandler" in globals():
                    EdsSetObjectEventHandler(self.cam, kEdsObjectEvent_All, None, None)
                if "EdsSetPropertyEventHandler" in globals():
                    EdsSetPropertyEventHandler(
                        self.cam, kEdsPropertyEvent_All, None, None
                    )
                if "EdsSetCameraStateEventHandler" in globals():
                    EdsSetCameraStateEventHandler(
                        self.cam, kEdsStateEvent_All, None, None
                    )
        except Exception:
            pass
        self._run_events = False
        t = getattr(self, "_ev_thread", None)
        if t and t.is_alive():
            t.join(timeout=0.7)
        self.debug_snapshot("E:events-off")
        # Close/Release/Terminate (with timeouts)
        if self._session and self.cam:
            ok_cl = self._call_with_timeout(
                "EdsCloseSession(probe)",
                lambda: EdsCloseSession(self.cam),
                timeout_s=0.6,
            )
            print(f"[E2:close-session] {'ok' if ok_cl else 'timeout'}")
        ok_rl = True
        if self.cam:
            ok_rl = self._call_with_timeout(
                "EdsRelease(CameraRef, probe)",
                lambda: EdsRelease(self.cam),
                timeout_s=0.6,
            )
        print(f"[E3:release] {'ok' if ok_rl else 'timeout'}")
        self.cam = None
        ok_tm = self._call_with_timeout(
            "EdsTerminateSDK(probe)", lambda: EdsTerminateSDK(), timeout_s=0.6
        )
        print(f"[F:terminate] {'ok' if ok_tm else 'timeout'}")
        print("[F:done] probe finished")

    # ------ Zoom / AF ------
    def set_evf_zoom(self, level: int, readback: bool = True):
        """Change zoom while pausing Live View; try both value families (1/5/10 and 0/1/2)."""
        mapping = {1: [1, 0], 5: [5, 1], 10: [10, 2]}
        self._lv_pause = True
        locked = self._try_uilock("ZOOM")
        try:
            last_exc = None
            for raw in mapping.get(level, [level]):
                try:
                    vv = EdsUInt32(raw)
                    _ok(
                        EdsSetPropertyData(
                            self.cam, kEdsPropID_Evf_Zoom, 0, C.sizeof(vv), byref(vv)
                        ),
                        f"Set Evf_Zoom target={level}x raw={raw}",
                    )
                    self._sleep_pump(0.08)
                    if readback:
                        cur = EdsUInt32(0)
                        try:
                            _ok(
                                EdsGetPropertyData(
                                    self.cam,
                                    kEdsPropID_Evf_Zoom,
                                    0,
                                    C.sizeof(cur),
                                    byref(cur),
                                ),
                                "Get Evf_Zoom",
                            )
                            return int(cur.value)
                        except Exception:
                            return raw
                    return raw
                except Exception as e:
                    last_exc = e
                    self._sleep_pump(0.05)
            if last_exc:
                raise last_exc
        finally:
            if locked:
                try:
                    _ok(
                        EdsSendStatusCommand(
                            self.cam, kEdsCameraStatusCommand_UIUnLock, 0
                        ),
                        "UIUnLock ZOOM",
                    )
                except Exception:
                    pass
            self._lv_pause = False

    def set_touch_af_or_zoompos(
        self, nx: float, ny: float, dwell: float = 0.8, *, allow_halfpress: bool = True
    ) -> bool:
        """Place AF point at the click: 1) TouchCoordinates + DoEvfAf;
        2) fallback ZoomPosition (zoom > 1x); 3) optional half-press.
        """
        # always start with PC|TFT
        try:
            self._ensure_evf_pc(keep_tft=True)
        except Exception:
            pass

        # 0..1 -> EVF coordinates
        pt_center = self._map_norm_to_evf(float(nx), float(ny))
        x_evf, y_evf = int(pt_center.x), int(pt_center.y)

        # (A) primary path - TouchCoordinates + short AF
        if "kEdsPropID_Evf_TouchCoordinates" in globals():
            if self._touch_focus_at_px(x_evf, y_evf, dwell=0.15):
                # Wait until TouchAF position converges; tolerate Y grid (~70 px)
                self._wait_center_converged(
                    x_evf, y_evf, tol_x=4, tol_y=70, timeout_s=0.9, prefer_taf=True
                )
                # and re-assert after convergence
                try:
                    self._ensure_evf_pc(keep_tft=True)
                except Exception:
                    pass
                try:
                    self.debug_probe_af(nx, ny)
                except Exception:
                    pass
                return True  # nothing else needed

        # (B) fallback - translate target center into top-left ZoomRect
        ok_zoompos = False
        if "kEdsPropID_Evf_ZoomPosition" in globals():
            # Only use ZoomPosition when zoom window exists (zoom > 1x)
            has_zoom_window = False
            if self._zr and self._vr:
                zx, zy, zw, zh = self._zr
                vx, vy, vw, vh = self._vr
                has_zoom_window = (zw < vw) or (zh < vh)
            if has_zoom_window:
                ok_zoompos = bool(self.set_zoom_pos_norm(nx, ny, retries=2, tol_px=2))
                self._wait_center_converged(
                    x_evf, y_evf, tol_x=4, tol_y=70, timeout_s=0.9, prefer_taf=False
                )
                try:
                    self._ensure_evf_pc(keep_tft=True)
                except Exception:
                    pass  # (C) fallback half-press to encourage AF
        ok_half = self._safe_halfpress(dwell) if allow_halfpress else False
        try:
            self._ensure_evf_pc(keep_tft=True)
        except Exception:
            pass

        try:
            self.debug_probe_af(nx, ny)
        except Exception:
            pass

        return bool(ok_zoompos or ok_half)

    def debug_probe_af(self, nx: float, ny: float):
        w, h = self.get_evf_coords()
        # Compute the desired center without calibration offset
        temp_ready = self._ofs_ready
        self._ofs_ready = False
        pt_desired = self._map_norm_to_evf(nx, ny)  # treat as center
        self._ofs_ready = temp_ready
        tx, ty = int(pt_desired.x), int(pt_desired.y)

        actual_px = None
        if getattr(self, "_taf", None):
            actual_px = self._taf
        elif self._zr:
            zx, zy, zw, zh = self._zr
            actual_px = (zx + zw // 2, zy + zh // 2)

        if actual_px:
            got_x, got_y = actual_px
            # Normalize actual point within VisibleRect like the target
            if self._vr:
                vx, vy, vw, vh = self._vr
                axn = (got_x - vx) / max(1, vw - 1)
                ayn = (got_y - vy) / max(1, vh - 1)
            else:
                axn = got_x / max(1, w - 1)
                ayn = got_y / max(1, h - 1)
            print(
                f"[AF PROBE] target_px=({tx},{ty})  actual_px=({got_x},{got_y})  "
                f"target_norm=({nx:.3f},{ny:.3f})  actual_norm=({axn:.3f},{ayn:.3f})  "
                f"diff_px=({got_x - tx},{got_y - ty})"
            )
        else:
            print(f"[AF PROBE] target_px=({tx},{ty}), actual=unavailable")

    def af_corners_test(self, dwell=0.5):
        """Run a 5-point calibration: top-left, top-right, bottom-right, bottom-left, center.
        Visually confirm on the camera that corners align.
        """
        pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)]
        print("[TEST] corners order: LT, RT, RB, LB, center")
        for nx, ny in pts:
            self.set_touch_af_or_zoompos(nx, ny, dwell=dwell)
            try:
                self.debug_probe_af(nx, ny)
            except Exception:
                pass
            time.sleep(0.3)

    def disable_liveview_pc(self):
        """Disable EVF to PC in a non-blocking way:
        - If EVF never delivered frames, skip touching Output/Mode (many bodies auto-restore on CloseSession).
        - Otherwise try both sequences via short threads with timeouts.
        """
        # If LV never actually produced frames, do not poke EVF props (avoid hangs).
        if int(getattr(self, "_lv_frames", 0) or 0) == 0 and not getattr(
            self, "_last_jpeg", None
        ):
            return

        DEV_PC = globals().get("kEdsEvfOutputDevice_PC", 0x02)
        DEV_TFT = globals().get("kEdsEvfOutputDevice_TFT", 0x01)
        pid_out = globals().get("kEdsPropID_Evf_OutputDevice")
        pid_mode = globals().get("kEdsPropID_Evf_Mode")

        # A) Output -> TFT (drop PC bit), then Mode = OFF
        try:
            if pid_out is not None:
                cur = EdsUInt32(0)
                try:
                    EdsGetPropertyData(self.cam, pid_out, 0, C.sizeof(cur), byref(cur))
                    val = (cur.value & ~DEV_PC) | DEV_TFT
                except Exception:
                    val = DEV_TFT  # best effort
                self._set_prop_u32_with_lock(
                    pid_out, val, "EvfOut->TFT", timeout_s=0.25
                )
            if pid_mode is not None:
                self._set_prop_u32_with_lock(pid_mode, 0, "EvfMode=OFF", timeout_s=0.25)
        except Exception:
            pass

        # B) Fallback sequence: Mode=OFF first, then Output->TFT
        try:
            if pid_mode is not None:
                self._set_prop_u32_with_lock(
                    pid_mode, 0, "EvfMode=OFF(b)", timeout_s=0.25
                )
            if pid_out is not None:
                self._set_prop_u32_with_lock(
                    pid_out, DEV_TFT, "EvfOut->TFT(b)", timeout_s=0.25
                )
        except Exception:
            pass

    def close(self):
        try:
            # 1) stop LV thread first
            self.stop_liveview()
            # 2) stop event pump BEFORE touching EVF props (avoids deadlocks)
            self._run_events = False
            t = getattr(self, "_ev_thread", None)
            if t and t.is_alive():
                t.join(timeout=0.7)
            self._ev_thread = None
            # 3) now safely switch EVF back to TFT (under UI lock)
            self.disable_liveview_pc()
            # 4) unregister event handlers (no late callbacks)
            try:
                if self.cam:
                    if "EdsSetObjectEventHandler" in globals():
                        EdsSetObjectEventHandler(
                            self.cam, kEdsObjectEvent_All, None, None
                        )
                    if "EdsSetPropertyEventHandler" in globals():
                        EdsSetPropertyEventHandler(
                            self.cam, kEdsPropertyEvent_All, None, None
                        )
                    if "EdsSetCameraStateEventHandler" in globals():
                        EdsSetCameraStateEventHandler(
                            self.cam, kEdsStateEvent_All, None, None
                        )
            except Exception:
                pass
            # 5) close session (best-effort, with timeout)
            if self._session and self.cam:
                self._call_with_timeout(
                    "EdsCloseSession", lambda: EdsCloseSession(self.cam), timeout_s=0.6
                )

        finally:
            # 6) release camera (IMPORTANT: before Terminate), best-effort
            if self.cam:
                self._call_with_timeout(
                    "EdsRelease(CameraRef)", lambda: EdsRelease(self.cam), timeout_s=0.6
                )
            self.cam = None
            # 7) drop callback refs
            self._obj_cb = None

            # 8) Terminate - last, best effort
            self._call_with_timeout(
                "EdsTerminateSDK", lambda: EdsTerminateSDK(), timeout_s=0.6
            )
            self._session = False


# Created by Dr. Z. Bakhtiyorov
