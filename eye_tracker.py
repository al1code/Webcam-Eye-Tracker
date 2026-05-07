"""
Global Ekran Göz Takibi & Arka Plan Isı Haritası — Enhanced Edition
====================================================================
Kontroller:
    C       -> Kalibrasyon Modunu Aç/Kapat (9-noktalı görsel mod)
    T       -> Gaze Trail Göster/Gizle
    S       -> Anlık snapshot kaydet
    ESC     -> Takibi Bitir ve Analiz Raporunu Göster
"""

from __future__ import annotations

import sys
import csv
import json
import os
import platform
import subprocess
import tempfile
import time
import threading
import hashlib
import urllib.request
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from scipy.ndimage import gaussian_filter

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QFont, QRadialGradient, QBrush,
)

# ──────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
MODEL_PATH   = "face_landmarker.task"
MODEL_URL    = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
# Set to expected SHA-256 hex digest to enable integrity check; None = skip
MODEL_SHA256 = None
TARGET_FPS   = 30

# Try resolutions from highest to lowest; pick first one camera actually supports
CAMERA_RESOLUTIONS = [(1920, 1080), (1280, 720), (640, 480)]

# Smoothing — lower SMOOTH_PREV = more responsive, higher = more stable
SMOOTH_PREV   = 0.88
SMOOTH_RAW    = 0.12
DEAD_ZONE_PX  = 15
MEDIAN_WINDOW = 7

# Gaze trail
TRAIL_LENGTH  = 30
TRAIL_ENABLED = True

# Blink detection (Eye Aspect Ratio)
EAR_THRESHOLD      = 0.21
BLINK_CONSEC_FRAMES = 2

# Iris / eye landmark indices for MediaPipe FaceMesh
RIGHT_IRIS      = [469, 470, 471, 472]
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
RIGHT_EYE_TOP   = 159
RIGHT_EYE_BOT   = 145

LEFT_IRIS       = [474, 475, 476, 477]
LEFT_EYE_OUTER  = 362
LEFT_EYE_INNER  = 263
LEFT_EYE_TOP    = 386
LEFT_EYE_BOT    = 374


# ──────────────────────────────────────────────────────────────
# Model download + optional integrity check
# ──────────────────────────────────────────────────────────────
def _verify_sha256(path: str, expected: str) -> bool:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def _download_model(url: str, dest: str) -> None:
    print(f"[i] Model indiriliyor: {dest}")
    urllib.request.urlretrieve(url, dest)
    if MODEL_SHA256 and not _verify_sha256(dest, MODEL_SHA256):
        os.remove(dest)
        raise ValueError(
            "Model SHA-256 doğrulaması başarısız — indirilen dosya güvenilir değil."
        )
    print("[✓] Model indirildi.")


if not os.path.exists(MODEL_PATH):
    _download_model(MODEL_URL, MODEL_PATH)

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
_lm_opts   = mp_vision.FaceLandmarkerOptions(
    base_options=_base_opts,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
)
face_landmarker = mp_vision.FaceLandmarker.create_from_options(_lm_opts)


# ──────────────────────────────────────────────────────────────
# Cross-platform screenshot (Windows / macOS / Linux)
# ──────────────────────────────────────────────────────────────
def take_screenshot() -> Optional[str]:
    """Return path to a temporary screenshot PNG, or None on failure."""
    tmp = os.path.join(
        tempfile.gettempdir(),
        f"eye_tracker_ss_{int(time.time())}.png",
    )
    system = platform.system()
    try:
        if system == "Windows":
            # Pillow ImageGrab is the most reliable on Windows
            from PIL import ImageGrab
            img = ImageGrab.grab()
            img.save(tmp)
        elif system == "Darwin":
            subprocess.run(
                ["screencapture", "-x", tmp], check=True, timeout=10
            )
        else:
            subprocess.run(["scrot", tmp], check=True, timeout=10)
        return tmp
    except Exception as exc:
        print(f"[!] Ekran görüntüsü alınamadı: {exc}")
        return None


# ──────────────────────────────────────────────────────────────
# Global State (protected by _lock)
# ──────────────────────────────────────────────────────────────
_lock = threading.Lock()

is_calibrating = False

CALIB_X_MIN, CALIB_X_MAX = 0.40, 0.60
CALIB_Y_MIN, CALIB_Y_MAX = 0.40, 0.60
calib_data_x: list[float] = []
calib_data_y: list[float] = []

session_gaze_points: list[tuple] = []   # (x, y, timestamp)
session_blinks:      list[float] = []   # timestamps of detected blinks
trail_points:        list[tuple] = []   # last N gaze points for trail


# ──────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────
def iris_center(landmarks, indices, w: int, h: int):
    xs = [landmarks[i].x * w for i in indices]
    ys = [landmarks[i].y * h for i in indices]
    return float(np.mean(xs)), float(np.mean(ys))


def eye_aspect_ratio(landmarks, outer, inner, top, bot, w: int, h: int) -> float:
    """Vertical/Horizontal ratio — drops on blink."""
    o = np.array([landmarks[outer].x * w, landmarks[outer].y * h])
    i = np.array([landmarks[inner].x * w, landmarks[inner].y * h])
    t = np.array([landmarks[top].x   * w, landmarks[top].y   * h])
    b = np.array([landmarks[bot].x   * w, landmarks[bot].y   * h])
    horiz = np.linalg.norm(o - i)
    vert  = np.linalg.norm(t - b)
    return float(vert / max(horiz, 1.0))


def get_eye_ratios(lm, w: int, h: int):
    r_iris    = iris_center(lm, RIGHT_IRIS, w, h)
    r_ox, r_ix = lm[RIGHT_EYE_OUTER].x * w, lm[RIGHT_EYE_INNER].x * w
    r_oy, r_iy = lm[RIGHT_EYE_OUTER].y * h, lm[RIGHT_EYE_INNER].y * h
    r_ew       = max(abs(r_ox - r_ix), 1)
    r_rx       = (r_iris[0] - min(r_ox, r_ix)) / r_ew
    r_ry       = (r_iris[1] - (r_oy + r_iy) / 2.0) / r_ew

    l_iris    = iris_center(lm, LEFT_IRIS, w, h)
    l_ox, l_ix = lm[LEFT_EYE_OUTER].x * w, lm[LEFT_EYE_INNER].x * w
    l_oy, l_iy = lm[LEFT_EYE_OUTER].y * h, lm[LEFT_EYE_INNER].y * h
    l_ew       = max(abs(l_ox - l_ix), 1)
    l_rx       = (l_iris[0] - min(l_ox, l_ix)) / l_ew
    l_ry       = (l_iris[1] - (l_oy + l_iy) / 2.0) / l_ew

    return (r_rx + l_rx) / 2.0, (r_ry + l_ry) / 2.0


def get_avg_ear(lm, w: int, h: int) -> float:
    r = eye_aspect_ratio(lm, RIGHT_EYE_OUTER, RIGHT_EYE_INNER,
                         RIGHT_EYE_TOP, RIGHT_EYE_BOT, w, h)
    l = eye_aspect_ratio(lm, LEFT_EYE_OUTER, LEFT_EYE_INNER,
                         LEFT_EYE_TOP, LEFT_EYE_BOT, w, h)
    return (r + l) / 2.0


# ──────────────────────────────────────────────────────────────
# Camera / MediaPipe Thread
# ──────────────────────────────────────────────────────────────
class EyeTrackingThread(QThread):
    gaze_signal  = pyqtSignal(int, int, bool)
    blink_signal = pyqtSignal()
    fps_signal   = pyqtSignal(float)

    def __init__(self, screen_w: int, screen_h: int):
        super().__init__()
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.running  = True

    def _open_camera(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        for rw, rh in CAMERA_RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  rw)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if aw >= rw * 0.9:
                print(f"[✓] Kamera çözünürlüğü: {aw}×{ah}")
                break
        return cap

    def run(self):
        global is_calibrating, calib_data_x, calib_data_y
        global CALIB_X_MIN, CALIB_X_MAX, CALIB_Y_MIN, CALIB_Y_MAX
        global session_gaze_points, session_blinks, trail_points

        cap = self._open_camera()

        smooth_x = self.screen_w // 2
        smooth_y = self.screen_h // 2
        disp_x   = smooth_x
        disp_y   = smooth_y
        buf_x:   list[int] = []
        buf_y:   list[int] = []

        blink_ctr  = 0
        fps_cnt    = 0
        fps_t0     = time.time()
        frame_int  = 1.0 / TARGET_FPS

        while self.running:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                continue

            frame  = cv2.flip(frame, 1)
            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = face_landmarker.detect(mp_img)

            fps_cnt += 1
            if time.time() - fps_t0 >= 1.0:
                self.fps_signal.emit(fps_cnt / (time.time() - fps_t0))
                fps_cnt = 0
                fps_t0  = time.time()

            if result.face_landmarks:
                lm = result.face_landmarks[0]

                # ── Blink detection ──
                ear = get_avg_ear(lm, w, h)
                if ear < EAR_THRESHOLD:
                    blink_ctr += 1
                else:
                    if blink_ctr >= BLINK_CONSEC_FRAMES:
                        with _lock:
                            session_blinks.append(time.time())
                        self.blink_signal.emit()
                    blink_ctr = 0

                ratio_x, ratio_y = get_eye_ratios(lm, w, h)

                with _lock:
                    calibrating = is_calibrating
                    x_min, x_max = CALIB_X_MIN, CALIB_X_MAX
                    y_min, y_max = CALIB_Y_MIN, CALIB_Y_MAX

                if calibrating:
                    with _lock:
                        calib_data_x.append(ratio_x)
                        calib_data_y.append(ratio_y)
                    self.gaze_signal.emit(0, 0, True)
                    continue

                # ── Map iris ratio → screen pixels ──
                rx = max(x_max - x_min, 0.001)
                ry = max(y_max - y_min, 0.001)
                raw_x = int(((ratio_x - x_min) / rx) * self.screen_w)
                raw_y = int(((ratio_y - y_min) / ry) * self.screen_h)

                buf_x.append(raw_x)
                buf_y.append(raw_y)
                if len(buf_x) > MEDIAN_WINDOW:
                    buf_x.pop(0)
                    buf_y.pop(0)
                med_x = int(np.median(buf_x))
                med_y = int(np.median(buf_y))

                smooth_x = int(SMOOTH_PREV * smooth_x + SMOOTH_RAW * med_x)
                smooth_y = int(SMOOTH_PREV * smooth_y + SMOOTH_RAW * med_y)

                if np.hypot(smooth_x - disp_x, smooth_y - disp_y) >= DEAD_ZONE_PX:
                    disp_x = smooth_x
                    disp_y = smooth_y

                tx = max(0, min(self.screen_w  - 1, disp_x))
                ty = max(0, min(self.screen_h - 1, disp_y))

                with _lock:
                    session_gaze_points.append((tx, ty, time.time()))
                    trail_points.append((tx, ty, time.time()))
                    if len(trail_points) > TRAIL_LENGTH:
                        trail_points.pop(0)

                self.gaze_signal.emit(tx, ty, False)

            elapsed = time.time() - t0
            st = frame_int - elapsed
            if st > 0:
                time.sleep(st)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


# ──────────────────────────────────────────────────────────────
# Analysis helpers
# ──────────────────────────────────────────────────────────────
def build_heatmap(points, w: int, h: int, sigma: int = 40) -> np.ndarray:
    hm = np.zeros((h, w), dtype=np.float32)
    for (x, y, *_) in points:
        if 0 <= x < w and 0 <= y < h:
            hm[y, x] += 1.0
    hm = gaussian_filter(hm, sigma=sigma)
    if hm.max() > 0:
        hm /= hm.max()
    return hm


def zone_analysis(points, w: int, h: int, cols: int = 3, rows: int = 3):
    cnt = np.zeros((rows, cols), dtype=np.float32)
    for (x, y, *_) in points:
        c = min(int(x / w * cols), cols - 1)
        r = min(int(y / h * rows), rows - 1)
        cnt[r, c] += 1
    total = max(cnt.sum(), 1)
    return cnt / total * 100


def fixation_analysis(points, radius: int = 60, min_dur: float = 0.15):
    if len(points) < 2:
        return []
    fixations = []
    cluster   = [points[0]]
    for pt in points[1:]:
        if np.hypot(pt[0] - cluster[0][0], pt[1] - cluster[0][1]) < radius:
            cluster.append(pt)
        else:
            dur = cluster[-1][2] - cluster[0][2]
            if dur >= min_dur:
                mx = int(np.mean([p[0] for p in cluster]))
                my = int(np.mean([p[1] for p in cluster]))
                fixations.append((mx, my, dur))
            cluster = [pt]
    return fixations


def export_session_data(points, blinks, out_dir: str = "heatmap_kayitlar") -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(out_dir, f"gaze_data_{ts}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "gaze_x", "gaze_y"])
        for (x, y, t) in points:
            w.writerow([f"{t:.4f}", x, y])

    total_time = (points[-1][2] - points[0][2]) if len(points) > 1 else 0
    summary = {
        "session_date":    datetime.now().isoformat(),
        "total_points":    len(points),
        "total_blinks":    len(blinks),
        "duration_seconds": round(total_time, 2),
        "blinks_per_minute": round(
            len(blinks) / (total_time / 60), 1
        ) if total_time > 0 else 0,
    }
    json_path = os.path.join(out_dir, f"session_summary_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"[✓] Gaze CSV: {csv_path}")
    print(f"[✓] Özet JSON: {json_path}")
    return ts


# ──────────────────────────────────────────────────────────────
# Final analysis window
# ──────────────────────────────────────────────────────────────
_ZONE_NAMES = [
    ["Sol Üst",  "Orta Üst", "Sağ Üst"],
    ["Sol Orta", "Merkez",   "Sağ Orta"],
    ["Sol Alt",  "Orta Alt", "Sağ Alt"],
]


def show_final_analysis(points, blinks, w: int, h: int) -> None:
    if len(points) < 10:
        print("[!] Yeterli veri toplanamadı.")
        return

    print(f"\n[i] {len(points)} bakış noktası analiz ediliyor...")
    ts = export_session_data(points, blinks)

    ss_path = take_screenshot()
    if ss_path and os.path.exists(ss_path):
        bg = cv2.imread(ss_path)
        bg = cv2.resize(bg, (w, h)) if bg is not None else np.zeros((h, w, 3), np.uint8)
        try:
            os.remove(ss_path)
        except OSError:
            pass
    else:
        bg = np.zeros((h, w, 3), np.uint8)

    # Heat map — TURBO colormap is perceptually superior to JET
    hm      = build_heatmap(points, w, h, sigma=40)
    colored = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_TURBO)
    alpha   = np.clip(hm * 2.2, 0, 0.80)
    a3      = np.stack([alpha] * 3, axis=2)
    canvas  = (bg * (1 - a3) + colored * a3).astype(np.uint8)

    # Fixation circles
    fixations = fixation_analysis(points)
    for (fx, fy, dur) in fixations:
        r = int(12 + dur * 18)
        cv2.circle(canvas, (fx, fy), r, (255, 255, 100), 2)
        cv2.circle(canvas, (fx, fy), 4,   (255, 255, 100), -1)
        cv2.putText(canvas, f"{dur:.1f}s", (fx + r + 4, fy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 100), 1)

    # Zone grid
    zone_pct = zone_analysis(points, w, h)
    cw, ch   = w // 3, h // 3
    for row in range(3):
        for col in range(3):
            x0, y0 = col * cw, row * ch
            pct     = zone_pct[row, col]
            intens  = int(min(pct / 30 * 255, 255))
            cv2.rectangle(canvas, (x0, y0), (x0 + cw - 1, y0 + ch - 1),
                          (0, intens, 255 - intens), 2)
            label = f"{pct:.1f}%"
            tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.putText(canvas, label,
                        (x0 + (cw - tw) // 2, y0 + (ch + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, _ZONE_NAMES[row][col], (x0 + 8, y0 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # Stats panel
    total_time   = points[-1][2] - points[0][2] if len(points) > 1 else 0
    n_fix        = len(fixations)
    avg_dur      = np.mean([f[2] for f in fixations]) if fixations else 0.0
    top_idx      = np.unravel_index(np.argmax(zone_pct), zone_pct.shape)
    top_name     = _ZONE_NAMES[top_idx[0]][top_idx[1]]
    bpm          = len(blinks) / (total_time / 60) if total_time > 0 else 0

    lines = [
        "GAZE ANALYTICS",
        f"Süre          : {total_time:.1f}s",
        f"Fiksasyon     : {n_fix}",
        f"Ort Fiksasyon : {avg_dur:.2f}s",
        f"En çok bakılan: {top_name} ({zone_pct[top_idx]:.1f}%)",
        f"Göz kırpma    : {len(blinks)}  ({bpm:.1f}/dak)",
    ]
    panel_h = len(lines) * 26 + 20
    overlay = canvas.copy()
    cv2.rectangle(overlay, (8, 8), (360, 14 + panel_h), (10, 10, 10), -1)
    canvas  = cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0)
    cv2.rectangle(canvas, (8, 8), (360, 14 + panel_h), (60, 180, 60), 1)
    for i, ln in enumerate(lines):
        color = (100, 255, 100) if i == 0 else (200, 240, 200)
        size  = 0.62 if i == 0 else 0.57
        cv2.putText(canvas, ln, (16, 32 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, size, color, 1)

    os.makedirs("heatmap_kayitlar", exist_ok=True)
    out_path = f"heatmap_kayitlar/oturum_{ts}.png"
    cv2.imwrite(out_path, canvas)
    print(f"[✓] Analiz görüntüsü: {out_path}")

    cv2.namedWindow("Göz Takibi Analizi", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Göz Takibi Analizi",
                          cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Göz Takibi Analizi", canvas)
    print("[i] Kapatmak için herhangi bir tuşa basın.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────
# 9-point calibration manager
# ──────────────────────────────────────────────────────────────
class CalibrationManager:
    DWELL = 1.5  # seconds to dwell on each target

    def __init__(self, sw: int, sh: int):
        m = 80
        self.targets = [
            (m, m), (sw // 2, m), (sw - m, m),
            (m, sh // 2), (sw // 2, sh // 2), (sw - m, sh // 2),
            (m, sh - m), (sw // 2, sh - m), (sw - m, sh - m),
        ]
        self.active      = False
        self.current_idx = 0
        self._t0         = 0.0

    def start(self):
        self.active      = True
        self.current_idx = 0
        self._t0         = time.time()

    def stop(self):
        self.active = False

    def current_target(self):
        if self.active and self.current_idx < len(self.targets):
            return self.targets[self.current_idx]
        return None

    def progress(self):
        """Returns (idx, total, elapsed_ratio)."""
        return (
            self.current_idx,
            len(self.targets),
            min((time.time() - self._t0) / self.DWELL, 1.0),
        )

    def tick(self) -> bool:
        """Advance to next point if dwell elapsed. Returns True when all done."""
        if not self.active:
            return False
        if (time.time() - self._t0) >= self.DWELL:
            self.current_idx += 1
            self._t0 = time.time()
            if self.current_idx >= len(self.targets):
                self.active = False
                return True
        return False


# ──────────────────────────────────────────────────────────────
# Transparent overlay widget
# ──────────────────────────────────────────────────────────────
class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint
            | Qt.FramelessWindowHint
            | Qt.WindowTransparentForInput
            | Qt.Tool,
        )
        self.setAttribute(Qt.WA_TranslucentBackground)

        sr = QApplication.primaryScreen().geometry()
        self.setGeometry(sr)
        self.sw, self.sh = sr.width(), sr.height()

        self.gaze_x       = self.sw // 2
        self.gaze_y       = self.sh // 2
        self.calibrating  = False
        self.fps          = 0.0
        self.blink_flash  = 0
        self.trail_on     = TRAIL_ENABLED
        self.t_start      = time.time()

        self.calib        = CalibrationManager(self.sw, self.sh)

        self.thread = EyeTrackingThread(self.sw, self.sh)
        self.thread.gaze_signal.connect(self._on_gaze)
        self.thread.blink_signal.connect(self._on_blink)
        self.thread.fps_signal.connect(self._on_fps)
        self.thread.start()

        self._timer = QTimer()
        self._timer.timeout.connect(self.update)
        self._timer.start(33)

        import keyboard
        keyboard.on_press_key("c",   self._toggle_calib)
        keyboard.on_press_key("t",   self._toggle_trail)
        keyboard.on_press_key("s",   self._snapshot)
        keyboard.on_press_key("esc", self._quit)

        print("\n" + "=" * 60)
        print(" GÖZ TAKİBİ — ENHANCED EDITION ".center(60, "="))
        print("=" * 60)
        print("[i] C   — Kalibrasyon (9-noktalı görsel mod)")
        print("[i] T   — Gaze Trail açma/kapama")
        print("[i] S   — Anlık heatmap snapshot")
        print("[i] ESC — Bitir + Analiz Raporu")
        print()

    # ── Slots ──────────────────────────────────────────────────
    def _on_gaze(self, x: int, y: int, cal: bool):
        self.gaze_x, self.gaze_y, self.calibrating = x, y, cal
        if cal and self.calib.active:
            self.calib.tick()

    def _on_blink(self):
        self.blink_flash = 4

    def _on_fps(self, fps: float):
        self.fps = fps

    # ── Key handlers ───────────────────────────────────────────
    def _toggle_calib(self, _=None):
        global is_calibrating, calib_data_x, calib_data_y
        global CALIB_X_MIN, CALIB_X_MAX, CALIB_Y_MIN, CALIB_Y_MAX

        with _lock:
            cur = is_calibrating

        if not cur:
            with _lock:
                is_calibrating = True
                calib_data_x.clear()
                calib_data_y.clear()
            self.calib.start()
            print("[i] Kalibrasyon başladı — 9 noktayı sırayla takip edin.")
        else:
            with _lock:
                is_calibrating = False
                dx = list(calib_data_x)
                dy = list(calib_data_y)
            self.calib.stop()
            if len(dx) > 10:
                CALIB_X_MIN = float(np.percentile(dx, 5))
                CALIB_X_MAX = float(np.percentile(dx, 95))
                CALIB_Y_MIN = float(np.percentile(dy, 5))
                CALIB_Y_MAX = float(np.percentile(dy, 95))
                print("[✓] Kalibrasyon tamamlandı!")
                print(f"    X: {CALIB_X_MIN:.3f} — {CALIB_X_MAX:.3f}")
                print(f"    Y: {CALIB_Y_MIN:.3f} — {CALIB_Y_MAX:.3f}")
            else:
                print("[!] Yeterli kalibrasyon verisi toplanamadı.")

    def _toggle_trail(self, _=None):
        self.trail_on = not self.trail_on
        print(f"[i] Trail: {'AÇık' if self.trail_on else 'KAPALI'}")

    def _snapshot(self, _=None):
        with _lock:
            pts = list(session_gaze_points)
        if len(pts) < 5:
            print("[!] Henüz yeterli veri yok.")
            return
        hm  = build_heatmap(pts, self.sw, self.sh, sigma=40)
        img = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_TURBO)
        os.makedirs("heatmap_kayitlar", exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        p   = f"heatmap_kayitlar/snapshot_{ts}.png"
        cv2.imwrite(p, img)
        print(f"[✓] Snapshot: {p}")

    def _quit(self, _=None):
        print("\n[i] Çıkılıyor...")
        import keyboard as kb
        kb.unhook_all()
        self._timer.stop()
        self.thread.stop()
        self.hide()
        with _lock:
            pts    = list(session_gaze_points)
            blinks = list(session_blinks)
        show_final_analysis(pts, blinks, self.sw, self.sh)
        QApplication.quit()

    # ── Paint ──────────────────────────────────────────────────
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        if self.calibrating and self.calib.active:
            self._paint_calib(p)
        else:
            if self.trail_on:
                self._paint_trail(p)
            self._paint_cursor(p)
        self._paint_hud(p)

    def _paint_calib(self, p: QPainter):
        target = self.calib.current_target()
        if not target:
            return
        idx, total, ratio = self.calib.progress()
        tx, ty = target

        # Dim overlay
        p.setBrush(QColor(0, 0, 0, 55))
        p.setPen(Qt.NoPen)
        p.drawRect(self.rect())

        OR = 32  # outer ring radius

        # Progress arc (clockwise from top)
        p.setPen(QPen(QColor(255, 210, 50, 210), 3))
        p.setBrush(Qt.NoBrush)
        p.drawArc(
            QRectF(tx - OR, ty - OR, OR * 2, OR * 2),
            90 * 16,
            -int(360 * ratio * 16),
        )

        # Static outer ring
        p.setPen(QPen(QColor(255, 255, 255, 90), 1))
        p.drawEllipse(tx - OR, ty - OR, OR * 2, OR * 2)

        # Centre dot
        p.setBrush(QColor(255, 80, 80, 230))
        p.setPen(Qt.NoPen)
        p.drawEllipse(tx - 9, ty - 9, 18, 18)

        # Cross-hair
        p.setPen(QPen(QColor(255, 255, 255, 130), 1))
        p.drawLine(tx - OR - 8, ty, tx - OR + 3, ty)
        p.drawLine(tx + OR - 3, ty, tx + OR + 8, ty)
        p.drawLine(tx, ty - OR - 8, tx, ty - OR + 3)
        p.drawLine(tx, ty + OR - 3, tx, ty + OR + 8)

        # Instruction text
        font = QFont("Arial", 14, QFont.Bold)
        p.setFont(font)
        p.setPen(QColor(255, 215, 50))
        p.drawText(
            self.rect().adjusted(0, 0, 0, -self.sh // 3),
            Qt.AlignHCenter | Qt.AlignBottom,
            f"KALİBRASYON  —  Nokta {idx + 1} / {total}\n"
            "Kafanızı sabit tutun, sadece gözlerinizle kırmızı noktaya bakın.",
        )

    def _paint_trail(self, p: QPainter):
        with _lock:
            pts = list(trail_points)
        now = time.time()
        for i, (x, y, t) in enumerate(pts):
            age   = now - t
            alpha = max(0, int(190 * (1.0 - age / 2.0)))
            size  = max(3, int(8 * (i / max(len(pts), 1))))
            if alpha <= 0:
                continue
            p.setBrush(QBrush(QColor(80, 200, 255, alpha)))
            p.setPen(Qt.NoPen)
            p.drawEllipse(x - size // 2, y - size // 2, size, size)

    def _paint_cursor(self, p: QPainter):
        x, y = self.gaze_x, self.gaze_y

        # Radial glow
        gr = QRadialGradient(x, y, 30)
        gr.setColorAt(0.0, QColor(255, 60, 60, 120))
        gr.setColorAt(1.0, QColor(255, 60, 60, 0))
        p.setBrush(QBrush(gr))
        p.setPen(Qt.NoPen)
        p.drawEllipse(x - 30, y - 30, 60, 60)

        # Outer ring (flashes white on blink)
        if self.blink_flash > 0:
            ring_col = QColor(255, 255, 255, 230)
            self.blink_flash -= 1
        else:
            ring_col = QColor(255, 80, 80, 170)
        p.setPen(QPen(ring_col, 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(x - 18, y - 18, 36, 36)

        # Cross-hair spokes
        p.setPen(QPen(QColor(255, 80, 80, 110), 1))
        for dx, dy in [(-25, 0), (20, 0), (0, -25), (0, 20)]:
            ex = dx + (5 if dx < 0 else -5 if dx > 0 else 0)
            ey = dy + (5 if dy < 0 else -5 if dy > 0 else 0)
            p.drawLine(x + dx, y + dy, x + ex, y + ey)

        # Centre dot
        p.setBrush(QColor(255, 255, 255, 230))
        p.setPen(Qt.NoPen)
        p.drawEllipse(x - 4, y - 4, 8, 8)

    def _paint_hud(self, p: QPainter):
        with _lock:
            n_blinks = len(session_blinks)
            n_pts    = len(session_gaze_points)
        elapsed = time.time() - self.t_start

        p.setBrush(QColor(0, 0, 0, 115))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(8, 8, 200, 96, 7, 7)

        font = QFont("Consolas", 10)
        p.setFont(font)
        p.setPen(QColor(140, 255, 140))
        for i, ln in enumerate([
            f"FPS: {self.fps:.0f}",
            f"Süre: {elapsed:.0f}s",
            f"Göz kırpma: {n_blinks}",
            f"Veri noktası: {n_pts}",
            f"Trail: {'ON' if self.trail_on else 'OFF'}",
        ]):
            p.drawText(16, 30 + i * 17, ln)


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app     = QApplication(sys.argv)
    overlay = Overlay()
    overlay.show()
    sys.exit(app.exec_())
