"""
Global Ekran Göz Takibi & Arka Plan Isı Haritası
================================================
Kontroller:
    C       -> Kalibrasyon Modunu Aç/Kapat
                (Ekranın dört köşesine bakın, bitince tekrar C)
    ESC     -> Takibi Bitir ve Analiz Raporunu Göster
"""

import sys
import cv2
import numpy as np
import mediapipe as mp
import keyboard
import urllib.request
import os
import time
import threading
from datetime import datetime
from scipy.ndimage import gaussian_filter

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen, QFont

# ──────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────
CAMERA_INDEX = 0
MODEL_PATH   = "face_landmarker.task"
MODEL_URL    = ("https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
TARGET_FPS   = 30

# Smoothing settings
SMOOTH_PREV   = 0.92   # önceki konuma ağırlık — yüksek = çok stabil
SMOOTH_RAW    = 0.08   # yeni ölçüme ağırlık
DEAD_ZONE_PX  = 18     # bu piksel yarıçapı içindeki titremeyi yoksay
MEDIAN_WINDOW = 7      # medyan filtre penceresi (son N frame)

# Iris landmark index
RIGHT_IRIS      = [469, 470, 471, 472]
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133

LEFT_IRIS       = [474, 475, 476, 477]
LEFT_EYE_OUTER  = 362
LEFT_EYE_INNER  = 263

# ──────────────────────────────────────────────
# Model Setup
# ──────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"[i] Model indiriliyor: {MODEL_PATH}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[✓] Model indirildi.")

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
face_landmarker_options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
)
face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_landmarker_options)

# ──────────────────────────────────────────────
# Global Situation
# ──────────────────────────────────────────────
_lock = threading.Lock()

is_calibrating = False

# Percentile calibration 
CALIB_X_MIN, CALIB_X_MAX = 0.40, 0.60
CALIB_Y_MIN, CALIB_Y_MAX = 0.40, 0.60
calib_data_x = []
calib_data_y = []

session_gaze_points = []   # [(x, y, timestamp), ...]

# ──────────────────────────────────────────────
# Supportive Functions
# ──────────────────────────────────────────────
def iris_center(landmarks, indices, w, h):
    xs = [landmarks[i].x * w for i in indices]
    ys = [landmarks[i].y * h for i in indices]
    return np.mean(xs), np.mean(ys)


def get_eye_ratios(lm, w, h):
    """
    Sağ ve sol iris ortalamasından ratio hesapla.
    ratio_x : normalize yatay iris konumu
    ratio_y : iris'in göz merkezine dikey sapması (göz genişliğine normalize)
    """
    # ── Right Eye ──
    r_iris    = iris_center(lm, RIGHT_IRIS, w, h)
    r_outer_x = lm[RIGHT_EYE_OUTER].x * w
    r_inner_x = lm[RIGHT_EYE_INNER].x * w
    r_outer_y = lm[RIGHT_EYE_OUTER].y * h
    r_inner_y = lm[RIGHT_EYE_INNER].y * h
    r_eye_w   = max(abs(r_outer_x - r_inner_x), 1)
    r_min_x   = min(r_outer_x, r_inner_x)
    r_ratio_x = (r_iris[0] - r_min_x) / r_eye_w
    r_ratio_y = (r_iris[1] - (r_outer_y + r_inner_y) / 2.0) / r_eye_w

    # ── Left Eye (x direction is reversed → 1 - ratio) ──
    l_iris    = iris_center(lm, LEFT_IRIS, w, h)
    l_outer_x = lm[LEFT_EYE_OUTER].x * w
    l_inner_x = lm[LEFT_EYE_INNER].x * w
    l_outer_y = lm[LEFT_EYE_OUTER].y * h
    l_inner_y = lm[LEFT_EYE_INNER].y * h
    l_eye_w   = max(abs(l_outer_x - l_inner_x), 1)
    l_min_x   = min(l_outer_x, l_inner_x)
    l_ratio_x = (l_iris[0] - l_min_x) / l_eye_w
    l_ratio_y = (l_iris[1] - (l_outer_y + l_inner_y) / 2.0) / l_eye_w

    return (r_ratio_x + l_ratio_x) / 2.0, (r_ratio_y + l_ratio_y) / 2.0


# ──────────────────────────────────────────────
# Camera / MediaPipe Thread
# ──────────────────────────────────────────────
class EyeTrackingThread(QThread):
    gaze_signal = pyqtSignal(int, int, bool)

    def __init__(self, screen_w, screen_h):
        super().__init__()
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.running  = True

    def run(self):
        global is_calibrating, calib_data_x, calib_data_y
        global CALIB_X_MIN, CALIB_X_MAX, CALIB_Y_MIN, CALIB_Y_MAX
        global session_gaze_points

        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        smooth_x = self.screen_w // 2
        smooth_y = self.screen_h // 2
        displayed_x = smooth_x
        displayed_y = smooth_y
        buf_x = []  
        buf_y = []
        frame_interval = 1.0 / TARGET_FPS

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

            if result.face_landmarks:
                lm = result.face_landmarks[0]
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

                # ── Convert to screen coordinates using percentage calibration. ──
                range_x = max(x_max - x_min, 0.001)
                range_y = max(y_max - y_min, 0.001)
                raw_x   = int(((ratio_x - x_min) / range_x) * self.screen_w)
                raw_y   = int(((ratio_y - y_min) / range_y) * self.screen_h)

                # 1. Median Filter 
                buf_x.append(raw_x)
                buf_y.append(raw_y)
                if len(buf_x) > MEDIAN_WINDOW:
                    buf_x.pop(0)
                    buf_y.pop(0)
                med_x = int(np.median(buf_x))
                med_y = int(np.median(buf_y))

                # 2. Heavy exponential smoothing
                smooth_x = int(SMOOTH_PREV * smooth_x + SMOOTH_RAW * med_x)
                smooth_y = int(SMOOTH_PREV * smooth_y + SMOOTH_RAW * med_y)

                # 3. Dead zone 
                dist = np.hypot(smooth_x - displayed_x, smooth_y - displayed_y)
                if dist >= DEAD_ZONE_PX:
                    displayed_x = smooth_x
                    displayed_y = smooth_y

                target_x = max(0, min(self.screen_w  - 1, displayed_x))
                target_y = max(0, min(self.screen_h - 1, displayed_y))

                with _lock:
                    session_gaze_points.append((target_x, target_y, time.time()))

                self.gaze_signal.emit(target_x, target_y, False)

            # Constant FPS 
            elapsed = time.time() - t0
            sleep_t = frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


# ──────────────────────────────────────────────
# Analysis & Heatmap
# ──────────────────────────────────────────────
def build_heatmap(points, w, h, sigma=35):
    """Gaussian blur ile ısı haritası oluştur."""
    hm = np.zeros((h, w), dtype=np.float32)
    for (x, y, *_) in points:
        if 0 <= x < w and 0 <= y < h:
            hm[y, x] += 1.0
    hm = gaussian_filter(hm, sigma=sigma)
    if hm.max() > 0:
        hm /= hm.max()
    return hm


def zone_analysis(points, w, h, cols=3, rows=3):
    """
    Ekranı cols×rows ızgaraya böl.
    Her bölge için bakış yüzdesini hesapla.
    """
    zone_counts = np.zeros((rows, cols), dtype=np.float32)
    for (x, y, *_) in points:
        col = min(int(x / w * cols), cols - 1)
        row = min(int(y / h * rows), rows - 1)
        zone_counts[row, col] += 1

    total = max(zone_counts.sum(), 1)
    zone_pct = zone_counts / total * 100
    return zone_pct


def fixation_analysis(points, w, h, radius=60, min_duration=0.15):
    """
    Basit fiksasyon tespiti: yakın nokta kümelerini bul ve süre filtrele.
    Döndürür: [(cx, cy, duration_secs), ...]
    """
    if len(points) < 2:
        return []

    fixations = []
    cluster   = [points[0]]

    for pt in points[1:]:
        cx   = cluster[0][0]
        cy   = cluster[0][1]
        dist = np.hypot(pt[0] - cx, pt[1] - cy)
        if dist < radius:
            cluster.append(pt)
        else:
            # Küme bitti — süre kontrolü
            t_start = cluster[0][2]
            t_end   = cluster[-1][2]
            dur     = t_end - t_start
            if dur >= min_duration:
                mean_x = int(np.mean([p[0] for p in cluster]))
                mean_y = int(np.mean([p[1] for p in cluster]))
                fixations.append((mean_x, mean_y, dur))
            cluster = [pt]

    return fixations


def show_final_analysis(points, w, h):
    if len(points) < 10:
        print("[!] Yeterli veri toplanamadi.")
        return

    print(f"\n[i] {len(points)} bakis noktasi analiz ediliyor...")

    try:
        import subprocess
        screenshot_path = "/tmp/screen_shot.png"
        # Linux için scrot, macOS için screencapture
        if sys.platform == "darwin":
            subprocess.run(["screencapture", "-x", screenshot_path], check=True)
        else:
            subprocess.run(["scrot", screenshot_path], check=True)
        bg = cv2.imread(screenshot_path)
        bg = cv2.resize(bg, (w, h))
    except Exception:
        bg = np.zeros((h, w, 3), dtype=np.uint8)

    # ── Heat Map ──
    hm      = build_heatmap(points, w, h, sigma=35)
    norm_hm = np.uint8(255 * hm)
    colored = cv2.applyColorMap(norm_hm, cv2.COLORMAP_JET)

    alpha_mask            = np.clip(hm * 2.0, 0, 0.75)
    alpha_3ch             = np.stack([alpha_mask] * 3, axis=2)
    blended               = (bg * (1 - alpha_3ch) + colored * alpha_3ch).astype(np.uint8)

    # ── Fixation Points ──
    fixations = fixation_analysis(points, w, h)
    for (fx, fy, dur) in fixations:
        radius  = int(15 + dur * 20)   # Uzun fiksasyon = büyük daire
        opacity = min(int(200 + dur * 30), 255)
        cv2.circle(blended, (fx, fy), radius, (255, 255, 0), 2)
        cv2.putText(blended, f"{dur:.1f}s", (fx + radius + 4, fy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

    # ── Region Analysis --Grid ──
    zone_pct = zone_analysis(points, w, h, cols=3, rows=3)
    cell_w   = w // 3
    cell_h   = h // 3

    for row in range(3):
        for col in range(3):
            x0  = col * cell_w
            y0  = row * cell_h
            x1  = x0 + cell_w
            y1  = y0 + cell_h
            pct = zone_pct[row, col]

            intensity = int(min(pct / 30 * 255, 255))
            color     = (0, intensity, 255 - intensity)
            cv2.rectangle(blended, (x0, y0), (x1 - 1, y1 - 1), color, 2)

            label = f"{pct:.1f}%"
            tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            tx = x0 + (cell_w - tw) // 2
            ty = y0 + (cell_h + th) // 2
            cv2.putText(blended, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ── Session Stats ──
    total_time  = points[-1][2] - points[0][2] if len(points) > 1 else 0
    n_fixations = len(fixations)
    avg_fix_dur = np.mean([f[2] for f in fixations]) if fixations else 0
    top_zone_idx = np.unravel_index(np.argmax(zone_pct), zone_pct.shape)

    zone_names = [["Sol Üst", "Orta Üst", "Sağ Üst"],
                  ["Sol Orta", "Merkez",   "Sağ Orta"],
                  ["Sol Alt",  "Orta Alt",  "Sağ Alt"]]
    top_zone_name = zone_names[top_zone_idx[0]][top_zone_idx[1]]

    stats_lines = [
        f"Oturum Suresi : {total_time:.1f}s",
        f"Fiksasyon Sayisi: {n_fixations}",
        f"Ort. Fiksasyon : {avg_fix_dur:.2f}s",
        f"En Cok Bakilan : {top_zone_name} ({zone_pct[top_zone_idx]:.1f}%)",
    ]
    panel_h = len(stats_lines) * 28 + 20
    cv2.rectangle(blended, (10, 10), (340, 10 + panel_h), (0, 0, 0), -1)
    for i, line in enumerate(stats_lines):
        cv2.putText(blended, line, (18, 38 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 1)

    # ── Save ──
    os.makedirs("heatmap_kayitlar", exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"heatmap_kayitlar/oturum_{ts}.png"
    cv2.imwrite(filepath, blended)
    print(f"[✓] Analiz goruntüsü kaydedildi: {filepath}")

    # ── Report ──
    cv2.namedWindow("Goz Takibi Analizi", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Goz Takibi Analizi",
                          cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Goz Takibi Analizi", blended)
    print("[i] Kapatmak icin herhangi bir tusa basin.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()



class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint  |
            Qt.WindowTransparentForInput |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)

        screen_rect = QApplication.primaryScreen().geometry()
        self.setGeometry(screen_rect)
        self.sw = screen_rect.width()
        self.sh = screen_rect.height()

        self.target_x       = self.sw // 2
        self.target_y       = self.sh // 2
        self.is_calibrating = False

        self.tracker_thread = EyeTrackingThread(self.sw, self.sh)
        self.tracker_thread.gaze_signal.connect(self.update_gaze)
        self.tracker_thread.start()

        keyboard.on_press_key("c",   self.toggle_calibration)
        keyboard.on_press_key("esc", self.quit_app)

        print("\n" + "=" * 52)
        print(" GÖZ TAKİBİ BAŞLADI ".center(52, "="))
        print("=" * 52)
        print("[i] C   : Kalibrasyon (Bas / Bitir)")
        print("[i] ESC : Bitir ve Analiz Raporunu Göster")
        print("\n[!] İpucu: Kalibrasyonda tüm ekrana (özellikle köşelere) bakın.\n")

    def update_gaze(self, x, y, calibrating):
        self.target_x       = x
        self.target_y       = y
        self.is_calibrating = calibrating
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.is_calibrating:
            font = QFont("Arial", 18, QFont.Bold)
            painter.setFont(font)
            painter.setPen(QColor(255, 165, 0))
            painter.drawText(
                self.rect(), Qt.AlignCenter,
                "KALİBRASYON MODU\n"
                "Kafanızı sabit tutun, gözlerinizle ekranın tüm köşelerine bakın.\n"
                "Bitince tekrar 'C' tuşuna basın."
            )
        else:
            painter.setBrush(QColor(255, 0, 0, 130))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.target_x - 20, self.target_y - 20, 40, 40)
            painter.setBrush(QColor(255, 0, 0, 255))
            painter.drawEllipse(self.target_x - 5, self.target_y - 5, 10, 10)

    def toggle_calibration(self, event=None):
        global is_calibrating, calib_data_x, calib_data_y
        global CALIB_X_MIN, CALIB_X_MAX, CALIB_Y_MIN, CALIB_Y_MAX

        with _lock:
            currently = is_calibrating

        if not currently:
            with _lock:
                is_calibrating = True
                calib_data_x.clear()
                calib_data_y.clear()
            print("[i] Kalibrasyon başladı. Tüm köşelere bakın...")
        else:
            with _lock:
                is_calibrating = False
                dx = list(calib_data_x)
                dy = list(calib_data_y)

            if len(dx) > 10:
                CALIB_X_MIN = float(np.percentile(dx, 5))
                CALIB_X_MAX = float(np.percentile(dx, 95))
                CALIB_Y_MIN = float(np.percentile(dy, 5))
                CALIB_Y_MAX = float(np.percentile(dy, 95))
                print("[✓] Kalibrasyon tamamlandı!")
                print(f"    X aralığı: {CALIB_X_MIN:.3f} – {CALIB_X_MAX:.3f}")
                print(f"    Y aralığı: {CALIB_Y_MIN:.3f} – {CALIB_Y_MAX:.3f}")
            else:
                print("[!] Yeterli kalibrasyon verisi toplanamadı.")

    def quit_app(self, event=None):
        print("\n[i] Çıkılıyor...")
        keyboard.unhook_all()
        self.tracker_thread.stop()
        self.hide()

        with _lock:
            pts = list(session_gaze_points)

        show_final_analysis(pts, self.sw, self.sh)
        QApplication.quit()


# ──────────────────────────────────────────────
if __name__ == "__main__":
    app     = QApplication(sys.argv)
    overlay = Overlay()
    overlay.show()
    sys.exit(app.exec_())