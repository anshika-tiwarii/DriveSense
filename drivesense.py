"""
DriveSense - Driver Drowsiness Detection System
Uses OpenCV, Dlib, and NumPy to monitor driver fatigue via eye-blink frequency
and facial landmark analysis. Real-time alarm with <150ms latency.
"""

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import time
import threading
import os
import sys

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
EAR_THRESHOLD      = 0.25   # Eye Aspect Ratio below this = eye closed
EAR_CONSEC_FRAMES  = 20     # Consecutive frames eye must be closed to trigger alarm
MAR_THRESHOLD      = 0.65   # Mouth Aspect Ratio above this = yawning
MAR_CONSEC_FRAMES  = 15     # Consecutive frames mouth must be open to trigger yawn
BLINK_RATE_WINDOW  = 60     # Seconds window for blink rate calculation
LOW_BLINK_RATE     = 8      # Blinks/min below this is a drowsiness sign
HIGH_BLINK_RATE    = 25     # Blinks/min above this is a stress/fatigue sign
ALARM_DURATION     = 2.0    # Seconds the alarm beeps

# Dlib landmark indices (iBUG 68-point model)
(L_START, L_END) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_START, R_END) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(M_START, M_END) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# ─────────────────────────────────────────────
#  COLOURS  (BGR)
# ─────────────────────────────────────────────
C_BG        = (15,  15,  20)
C_GREEN     = (80,  220, 130)
C_YELLOW    = (30,  210, 255)
C_RED       = (60,  60,  220)
C_WHITE     = (240, 240, 240)
C_GREY      = (120, 120, 130)
C_TEAL      = (180, 200, 60)
C_PANEL_BG  = (28,  28,  35)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2],  mouth[10])
    B = dist.euclidean(mouth[4],  mouth[8])
    C = dist.euclidean(mouth[0],  mouth[6])
    return (A + B) / (2.0 * C)


def draw_landmarks(frame, shape, indices, color):
    pts = shape[indices[0]:indices[1]]
    hull = cv2.convexHull(pts)
    cv2.drawContours(frame, [hull], -1, color, 1)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 1, color, -1)


def beep_alarm():
    """Cross-platform audio alert."""
    for _ in range(4):
        if sys.platform == "win32":
            import winsound
            winsound.Beep(1000, 200)
        elif sys.platform == "darwin":
            os.system("afplay /System/Library/Sounds/Ping.aiff &")
        else:
            # Linux: try multiple methods
            os.system("paplay /usr/share/sounds/alsa/Front_Left.wav 2>/dev/null || "
                      "beep -f 1000 -l 200 2>/dev/null || "
                      "echo -e '\\a' 2>/dev/null")
        time.sleep(0.25)


def draw_panel(frame, x, y, w, h, alpha=0.6):
    """Draw a semi-transparent dark panel."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), C_PANEL_BG, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_TEAL, 1)


def draw_gauge(frame, label, value, max_val, x, y, w, h, color):
    """Draw a horizontal progress bar gauge."""
    pct = min(value / max_val, 1.0)
    bar_w = int((w - 10) * pct)
    # Background bar
    cv2.rectangle(frame, (x + 5, y + h//2 - 4), (x + w - 5, y + h//2 + 4),
                  (50, 50, 60), -1)
    # Filled bar
    if bar_w > 0:
        cv2.rectangle(frame, (x + 5, y + h//2 - 4),
                      (x + 5 + bar_w, y + h//2 + 4), color, -1)
    cv2.putText(frame, f"{label}: {value:.2f}", (x + 5, y + h//2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_WHITE, 1)


def status_color(level):
    if level == "ALERT":
        return C_RED
    if level == "WARNING":
        return C_YELLOW
    return C_GREEN


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────
class DriveSense:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat",
                 camera_index=0):
        print("[DriveSense] Initialising detector…")
        self.detector  = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # State
        self.ear_counter   = 0
        self.mar_counter   = 0
        self.total_blinks  = 0
        self.total_yawns   = 0
        self.alarm_on      = False
        self.session_start = time.time()
        self.blink_times   = []     # timestamps of each blink
        self.ear_history   = []     # rolling EAR for graph
        self.status        = "NORMAL"
        self.status_msg    = "Monitoring active"
        self.fps_list      = []
        self.last_alarm_t  = 0

        print("[DriveSense] Ready. Press Q to quit, R to reset stats.")

    # ── alarm thread ──────────────────────────
    def trigger_alarm(self, reason):
        if self.alarm_on:
            return
        now = time.time()
        if now - self.last_alarm_t < ALARM_DURATION + 1:
            return
        self.alarm_on   = True
        self.last_alarm_t = now
        self.status     = "ALERT"
        self.status_msg = f"⚠  {reason}"
        t = threading.Thread(target=self._alarm_worker, daemon=True)
        t.start()

    def _alarm_worker(self):
        beep_alarm()
        self.alarm_on = False

    # ── per-frame processing ──────────────────
    def process_frame(self, frame):
        t0     = time.perf_counter()
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects  = self.detector(gray, 0)

        ear = mar = 0.0
        face_found = False

        for rect in rects:
            face_found = True
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            left_eye  = shape[L_START:L_END]
            right_eye = shape[R_START:R_END]
            mouth_pts = shape[M_START:M_END]

            left_ear  = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(mouth_pts)

            # Draw landmarks
            draw_landmarks(frame, shape, (L_START, L_END), C_TEAL)
            draw_landmarks(frame, shape, (R_START, R_END), C_TEAL)
            draw_landmarks(frame, shape, (M_START, M_END), C_GREEN)

            # Face bounding box
            (fx, fy, fw, fh) = (rect.left(), rect.top(),
                                 rect.width(), rect.height())
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh),
                          status_color(self.status), 1)

            # ── eye closure detection ──
            if ear < EAR_THRESHOLD:
                self.ear_counter += 1
                if self.ear_counter >= EAR_CONSEC_FRAMES:
                    self.trigger_alarm("Drowsiness – Eyes closed!")
            else:
                if self.ear_counter >= 3:          # completed blink
                    self.total_blinks += 1
                    self.blink_times.append(time.time())
                self.ear_counter = 0

            # ── yawn detection ──
            if mar > MAR_THRESHOLD:
                self.mar_counter += 1
                if self.mar_counter >= MAR_CONSEC_FRAMES:
                    if self.mar_counter == MAR_CONSEC_FRAMES:   # first trigger
                        self.total_yawns += 1
                        self.trigger_alarm("Yawning detected!")
            else:
                self.mar_counter = 0

        # Rolling EAR history (last 90 values → graph)
        self.ear_history.append(ear)
        if len(self.ear_history) > 90:
            self.ear_history.pop(0)

        # Blink rate (last BLINK_RATE_WINDOW seconds)
        now = time.time()
        cutoff = now - BLINK_RATE_WINDOW
        self.blink_times = [t for t in self.blink_times if t > cutoff]
        blink_rate = len(self.blink_times) * (60.0 / BLINK_RATE_WINDOW)

        # Blink-rate drowsiness heuristic
        # Only fire after 60s (full window) AND blink_rate must be > 0 to avoid
        # false triggers at session start when no blinks have been counted yet
        elapsed = now - self.session_start
        if elapsed > BLINK_RATE_WINDOW and blink_rate > 0 and blink_rate < LOW_BLINK_RATE and face_found:
            self.trigger_alarm("Low blink rate – possible microsleep!")

        # Update status — clear ALERT once alarm finishes AND eyes are open
        alert_cooldown = 3.0  # seconds after alarm finishes before returning to NORMAL
        time_since_alarm = now - self.last_alarm_t
        alert_cleared = (not self.alarm_on) and (time_since_alarm > alert_cooldown)

        if alert_cleared:
            if self.ear_counter > EAR_CONSEC_FRAMES // 2:
                self.status = "WARNING"
                self.status_msg = "Eyes closing…"
            elif self.mar_counter > MAR_CONSEC_FRAMES // 2:
                self.status = "WARNING"
                self.status_msg = "Yawning…"
            else:
                self.status = "NORMAL"
                self.status_msg = "Monitoring active"

        # FPS
        elapsed_frame = time.perf_counter() - t0
        fps = 1.0 / elapsed_frame if elapsed_frame > 0 else 0
        self.fps_list.append(fps)
        if len(self.fps_list) > 30:
            self.fps_list.pop(0)
        avg_fps = sum(self.fps_list) / len(self.fps_list)

        latency_ms = elapsed_frame * 1000

        return frame, ear, mar, blink_rate, avg_fps, latency_ms, face_found

    # ── HUD overlay ──────────────────────────
    def draw_hud(self, frame, ear, mar, blink_rate, fps, latency_ms, face_found):
        H, W = frame.shape[:2]

        # ── top banner ──
        draw_panel(frame, 0, 0, W, 42, alpha=0.75)
        cv2.putText(frame, "DRIVESENSE", (12, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, C_TEAL, 1)
        elapsed = int(time.time() - self.session_start)
        mins, secs = divmod(elapsed, 60)
        cv2.putText(frame, f"Session  {mins:02d}:{secs:02d}", (180, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_GREY, 1)

        # Latency badge (right of banner)
        lat_color = C_GREEN if latency_ms < 150 else C_RED
        cv2.putText(frame, f"{latency_ms:.1f} ms", (W - 100, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, lat_color, 1)
        cv2.putText(frame, f"{fps:.0f} FPS", (W - 185, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_GREY, 1)

        # ── left stats panel ──
        panel_w, panel_h = 230, 220
        draw_panel(frame, 8, 50, panel_w, panel_h)

        y = 72
        def stat(label, val, color=C_WHITE):
            nonlocal y
            cv2.putText(frame, label, (16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GREY, 1)
            cv2.putText(frame, str(val), (130, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            y += 22

        stat("EAR",        f"{ear:.3f}",
             C_RED if ear < EAR_THRESHOLD and face_found else C_GREEN)
        stat("MAR",        f"{mar:.3f}",
             C_YELLOW if mar > MAR_THRESHOLD and face_found else C_GREEN)
        stat("Blink rate", f"{blink_rate:.1f}/min",
             C_YELLOW if blink_rate < LOW_BLINK_RATE and blink_rate > 0 else C_WHITE)
        stat("Total blinks",  self.total_blinks)
        stat("Total yawns",   self.total_yawns)
        stat("Eye frames",    self.ear_counter,
             C_RED if self.ear_counter > 5 else C_WHITE)
        stat("Face detected", "YES" if face_found else "NO",
             C_GREEN if face_found else C_RED)

        # EAR gauge
        draw_gauge(frame, "EAR", ear, 0.5, 16, y, panel_w - 24, 24,
                   C_RED if ear < EAR_THRESHOLD else C_GREEN)
        y += 28
        draw_gauge(frame, "MAR", mar, 1.0, 16, y, panel_w - 24, 24,
                   C_YELLOW if mar > MAR_THRESHOLD else C_GREEN)

        # ── EAR mini-graph (bottom-left) ──
        graph_x, graph_y = 8, H - 100
        graph_w, graph_h = 230, 90
        draw_panel(frame, graph_x, graph_y, graph_w, graph_h, alpha=0.7)
        cv2.putText(frame, "EAR History", (graph_x + 5, graph_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_GREY, 1)
        # threshold line
        th_y = graph_y + graph_h - int(EAR_THRESHOLD / 0.5 * (graph_h - 20)) - 5
        cv2.line(frame, (graph_x + 2, th_y), (graph_x + graph_w - 2, th_y),
                 C_RED, 1)
        if len(self.ear_history) > 1:
            pts = []
            for i, v in enumerate(self.ear_history):
                px = graph_x + 2 + int(i * (graph_w - 4) / 90)
                py = graph_y + graph_h - 5 - int(min(v, 0.5) / 0.5 * (graph_h - 20))
                pts.append((px, py))
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], C_TEAL, 1)

        # ── status bar ──
        s_color = status_color(self.status)
        draw_panel(frame, 0, H - 36, W, 36, alpha=0.8)
        cv2.putText(frame, f"[ {self.status} ]  {self.status_msg}",
                    (12, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, s_color, 1)

        # ── flashing ALERT overlay ──
        if self.alarm_on:
            flash = np.zeros_like(frame, dtype=np.uint8)
            flash[:] = (0, 0, 180)
            t = int(time.time() * 6) % 2
            if t == 0:
                cv2.addWeighted(flash, 0.18, frame, 0.82, 0, frame)
            cv2.putText(frame, "!  DROWSINESS ALERT  !",
                        (W // 2 - 185, H // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, C_RED, 2)

        # ── controls hint (top-right) ──
        hints = ["Q - Quit", "R - Reset stats", "S - Screenshot"]
        for i, h in enumerate(hints):
            cv2.putText(frame, h, (W - 155, 62 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_GREY, 1)

        return frame

    # ── main loop ────────────────────────────
    def run(self):
        cv2.namedWindow("DriveSense", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("DriveSense", 1280, 720)
        shot_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[DriveSense] Camera read failed.")
                break

            frame, ear, mar, blink_rate, fps, latency_ms, face_found = \
                self.process_frame(frame)
            frame = self.draw_hud(frame, ear, mar, blink_rate,
                                  fps, latency_ms, face_found)

            cv2.imshow("DriveSense", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                self.total_blinks = 0
                self.total_yawns  = 0
                self.blink_times  = []
                self.session_start = time.time()
                self.status_msg   = "Stats reset"
                print("[DriveSense] Stats reset.")
            elif key == ord('s'):
                shot_count += 1
                fname = f"drivesense_shot_{shot_count}.png"
                cv2.imwrite(fname, frame)
                print(f"[DriveSense] Screenshot saved: {fname}")

        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\n[DriveSense] Session ended.")
        print(f"  Total blinks : {self.total_blinks}")
        print(f"  Total yawns  : {self.total_yawns}")
        elapsed = int(time.time() - self.session_start)
        print(f"  Session time : {elapsed // 60}m {elapsed % 60}s")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DriveSense – Driver Drowsiness Detection")
    parser.add_argument("--predictor", default="shape_predictor_68_face_landmarks.dat",
                        help="Path to dlib 68-point landmark predictor file")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    args = parser.parse_args()

    if not os.path.exists(args.predictor):
        print(f"""
[DriveSense] ERROR: Landmark predictor not found at '{args.predictor}'

Download it with:
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

Or use the setup script:
    python setup.py
""")
        sys.exit(1)

    app = DriveSense(predictor_path=args.predictor, camera_index=args.camera)
    app.run()
