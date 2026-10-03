"""
ui/dashboard.py
------------------
Tkinter dashboard for the real-time drowsiness detection application.

Layout:
    Left   -> live webcam feed with facial landmarks drawn on top.
    Right  -> status banner, drowsiness score, EAR/MAR/head-pose readouts
              and session statistics, plus Start/Stop/Reset/Exit controls.

The whole pipeline (camera read -> face mesh -> analyzers -> scoring) runs
inside a single Tkinter `after()` loop on the main thread, which keeps the
application simple while remaining responsive at typical webcam frame rates.
"""

import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

import config
from detection.drowsiness_engine import DrowsinessEngine
from detection.face_detector import FaceDetector
from utils.alarm import AlarmManager


class DrowsinessDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title(config.WINDOW_TITLE)
        self.root.configure(bg=config.WINDOW_BG)
        self.root.minsize(980, 600)

        self.engine = DrowsinessEngine()
        self.alarm_manager = AlarmManager()
        self.face_detector = None  # created lazily when Start is pressed

        self.capture = None
        self.is_running = False
        self._after_job = None
        self._consecutive_read_failures = 0
        self._last_frame_time = time.time()
        self._fps = 0.0

        self._build_ui()
        self._show_placeholder_frame("Press START to begin monitoring")
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        container = tk.Frame(self.root, bg=config.WINDOW_BG)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ---------------- Left: video panel ----------------
        left_frame = tk.Frame(container, bg=config.PANEL_BG, bd=0)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        tk.Label(
            left_frame,
            text="Live Camera Feed",
            bg=config.PANEL_BG,
            fg=config.TEXT_COLOR,
            font=("Segoe UI", 13, "bold"),
        ).pack(pady=(8, 4))

        self.video_label = tk.Label(left_frame, bg="black")
        self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.fps_label = tk.Label(
            left_frame, text="FPS: --", bg=config.PANEL_BG, fg=config.TEXT_COLOR,
            font=("Consolas", 10),
        )
        self.fps_label.pack(pady=(0, 8))

        # ---------------- Right: status / stats panel ----------------
        right_frame = tk.Frame(container, bg=config.PANEL_BG, width=340)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)

        tk.Label(
            right_frame,
            text="Drowsiness Monitor",
            bg=config.PANEL_BG,
            fg=config.TEXT_COLOR,
            font=("Segoe UI", 14, "bold"),
        ).pack(pady=(10, 6))

        # State banner
        self.state_banner = tk.Label(
            right_frame,
            text=config.STATE_NORMAL,
            bg=config.STATE_COLORS[config.STATE_NORMAL],
            fg="white",
            font=("Segoe UI", 20, "bold"),
            pady=12,
        )
        self.state_banner.pack(fill=tk.X, padx=14, pady=(0, 10))

        # Score
        score_frame = tk.Frame(right_frame, bg=config.PANEL_BG)
        score_frame.pack(fill=tk.X, padx=14, pady=(0, 10))
        tk.Label(
            score_frame, text="Drowsiness Score", bg=config.PANEL_BG,
            fg=config.TEXT_COLOR, font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")
        self.score_value_label = tk.Label(
            score_frame, text="0 / 100", bg=config.PANEL_BG,
            fg=config.ACCENT_COLOR, font=("Segoe UI", 16, "bold"),
        )
        self.score_value_label.pack(anchor="w")
        self.score_bar = ttk.Progressbar(
            score_frame, orient="horizontal", mode="determinate", maximum=100
        )
        self.score_bar.pack(fill=tk.X, pady=(4, 0))

        # Live readouts
        readout_frame = tk.LabelFrame(
            right_frame, text="Live Readouts", bg=config.PANEL_BG,
            fg=config.TEXT_COLOR, font=("Segoe UI", 10, "bold"),
        )
        readout_frame.pack(fill=tk.X, padx=14, pady=(0, 10))

        self.readout_vars = {}
        for key, label in [
            ("ear", "Eye Aspect Ratio (EAR)"),
            ("eye_state", "Eye State"),
            ("mar", "Mouth Aspect Ratio (MAR)"),
            ("mouth_state", "Mouth / Yawn State"),
            ("head_pose", "Head Pose"),
        ]:
            row = tk.Frame(readout_frame, bg=config.PANEL_BG)
            row.pack(fill=tk.X, padx=8, pady=2)
            tk.Label(
                row, text=f"{label}:", bg=config.PANEL_BG, fg=config.TEXT_COLOR,
                font=("Segoe UI", 9), anchor="w", width=22,
            ).pack(side=tk.LEFT)
            value_label = tk.Label(
                row, text="--", bg=config.PANEL_BG, fg="#9be28a",
                font=("Consolas", 9, "bold"), anchor="w",
            )
            value_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.readout_vars[key] = value_label

        # Session statistics
        stats_frame = tk.LabelFrame(
            right_frame, text="Session Statistics", bg=config.PANEL_BG,
            fg=config.TEXT_COLOR, font=("Segoe UI", 10, "bold"),
        )
        stats_frame.pack(fill=tk.X, padx=14, pady=(0, 10))

        self.stat_vars = {}
        for key, label in [
            ("blink_count", "Blink Count"),
            ("long_blink_count", "Long Blink Count"),
            ("yawn_count", "Yawn Count"),
            ("warning_count", "Warning Count"),
            ("drowsy_count", "Drowsiness Count"),
            ("max_closure_duration", "Max Eye-Closure (s)"),
            ("session_duration", "Session Duration (s)"),
        ]:
            row = tk.Frame(stats_frame, bg=config.PANEL_BG)
            row.pack(fill=tk.X, padx=8, pady=2)
            tk.Label(
                row, text=f"{label}:", bg=config.PANEL_BG, fg=config.TEXT_COLOR,
                font=("Segoe UI", 9), anchor="w", width=22,
            ).pack(side=tk.LEFT)
            value_label = tk.Label(
                row, text="0", bg=config.PANEL_BG, fg=config.TEXT_COLOR,
                font=("Consolas", 9, "bold"), anchor="w",
            )
            value_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.stat_vars[key] = value_label

        # Status / error line
        self.status_label = tk.Label(
            right_frame, text="Idle. Press Start to begin.", bg=config.PANEL_BG,
            fg="#bbbbbb", font=("Segoe UI", 9, "italic"), wraplength=300,
            justify="left",
        )
        self.status_label.pack(fill=tk.X, padx=14, pady=(0, 10))

        # Buttons
        button_frame = tk.Frame(right_frame, bg=config.PANEL_BG)
        button_frame.pack(fill=tk.X, padx=14, pady=(4, 14), side=tk.BOTTOM)

        self.start_button = tk.Button(
            button_frame, text="Start", command=self.on_start, bg="#2ecc71",
            fg="white", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, width=9,
        )
        self.start_button.grid(row=0, column=0, padx=3, pady=3)

        self.stop_button = tk.Button(
            button_frame, text="Stop", command=self.on_stop, bg="#e67e22",
            fg="white", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, width=9,
            state=tk.DISABLED,
        )
        self.stop_button.grid(row=0, column=1, padx=3, pady=3)

        self.reset_button = tk.Button(
            button_frame, text="Reset Session", command=self.on_reset,
            bg="#3498db", fg="white", font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, width=12,
        )
        self.reset_button.grid(row=1, column=0, padx=3, pady=3)

        self.exit_button = tk.Button(
            button_frame, text="Exit", command=self.on_exit, bg="#e74c3c",
            fg="white", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, width=9,
        )
        self.exit_button.grid(row=1, column=1, padx=3, pady=3)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def on_start(self):
        if self.is_running:
            return

        if self.face_detector is None:
            try:
                self.face_detector = FaceDetector()
            except Exception as exc:
                messagebox.showerror("Initialization Error", f"Could not load MediaPipe Face Mesh:\n{exc}")
                return

        opened = self._open_camera()
        if not opened:
            messagebox.showerror(
                "Webcam Error",
                "Could not access the webcam. Make sure it is connected, not "
                "being used by another application, and that camera "
                "permissions are granted.",
            )
            self.status_label.config(text="Webcam not available.")
            return

        self.is_running = True
        self._consecutive_read_failures = 0
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Monitoring in progress...")
        self._last_frame_time = time.time()
        self._update_frame()

    def on_stop(self, status_message="Stopped. Press Start to resume."):
        if not self.is_running:
            return
        self._stop_loop()
        self._release_camera()
        self.alarm_manager.stop()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text=status_message)
        self._show_placeholder_frame("Camera stopped")

    def on_reset(self):
        self.engine.reset_session()
        self.alarm_manager.stop()
        self._refresh_stats_only()
        self.status_label.config(text="Session statistics have been reset.")

    def on_exit(self):
        self._stop_loop()
        self._release_camera()
        try:
            self.alarm_manager.shutdown()
        except Exception:
            pass
        if self.face_detector is not None:
            self.face_detector.close()
        self.root.destroy()

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------
    def _open_camera(self):
        attempts = config.CAMERA_OPEN_RETRIES + 1
        for _ in range(attempts):
            try:
                capture = cv2.VideoCapture(config.CAMERA_INDEX)
            except Exception:
                capture = None

            if capture is not None and capture.isOpened():
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                self.capture = capture
                return True

            if capture is not None:
                capture.release()
            time.sleep(0.3)

        self.capture = None
        return False

    def _release_camera(self):
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
            self.capture = None

    def _stop_loop(self):
        self.is_running = False
        if self._after_job is not None:
            try:
                self.root.after_cancel(self._after_job)
            except Exception:
                pass
            self._after_job = None

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------
    def _update_frame(self):
        if not self.is_running or self.capture is None:
            return

        try:
            ok, frame = self.capture.read()
        except Exception as exc:
            ok, frame = False, None
            self.status_label.config(text=f"Camera read error: {exc}")

        if not ok or frame is None:
            self._consecutive_read_failures += 1
            if self._consecutive_read_failures >= 30:
                self.on_stop(status_message="Lost connection to the webcam. Stopped.")
                return
            self._after_job = self.root.after(config.UI_REFRESH_MS, self._update_frame)
            return

        self._consecutive_read_failures = 0

        try:
            frame = cv2.flip(frame, 1)
            display_frame, result = self._process_frame(frame)
            self._render_frame(display_frame)
            self._apply_result(result)
        except Exception as exc:
            # Never let a processing hiccup crash the whole application.
            self.status_label.config(text=f"Processing error (recovered): {exc}")

        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            self._fps = (0.9 * self._fps) + (0.1 * (1.0 / dt)) if self._fps else (1.0 / dt)
            self.fps_label.config(text=f"FPS: {self._fps:.1f}")

        self._after_job = self.root.after(config.UI_REFRESH_MS, self._update_frame)

    def _process_frame(self, frame):
        landmarks, mp_results = self.face_detector.process(frame)
        result = self.engine.update(landmarks, frame.shape)

        display_frame = frame.copy()
        if landmarks is not None:
            self.face_detector.draw_landmarks(display_frame, mp_results)
            self._draw_overlay_text(display_frame, result)
        else:
            cv2.putText(
                display_frame, "NO FACE DETECTED", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
            )

        return display_frame, result

    @staticmethod
    def _draw_overlay_text(frame, result):
        eye = result["eye"] or {}
        mouth = result["mouth"] or {}
        head = result["head"] or {}
        lines = [
            f"EAR: {eye.get('ear', 0):.2f}",
            f"MAR: {mouth.get('mar', 0):.2f}",
            f"Head: {head.get('label', '--')}",
            f"Score: {result['score']:.0f}",
            f"State: {result['state']}",
        ]
        # OpenCV uses BGR ordering, so these are the BGR equivalents of the
        # hex colours used for the Tkinter state banner (config.STATE_COLORS).
        color = {
            config.STATE_NORMAL: (113, 204, 46),   # green
            config.STATE_WARNING: (18, 156, 243),  # amber
            config.STATE_DROWSY: (60, 76, 231),    # red
        }.get(result["state"], (255, 255, 255))

        y = 30
        for line in lines:
            cv2.putText(
                frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )
            y += 26

    def _render_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.imgtk = photo  # keep a reference alive
        self.video_label.configure(image=photo)

    # ------------------------------------------------------------------
    # Result -> UI updates
    # ------------------------------------------------------------------
    def _apply_result(self, result):
        state = result["state"]
        self.state_banner.config(text=state, bg=config.STATE_COLORS[state])
        self.score_value_label.config(text=f"{result['score']:.0f} / 100")
        self.score_bar["value"] = result["score"]

        eye = result["eye"]
        mouth = result["mouth"]
        head = result["head"]

        if result["face_found"] and eye and mouth and head:
            self.readout_vars["ear"].config(text=f"{eye['ear']:.3f}")
            self.readout_vars["eye_state"].config(
                text="CLOSED" if eye["eye_closed"] else "OPEN"
            )
            self.readout_vars["mar"].config(text=f"{mouth['mar']:.3f}")
            self.readout_vars["mouth_state"].config(
                text="YAWNING" if mouth["is_yawning"] else (
                    "OPEN" if mouth["mouth_open"] else "CLOSED"
                )
            )
            self.readout_vars["head_pose"].config(
                text=f"{head['label']} (p{head['pitch']:.0f} / y{head['yaw']:.0f})"
            )
            if not result["no_face_timeout"]:
                self.status_label.config(text="Monitoring in progress...")
        else:
            for key in ("ear", "eye_state", "mar", "mouth_state", "head_pose"):
                self.readout_vars[key].config(text="--")
            if result["no_face_timeout"]:
                self.status_label.config(
                    text="No face detected. Please face the camera."
                )

        self._refresh_stats_only()

        if state == config.STATE_DROWSY:
            self.alarm_manager.start()
        else:
            self.alarm_manager.stop()

    def _refresh_stats_only(self):
        stats = self.engine.get_stats()
        self.stat_vars["blink_count"].config(text=str(stats["blink_count"]))
        self.stat_vars["long_blink_count"].config(text=str(stats["long_blink_count"]))
        self.stat_vars["yawn_count"].config(text=str(stats["yawn_count"]))
        self.stat_vars["warning_count"].config(text=str(stats["warning_count"]))
        self.stat_vars["drowsy_count"].config(text=str(stats["drowsy_count"]))
        self.stat_vars["max_closure_duration"].config(
            text=f"{stats['max_closure_duration']:.2f}"
        )
        self.stat_vars["session_duration"].config(
            text=f"{stats['session_duration']:.0f}"
        )
        if not self.is_running:
            self.score_value_label.config(text=f"{self.engine.score:.0f} / 100")
            self.score_bar["value"] = self.engine.score
            self.state_banner.config(
                text=self.engine.state, bg=config.STATE_COLORS[self.engine.state]
            )

    # ------------------------------------------------------------------
    def _show_placeholder_frame(self, message):
        placeholder = Image.new("RGB", (config.FRAME_WIDTH, config.FRAME_HEIGHT), "black")
        photo = ImageTk.PhotoImage(image=placeholder)
        self.video_label.imgtk = photo
        self.video_label.configure(image=photo, text=message, compound="center",
                                    fg="white", font=("Segoe UI", 14))
