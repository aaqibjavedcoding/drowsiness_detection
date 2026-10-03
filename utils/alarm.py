"""
utils/alarm.py
----------------
Thin wrapper around pygame's mixer used to play a looping alarm sound when
the driver is classified as DROWSY. All failures (missing audio device,
missing sound file, etc.) are caught so the main application never crashes
just because audio is unavailable on the host machine.
"""

import os

import config

try:
    import pygame
except Exception:  # pragma: no cover - pygame should always be installed
    pygame = None


class AlarmManager:
    """Starts/stops a looping alarm sound and reports its current state."""

    def __init__(self, sound_path=None):
        self.sound_path = sound_path or config.ALARM_SOUND_PATH
        self.enabled = False
        self.sound = None
        self._playing = False
        self._init_mixer()

    def _init_mixer(self):
        if pygame is None:
            print("[AlarmManager] pygame is not available; alarm disabled.")
            return
        try:
            pygame.mixer.init()
            if os.path.exists(self.sound_path):
                self.sound = pygame.mixer.Sound(self.sound_path)
                self.enabled = True
            else:
                print(f"[AlarmManager] Sound file not found: {self.sound_path}")
        except Exception as exc:
            # No sound card / audio backend available (e.g. headless server).
            print(f"[AlarmManager] Audio device unavailable: {exc}")

    def start(self):
        """Begin looping the alarm sound (no-op if already playing)."""
        if not self.enabled or self.sound is None or self._playing:
            return
        try:
            self.sound.play(loops=-1)
            self._playing = True
        except Exception as exc:
            print(f"[AlarmManager] Could not start alarm: {exc}")

    def stop(self):
        """Stop the alarm sound if it is currently playing."""
        if not self._playing:
            return
        try:
            self.sound.stop()
        except Exception as exc:
            print(f"[AlarmManager] Could not stop alarm: {exc}")
        finally:
            self._playing = False

    def is_playing(self):
        return self._playing

    def shutdown(self):
        """Release the mixer resources on application exit."""
        self.stop()
        if pygame is not None:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
