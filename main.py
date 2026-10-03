"""
main.py
--------
Entry point for the Drowsiness Detection application (desktop / Tkinter).

Run with:
    python main.py

Requirements for this mode: a real desktop session (a display server) **and**
a webcam attached to the same machine. If you are on a headless server, inside
a container, a remote sandbox or connected over SSH, use the browser version
instead - it takes the camera from the machine running the browser:

    python web_app.py        # then open http://localhost:5000
"""

import os
import sys

WEB_HINT = (
    "\nTip: this machine cannot show a desktop window.\n"
    "Run the browser dashboard instead:\n\n"
    "    python web_app.py\n\n"
    "then open http://localhost:5000 in your browser and press "
    "'Start Monitoring'.\n"
)


def _check_display():
    """Return an error message if no GUI display is available, else None."""
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return None
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return None
    return (
        "No graphical display was found (DISPLAY / WAYLAND_DISPLAY are unset), "
        "so the Tkinter window cannot be opened."
    )


def main():
    display_error = _check_display()
    if display_error:
        print(f"ERROR: {display_error}{WEB_HINT}")
        return 1

    try:
        import tkinter as tk
    except ImportError:
        print(
            "ERROR: Python was built without Tkinter support.\n"
            "Install it (Debian/Ubuntu: sudo apt-get install python3-tk) or "
            f"use the browser version.{WEB_HINT}"
        )
        return 1

    from ui.dashboard import DrowsinessDashboard

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"ERROR: could not open a window ({exc}).{WEB_HINT}")
        return 1

    app = DrowsinessDashboard(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_exit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
