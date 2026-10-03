"""
main.py
--------
Entry point for the Drowsiness Detection application.

Run with:
    python main.py
"""

import tkinter as tk

from ui.dashboard import DrowsinessDashboard


def main():
    root = tk.Tk()
    app = DrowsinessDashboard(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_exit()


if __name__ == "__main__":
    main()
