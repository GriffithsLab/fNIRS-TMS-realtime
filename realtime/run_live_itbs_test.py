"""Repeat the live fNIRS peak/trough-triggered MagVenture test, standalone.

Launches three processes:
  1. stream_snirf_lsl.py -- replays the recorded SNIRF file over LSL in its
     own console window, at real time (--speed 1, looping).
  2. viewer.py            -- the graphical real-time viewer, in its own
     window, so you can watch the activity trace / peak-trough flashes /
     probe map live while the stimulator fires.
  3. itbs_trigger.py      -- runs in THIS window/console, in the foreground,
     so its regular status line and every FIRE event print directly where
     you're watching. Connects live, in --no-arm mode (connect + fire only
     -- never arms the device or changes its amplitude; the stimulator must
     already be armed and configured with the intended protocol on its own
     front panel).

Speed is intentionally fixed at 1x (true real time) for the streamer: the
OFF period between triggers is measured in stream time, so any faster
playback speed would compress real trigger pulses closer together than
intended -- this script does not expose a --speed option for that reason.

Before touching any hardware, this script requires you to type "yes" to an
interactive safety confirmation. There is no way to skip it via a flag.

Prerequisites
-------------
- Run from inside the 'nirx-realtime' conda environment (the one with mne,
  mne_lsl, pyqtgraph, magicpy installed): conda activate nirx-realtime
- The MagVenture must be:
    * powered on, USB serial adapter plugged in
    * visible in Device Manager as the COM port you pass with --port
    * already ARMED and CONFIGURED (waveform, burst pattern, amplitude) on
      its own front panel -- this script does not touch any of that
- Coil positioned somewhere safe to fire: away from anyone, or on a subject
  only as a deliberate, supervised test.

Usage
-----
    python run_live_itbs_test.py
    python run_live_itbs_test.py --port COM5 --on-sec 2 --off-sec 8
    python run_live_itbs_test.py --no-viewer          # skip the GUI window

To test the software without touching hardware at all, skip this script
entirely and just run the dry run directly (no flags needed, nothing is
ever connected or fired):
    python stream_snirf_lsl.py --speed 20 --loop
    python itbs_trigger.py

To stop a live run: Ctrl+C in this window. That lets itbs_trigger.py clean
up first (disconnects the serial port; never disarms, since --no-arm never
armed it), then this script closes the streamer and viewer windows it
opened.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def launch_in_new_window(args: list[str]) -> subprocess.Popen:
    """Launch a script in its own console window (Windows-only)."""
    return subprocess.Popen(
        [sys.executable, *args],
        cwd=HERE,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="COM4", help="serial port the MagVenture is connected on (default: COM4)")
    p.add_argument("--on-sec", type=float, default=2.0, help="nominal ON burst duration in seconds (default: 2.0)")
    p.add_argument("--off-sec", type=float, default=8.0,
                    help="minimum OFF/refractory period after each burst before the next trigger is considered (default: 8.0)")
    p.add_argument("--grace-sec", type=float, default=3.0,
                    help="extra seconds to wait for a real peak/trough before firing on a schedule fallback (default: 3.0)")
    p.add_argument("--no-viewer", action="store_true", help="don't launch the graphical viewer window")
    args = p.parse_args()

    print()
    print("=== Live fNIRS-triggered MagVenture test ===")
    print(f"Port:   {args.port}")
    print(f"Timing: {args.on_sec:g}s ON / {args.off_sec:g}s OFF minimum / {args.grace_sec:g}s grace")
    print("Mode:   --no-arm (connect + fire only; device must already be armed/configured)")
    print()
    print("This sends REAL trigger pulses to the stimulator. Before continuing,")
    print("confirm ALL of the following are true right now:")
    print(f"  1. The stimulator is powered on and {args.port} is the correct serial port.")
    print("  2. The stimulator is already armed and configured on its own front")
    print("     panel with the protocol, waveform, and amplitude you intend to fire.")
    print("  3. The coil is positioned somewhere safe to fire right now -- away")
    print("     from anyone, or on a subject only as a deliberate, supervised test.")
    print()
    confirm = input("Type 'yes' to confirm all of the above and proceed: ")
    if confirm.strip().lower() != "yes":
        print("Aborted -- nothing was connected or fired.")
        return 1

    print("\nStarting streamer in a new window (real-time speed, looping)...")
    streamer = launch_in_new_window(["stream_snirf_lsl.py", "--speed", "1", "--loop"])
    time.sleep(3)
    if streamer.poll() is not None:
        print("Streamer exited immediately -- check its window for the error.")
        return 1

    viewer = None
    if not args.no_viewer:
        print("Starting graphical viewer in a new window...")
        viewer = launch_in_new_window(["viewer.py"])

    print(f"Starting trigger controller (live, --no-arm) on {args.port} ...\n")
    try:
        subprocess.run(
            [
                sys.executable, "itbs_trigger.py",
                "--live", "--port", args.port, "--no-arm",
                "--on-sec", str(args.on_sec), "--off-sec", str(args.off_sec), "--grace-sec", str(args.grace_sec),
            ],
            cwd=HERE,
        )
    except KeyboardInterrupt:
        pass
    finally:
        which = "streamer and viewer windows" if viewer is not None else "streamer window"
        print(f"\nTrigger controller stopped. Closing {which}...")
        streamer.terminate()
        if viewer is not None:
            viewer.terminate()

    return 0


if __name__ == "__main__":
    sys.exit(main())
