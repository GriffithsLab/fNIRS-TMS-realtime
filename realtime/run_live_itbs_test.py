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
    python run_live_itbs_test.py --target-phase trough
    python run_live_itbs_test.py --port COM5 --target-phase peak
    python run_live_itbs_test.py --no-viewer --target-phase trough   # skip the GUI window

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
    p.add_argument("--channels", nargs="+", default=["grandaverage"], metavar="LABEL",
                    help="which source-detector pair(s) feed activity_val. One label (e.g. S1_D2) -> that "
                         "channel only; several labels -> their average; 'grandaverage' (default) -> average "
                         "across every available pair.")
    p.add_argument("--on-sec", type=float, default=2.0, help="nominal ON burst duration in seconds (default: 2.0)")
    p.add_argument("--off-sec", type=float, default=8.0,
                    help="[legacy-reactive] minimum OFF/refractory period after each burst before the next "
                         "trigger is considered. [adaptive] the grid's phase-alignment offset -- NOTE this does "
                         "not fix the negative-bound shift saturation seen across real sessions, see "
                         "itbs_trigger.DEFAULT_OFF_SEC's comment (default: 8.0, standard 2s-ON/8s-OFF cadence)")
    p.add_argument("--grace-sec", type=float, default=3.0,
                    help="[legacy-reactive only] extra seconds to wait for a real peak/trough before firing on a "
                         "schedule fallback (default: 3.0)")
    p.add_argument("--legacy-reactive", action="store_true",
                    help="use the old reactive detector instead of the adaptive phase-predictive one (no "
                         "filtering, not robust to cardiac/respiratory noise -- kept only for A/B comparison)")
    p.add_argument("--lead-sec", type=float, default=5.0,
                    help="[adaptive only] fire this many seconds before the predicted hemodynamic peak/trough "
                         "(default: 5.0 -- a literature-derived placeholder, validate empirically first)")
    p.add_argument("--max-shift-sec", type=float, default=3.0,
                    help="[adaptive only] max seconds a burst's onset may be shifted from its fixed 10s-grid slot "
                         "(default: 3.0)")
    p.add_argument("--target-phase", choices=("peak", "trough"), default=None,
                    help="[adaptive only] which extremum to commit to targeting for the whole run. Required "
                         "unless --legacy-reactive is used -- 'either' is not supported (its true scheduling "
                         "offset is too large to recalibrate away).")
    p.add_argument("--require-prediction", action="store_true",
                    help="[adaptive only] no fallback firing -- stay silent until a real prediction is available.")
    p.add_argument("--n-bursts", type=int, default=1,
                    help="[adaptive only] number of on_sec bursts to fire back-to-back at each triggering event "
                         "(default: 1; e.g. 2 to match standard iTBS dosage when targeting only one extremum type)")
    p.add_argument("--burst-gap-sec", type=float, default=0.0,
                    help="[adaptive only] gap between consecutive bursts within a multi-burst group (default: 0.0s)")
    p.add_argument("--no-viewer", action="store_true", help="don't launch the graphical viewer window")
    args = p.parse_args()

    if not args.legacy_reactive and args.target_phase is None:
        print("--target-phase peak or trough is required (not the removed 'either') unless --legacy-reactive is used. Aborting.")
        return 1

    mode = "legacy-reactive" if args.legacy_reactive else f"adaptive (lead {args.lead_sec:g}s, max shift {args.max_shift_sec:g}s)"
    print()
    print("=== Live fNIRS-triggered MagVenture test ===")
    print(f"Port:   {args.port}")
    print(f"Timing: {args.on_sec:g}s ON / {args.off_sec:g}s OFF minimum / {args.grace_sec:g}s grace")
    print(f"Mode:   {mode}")
    print("        --no-arm (connect + fire only; device must already be armed/configured)")
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
    trigger_args = [
        sys.executable, "itbs_trigger.py",
        "--live", "--port", args.port, "--no-arm",
        "--on-sec", str(args.on_sec), "--off-sec", str(args.off_sec), "--grace-sec", str(args.grace_sec),
        "--lead-sec", str(args.lead_sec), "--max-shift-sec", str(args.max_shift_sec),
        "--n-bursts", str(args.n_bursts), "--burst-gap-sec", str(args.burst_gap_sec),
        "--channels", *args.channels,
    ]
    if args.target_phase is not None:  # None only possible with --legacy-reactive, which ignores it anyway
        trigger_args += ["--target-phase", args.target_phase]
    if args.legacy_reactive:
        trigger_args.append("--legacy-reactive")
    if args.require_prediction:
        trigger_args.append("--require-prediction")
    try:
        subprocess.run(trigger_args, cwd=HERE)
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
