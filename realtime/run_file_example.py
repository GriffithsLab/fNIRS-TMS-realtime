"""Launch a SNIRF replay, the graphical viewer, and the dry-run trigger controller.

Nothing in this launcher connects to or fires TMS hardware. Press Ctrl+C in
the controller console to stop the run and close the streamer and viewer.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent


def launch(args: list[str]) -> subprocess.Popen:
    options = {"creationflags": subprocess.CREATE_NEW_CONSOLE} if sys.platform == "win32" else {}
    return subprocess.Popen([sys.executable, *args], cwd=HERE, **options)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", help="SNIRF file (default: streamer's example recording)")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (default: 1; use 1 for valid timing)")
    parser.add_argument("--chunk-ms", type=float, default=40.0, help="streamer chunk size (default: 40 ms)")
    parser.add_argument("--no-loop", action="store_true", help="stop instead of repeating the recording")
    parser.add_argument("--amp-name", default="SNIRF_Amplitude")
    parser.add_argument("--marker-name", default="SNIRF_Markers")
    parser.add_argument("--on-sec", type=float, default=2.0)
    parser.add_argument("--off-sec", type=float, default=8.0)
    parser.add_argument("--grace-sec", type=float, default=3.0)
    parser.add_argument("--legacy-reactive", action="store_true")
    parser.add_argument("--lead-sec", type=float, default=5.0)
    parser.add_argument("--max-shift-sec", type=float, default=3.0)
    parser.add_argument("--center-period-sec", type=float, default=10.0)
    parser.add_argument("--period-lo-sec", type=float, default=7.0)
    parser.add_argument("--period-hi-sec", type=float, default=17.0)
    parser.add_argument("--buffer-sec", type=float, default=40.0)
    parser.add_argument("--pad-sec", type=float, default=12.0)
    parser.add_argument("--ar-order", type=int, default=20)
    parser.add_argument("--freq-window-sec", type=float, default=5.0)
    parser.add_argument("--recompute-hz", type=float, default=5.0)
    parser.add_argument("--target-phase", choices=("either", "peak", "trough"), default="either")
    parser.add_argument("--require-prediction", action="store_true")
    parser.add_argument("--n-bursts", type=int, default=1)
    parser.add_argument("--burst-gap-sec", type=float, default=0.0)
    args = parser.parse_args()

    if args.speed <= 0:
        parser.error("--speed must be greater than zero")
    if args.require_prediction and args.target_phase == "either":
        parser.error("--require-prediction needs --target-phase peak or trough")

    streamer_args = ["stream_snirf_lsl.py"]
    if args.file:
        streamer_args.append(args.file)
    streamer_args += ["--speed", str(args.speed), "--chunk-ms", str(args.chunk_ms)]
    if not args.no_loop:
        streamer_args.append("--loop")

    print("Starting SNIRF file streamer and viewer...")
    streamer = launch(streamer_args)
    viewer = None
    try:
        time.sleep(2)
        if streamer.poll() is not None:
            print("Streamer exited immediately; check its window for the error.")
            return 1
        predictive_args = [
            "--lead-sec", str(args.lead_sec),
            "--max-shift-sec", str(args.max_shift_sec),
            "--center-period-sec", str(args.center_period_sec),
            "--period-lo-sec", str(args.period_lo_sec),
            "--period-hi-sec", str(args.period_hi_sec),
            "--buffer-sec", str(args.buffer_sec),
            "--pad-sec", str(args.pad_sec),
            "--ar-order", str(args.ar_order),
            "--freq-window-sec", str(args.freq_window_sec),
            "--recompute-hz", str(args.recompute_hz),
            "--target-phase", args.target_phase,
            "--n-bursts", str(args.n_bursts),
            "--burst-gap-sec", str(args.burst_gap_sec),
        ]
        mode_args = ["--legacy-reactive"] if args.legacy_reactive else []
        prediction_args = ["--require-prediction"] if args.require_prediction else []
        viewer = launch([
            "viewer.py", "--amp-name", args.amp_name, "--marker-name", args.marker_name,
            *predictive_args, *mode_args, *prediction_args,
        ])
        print("Starting trigger controller in DRY RUN mode. Press Ctrl+C to stop.\n")
        trigger_args = [
            "itbs_trigger.py", "--amp-name", args.amp_name,
            "--on-sec", str(args.on_sec), "--off-sec", str(args.off_sec),
            "--grace-sec", str(args.grace_sec),
            *predictive_args, *mode_args, *prediction_args,
        ]
        return subprocess.run([sys.executable, *trigger_args], cwd=HERE).returncode
    except KeyboardInterrupt:
        return 0
    finally:
        print("\nStopping streamer and viewer...")
        streamer.terminate()
        if viewer is not None:
            viewer.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
