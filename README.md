# TMS-fNIRS-realtime

Real-time fNIRS activity monitoring with peak/trough-triggered iTBS
stimulation. 

All code lives in [`realtime/`](realtime/).

## Components

| File | What it does |
|---|---|
| `stream_snirf_lsl.py` | Replays a recorded `.snirf` file over LSL as two streams (`SNIRF_Amplitude`, `SNIRF_Markers`), exactly as NIRx Aurora would produce live -- true sample rate, real montage geometry embedded in the stream metadata, TMS/iTBS annotation events fired at the right moment. Supports `--speed` for faster-than-real-time playback and `--loop`. |
| `activity_core.py` | Shared decoding/analysis library used by both the viewer and the trigger controller: SNIRF stream metadata parsing, modified Beer-Lambert HbO/HbR conversion (using the OMLC/Prahl extinction-coefficient table, the same source MNE-Python bundles), and the causal peak/trough detector -- a real-time-safe version of the notebook's `find_peaks_numpy`. |
| `extinction_coef_600_1000nm.csv` | Hemoglobin molar extinction coefficients (HbO2, Hb) for 600-1000nm, extracted from MNE-Python's bundled OMLC/Prahl table, used by `activity_core.py`. |
| `viewer.py` | PyQt5/pyqtgraph real-time GUI: live probe map colour-coded by HbO, per-channel HbO/HbR scrolling traces, and an activity panel replicating the notebook's three trigger-window strategies (standard fixed-rate, activity-threshold, peak/trough-centered) with a clear flash the instant a peak or trough is confirmed. |
| `itbs_trigger.py` | Headless controller that fires a MagVenture TMS stimulator (via [`magicpy`](https://magicpy.readthedocs.io/en/latest/magicpy.html)) on every confirmed peak/trough, on a nominal 2s-ON/8s-OFF cadence whose exact timing follows the real detected activity rather than a fixed clock. **Dry run by default** -- prints what it would fire without touching hardware. |
| `run_live_itbs_test.py` | Standalone launcher for a full live test: starts the streamer and the viewer in their own windows, then runs `itbs_trigger.py --live` in the foreground -- gated behind an interactive safety confirmation. |

## Quick start

Dry run, no hardware required:

```bash
cd realtime
python stream_snirf_lsl.py --speed 20 --loop   # terminal 1
python viewer.py                                # terminal 2 (optional GUI)
python itbs_trigger.py                          # terminal 3 (prints only)
```

Live test against a MagVenture stimulator (already armed and configured on
its own front panel):

```bash
cd realtime
python run_live_itbs_test.py --port COM4
```

## Safety

`itbs_trigger.py` sends real trigger pulses to TMS hardware, which stimulates
the human brain, when run with `--live`. Defaults are deliberately
conservative:

- Dry run unless `--live` is explicitly passed.
- `--live` requires `--port`; it also requires `--amplitude` unless
  `--no-arm` is given (amplitude is never guessed).
- `--no-arm` skips `arm()`/`set_amplitude()`/`disarm()` entirely -- use it
  when the stimulator is already armed and configured on its own panel, so
  the script only ever calls `connect()`, `fire()`, and `disconnect()`.
- `run_live_itbs_test.py` requires typing `yes` to an interactive safety
  checklist (correct port, device already configured, coil positioned
  safely) before connecting to anything -- there is no flag to skip it.

None of this verifies coil placement, subject safety, or on-device protocol
configuration -- that remains the operator's responsibility every time.
