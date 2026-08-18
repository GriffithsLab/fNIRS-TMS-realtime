# fNIRS-TMS-realtime

Real-time occipital fNIRS activity monitoring with peak/trough-triggered iTBS
stimulation, built as a live version of the offline analysis in
[`Occipital_fNIRS_iTBS_Analysis_Executed.ipynb`](Occipital_fNIRS_iTBS_Analysis_Executed.ipynb)
(peak/trough detection → categorization → activity-based iTBS triggers).

All code lives in [`realtime/`](realtime/).

## Components

| File | What it does |
|---|---|
| `stream_snirf_lsl.py` | Replays a recorded `.snirf` file over LSL as two streams (`SNIRF_Amplitude`, `SNIRF_Markers`), exactly as NIRx Aurora would produce live -- true sample rate, real montage geometry embedded in the stream metadata, TMS/iTBS annotation events fired at the right moment. Supports `--speed` for faster-than-real-time playback and `--loop`. |
| `activity_core.py` | Shared decoding/analysis library used by both the viewer and the trigger controller: SNIRF stream metadata parsing, modified Beer-Lambert HbO/HbR conversion (using the OMLC/Prahl extinction-coefficient table, the same source MNE-Python bundles), real-time short-channel/systemic-artifact removal (`ShortChannelArtifactRemover`, see below), and the causal peak/trough detector -- a real-time-safe version of the notebook's `find_peaks_numpy`. |
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

## Real-time artifact removal

Channels with a short source-detector distance (≤15mm by default, covering
typical 8mm short-separation probes) see scalp/systemic physiology --
heartbeat, respiration, Mayer waves, general scalp blood flow -- with
negligible cerebral signal, while long channels see that same superficial
signal added on top of the cerebral response. `activity_core.py` classifies
channels this way automatically from the montage geometry embedded in the
LSL stream and, on every sample, runs a causal multi-reference Recursive
Least Squares (RLS) adaptive filter (`RLSFilter`) per long channel per
chromophore, using its nearest short channels as the reference signal. This
removes the short-channel-correlated superficial/systemic component from
each long channel's HbO/HbR in place, before it reaches the activity trace,
peak/trough detector, or trigger logic -- so `viewer.py` and `itbs_trigger.py`
both act on cleaned, cerebral-only signal whenever short channels are
present in the stream. (There is no ready-made real-time RLS class in
mne/mne-realtime to reuse -- mne-realtime is archived and only ever offered
epoch-buffer tools, and mne-nirs's `short_channel_regression()` is an
offline batch GLM -- so this is a small self-contained online
implementation of the standard algorithm from the adaptive-filtering/fNIRS
literature.)

Both `viewer.py` and `itbs_trigger.py` accept:

- `--short-channel-mm` -- source-detector distance (mm) at or below which a
  channel counts as short-separation (default: 15).
- `--no-artifact-removal` -- disable this entirely and fall back to raw
  Beer-Lambert HbO/HbR.

If no channel in the stream is short enough, artifact removal is
automatically skipped (a message is printed) and behavior is unchanged.

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
