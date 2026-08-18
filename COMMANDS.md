# Command quick reference

Run these from the repository root after activating the `nirx-realtime` Conda environment:

```powershell
conda activate nirx-realtime
Set-Location realtime
```

## See available options

```powershell
python stream_snirf_lsl.py --help
python viewer.py --help
python itbs_trigger.py --help
python run_live_itbs_test.py --help
```

You will see each program's flags and defaults. Check names, timing, phase-target, and hardware options before changing them.

## Check the phase predictor (no hardware)

```powershell
python test_phase_predictor.py
```

You will see prediction-error summaries for several noise levels, followed by invariant checks. Look for `OK` results and the final `All invariant checks passed`; also judge whether the reported median and 90th-percentile errors are acceptable for the experiment.

## Replay and monitor a recording (no hardware)

To launch the streamer, viewer, and dry-run trigger controller together at timing-correct 1x playback:

```powershell
python run_file_example.py
```

Pass a different recording if needed: `python run_file_example.py "C:\path\to\recording.snirf"`. Controller parameters can be supplied on the same command, for example: `python run_file_example.py --target-phase trough --require-prediction --lead-sec 5 --max-shift-sec 3`. Run `python run_file_example.py --help` for all options. Press `Ctrl+C` in the controller console to stop and close everything.

Use `--speed 1` when evaluating trigger timing. Faster playback compresses the recorded physiology into wall-clock time while the predictor's 7-17 second passband and trigger schedule remain expressed in seconds, so predictions and trigger times are not equivalent to a real-time run. Higher speeds are useful only for checking data flow and the GUI.

Alternatively, start the components separately in different terminals. Activate the environment and `Set-Location realtime` in each:

```powershell
# Terminal 1: replay the default SNIRF recording quickly and repeatedly
python stream_snirf_lsl.py --speed 20 --loop

# Terminal 2: graphical display (optional)
python viewer.py

# Terminal 3: dry-run trigger controller
python itbs_trigger.py
```

The streamer prints the stream names, channel/sample counts, playback progress, and markers. The viewer opens live probe, trace, and activity panels. The controller prints `[DRY RUN]`, connects to `SNIRF_Amplitude`, shows prediction/status updates, and prints `FIRE` when it *would* trigger. Look for a stable connection, moving traces, sensible peak/trough timing, and no unexpected pauses or repeated errors. Stop each process with `Ctrl+C`.

To replay another file:

```powershell
python stream_snirf_lsl.py "C:\path\to\recording.snirf" --speed 20 --loop
```

## Live stimulator test (fires real hardware)

```powershell
python run_live_itbs_test.py --port COM4
```

This opens the streamer and viewer, then runs the trigger controller against the MagVenture. It first displays a safety checklist and does nothing unless you type `yes`. Before doing so, verify the COM port, on-device protocol/amplitude, armed state, coil position, and subject/operator safety. During the run, look for `[LIVE] connected`, regular status updates, and `FIRE` lines at plausible times. Press `Ctrl+C` in the controller window to disconnect and close the child windows.

Use `--no-viewer` to omit the GUI. Never use accelerated playback for a live run.
