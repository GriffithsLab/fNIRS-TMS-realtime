"""Replay a SNIRF recording over LSL as if it were live NIRx Aurora acquisition.

Reads a .snirf file with MNE, then pushes the raw wavelength-amplitude data
sample-by-sample (in small chunks) over an LSL stream at the recording's true
sample rate (or a scaled speed), and pushes the file's annotations/events on
a second, irregular-rate "Markers" LSL stream at the moment they occur during
playback. Two independent LSL applications (e.g. LabRecorder, or the viewer.py
in this folder) can then pick up the streams exactly as they would from real
NIRx hardware.

Usage
-----
    python stream_snirf_lsl.py path\\to\\file.snirf
    python stream_snirf_lsl.py path\\to\\file.snirf --speed 10   # 10x faster than real time
    python stream_snirf_lsl.py path\\to\\file.snirf --loop
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import mne
import numpy as np
from mne_lsl.lsl import StreamInfo, StreamOutlet, local_clock

CH_NAME_RE = re.compile(r"^(S\d+)_(D\d+)\s+(\d+)$")


def parse_channel_name(name: str) -> tuple[str, str, int]:
    """'S1_D2 760' -> ('S1', 'D2', 760)."""
    m = CH_NAME_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized SNIRF channel name format: {name!r}")
    src, det, wavelength = m.groups()
    return src, det, int(wavelength)


def build_amplitude_info(raw: mne.io.BaseRaw, speed: float, source_id: str, fname: str) -> StreamInfo:
    ch_names = raw.ch_names
    sfreq = raw.info["sfreq"]

    montage = raw.get_montage()
    ch_pos = montage.get_positions()["ch_pos"] if montage is not None else {}

    info = StreamInfo(
        name="SNIRF_Amplitude",
        stype="NIRS",
        n_channels=len(ch_names),
        sfreq=sfreq * speed,
        dtype="float32",
        source_id=source_id,
    )
    desc = info.desc
    desc.append_child_value("manufacturer", "NIRx (simulated replay)")
    desc.append_child_value("source_file", Path(fname).name)
    desc.append_child_value("orig_sfreq", f"{sfreq:.6f}")
    desc.append_child_value("playback_speed", f"{speed:g}")

    chs = desc.append_child("channels")
    for name in ch_names:
        src, det, wavelength = parse_channel_name(name)
        sp = np.asarray(ch_pos.get(src, [np.nan, np.nan, np.nan]), dtype=float)
        dp = np.asarray(ch_pos.get(det, [np.nan, np.nan, np.nan]), dtype=float)
        distance_m = float(np.linalg.norm(sp - dp))

        c = chs.append_child("channel")
        c.append_child_value("label", name)
        c.append_child_value("type", "fnirs_cw_amplitude")
        c.append_child_value("unit", "V")
        c.append_child_value("source", src)
        c.append_child_value("detector", det)
        c.append_child_value("wavelength_nm", str(wavelength))
        c.append_child_value("src_x", f"{sp[0]:.6f}")
        c.append_child_value("src_y", f"{sp[1]:.6f}")
        c.append_child_value("src_z", f"{sp[2]:.6f}")
        c.append_child_value("det_x", f"{dp[0]:.6f}")
        c.append_child_value("det_y", f"{dp[1]:.6f}")
        c.append_child_value("det_z", f"{dp[2]:.6f}")
        c.append_child_value("distance_m", f"{distance_m:.6f}")

    return info


def build_marker_info(source_id: str) -> StreamInfo:
    info = StreamInfo(
        name="SNIRF_Markers",
        stype="Markers",
        n_channels=1,
        sfreq=0.0,
        dtype="string",
        source_id=source_id,
    )
    info.set_channel_names(["event"])
    return info


def run(fname: str, speed: float, loop: bool, chunk_ms: float) -> None:
    print(f"Loading {fname} ...")
    raw = mne.io.read_raw_snirf(fname, preload=True, verbose="ERROR")
    data = raw.get_data().astype(np.float32).T  # (n_times, n_channels)
    sfreq = raw.info["sfreq"]
    n_times, n_channels = data.shape
    duration = n_times / sfreq

    events = list(zip(raw.annotations.onset, raw.annotations.description))
    events.sort(key=lambda e: e[0])

    source_id = f"snirf-replay-{Path(fname).stem}"
    amp_info = build_amplitude_info(raw, speed, source_id, fname)
    marker_info = build_marker_info(source_id)

    amp_outlet = StreamOutlet(amp_info, chunk_size=max(1, round(chunk_ms / 1000 * sfreq)))
    marker_outlet = StreamOutlet(marker_info)

    print(f"Streaming '{amp_info.name}': {n_channels} channels @ {sfreq:.4f} Hz "
          f"(declared LSL rate {sfreq * speed:.4f} Hz), duration {duration:.1f}s, "
          f"speed x{speed:g}{' [looping]' if loop else ''}")
    print(f"Streaming '{marker_info.name}': {len(events)} annotation events")
    print("Waiting for a viewer/consumer to connect is not required -- press Ctrl+C to stop.\n")

    chunk_samples = max(1, round(chunk_ms / 1000 * sfreq))

    try:
        while True:
            t_start_wall = time.perf_counter()
            t0_lsl = local_clock()
            next_event_idx = 0
            idx = 0
            while idx < n_times:
                end = min(idx + chunk_samples, n_times)
                chunk = data[idx:end]

                target_elapsed = (end - 1) / sfreq / speed
                now_elapsed = time.perf_counter() - t_start_wall
                sleep_s = target_elapsed - now_elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

                push_time = t0_lsl + (time.perf_counter() - t_start_wall)
                if len(chunk) == 1:
                    amp_outlet.push_sample(chunk[0], timestamp=push_time)
                else:
                    amp_outlet.push_chunk(chunk, timestamp=push_time)

                playback_time = end / sfreq
                while next_event_idx < len(events) and events[next_event_idx][0] <= playback_time:
                    onset, desc = events[next_event_idx]
                    marker_outlet.push_sample([desc], timestamp=local_clock())
                    print(f"  [{onset:8.2f}s] marker: {desc}")
                    next_event_idx += 1

                idx = end
                pct = 100 * idx / n_times
                print(f"\r  playing {idx/sfreq:7.1f}s / {duration:7.1f}s  ({pct:5.1f}%)", end="", flush=True)

            print()
            if not loop:
                break
            print("Reached end of recording, looping...\n")
    except KeyboardInterrupt:
        print("\nStopped.")


def main() -> None:
    default_file = Path(__file__).resolve().parents[2] / "data" / "collegestreet" / "2025-03-05_001.snirf"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", default=str(default_file), help="Path to .snirf file")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default: 1.0 = real time)")
    p.add_argument("--loop", action="store_true", help="Loop playback indefinitely")
    p.add_argument("--chunk-ms", type=float, default=40.0, help="Push chunk size in ms (default: 40)")
    args = p.parse_args()

    run(args.file, args.speed, args.loop, args.chunk_ms)


if __name__ == "__main__":
    main()
