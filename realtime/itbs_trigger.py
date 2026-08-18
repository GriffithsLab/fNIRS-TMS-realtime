"""Real-time peak/trough-triggered iTBS controller for a MagVenture stimulator.

Connects to the 'SNIRF_Amplitude' LSL stream (from stream_snirf_lsl.py, or a
live NIRx acquisition), runs the same causal peak/trough detector used by
viewer.py (activity_core.ActivityAnalyzer), and fires the stimulator's
preprogrammed theta-burst protocol on every confirmed peak or trough -- the
live version of the "Peak/Trough-Based Triggers" strategy from
Occipital_fNIRS_iTBS_Analysis_Executed.ipynb, applied to real hardware via
magicpy (https://magicpy.readthedocs.io/en/latest/magicpy.html).

A single fire() sends one external trigger; it assumes the stimulator's own
Protocol/Treatment page is already configured with the desired 2-second
theta-burst train, exactly as iTBS stimulators are normally externally
triggered -- this script does not attempt to time individual 50 Hz/5 Hz
sub-pulses itself.

Timing
------
Nominal cadence: a 2-second ON burst (--on-sec) followed by an 8-second OFF/
refractory period (--off-sec) before the next trigger is even considered --
"every eight seconds after the 2 second ON period". The exact trigger moment
is not a rigid clock tick, though: once the OFF period has elapsed, the
controller waits for the peak/trough detector to confirm the next real peak
or trough and fires on that -- so timing tracks measured activity rather
than a metronome. Peak/trough confirmation is itself causal and lags real
time by MIN_PEAK_DISTANCE_SEC (5s, see activity_core.ActivityAnalyzer), so
the grace fallback is measured from OFF-period-end + that 5s floor, not from
OFF-period-end directly -- otherwise it would always win the race against a
real peak/trough that hasn't had a chance to be confirmed yet. If nothing is
confirmed within --grace-sec beyond that floor, it fires anyway so the
cadence never stalls waiting on a peak that doesn't come.

Safety
------
This drives real TMS hardware, which stimulates the human brain. By default
the script runs in --dry-run mode (the default -- no flag needed) and only
prints what it *would* trigger; nothing is armed, connected, or fired. Pass
--live together with --port and --amplitude to actually connect to a
MagVenture stimulator and fire it. Read magicpy's documentation and your
stimulator's manual first. This script does not verify coil placement,
subject safety, or on-device protocol configuration -- that is the
operator's responsibility, every time.

Usage
-----
    python itbs_trigger.py                                    # dry run, no hardware touched
    python itbs_trigger.py --live --port COM3 --amplitude 50   # fires real TMS pulses
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from mne_lsl.lsl import StreamInlet, resolve_streams

from activity_core import (
    BASELINE_SEC,
    EXTINCTION_CSV,
    MIN_PEAK_DISTANCE_SEC,
    SHORT_CHANNEL_MAX_DISTANCE_M,
    STANDARD_BURST_DURATION_SEC,
    ActivityAnalyzer,
    ChannelPair,
    ExtinctionTable,
    ShortChannelArtifactRemover,
    parse_channels,
)

DEFAULT_ON_SEC = STANDARD_BURST_DURATION_SEC  # 2.0
DEFAULT_OFF_SEC = 8.0
DEFAULT_GRACE_SEC = 3.0


class TriggerController:
    """Fires on the next confirmed peak/trough once an ON+OFF window has elapsed."""

    def __init__(self, on_sec: float, off_sec: float, grace_sec: float, on_fire):
        self.on_sec = on_sec
        self.off_sec = off_sec
        self.grace_sec = grace_sec
        self.on_fire = on_fire  # callback(t, event_or_None)
        self.next_earliest_t: float | None = None

    def start(self, t0: float):
        self.next_earliest_t = t0 + self.off_sec

    def handle_event(self, ev: dict):
        if self.next_earliest_t is not None and ev["t"] >= self.next_earliest_t:
            self._fire(ev["t"], ev)

    def check_timeout(self, t_now: float):
        # A peak/trough true to time `x` is only ever reported once the clock reaches
        # x + MIN_PEAK_DISTANCE_SEC (see activity_core.ActivityAnalyzer) -- that's the
        # causal detection lag. So a peak occurring right at next_earliest_t cannot be
        # confirmed before next_earliest_t + MIN_PEAK_DISTANCE_SEC; the grace window has
        # to be measured from there, or every timeout fires before a real peak/trough
        # from that window ever gets a chance to be seen.
        if self.next_earliest_t is None:
            return
        deadline = self.next_earliest_t + MIN_PEAK_DISTANCE_SEC + self.grace_sec
        if t_now >= deadline:
            self._fire(t_now, None)

    def _fire(self, t: float, ev: dict | None):
        self.next_earliest_t = t + self.on_sec + self.off_sec
        self.on_fire(t, ev)


class Stimulator:
    """Wraps magicpy.MagVenture; a no-op dry run unless --live is given.

    With no_arm=True, only connect()/fire()/disconnect() are ever called --
    no arm(), no set_amplitude(), no disarm(). Use this when the stimulator
    is already armed and configured (waveform, burst pattern, amplitude) on
    its own front panel and you only want external trigger pulses sent to
    it, without this script touching or later undoing that configuration.
    """

    def __init__(self, live: bool, port: str | None, ttl_port, amplitude: int | None, no_arm: bool):
        self.live = live
        self.no_arm = no_arm
        self.dev = None
        if not live:
            print("[DRY RUN] no hardware connection -- pass --live --port ... to fire real pulses")
            return

        import magicpy as mp  # imported lazily so dry-run mode never needs pyserial/magicpy to work

        self.dev = mp.MagVenture(port, ttl_port=ttl_port)
        self.dev.connect()
        if no_arm:
            print(f"[LIVE] connected on {port} -- skipping arm()/set_amplitude() "
                  f"(device assumed already armed and configured)")
        else:
            self.dev.arm()
            if amplitude is not None:
                self.dev.set_amplitude(amplitude)
            print(f"[LIVE] connected + armed on {port}" + (f", amplitude {amplitude}%" if amplitude is not None else ""))

    def fire(self):
        if self.live:
            self.dev.fire(get_response=False)

    def close(self):
        if self.live and self.dev is not None:
            try:
                if not self.no_arm:
                    self.dev.disarm()
            finally:
                self.dev.disconnect()


def run(amp_name, on_sec, off_sec, grace_sec, live, port, ttl_port, amplitude, no_arm,
        short_max_distance_m=SHORT_CHANNEL_MAX_DISTANCE_M, artifact_removal=True):
    if live and port is None:
        raise SystemExit("--live requires --port")
    if live and not no_arm and amplitude is None:
        raise SystemExit("--live requires --amplitude unless --no-arm is given (choose the output %% deliberately -- it is never guessed for you)")

    print(f"Resolving '{amp_name}' LSL stream...")
    found = []
    attempts = 0
    while not found:
        found = resolve_streams(timeout=1.0, name=amp_name)
        attempts += 1
        if not found and attempts % 5 == 0:
            print(f"  still waiting for '{amp_name}'...")
    inlet = StreamInlet(found[0])
    inlet.open_stream()
    sinfo = inlet.get_sinfo()
    sfreq = sinfo.sfreq
    metas = parse_channels(sinfo)
    print(f"Connected: {sinfo.n_channels} channels @ {sfreq:.2f} Hz")

    ext = ExtinctionTable(EXTINCTION_CSV)
    by_pair: dict[tuple, dict[float, int]] = {}
    pos_by_pair = {}
    for m in metas:
        key = (m.source, m.detector)
        by_pair.setdefault(key, {})[m.wavelength] = m.index
        pos_by_pair[key] = (m.src_pos, m.det_pos, m.distance_m)
    pairs = []
    for (src, det), idx_by_wl in by_pair.items():
        src_pos, det_pos, dist = pos_by_pair[(src, det)]
        pair = ChannelPair(src, det, idx_by_wl, src_pos, det_pos, dist)
        pair.set_extinction(ext)
        pairs.append(pair)

    remover = None
    if artifact_removal:
        candidate = ShortChannelArtifactRemover(pairs, short_max_distance_m)
        if candidate.active:
            remover = candidate
            print(f"[artifact removal] {len(remover.short_pairs)} short / "
                  f"{len(remover.long_pairs)} long channels (threshold {short_max_distance_m*1000:.1f}mm)")
        else:
            print(f"[artifact removal] no short channels found under {short_max_distance_m*1000:.1f}mm -- disabled")
    activity_pairs = remover.long_pairs if remover is not None else pairs

    analyzer = ActivityAnalyzer(sfreq)

    def clear_line():
        print("\r" + " " * 110 + "\r", end="", flush=True)

    def on_fire(t, ev):
        clear_line()
        if ev is None:
            print(f"[{time.strftime('%H:%M:%S')}] FIRE @ t={t:7.1f}s  (grace timeout -- no peak/trough in window)")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] FIRE @ t={t:7.1f}s  "
                  f"({ev['kind']}, {'valid' if ev['valid'] else 'invalid'}, v={ev['v']:+.2f}uM)")
        stim.fire()

    controller = TriggerController(on_sec, off_sec, grace_sec, on_fire)
    print(f"Schedule: {on_sec:g}s ON, {off_sec:g}s OFF minimum, exact moment follows the next detected "
          f"peak/trough (grace fallback after {grace_sec:g}s extra)")

    stim = None
    baseline_ready = False
    baseline_deadline = BASELINE_SEC
    n_samples_seen = 0
    n_peaks = 0
    n_troughs = 0
    t = 0.0
    activity_val = 0.0

    try:
        stim = Stimulator(live, port, ttl_port, amplitude, no_arm)
        while True:
            samples, _timestamps = inlet.pull_chunk(timeout=0.5, max_samples=2048)
            for sample in samples:
                n_samples_seen += 1
                t = n_samples_seen / sfreq

                if not baseline_ready:
                    for pair in pairs:
                        pair.add_baseline_sample(sample)
                    if t >= baseline_deadline:
                        for pair in pairs:
                            pair.finalize_baseline()
                        baseline_ready = True
                        controller.start(t)
                        clear_line()
                        print(f"Baseline complete at t={t:.1f}s -- trigger loop armed "
                              f"(first trigger opportunity at t={controller.next_earliest_t:.1f}s)")
                    continue

                for pair in pairs:
                    pair.update(sample)
                if remover is not None:
                    remover.process()  # in-place artifact removal on long-channel last_hbo/last_hbr
                activity_val = -float(np.mean([p.last_hbo for p in activity_pairs]))

                for ev in analyzer.push(t, activity_val):
                    if ev["kind"] == "peak":
                        n_peaks += 1
                    else:
                        n_troughs += 1
                    controller.handle_event(ev)
                controller.check_timeout(t)

            # Heartbeat: refreshed roughly every pull_chunk cycle (~2 Hz at real-time
            # speed, since timeout=0.5s dominates when data arrives slower than that).
            # Uses \r so it never scrolls the terminal; FIRE/baseline lines clear it
            # first via clear_line() so they always land on their own clean line.
            if not baseline_ready:
                remaining = max(0.0, baseline_deadline - t)
                line = f"  collecting baseline...  t={t:5.1f}s  ({remaining:4.1f}s remaining)"
            else:
                eta = max(0.0, controller.next_earliest_t - t) if controller.next_earliest_t is not None else 0.0
                line = (f"  t={t:7.1f}s  activity={activity_val:+6.2f}uM  "
                        f"next trigger in ~{eta:4.1f}s  (peaks:{n_peaks} troughs:{n_troughs})")
            print("\r" + line + " " * 6, end="", flush=True)
    except KeyboardInterrupt:
        clear_line()
        print("Stopping.")
    finally:
        if stim is not None:
            stim.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--amp-name", default="SNIRF_Amplitude")
    p.add_argument("--on-sec", type=float, default=DEFAULT_ON_SEC, help="ON burst duration (default: 2.0s)")
    p.add_argument("--off-sec", type=float, default=DEFAULT_OFF_SEC,
                    help="minimum OFF/refractory period after each burst before the next trigger is considered (default: 8.0s)")
    p.add_argument("--grace-sec", type=float, default=DEFAULT_GRACE_SEC,
                    help="fire anyway if no peak/trough is confirmed within this many extra seconds beyond the "
                         "5s detection lag after the OFF period ends (default: 3.0s)")
    p.add_argument("--live", action="store_true",
                    help="actually connect to and fire a MagVenture stimulator via magicpy (default: dry run, prints only)")
    p.add_argument("--port", default=None, help="serial port for the stimulator, e.g. COM3 (required with --live)")
    p.add_argument("--ttl-port", default=None,
                    help="optional parallel port for low-latency TTL firing, passed through to magicpy (default: serial fire, higher jitter)")
    p.add_argument("--amplitude", type=int, default=None,
                    help="stimulator output in %% (0-100); required with --live unless --no-arm is given, "
                         "choose deliberately, never guessed")
    p.add_argument("--no-arm", action="store_true",
                    help="skip arm()/set_amplitude()/disarm() entirely -- only connect() and fire(). "
                         "Use when the stimulator is already armed and configured on its own front panel.")
    p.add_argument("--short-channel-mm", type=float, default=SHORT_CHANNEL_MAX_DISTANCE_M * 1000,
                    help=f"source-detector distance (mm) at or below which a channel is treated as a "
                         f"short-separation channel used to regress out superficial/systemic artifact from the "
                         f"long channels feeding the trigger detector (default: {SHORT_CHANNEL_MAX_DISTANCE_M*1000:g})")
    p.add_argument("--no-artifact-removal", action="store_true",
                    help="disable real-time short-channel/systemic-artifact removal (RLS regression)")
    args = p.parse_args()

    run(args.amp_name, args.on_sec, args.off_sec, args.grace_sec, args.live, args.port, args.ttl_port,
        args.amplitude, args.no_arm, short_max_distance_m=args.short_channel_mm / 1000,
        artifact_removal=not args.no_artifact_removal)


if __name__ == "__main__":
    main()
