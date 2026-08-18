"""Real-time phase-predictive adaptive iTBS controller for a MagVenture stimulator.

Connects to the 'SNIRF_Amplitude' LSL stream (from stream_snirf_lsl.py, or a
live NIRx acquisition) and drives a fixed 2-second-ON / 8-second-OFF / 10-
second-period intermittent theta-burst schedule, exactly like standard iTBS --
except each burst's onset is shifted forward or backward by a few seconds
(bounded by --max-shift-sec) to land on the peak or trough of the *actual*
ongoing hemodynamic oscillation, compensating for the ~5s hemodynamic lag
(--lead-sec) between a neural event and its delayed HbO signature. This is
done by activity_core.PhasePredictor: a narrowband-bandpass + AR-extrapolated
+ Hilbert-phase estimator that continuously predicts when the next peak/
trough will occur, structurally immune to cardiac (~0.8-2 Hz) and
respiratory (~0.15-0.4 Hz) contamination since those bands sit well outside
its passband. See AdaptiveTriggerController below for the scheduling math.

A single fire() sends one external trigger; it assumes the stimulator's own
Protocol/Treatment page is already configured with the desired 2-second
theta-burst train, exactly as iTBS stimulators are normally externally
triggered -- this script does not attempt to time individual 50 Hz/5 Hz
sub-pulses itself.

Legacy reactive mode
---------------------
Pass --legacy-reactive to fall back to the original, simpler behavior: wait
for the OFF period to elapse, then fire on the next confirmed peak/trough
from activity_core.ActivityAnalyzer's causal 2-point extremum test (with a
fixed MIN_PEAK_DISTANCE_SEC confirmation lag), or on a grace-period timeout
if nothing is confirmed in time. This path has no filtering and is not
robust to cardiac/respiratory noise -- kept only as a known-working fallback
for A/B comparison, not the recommended mode.

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

The --lead-sec default (5.0s) is a literature-derived placeholder (canonical
fNIRS/BOLD HbO hemodynamic-response peak latency) and MUST be validated
empirically for this specific paradigm before being trusted for real
stimulation -- see the verification plan in the project's implementation
notes.

Usage
-----
    python itbs_trigger.py                                    # dry run, adaptive mode, no hardware touched
    python itbs_trigger.py --legacy-reactive                   # dry run, old reactive behavior
    python itbs_trigger.py --live --port COM3 --amplitude 50   # fires real TMS pulses
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from mne_lsl.lsl import StreamInlet, resolve_streams

from activity_core import (
    BASELINE_SEC,
    DEFAULT_AR_ORDER,
    DEFAULT_BUFFER_SEC,
    DEFAULT_CENTER_PERIOD_SEC,
    DEFAULT_FREQ_WINDOW_SEC,
    DEFAULT_LEAD_SEC,
    DEFAULT_MAX_SHIFT_SEC,
    DEFAULT_PAD_SEC,
    DEFAULT_PERIOD_HI_SEC,
    DEFAULT_PERIOD_LO_SEC,
    DEFAULT_RECOMPUTE_HZ,
    EXTINCTION_CSV,
    MIN_PEAK_DISTANCE_SEC,
    STANDARD_BURST_DURATION_SEC,
    STANDARD_BURST_INTERVAL_SEC,
    GRAND_AVERAGE,
    ActivityAnalyzer,
    ChannelPair,
    ExtinctionTable,
    PhasePredictor,
    parse_channels,
    select_pairs,
)

DEFAULT_ON_SEC = STANDARD_BURST_DURATION_SEC  # 2.0
# In --legacy-reactive mode this is literally "OFF/refractory wait before the next trigger is
# considered". In adaptive mode: nominal_t(n) = t0 + off_sec + n*interval_sec, so off_sec sets the
# grid's initial phase alignment relative to session start.
#
# IMPORTANT: off_sec was investigated as a lever to fix the near-universal negative-bound shift
# saturation found across all 13 real sessions (reports/adaptive_itbs_*.ipynb), and it does NOT
# work -- a sweep of off_sec from -2s to +9s left the median (predicted_t - lead_sec - nominal_t)
# offset essentially unchanged (~-3s throughout, for both peak and trough targeting; see
# scratch diagnostics from that investigation). The likely cause is structural, not a calibration
# offset: the mean forward wait from "now" to the next trough (renewal-theory expectation, ~half
# the ~10s period) empirically measured at ~5.0-5.2s -- almost exactly DEFAULT_LEAD_SEC -- which
# makes target = predicted_t - lead_sec track "now" itself throughout the window. Since
# scheduled_t = nominal_t + shift(t_now) and shift(t_now) ~= t_now - nominal_t once target~now,
# the firing condition t_now >= scheduled_t becomes approximately self-referential (t_now >=
# t_now) and triggers almost immediately at window-open (shift = -max_shift_sec), independent of
# where in absolute time that window happens to sit -- i.e. independent of off_sec. Fixing this
# properly likely needs an architecture change (e.g. not continuously re-evaluating shift against
# a target that trivially tracks t_now), not a constant tweak -- left at the original standard
# iTBS value (8.0 = 2s ON + 8s OFF) pending that redesign.
DEFAULT_OFF_SEC = 8.0
DEFAULT_GRACE_SEC = 3.0  # legacy-reactive mode only


class TriggerController:
    """Legacy reactive controller: fires on the next confirmed peak/trough
    once an ON+OFF window has elapsed. Kept only for --legacy-reactive."""

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
        if self.next_earliest_t is None:
            return
        deadline = self.next_earliest_t + MIN_PEAK_DISTANCE_SEC + self.grace_sec
        if t_now >= deadline:
            self._fire(t_now, None)

    def _fire(self, t: float, ev: dict | None):
        self.next_earliest_t = t + self.on_sec + self.off_sec
        self.on_fire(t, ev)


class AdaptiveTriggerController:
    """Fixed-grid iTBS scheduler: each slot's nominal onset
    (t0 + off_sec + n*interval_sec, anchored to session start so shifts
    never accumulate/drift) is shifted toward PhasePredictor's lead-time-
    compensated extremum prediction, bounded to +/- max_shift_sec.

    require_prediction=False (default): falls back to firing unshifted on
    the nominal grid whenever no prediction is available yet (e.g. predictor
    still warming up).

    require_prediction=True: never falls back -- a slot with no prediction
    available is silently skipped (no fire that cycle).

    target_phase must always be "peak" or "trough" -- "either" (opportunistically
    targeting whichever of peak/trough is sooner) is not supported here: a
    read-only diagnostic across 13 real sessions found its true scheduling
    offset (~-5.4s median) is roughly double peak/trough's (~-3s), too large
    to fix with a single off_sec recalibration without leaving almost no
    OFF/refractory gap -- so the operator must commit to a single extremum
    type up front. PhasePredictor itself still supports "either" for
    general-purpose phase/period analysis (see the report notebooks); this
    restriction is specific to the trigger-driving role.

    n_bursts (>=1): how many on_sec bursts to deliver, back-to-back
    (separated by burst_gap_sec), once a slot fires. The grid always
    advances on its own fixed clock regardless of whether a given slot
    fired -- this is what makes require_prediction's "skip a slot" safe:
    a late-arriving prediction is always evaluated against the CURRENT
    slot, never a stale one left over from a silent skip.
    """

    def __init__(self, on_sec: float, off_sec: float, interval_sec: float,
                 lead_sec: float, max_shift_sec: float, on_fire, *,
                 target_phase: str, require_prediction: bool = False, n_bursts: int = 1,
                 burst_gap_sec: float = 0.0):
        if target_phase not in ("peak", "trough"):
            raise ValueError(
                f"target_phase must be 'peak' or 'trough', got {target_phase!r} -- 'either' is not "
                f"supported for triggering (its scheduling offset is too large to recalibrate away; "
                f"see the class docstring)"
            )
        self.on_sec = on_sec
        self.off_sec = off_sec
        self.interval_sec = interval_sec
        self.lead_sec = lead_sec
        self.max_shift_sec = max_shift_sec
        self.on_fire = on_fire  # callback(t, info: dict)
        self.require_prediction = require_prediction
        self.n_bursts = max(1, n_bursts)
        self.burst_gap_sec = burst_gap_sec
        self.target_phase = target_phase
        self.t0: float | None = None
        self.n = 0
        self.nominal_t: float | None = None
        self.scheduled_t: float | None = None
        self.fired_this_slot = False
        self._pending_bursts: list[float] = []  # remaining scheduled times for an in-progress burst group
        self._group_info: dict | None = None
        self._burst_counter = 0

    def start(self, t0: float):
        # NOTE: near-universal negative-bound shift saturation was found across all 13 real
        # sessions tested (see DEFAULT_OFF_SEC's comment) and does NOT respond to off_sec -- it
        # looks structural (target ~= t_now makes the firing condition self-referential and
        # trigger at window-open almost every time), not a per-session anchor-calibration problem.
        # A real fix likely needs to change how shift/firing is computed relative to a
        # continuously-updated target, not just adjust a scheduling constant -- deferred pending a
        # proper redesign, not attempted here.
        self.t0 = t0
        self.n = 0
        self.nominal_t = t0 + self.off_sec
        self.scheduled_t = self.nominal_t
        self.fired_this_slot = False
        self._pending_bursts = []
        self._group_info = None

    def step(self, t_now: float, prediction: dict | None):
        if self.nominal_t is None:
            return

        if self._pending_bursts:
            if t_now >= self._pending_bursts[0]:
                t_fire = self._pending_bursts.pop(0)
                self._deliver_burst(t_fire)
                if not self._pending_bursts:
                    self._group_info = None
            return  # don't evaluate new scheduling while a burst group is still in progress

        # Advance the grid on its own fixed clock, independent of whether the
        # slot actually fired -- so a slot skipped under require_prediction
        # never leaves nominal_t stale for a prediction that arrives later.
        while t_now >= self.nominal_t + self.max_shift_sec:
            self.n += 1
            self.nominal_t = self.t0 + self.off_sec + self.n * self.interval_sec
            self.fired_this_slot = False

        have_pred = prediction is not None
        if have_pred and prediction["predicted_kind"] != self.target_phase:
            raise RuntimeError(
                f"AdaptiveTriggerController.target_phase={self.target_phase!r} but received a prediction for "
                f"{prediction['predicted_kind']!r}. The PhasePredictor feeding this controller must be built "
                f"with the same target_phase=... — otherwise this could end up firing on both peak and trough, "
                f"which target_phase='peak'/'trough' is specifically meant to prevent."
            )
        shift = 0.0
        if have_pred:
            target = prediction["predicted_t"] - self.lead_sec
            shift = float(np.clip(target - self.nominal_t, -self.max_shift_sec, self.max_shift_sec))

        if self.require_prediction and not have_pred:
            self.scheduled_t = None
            return

        self.scheduled_t = self.nominal_t + shift
        if not self.fired_this_slot and t_now >= self.scheduled_t:
            self._begin_fire(t_now, shift, prediction)

    def _begin_fire(self, t: float, shift: float, prediction: dict | None):
        self.fired_this_slot = True
        self._group_info = {"nominal_t": self.nominal_t, "scheduled_t": self.scheduled_t, "shift": shift,
                             "prediction": prediction}
        self._burst_counter = 0
        self._pending_bursts = [t + k * (self.on_sec + self.burst_gap_sec) for k in range(self.n_bursts)]
        t_fire = self._pending_bursts.pop(0)
        self._deliver_burst(t_fire)

    def _deliver_burst(self, t: float):
        self._burst_counter += 1
        info = dict(self._group_info)
        info["burst_index"] = self._burst_counter
        info["n_bursts"] = self.n_bursts
        self.on_fire(t, info)


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


def run(args):
    if args.live and args.port is None:
        raise SystemExit("--live requires --port")
    if args.live and not args.no_arm and args.amplitude is None:
        raise SystemExit("--live requires --amplitude unless --no-arm is given (choose the output %% deliberately -- it is never guessed for you)")
    if not args.legacy_reactive and args.target_phase is None:
        raise SystemExit("--target-phase peak or trough is required (not the removed 'either') unless --legacy-reactive is used")

    print(f"Resolving '{args.amp_name}' LSL stream...")
    found = []
    attempts = 0
    while not found:
        found = resolve_streams(timeout=1.0, name=args.amp_name)
        attempts += 1
        if not found and attempts % 5 == 0:
            print(f"  still waiting for '{args.amp_name}'...")
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

    try:
        analysis_pairs = select_pairs(pairs, args.channels)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if len(analysis_pairs) == len(pairs):
        print(f"Analysis channels: grand average across all {len(pairs)} pairs")
    elif len(analysis_pairs) == 1:
        print(f"Analysis channel: {analysis_pairs[0].label} only")
    else:
        print(f"Analysis channels: average of {[p.label for p in analysis_pairs]}")

    analyzer = ActivityAnalyzer(sfreq)  # kept running for QC peak/trough counts in both modes

    predictor = None
    if not args.legacy_reactive:
        predictor = PhasePredictor(
            sfreq,
            center_period_sec=args.center_period_sec,
            period_lo_sec=args.period_lo_sec,
            period_hi_sec=args.period_hi_sec,
            buffer_sec=args.buffer_sec,
            pad_sec=args.pad_sec,
            ar_order=args.ar_order,
            freq_window_sec=args.freq_window_sec,
            recompute_hz=args.recompute_hz,
            target_phase=args.target_phase,
        )

    def clear_line():
        print("\r" + " " * 120 + "\r", end="", flush=True)

    def on_fire_legacy(t, ev):
        clear_line()
        if ev is None:
            print(f"[{time.strftime('%H:%M:%S')}] FIRE @ t={t:7.1f}s  (grace timeout -- no peak/trough in window)")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] FIRE @ t={t:7.1f}s  "
                  f"({ev['kind']}, {'valid' if ev['valid'] else 'invalid'}, v={ev['v']:+.2f}uM)")
        stim.fire()

    def on_fire_adaptive(t, info):
        clear_line()
        pred = info["prediction"]
        burst_tag = f"burst {info['burst_index']}/{info['n_bursts']}" if info["n_bursts"] > 1 else "burst"
        if pred is None:
            print(f"[{time.strftime('%H:%M:%S')}] FIRE @ t={t:7.1f}s  ({burst_tag}, "
                  f"predictor still warming up -- fixed-grid fallback, shift=+0.0s)")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] FIRE @ t={t:7.1f}s  ({burst_tag}, "
                  f"nominal {info['nominal_t']:.1f}s, shift {info['shift']:+.1f}s -> {info['scheduled_t']:.1f}s, "
                  f"targeting {pred['predicted_kind']} ~{pred['predicted_t']:.1f}s, "
                  f"period~{1.0 / pred['freq_inst_hz']:.1f}s)")
        stim.fire()

    if args.legacy_reactive:
        controller = TriggerController(args.on_sec, args.off_sec, args.grace_sec, on_fire_legacy)
        print(f"Mode: legacy-reactive. Schedule: {args.on_sec:g}s ON, {args.off_sec:g}s OFF minimum, exact moment "
              f"follows the next detected peak/trough (grace fallback after {args.grace_sec:g}s extra)")
    else:
        controller = AdaptiveTriggerController(
            args.on_sec, args.off_sec, STANDARD_BURST_INTERVAL_SEC, args.lead_sec, args.max_shift_sec,
            on_fire_adaptive, require_prediction=args.require_prediction, n_bursts=args.n_bursts,
            burst_gap_sec=args.burst_gap_sec, target_phase=args.target_phase,
        )
        fallback_desc = "no fallback -- stays silent until a real prediction arrives" if args.require_prediction \
            else "falls back to unshifted firing while the predictor warms up"
        burst_desc = f", {args.n_bursts} bursts/event ({args.burst_gap_sec:g}s gap)" if args.n_bursts > 1 else ""
        print(f"Mode: adaptive phase-predictive. Fixed {STANDARD_BURST_INTERVAL_SEC:g}s grid, "
              f"each event shifted up to +/-{args.max_shift_sec:g}s toward "
              f"(predicted {args.target_phase} - {args.lead_sec:g}s lead), {fallback_desc}{burst_desc}")

        group_duration = args.n_bursts * args.on_sec + (args.n_bursts - 1) * args.burst_gap_sec
        margin = STANDARD_BURST_INTERVAL_SEC - 2 * args.max_shift_sec
        if group_duration > margin:
            print(f"WARNING: a full burst group ({group_duration:g}s) exceeds the "
                  f"{margin:g}s safety margin (interval - 2*max_shift_sec) reserved to keep consecutive events "
                  f"from colliding -- consider reducing --n-bursts/--burst-gap-sec or --max-shift-sec, or "
                  f"increasing the effective interval via --off-sec.")

    stim = None
    baseline_ready = False
    baseline_deadline = BASELINE_SEC
    n_samples_seen = 0
    n_peaks = 0
    n_troughs = 0
    t = 0.0
    activity_val = 0.0
    pred = None

    try:
        stim = Stimulator(args.live, args.port, args.ttl_port, args.amplitude, args.no_arm)
        while True:
            samples, _timestamps = inlet.pull_chunk(timeout=0.5, max_samples=2048)
            for sample in samples:
                n_samples_seen += 1
                t = n_samples_seen / sfreq

                if not baseline_ready:
                    for pair in analysis_pairs:
                        pair.add_baseline_sample(sample)
                    if t >= baseline_deadline:
                        for pair in analysis_pairs:
                            pair.finalize_baseline()
                        baseline_ready = True
                        controller.start(t)
                        clear_line()
                        print(f"Baseline complete at t={t:.1f}s -- trigger loop armed")
                    continue

                for pair in analysis_pairs:
                    pair.update(sample)
                activity_val = -float(np.mean([p.last_hbo for p in analysis_pairs]))

                for ev in analyzer.push(t, activity_val):
                    if ev["kind"] == "peak":
                        n_peaks += 1
                    else:
                        n_troughs += 1
                    if args.legacy_reactive:
                        controller.handle_event(ev)

                if args.legacy_reactive:
                    controller.check_timeout(t)
                else:
                    pred = predictor.update(t, activity_val)
                    controller.step(t, pred)

            if not baseline_ready:
                remaining = max(0.0, baseline_deadline - t)
                line = f"  collecting baseline...  t={t:5.1f}s  ({remaining:4.1f}s remaining)"
            elif args.legacy_reactive:
                eta = max(0.0, controller.next_earliest_t - t) if controller.next_earliest_t is not None else 0.0
                line = (f"  t={t:7.1f}s  activity={activity_val:+6.2f}uM  "
                        f"next trigger in ~{eta:4.1f}s  (peaks:{n_peaks} troughs:{n_troughs})")
            else:
                period_str = f"{1.0 / pred['freq_inst_hz']:.1f}s" if pred else "warming up"
                if controller.scheduled_t is not None:
                    eta = max(0.0, controller.scheduled_t - t)
                    shift_str = f"{(controller.scheduled_t - controller.nominal_t):+.1f}s"
                    eta_str = f"~{eta:4.1f}s"
                else:
                    shift_str = "n/a"  # require_prediction: nothing scheduled for this slot yet
                    eta_str = " ?.?s"
                line = (f"  t={t:7.1f}s  activity={activity_val:+6.2f}uM  period~{period_str}  "
                        f"shift {shift_str} (+/-{args.max_shift_sec:.1f}s)  next trigger in {eta_str}  "
                        f"(peaks:{n_peaks} troughs:{n_troughs})")
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
    p.add_argument("--channels", nargs="+", default=[GRAND_AVERAGE], metavar="LABEL",
                    help="which source-detector pair(s) feed activity_val = -mean(HbO). One label (e.g. S1_D2) "
                         "-> that channel only; several labels -> their average; 'grandaverage' (default) -> "
                         "average across every available pair. Labels are printed at startup once connected.")
    p.add_argument("--on-sec", type=float, default=DEFAULT_ON_SEC, help="ON burst duration (default: 2.0s)")
    p.add_argument("--off-sec", type=float, default=DEFAULT_OFF_SEC,
                    help="[legacy-reactive] minimum OFF/refractory period after each burst. [adaptive] the "
                         "grid's initial phase-alignment offset (not a literal OFF duration -- burst spacing is "
                         "always interval-on_sec regardless); NOTE this does not fix the negative-bound shift "
                         "saturation seen across real sessions -- see DEFAULT_OFF_SEC's comment.")

    p.add_argument("--legacy-reactive", action="store_true",
                    help="use the old reactive ActivityAnalyzer+TriggerController path instead of the adaptive "
                         "phase-predictive one (no filtering, not robust to cardiac/respiratory noise -- kept "
                         "only for A/B comparison)")
    p.add_argument("--grace-sec", type=float, default=DEFAULT_GRACE_SEC,
                    help="[legacy-reactive only] fire anyway if no peak/trough is confirmed within this many extra "
                         "seconds beyond the 5s detection lag after the OFF period ends (default: 3.0s)")

    p.add_argument("--lead-sec", type=float, default=DEFAULT_LEAD_SEC,
                    help="[adaptive only] fire this many seconds before the predicted hemodynamic peak/trough, "
                         "compensating for the neural-to-hemodynamic response lag (default: 5.0s -- a literature-"
                         "derived HRF-peak-latency placeholder; validate empirically before trusting it)")
    p.add_argument("--max-shift-sec", type=float, default=DEFAULT_MAX_SHIFT_SEC,
                    help="[adaptive only] max seconds a burst's onset may be shifted from its fixed 10s-grid slot "
                         "(default: 3.0s, the largest bound that still guarantees a safety margin against burst "
                         "collisions given on_sec=2s/interval=10s)")
    p.add_argument("--center-period-sec", type=float, default=DEFAULT_CENTER_PERIOD_SEC,
                    help="[adaptive only] nominal oscillation period, informational (default: 10.0s)")
    p.add_argument("--period-lo-sec", type=float, default=DEFAULT_PERIOD_LO_SEC,
                    help="[adaptive only] passband fast edge in seconds (default: 7.0s -> 0.143 Hz)")
    p.add_argument("--period-hi-sec", type=float, default=DEFAULT_PERIOD_HI_SEC,
                    help="[adaptive only] passband slow edge in seconds (default: 17.0s -> 0.059 Hz)")
    p.add_argument("--buffer-sec", type=float, default=DEFAULT_BUFFER_SEC,
                    help="[adaptive only] rolling buffer length for phase/frequency estimation (default: 40.0s)")
    p.add_argument("--pad-sec", type=float, default=DEFAULT_PAD_SEC,
                    help="[adaptive only] AR-extrapolated padding length before bandpass filtering (default: 12.0s)")
    p.add_argument("--ar-order", type=int, default=DEFAULT_AR_ORDER,
                    help="[adaptive only] autoregressive model order (default: 20)")
    p.add_argument("--freq-window-sec", type=float, default=DEFAULT_FREQ_WINDOW_SEC,
                    help="[adaptive only] window for local instantaneous-frequency slope fit (default: 5.0s)")
    p.add_argument("--recompute-hz", type=float, default=DEFAULT_RECOMPUTE_HZ,
                    help="[adaptive only] how often to recompute the phase/frequency estimate (default: 5.0 Hz)")
    p.add_argument("--target-phase", choices=("peak", "trough"), default=None,
                    help="[adaptive only] which extremum to commit to targeting for the whole run. Required "
                         "unless --legacy-reactive is used -- 'either' (opportunistic peak-or-trough) is not "
                         "supported here: a cross-session diagnostic found its true scheduling offset is roughly "
                         "double peak/trough's, too large to recalibrate away without leaving almost no OFF gap.")
    p.add_argument("--require-prediction", action="store_true",
                    help="[adaptive only] never fall back to firing on the unshifted nominal grid -- stay silent "
                         "and skip a slot entirely if no real prediction is available for it (e.g. still warming "
                         "up).")
    p.add_argument("--n-bursts", type=int, default=1,
                    help="[adaptive only] number of on_sec bursts to fire back-to-back at each triggering event "
                         "(default: 1). Use e.g. 2 to double up when targeting only peaks or only troughs, to "
                         "approximate standard iTBS's total pulse count/dosage despite triggering on just one "
                         "extremum type per cycle instead of both.")
    p.add_argument("--burst-gap-sec", type=float, default=0.0,
                    help="[adaptive only] gap between consecutive bursts within a multi-burst group (default: "
                         "0.0s -- back-to-back/immediate succession)")

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
    args = p.parse_args()

    run(args)


if __name__ == "__main__":
    main()
