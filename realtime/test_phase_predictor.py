"""Synthetic self-test for activity_core.PhasePredictor and
itbs_trigger.AdaptiveTriggerController -- run this BEFORE ever trusting the
adaptive trigger against real data, let alone real hardware.

Generates a noisy, aperiodic ~10s-period oscillation (the kind of signal
described for this paradigm: on average a 10s period, but sometimes faster,
sometimes slower) with injected cardiac (~1.2 Hz) and respiratory (~0.25 Hz)
contamination, feeds it sample-by-sample into PhasePredictor exactly as
itbs_trigger.py would, and checks:

  1. How far off the predicted peak/trough times are from the known ground
     truth, at several contamination levels (clean / moderate / heavy) --
     reported, not pass/fail, since what error is "acceptable" for this
     specific paradigm is a judgment call for the operator to make once they
     see real numbers, not something this script can decide for them.
  2. Hard structural invariants of AdaptiveTriggerController that must ALWAYS
     hold regardless of how noisy the input is: every fire's shift stays
     within +/- max_shift_sec of its nominal slot, and consecutive fires
     never invert order or land closer together than on_sec.

Uses the sample recording's actual sample rate (data/collegestreet/
2025-03-05_001.snirf, relative to realtime_dev/) if available, since
PhasePredictor's behavior depends on sfreq; falls back to a representative
NIRx CW rate if the file or mne isn't available.

Usage
-----
    python test_phase_predictor.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from activity_core import PhasePredictor, STANDARD_BURST_DURATION_SEC, STANDARD_BURST_INTERVAL_SEC
from itbs_trigger import AdaptiveTriggerController

SNIRF_PATH = Path(__file__).resolve().parents[2] / "data" / "collegestreet" / "2025-03-05_001.snirf"
FALLBACK_SFREQ = 7.81  # representative NIRx CW sample rate, used only if the sample file/mne is unavailable


def _resolve_sfreq() -> float:
    try:
        import mne
        raw = mne.io.read_raw_snirf(str(SNIRF_PATH), preload=False, verbose=False)
        sfreq = float(raw.info["sfreq"])
        print(f"Using sample rate from {SNIRF_PATH.name}: {sfreq:.3f} Hz")
        return sfreq
    except Exception as exc:  # file missing, mne missing, unreadable, etc.
        print(f"Could not read {SNIRF_PATH} ({exc!r}); falling back to {FALLBACK_SFREQ} Hz")
        return FALLBACK_SFREQ


def generate_synthetic(sfreq: float, duration_sec: float, seed: int,
                        period_lo: float = 7.0, period_hi: float = 13.0,
                        cardiac_amp: float = 0.0, resp_amp: float = 0.0, noise_std: float = 0.05):
    """v(t) = cos(phi(t)) + cardiac + respiration + noise, where phi(t) is
    built by integrating an instantaneous period that smoothly random-walks
    in [period_lo, period_hi] -- "on average 10s, but noisy/aperiodic"."""
    rng = np.random.default_rng(seed)
    n = int(round(duration_sec * sfreq))
    t = np.arange(n) / sfreq

    # smooth random walk of instantaneous period, mean-reverting toward 10s
    period = np.empty(n)
    period[0] = 10.0
    for i in range(1, n):
        period[i] = period[i - 1] + 0.02 * (10.0 - period[i - 1]) / sfreq + rng.normal(0, 0.15) / np.sqrt(sfreq)
        period[i] = np.clip(period[i], period_lo, period_hi)
    omega = 2 * np.pi / period
    phi = np.cumsum(omega) / sfreq  # continuous unwrapped phase

    v_clean = np.cos(phi)
    cardiac = cardiac_amp * np.sin(2 * np.pi * 1.2 * t + rng.uniform(0, 2 * np.pi))
    resp = resp_amp * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 2 * np.pi))
    noise = rng.normal(0, noise_std, n)
    v = v_clean + cardiac + resp + noise

    # ground-truth extremum times: phi crosses k*2pi (peaks) / (k+0.5)*2pi (troughs)
    peak_times = _crossing_times(t, phi, np.arange(np.floor(phi[0] / (2 * np.pi)), np.ceil(phi[-1] / (2 * np.pi)) + 1) * 2 * np.pi)
    trough_times = _crossing_times(t, phi, (np.arange(np.floor(phi[0] / (2 * np.pi) - 0.5), np.ceil(phi[-1] / (2 * np.pi) + 0.5)) + 0.5) * 2 * np.pi)
    return t, v, peak_times, trough_times


def _crossing_times(t: np.ndarray, phi: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Linearly interpolate the time at which monotonic-increasing phi(t)
    crosses each target value."""
    out = []
    for target in targets:
        idx = np.searchsorted(phi, target)
        if idx <= 0 or idx >= len(phi):
            continue
        t0, t1 = t[idx - 1], t[idx]
        p0, p1 = phi[idx - 1], phi[idx]
        frac = (target - p0) / (p1 - p0) if p1 != p0 else 0.0
        out.append(t0 + frac * (t1 - t0))
    return np.array(out)


def _next_true_time(true_times: np.ndarray, t_now: float) -> float | None:
    future = true_times[true_times > t_now]
    return float(future[0]) if len(future) else None


def evaluate(sfreq: float, duration_sec: float, seed: int, cardiac_amp: float, resp_amp: float, noise_std: float,
             label: str):
    t, v, peak_times, trough_times = generate_synthetic(
        sfreq, duration_sec, seed, cardiac_amp=cardiac_amp, resp_amp=resp_amp, noise_std=noise_std
    )

    predictor = PhasePredictor(sfreq, recompute_hz=2.0)
    errors = []
    for ti, vi in zip(t, v):
        pred = predictor.update(float(ti), float(vi))
        if pred is None:
            continue
        true_times = peak_times if pred["predicted_kind"] == "peak" else trough_times
        nxt = _next_true_time(true_times, pred["t_now"])
        if nxt is not None:
            errors.append(pred["predicted_t"] - nxt)

    errors = np.array(errors)
    if len(errors) == 0:
        print(f"[{label}] no predictions collected (duration too short?)")
        return

    print(f"[{label}] n={len(errors)}  median|err|={np.median(np.abs(errors)):.2f}s  "
          f"p90|err|={np.percentile(np.abs(errors), 90):.2f}s  max|err|={np.max(np.abs(errors)):.2f}s")


def check_controller_invariants(sfreq: float, duration_sec: float, seed: int,
                                 lead_sec: float, max_shift_sec: float):
    """Hard structural checks that must hold regardless of prediction noise:
    every fire's shift stays within the bound, and consecutive fires never
    invert order or land closer than on_sec apart."""
    t, v, _peak_times, _trough_times = generate_synthetic(
        sfreq, duration_sec, seed, cardiac_amp=0.4, resp_amp=0.3, noise_std=0.1
    )
    predictor = PhasePredictor(sfreq, recompute_hz=2.0, target_phase="trough")

    fires = []

    def on_fire(t_fire, info):
        fires.append((t_fire, info["shift"]))

    controller = AdaptiveTriggerController(
        STANDARD_BURST_DURATION_SEC, STANDARD_BURST_INTERVAL_SEC - STANDARD_BURST_DURATION_SEC,
        STANDARD_BURST_INTERVAL_SEC, lead_sec, max_shift_sec, on_fire, target_phase="trough",
    )
    controller.start(t[0])

    for ti, vi in zip(t, v):
        pred = predictor.update(float(ti), float(vi))
        controller.step(float(ti), pred)

    assert fires, "controller never fired -- something is wrong with the synthetic drive"

    shifts = np.array([s for _, s in fires])
    assert np.all(np.abs(shifts) <= max_shift_sec + 1e-9), \
        f"shift bound violated: max |shift| = {np.max(np.abs(shifts)):.3f}s > {max_shift_sec}s"

    fire_times = np.array([tf for tf, _ in fires])
    gaps = np.diff(fire_times)
    assert np.all(gaps > 0), "consecutive fire times were not strictly increasing (order inversion)"
    assert np.all(gaps >= STANDARD_BURST_DURATION_SEC - 1e-9), \
        f"two fires landed closer than on_sec={STANDARD_BURST_DURATION_SEC}s apart: min gap = {np.min(gaps):.3f}s"

    print(f"[invariants] n_fires={len(fires)}  max|shift|={np.max(np.abs(shifts)):.2f}s "
          f"(bound {max_shift_sec}s)  min_gap={np.min(gaps):.2f}s (on_sec {STANDARD_BURST_DURATION_SEC}s)  -- OK")


def check_require_prediction_and_n_bursts(sfreq: float, duration_sec: float, seed: int,
                                           lead_sec: float, max_shift_sec: float):
    """Checks specific to the require_prediction / target_phase / n_bursts
    options: no firing before the predictor has real data, every fired
    prediction matches the committed target_phase, and multi-burst groups
    are spaced/tagged correctly -- all on realistic (noisy) synthetic data,
    not just the isolated controller-only unit test."""
    t, v, _peak_times, _trough_times = generate_synthetic(
        sfreq, duration_sec, seed, cardiac_amp=0.3, resp_amp=0.2, noise_std=0.08
    )
    predictor = PhasePredictor(sfreq, recompute_hz=2.0, target_phase="trough")

    fires = []

    def on_fire(t_fire, info):
        fires.append((t_fire, info))

    n_bursts = 2
    controller = AdaptiveTriggerController(
        STANDARD_BURST_DURATION_SEC, STANDARD_BURST_INTERVAL_SEC - STANDARD_BURST_DURATION_SEC,
        STANDARD_BURST_INTERVAL_SEC, lead_sec, max_shift_sec, on_fire,
        require_prediction=True, n_bursts=n_bursts, burst_gap_sec=0.0, target_phase="trough",
    )
    controller.start(t[0])

    first_valid_t = None  # ground truth: the actual moment PhasePredictor.update() first returns non-None
    for ti, vi in zip(t, v):
        pred = predictor.update(float(ti), float(vi))
        if pred is not None and first_valid_t is None:
            first_valid_t = float(ti)
        controller.step(float(ti), pred)

    assert fires, "expected at least one fire once the predictor warms up on realistic data"
    assert first_valid_t is not None, "predictor never produced a prediction -- test duration too short?"

    early_fires = [(tf, info) for tf, info in fires if tf < first_valid_t]
    assert not early_fires, \
        f"require_prediction fired {len(early_fires)} time(s) before the predictor ever produced a real " \
        f"prediction (first valid at t={first_valid_t:.2f}s): {[tf for tf, _ in early_fires]}"

    kinds = {info["prediction"]["predicted_kind"] for _, info in fires}
    assert kinds == {"trough"}, f"target_phase='trough' fired on kinds {kinds}, expected only 'trough'"

    burst_idxs = [info["burst_index"] for _, info in fires]
    n_bursts_seen = {info["n_bursts"] for _, info in fires}
    assert n_bursts_seen == {n_bursts}, f"expected every fire tagged n_bursts={n_bursts}, saw {n_bursts_seen}"
    assert set(burst_idxs) <= {1, 2}, f"expected burst_index in {{1,2}}, saw {set(burst_idxs)}"

    print(f"[require_prediction/n_bursts] n_fires={len(fires)}  first_fire_t={fires[0][0]:.1f}s "
          f"(first valid prediction ~{first_valid_t:.1f}s)  all kinds='trough'  n_bursts={n_bursts} tags OK -- OK")


def main():
    sfreq = _resolve_sfreq()
    duration_sec = 600.0
    seed = 0

    print("\n--- Prediction accuracy sweep (noise/contamination increasing) ---")
    evaluate(sfreq, duration_sec, seed, cardiac_amp=0.0, resp_amp=0.0, noise_std=0.02, label="clean")
    evaluate(sfreq, duration_sec, seed, cardiac_amp=0.2, resp_amp=0.15, noise_std=0.05, label="moderate contamination")
    evaluate(sfreq, duration_sec, seed, cardiac_amp=0.4, resp_amp=0.3, noise_std=0.1, label="heavy contamination")
    evaluate(sfreq, duration_sec, seed, cardiac_amp=0.6, resp_amp=0.45, noise_std=0.15, label="very heavy contamination")

    print("\n--- AdaptiveTriggerController structural invariants ---")
    check_controller_invariants(sfreq, duration_sec, seed, lead_sec=5.0, max_shift_sec=3.0)

    print("\n--- require_prediction / target_phase / n_bursts options ---")
    check_require_prediction_and_n_bursts(sfreq, duration_sec, seed, lead_sec=5.0, max_shift_sec=3.0)

    print("\nAll invariant checks passed. Review the accuracy numbers above against your own tolerance "
          "for this paradigm before trusting --live firing.")


if __name__ == "__main__":
    main()
