"""Shared fNIRS stream-decoding and activity/peak-trough-detection logic.

Used by both viewer.py (GUI) and itbs_trigger.py (headless MagVenture trigger
controller) so the two consumers see byte-for-byte identical peak/trough
detection -- what the viewer flashes on is exactly what the stimulator fires
on.

Parameters below match Occipital_fNIRS_iTBS_Analysis_Executed.ipynb
(TIME_WINDOW_SEC, MIN_PEAK_DISTANCE_SEC, ACTIVITY_THRESHOLD_PERCENTILE,
STANDARD_BURST_DURATION_SEC/INTERVAL_SEC).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

PPF = 6.0  # partial pathlength factor, standard adult default (same as MNE's default)
BASELINE_SEC = 8.0
EXTINCTION_CSV = Path(__file__).with_name("extinction_coef_600_1000nm.csv")

ACTIVITY_STATS_WINDOW_SEC = 300.0
MIN_PEAK_DISTANCE_SEC = 5.0
ACTIVITY_THRESHOLD_PERCENTILE = 75
STANDARD_BURST_DURATION_SEC = 2.0
STANDARD_BURST_INTERVAL_SEC = 10.0
PEAK_TRIGGER_HALF_WIDTH_SEC = 1.0  # notebook uses 2s windows centered on each peak/trough

# --------------------------------------------------------------------------- #
# PhasePredictor defaults (adaptive, phase-predictive iTBS triggering)
# --------------------------------------------------------------------------- #
DEFAULT_LEAD_SEC = 5.0             # HRF-peak-latency placeholder (Boynton 1996 / SPM canonical HRF / fNIRS HbO
                                    # studies all report ~4-6s) -- MUST be validated empirically for this paradigm.
DEFAULT_MAX_SHIFT_SEC = 3.0        # largest bound that still guarantees on_sec(2s) + 2s safety margin of true OFF
                                    # gap in the worst case: interval_sec(10) - 2*max_shift_sec >= on_sec(2) + 2
DEFAULT_CENTER_PERIOD_SEC = STANDARD_BURST_INTERVAL_SEC  # 10.0 -- reuse the existing nominal iTBS period
DEFAULT_PERIOD_LO_SEC = 7.0        # passband fast edge -> 1/7   = 0.143 Hz, guard band below respiration's ~0.15 Hz
DEFAULT_PERIOD_HI_SEC = 17.0       # passband slow edge -> 1/17  = 0.059 Hz
DEFAULT_BUFFER_SEC = 40.0          # ~4 nominal cycles: enough for a stable AR fit, short enough to track drift
DEFAULT_PAD_SEC = 12.0             # ~1 filter settling time (1/bandwidth ~= 12.5s for a ~0.08Hz-wide order-3 filter)
DEFAULT_AR_ORDER = 20              # clamped to buffer_n // 4 to keep the least-squares fit well-conditioned
DEFAULT_FREQ_WINDOW_SEC = 5.0
DEFAULT_RECOMPUTE_HZ = 5.0         # throttle: recomputing every incoming sample is cheap but wasteful/flickery
DEFAULT_FILTER_ORDER = 3


# --------------------------------------------------------------------------- #
# Absorption spectrum lookup
# --------------------------------------------------------------------------- #
class ExtinctionTable:
    """Linear-interpolated molar extinction coefficients (HbO2, Hb) vs wavelength.

    Source: OMLC / Prahl hemoglobin extinction spectra (https://omlc.org/spectra/hemoglobin/),
    the same table bundled with MNE-Python for its beer_lambert_law() implementation.
    """

    def __init__(self, csv_path: Path):
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        self.wl = data[:, 0]
        self.eps_hbo2 = data[:, 1]
        self.eps_hb = data[:, 2]

    def __call__(self, wavelength_nm: float) -> tuple[float, float]:
        hbo2 = float(np.interp(wavelength_nm, self.wl, self.eps_hbo2))
        hb = float(np.interp(wavelength_nm, self.wl, self.eps_hb))
        return hbo2, hb


# --------------------------------------------------------------------------- #
# Stream metadata parsing
# --------------------------------------------------------------------------- #
class ChannelMeta:
    __slots__ = ("index", "label", "source", "detector", "wavelength", "src_pos", "det_pos", "distance_m")

    def __init__(self, index, label, source, detector, wavelength, src_pos, det_pos, distance_m):
        self.index = index
        self.label = label
        self.source = source
        self.detector = detector
        self.wavelength = wavelength
        self.src_pos = src_pos
        self.det_pos = det_pos
        self.distance_m = distance_m


def _xml_children(elem, name):
    child = elem.child(name)
    while not child.empty():
        yield child
        child = child.next_sibling(name)


def parse_channels(sinfo) -> list[ChannelMeta]:
    desc = sinfo.desc
    chs_elem = desc.child("channels")
    metas = []
    for i, c in enumerate(_xml_children(chs_elem, "channel")):
        label = c.child_value("label")
        source = c.child_value("source")
        detector = c.child_value("detector")
        wavelength = float(c.child_value("wavelength_nm"))
        src_pos = np.array([float(c.child_value("src_x")), float(c.child_value("src_y")), float(c.child_value("src_z"))])
        det_pos = np.array([float(c.child_value("det_x")), float(c.child_value("det_y")), float(c.child_value("det_z"))])
        distance_m = float(c.child_value("distance_m"))
        metas.append(ChannelMeta(i, label, source, detector, wavelength, src_pos, det_pos, distance_m))
    return metas


class ChannelPair:
    """A source-detector pair with two wavelength channel indices -> HbO/HbR."""

    def __init__(self, source, detector, idx_by_wl: dict[float, int], src_pos, det_pos, distance_m):
        self.source = source
        self.detector = detector
        self.idx_by_wl = idx_by_wl
        self.wavelengths = sorted(idx_by_wl)
        self.src_pos = src_pos
        self.det_pos = det_pos
        self.distance_m = distance_m
        self.mid = (src_pos + det_pos) / 2.0
        self.label = f"{source}_{detector}"

        self.baseline = None  # I0 per wavelength, shape (n_wl,)
        self.baseline_buf = []
        self.iEL = None  # (2, n_wl) pseudo-inverse
        self.last_hbo = 0.0
        self.last_hbr = 0.0

    def set_extinction(self, ext: ExtinctionTable):
        abs_coef = np.array([ext(wl) for wl in self.wavelengths])  # (n_wl, 2) -> [HbO2, Hb]
        EL = abs_coef * self.distance_m * PPF
        self.iEL = np.linalg.pinv(EL)  # (2, n_wl)

    def add_baseline_sample(self, sample: np.ndarray):
        raw = np.array([sample[self.idx_by_wl[wl]] for wl in self.wavelengths])
        self.baseline_buf.append(raw)

    def finalize_baseline(self):
        self.baseline = np.mean(self.baseline_buf, axis=0)
        self.baseline_buf = []

    def update(self, sample: np.ndarray) -> tuple[float, float]:
        """Feed one full sample (all channels); returns (dHbO, dHbR) in micromolar."""
        raw = np.array([sample[self.idx_by_wl[wl]] for wl in self.wavelengths])
        with np.errstate(divide="ignore", invalid="ignore"):
            od = -np.log10(np.clip(raw, 1e-12, None) / self.baseline)
        conc = self.iEL @ od * 1e-3  # mol/L, matches MNE's beer_lambert_law scaling
        self.last_hbo = float(conc[0] * 1e6)  # -> micromolar
        self.last_hbr = float(conc[1] * 1e6)
        return self.last_hbo, self.last_hbr


GRAND_AVERAGE = "grandaverage"


def select_pairs(pairs: list[ChannelPair], channels: list[str] | None) -> list[ChannelPair]:
    """Select which source-detector pairs feed the aggregate activity_val
    signal (activity_val = -mean(HbO) over whatever this returns).

    channels is a list of ChannelPair.label values (e.g. "S1_D2"):
      - None, empty, or a single "grandaverage" (case-insensitive) entry ->
        every pair (global average -- the original, only behavior).
      - a single channel label -> that one pair only, no averaging.
      - multiple channel labels -> the average of just those pairs.

    Raises ValueError (with the list of valid labels) if any requested label
    doesn't match an available pair -- callers should surface this clearly
    rather than silently falling back to something else.
    """
    if not channels or (len(channels) == 1 and channels[0].strip().lower() == GRAND_AVERAGE):
        return list(pairs)
    by_label = {p.label: p for p in pairs}
    missing = [c for c in channels if c not in by_label]
    if missing:
        available = ", ".join(sorted(by_label))
        raise ValueError(f"Unknown channel label(s) {missing} -- available labels: {available}")
    return [by_label[c] for c in channels]


# --------------------------------------------------------------------------- #
# Causal peak/trough detection (real-time analogue of the notebook's
# find_peaks_numpy, which flags sample i as a peak when
# signal[i] > signal[i-distance] and signal[i] > signal[i+distance]).
# That definition inherently needs `distance` samples of look-ahead to
# classify point i, so a live/causal version confirms a peak exactly
# `distance` samples (= MIN_PEAK_DISTANCE_SEC) after it occurs -- the same
# fixed latency the offline algorithm has, just paid in real time instead of
# in hindsight.
# --------------------------------------------------------------------------- #
class ActivityAnalyzer:
    def __init__(self, sfreq: float):
        self.d = max(1, round(MIN_PEAK_DISTANCE_SEC * sfreq))
        self.maxlen = max(self.d * 2 + 2, round(ACTIVITY_STATS_WINDOW_SEC * sfreq))
        self.t_buf: list[float] = []
        self.v_buf: list[float] = []
        self.last_peak_t = -1e18
        self.last_trough_t = -1e18

    def push(self, t: float, v: float) -> list[dict]:
        self.t_buf.append(t)
        self.v_buf.append(v)
        if len(self.v_buf) > self.maxlen * 2:
            self.t_buf = self.t_buf[-self.maxlen:]
            self.v_buf = self.v_buf[-self.maxlen:]

        events = []
        d = self.d
        n = len(self.v_buf)
        if n >= 2 * d + 1:
            i = n - 1 - d
            v_lo, v_i, v_hi = self.v_buf[i - d], self.v_buf[i], self.v_buf[i + d]
            t_i = self.t_buf[i]
            if v_i > v_lo and v_i > v_hi and t_i - self.last_peak_t >= MIN_PEAK_DISTANCE_SEC:
                self.last_peak_t = t_i
                events.append({"kind": "peak", "valid": v_i > 0, "t": t_i, "v": v_i})
            if v_i < v_lo and v_i < v_hi and t_i - self.last_trough_t >= MIN_PEAK_DISTANCE_SEC:
                self.last_trough_t = t_i
                events.append({"kind": "trough", "valid": v_i < 0, "t": t_i, "v": v_i})
        return events

    def percentile_threshold(self) -> float | None:
        if len(self.v_buf) < 20:
            return None
        return float(np.percentile(self.v_buf, ACTIVITY_THRESHOLD_PERCENTILE))


# --------------------------------------------------------------------------- #
# Phase-predictive extremum forecasting (adaptive iTBS triggering).
#
# ActivityAnalyzer above only ever reports a peak/trough after it already
# happened (with a fixed MIN_PEAK_DISTANCE_SEC confirmation lag), and is not
# robust to cardiac (~0.8-2 Hz) / respiratory (~0.15-0.4 Hz) noise since its
# extremum test compares just two isolated samples. PhasePredictor instead
# narrowband-bandpasses the signal (which structurally excludes both noise
# bands), estimates its instantaneous phase/frequency, and linearly
# extrapolates forward to predict WHEN the next peak or trough will occur --
# so a caller can act *before* the extremum, not after it.
#
# Edge-artifact handling follows the same principle used for real-time
# phase-triggered EEG-TMS stimulation (e.g. the phastimate approach): a
# zero-phase bandpass filter run on a plain causal buffer would distort the
# most recent ("now") samples the most, since they sit right at the filter's
# edge. Padding the buffer with AR-model-extrapolated synthetic future
# samples before filtering pushes that edge distortion into the synthetic
# region instead, leaving the real "now" sample comparatively clean.
# --------------------------------------------------------------------------- #
def _fit_ar_lstsq(x: np.ndarray, order: int) -> np.ndarray:
    """Least-squares AR(order) fit: x[i] ~= sum_k coefs[k] * x[i-1-k]."""
    n = len(x)
    Y = x[order:]
    X = np.column_stack([x[order - 1 - k : n - 1 - k] for k in range(order)])
    coefs, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return coefs


def _ar_extrapolate(x: np.ndarray, coefs: np.ndarray, n_ahead: int) -> np.ndarray:
    """Recursively synthesize n_ahead future samples from a fitted AR model."""
    order = len(coefs)
    hist = list(x[-order:])
    out = np.empty(n_ahead)
    for i in range(n_ahead):
        nxt = float(np.dot(coefs, hist[::-1]))
        out[i] = nxt
        hist = hist[1:] + [nxt]
    return out


class PhasePredictor:
    """Causal, continuously self-correcting predictor of the next peak/trough
    time on the aggregated activity_val scalar.

    On each (throttled) recompute:
      1. Linearly detrend the rolling raw buffer.
      2. Fit an AR(order) model to it via plain-numpy least squares.
      3. Use the AR recursion to synthesize `pad_sec` of future samples.
      4. Zero-phase bandpass (sosfiltfilt) the extended real+synthetic
         buffer -- edge distortion lands in the synthetic tail, not on the
         real "now" sample.
      5. Hilbert-transform the filtered buffer; read phase/amplitude at the
         real "now" sample.
      6. Estimate local instantaneous frequency from the recent unwrapped-
         phase slope, clamped to the passband to resist noisy data.
      7. Linearly extrapolate phase forward to the next peak (phase 0) and
         next trough (phase pi); report whichever is sooner (or whichever
         `target_phase` requests).

    The passband (default ~7-17s period, i.e. ~0.059-0.143 Hz) is what makes
    this simultaneously reject cardiac and respiratory oscillations before
    phase is ever estimated -- no separate filtering stage is needed.
    """

    def __init__(
        self,
        sfreq: float,
        center_period_sec: float = DEFAULT_CENTER_PERIOD_SEC,
        period_lo_sec: float = DEFAULT_PERIOD_LO_SEC,
        period_hi_sec: float = DEFAULT_PERIOD_HI_SEC,
        buffer_sec: float = DEFAULT_BUFFER_SEC,
        pad_sec: float = DEFAULT_PAD_SEC,
        ar_order: int = DEFAULT_AR_ORDER,
        freq_window_sec: float = DEFAULT_FREQ_WINDOW_SEC,
        recompute_hz: float = DEFAULT_RECOMPUTE_HZ,
        target_phase: str = "either",
        filter_order: int = DEFAULT_FILTER_ORDER,
    ):
        if target_phase not in ("either", "peak", "trough"):
            raise ValueError(f"target_phase must be 'either', 'peak', or 'trough', got {target_phase!r}")
        self.center_period_sec = center_period_sec  # informational only; passband is set by period_lo_sec/period_hi_sec
        self.sfreq = sfreq
        self.f_lo = 1.0 / period_hi_sec
        self.f_hi = 1.0 / period_lo_sec
        self.buffer_n = max(int(round(buffer_sec * sfreq)), ar_order * 4)
        self.pad_n = max(1, int(round(pad_sec * sfreq)))
        self.ar_order = max(1, min(ar_order, self.buffer_n // 4))
        self.freq_window_n = max(4, int(round(freq_window_sec * sfreq)))
        self.recompute_every = max(1, int(round(sfreq / recompute_hz))) if recompute_hz > 0 else 1
        self.target_phase = target_phase
        self.sos = butter(filter_order, [self.f_lo, self.f_hi], btype="bandpass", fs=sfreq, output="sos")
        self.t_buf: list[float] = []
        self.v_buf: list[float] = []
        self._n_since_recompute = 0
        self.last: dict | None = None

    def reset(self):
        self.t_buf = []
        self.v_buf = []
        self._n_since_recompute = 0
        self.last = None

    def update(self, t: float, v: float) -> dict | None:
        """Feed one new sample; returns the latest prediction dict, or None
        while the buffer is still warming up (~buffer_sec after baseline)."""
        self.t_buf.append(t)
        self.v_buf.append(v)
        if len(self.v_buf) > self.buffer_n:
            del self.t_buf[: len(self.t_buf) - self.buffer_n]
            del self.v_buf[: len(self.v_buf) - self.buffer_n]

        if len(self.v_buf) < self.buffer_n:
            return None

        self._n_since_recompute += 1
        if self.last is not None and self._n_since_recompute < self.recompute_every:
            return self.last

        self._n_since_recompute = 0
        try:
            self.last = self._recompute()
        except np.linalg.LinAlgError:
            pass  # keep the previous prediction if the AR fit degenerates on this window
        return self.last

    def _recompute(self) -> dict:
        v = np.asarray(self.v_buf)
        n = len(v)
        t_now = self.t_buf[-1]

        x = np.arange(n)
        trend = np.polyval(np.polyfit(x, v, 1), x)
        detrended = v - trend

        ar_coefs = _fit_ar_lstsq(detrended, self.ar_order)
        extrap = _ar_extrapolate(detrended, ar_coefs, self.pad_n)
        cap = 5.0 * (np.std(detrended) + 1e-12)  # guard a mildly unstable AR fit from blowing up the padding
        extrap = np.clip(extrap, -cap, cap)
        extended = np.concatenate([detrended, extrap])

        filtered = sosfiltfilt(self.sos, extended)
        analytic = hilbert(filtered)
        phase = np.angle(analytic)
        amp = np.abs(analytic)
        phase_now = float(phase[n - 1])
        amp_now = float(amp[n - 1])

        w0 = max(0, n - self.freq_window_n)
        unwrapped = np.unwrap(phase[w0:n])
        dt = self.t_buf[-1] - self.t_buf[w0]
        if dt <= 0:
            dt = 1.0 / self.sfreq
        omega_raw = (unwrapped[-1] - unwrapped[0]) / dt
        omega = float(np.clip(omega_raw, 2 * np.pi * self.f_lo, 2 * np.pi * self.f_hi))
        f_inst = omega / (2 * np.pi)

        dt_peak = ((-phase_now) % (2 * np.pi)) / omega
        dt_trough = ((np.pi - phase_now) % (2 * np.pi)) / omega
        if self.target_phase == "peak":
            kind, dtp = "peak", dt_peak
        elif self.target_phase == "trough":
            kind, dtp = "trough", dt_trough
        else:
            kind, dtp = ("peak", dt_peak) if dt_peak <= dt_trough else ("trough", dt_trough)

        return {
            "t_now": t_now,
            "phase_now": phase_now,
            "amp_now": amp_now,
            "freq_inst_hz": f_inst,
            "predicted_kind": kind,
            "predicted_t": t_now + dtp,
        }
