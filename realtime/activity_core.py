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
# Real-time short-channel / systemic-physiology artifact removal
# --------------------------------------------------------------------------- #
# Source-detector distance below which a channel is treated as a short-
# separation channel (sees scalp/systemic signal only, negligible cerebral
# contribution). 15mm comfortably covers the common 8mm short-channel
# hardware while staying well under standard ~30mm long-channel spacing;
# mne-nirs's short_channel_regression() default threshold is 10mm.
SHORT_CHANNEL_MAX_DISTANCE_M = 0.015
N_NEAREST_SHORT_CHANNELS = 3  # short channels used as reference per long channel
RLS_FORGETTING_FACTOR = 0.999  # lambda; near 1 -> slow, stable adaptation
RLS_REGULARIZATION = 1e-2  # initial inverse-correlation matrix P0 = I / this


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


# --------------------------------------------------------------------------- #
# Real-time short-channel artifact removal
#
# Short-separation channels (source-detector distance too small to reach
# cortex) pick up scalp/systemic physiology -- heartbeat, respiration, Mayer
# waves, general scalp blood flow -- without a cerebral component, while
# long channels see that same superficial signal added on top of the
# cerebral one. Regressing the short-channel signal out of each long
# channel therefore removes both "short channel signal" and the systemic
# physiological artifacts it carries in one step (Zhang et al. 2007; Gagnon
# et al. 2014; the same rationale mne-nirs's short_channel_regression()
# uses, offline, via a batched GLM).
#
# There is no ready-made "real-time RLS" class in mne or mne-realtime to
# reuse here -- mne-realtime is archived and only ever provided epoch-buffer
# tools (RtEpochs/RtRegression for buffered, block-wise fits), and
# mne-nirs's short_channel_regression() is an offline batch GLM over a
# fixed recording, not a per-sample online estimator. Recursive Least
# Squares is the standard causal, online algorithm for exactly this
# adaptive-noise-cancellation problem (Haykin, "Adaptive Filter Theory"),
# so RLSFilter below is a small self-contained implementation of it.
# --------------------------------------------------------------------------- #
class RLSFilter:
    """Causal multi-reference Recursive Least Squares adaptive filter.

    Predicts the desired (contaminated) signal `d` from a reference vector
    `x` on every sample, adapting its weights after each prediction so the
    filter tracks slowly-changing coupling between references and target.
    Unlike a batch regression, it needs no fixed window and never looks
    ahead, so it is safe to use in a live stream.
    """

    def __init__(self, n_refs: int, forgetting_factor: float = RLS_FORGETTING_FACTOR, delta: float = RLS_REGULARIZATION):
        self.lam = forgetting_factor
        self.w = np.zeros(n_refs)
        self.P = np.eye(n_refs) / delta

    def update(self, x: np.ndarray, d: float) -> tuple[float, float]:
        """One RLS step. Returns (cleaned_estimate, predicted_artifact)."""
        Px = self.P @ x
        gain = Px / (self.lam + float(x @ Px))
        y_hat = float(self.w @ x)  # predicted superficial/systemic contribution
        e = d - y_hat  # artifact-removed residual
        self.w = self.w + gain * e
        self.P = (self.P - np.outer(gain, Px)) / self.lam
        return e, y_hat


class ShortChannelArtifactRemover:
    """Removes short-channel-correlated superficial/systemic signal from
    every long channel's HbO/HbR, in place, sample by sample.

    Each long channel is paired with its `N_NEAREST_SHORT_CHANNELS` nearest
    short channels (by probe midpoint distance) as the RLS reference
    vector -- one independent RLS filter per long channel per chromophore
    (HbO, HbR), so coupling strength can differ by hemoglobin species and
    by scalp location.
    """

    def __init__(
        self,
        pairs: list[ChannelPair],
        short_max_distance_m: float = SHORT_CHANNEL_MAX_DISTANCE_M,
        n_nearest: int = N_NEAREST_SHORT_CHANNELS,
    ):
        self.short_pairs = [p for p in pairs if p.distance_m <= short_max_distance_m]
        self.long_pairs = [p for p in pairs if p.distance_m > short_max_distance_m]
        self._nearest: dict[str, list[ChannelPair]] = {}
        self._filters: dict[str, tuple[RLSFilter, RLSFilter]] = {}
        for lp in self.long_pairs:
            if not self.short_pairs:
                continue
            order = sorted(self.short_pairs, key=lambda sp: float(np.linalg.norm(lp.mid - sp.mid)))
            nearest = order[:n_nearest]
            self._nearest[lp.label] = nearest
            self._filters[lp.label] = (RLSFilter(len(nearest)), RLSFilter(len(nearest)))

    @property
    def active(self) -> bool:
        """True once there is at least one short and one long channel to work with."""
        return bool(self.short_pairs) and bool(self.long_pairs)

    def process(self) -> None:
        """Call once per sample, after ChannelPair.update() has run on every
        pair (short and long alike). Overwrites each long channel's
        last_hbo/last_hbr in place with the artifact-removed estimate."""
        for lp in self.long_pairs:
            filt = self._filters.get(lp.label)
            if filt is None:
                continue
            hbo_filt, hbr_filt = filt
            refs = self._nearest[lp.label]
            x_hbo = np.array([r.last_hbo for r in refs])
            x_hbr = np.array([r.last_hbr for r in refs])
            lp.last_hbo, _ = hbo_filt.update(x_hbo, lp.last_hbo)
            lp.last_hbr, _ = hbr_filt.update(x_hbr, lp.last_hbr)


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
