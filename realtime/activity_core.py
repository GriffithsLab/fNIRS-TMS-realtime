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
