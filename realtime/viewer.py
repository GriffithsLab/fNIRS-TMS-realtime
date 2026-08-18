"""Slick real-time viewer for an LSL fNIRS stream produced by stream_snirf_lsl.py.

Connects to the 'SNIRF_Amplitude' and 'SNIRF_Markers' LSL streams, converts
raw dual-wavelength intensity to HbO/HbR concentration changes on the fly
(modified Beer-Lambert law, using the OMLC/Prahl hemoglobin extinction
spectrum bundled as extinction_coef_600_1000nm.csv, same source table used by
MNE-Python), and displays:

  - a live probe map (sources/detectors/channels) colour-coded by current
    HbO signal, click a channel to add/remove it from the scrolling traces
  - a scrolling multi-channel HbO/HbR time series with TMS/iTBS event markers
  - a synchronized time series of the predictor's estimated wrapped phase
  - a live "activity" panel replicating the peak/trough-triggered iTBS
    analysis from Occipital_fNIRS_iTBS_Analysis_Executed.ipynb: a combined
    activity trace (-mean HbO across channels), causal peak/trough detection
    with a clear flash the instant a peak or trough is confirmed, a rolling
    75th-percentile activity threshold, and all three of the notebook's
    trigger-window strategies (standard fixed-rate, activity-threshold, and
    peak/trough-centered) shaded live as they occur

Usage
-----
    python viewer.py
    python viewer.py --amp-name SNIRF_Amplitude --marker-name SNIRF_Markers
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
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
    GRAND_AVERAGE,
    PEAK_TRIGGER_HALF_WIDTH_SEC,
    STANDARD_BURST_DURATION_SEC,
    STANDARD_BURST_INTERVAL_SEC,
    ActivityAnalyzer,
    ChannelPair,
    ExtinctionTable,
    PhasePredictor,
    parse_channels,
    select_pairs,
)
from itbs_trigger import AdaptiveTriggerController

WINDOW_SEC = 30.0
ACTIVITY_DISPLAY_WINDOW_SEC = 60.0
MAX_DISPLAY_WINDOW_SEC = 600.0
BG = "#0b0d12"
FG = "#c9d1d9"
GRID = "#232733"
HBO_COLOR = "#ff5c5c"
HBR_COLOR = "#4da3ff"
ACCENT = "#39d6a8"
MUTED = "#4a5164"
STANDARD_TRIGGER_BRUSH = (232, 184, 75, 45)     # gold  - fixed-rate triggers
ACTIVITY_TRIGGER_BRUSH = (255, 92, 92, 60)      # red   - above-threshold triggers
PEAKTROUGH_TRIGGER_BRUSH = (150, 110, 230, 70)  # purple - peak/trough-centered triggers

PREDICTED_LINE_COLOR = "#39d6a8"   # dashed: raw predicted hemodynamic extremum
SCHEDULED_LINE_COLOR = "#f5c542"   # solid: live scheduled (clipped) fire time
NOMINAL_TICK_COLOR = "#5a6275"     # faint dotted: fixed nominal 10s grid
SHIFT_BAND_BRUSH = (245, 197, 66, 28)  # translucent band around nominal +/- max_shift_sec
PREDICTION_PAIR_BRUSH = (57, 214, 168, 30)


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class Viewer(QtWidgets.QMainWindow):
    def __init__(self, amp_name: str, marker_name: str, *, legacy_reactive: bool = False,
                 lead_sec: float = DEFAULT_LEAD_SEC, max_shift_sec: float = DEFAULT_MAX_SHIFT_SEC,
                 center_period_sec: float = DEFAULT_CENTER_PERIOD_SEC, period_lo_sec: float = DEFAULT_PERIOD_LO_SEC,
                 period_hi_sec: float = DEFAULT_PERIOD_HI_SEC, buffer_sec: float = DEFAULT_BUFFER_SEC,
                 pad_sec: float = DEFAULT_PAD_SEC, ar_order: int = DEFAULT_AR_ORDER,
                 freq_window_sec: float = DEFAULT_FREQ_WINDOW_SEC, recompute_hz: float = DEFAULT_RECOMPUTE_HZ,
                 target_phase: str | None = None, require_prediction: bool = False, n_bursts: int = 1,
                 burst_gap_sec: float = 0.0, channels: list[str] | None = None):
        if not legacy_reactive and target_phase is None:
            raise ValueError("target_phase='peak' or 'trough' is required (not 'either') unless legacy_reactive=True")
        super().__init__()
        self.setWindowTitle("fNIRS Real-Time Viewer  —  LSL")
        self.resize(1500, 900)

        self.amp_inlet: StreamInlet | None = None
        self.marker_inlet: StreamInlet | None = None
        self.pairs: list[ChannelPair] = []
        self.channels = channels if channels else [GRAND_AVERAGE]
        self.analysis_pairs: list[ChannelPair] = []
        self.analysis_channels_desc = ""
        self.n_channels = 0
        self.sfreq = 0.0
        self.baseline_ready = False
        self.baseline_deadline = None
        self.t0 = None
        self.n_samples_seen = 0
        self.selected: set[str] = set()
        self.recent_events: deque = deque(maxlen=8)

        self.ext = ExtinctionTable(EXTINCTION_CSV)

        # activity / peak-trough analysis (real-time analogue of the notebook)
        self.analyzer: ActivityAnalyzer | None = None
        self.activity_t: list[float] = []
        self.activity_v: list[float] = []
        self.phase_t: list[float] = []
        self.phase_v: list[float] = []
        self._last_phase_t: float | None = None
        self.display_window_sec = ACTIVITY_DISPLAY_WINDOW_SEC
        self.activity_items: list[tuple[float, object]] = []  # (t_end, graphics item) on activity_plot
        self.n_peaks = 0
        self.n_troughs = 0
        self.next_standard_trigger_t = 0.0
        self.was_above_threshold = False
        self.pending_threshold_start: float | None = None

        # adaptive phase-predictive trigger mirroring, for display only (no hardware fired here)
        self.legacy_reactive = legacy_reactive
        self.lead_sec = lead_sec
        self.max_shift_sec = max_shift_sec
        self.center_period_sec = center_period_sec
        self.period_lo_sec = period_lo_sec
        self.period_hi_sec = period_hi_sec
        self.buffer_sec = buffer_sec
        self.pad_sec = pad_sec
        self.ar_order = ar_order
        self.freq_window_sec = freq_window_sec
        self.recompute_hz = recompute_hz
        self.target_phase = target_phase
        self.require_prediction = require_prediction
        self.n_bursts = n_bursts
        self.burst_gap_sec = burst_gap_sec
        self.predictor: PhasePredictor | None = None
        self.controller: AdaptiveTriggerController | None = None
        self.last_pred: dict | None = None
        self._last_nominal_drawn: float | None = None

        self._build_ui()
        self._apply_theme()

        self.connect_timer = QtCore.QTimer(self)
        self.connect_timer.timeout.connect(lambda: self._try_connect(amp_name, marker_name))
        self.connect_timer.start(250)

        self.pull_timer = QtCore.QTimer(self)
        self.pull_timer.timeout.connect(self._on_tick)
        self.pull_timer.start(33)  # ~30 Hz UI refresh

        self._fps_last_t = time.perf_counter()
        self._fps_count = 0
        self._sample_rate_est = 0.0
        self._last_rate_check = (time.perf_counter(), 0)

    # ---------------------------------------------------------------- UI --
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        # --- status bar ---
        bar = QtWidgets.QHBoxLayout()
        self.status_lbl = QtWidgets.QLabel("● searching for LSL streams...")
        self.status_lbl.setObjectName("status")
        self.elapsed_lbl = QtWidgets.QLabel("--:--")
        self.rate_lbl = QtWidgets.QLabel("-- Hz")
        self.counts_lbl = QtWidgets.QLabel("peaks 0  ·  troughs 0")
        self.adaptive_lbl = QtWidgets.QLabel("")
        self.marker_lbl = QtWidgets.QLabel("")
        self.marker_lbl.setObjectName("marker")
        widgets = [self.status_lbl, self.elapsed_lbl, self.rate_lbl, self.counts_lbl]
        if not self.legacy_reactive:
            widgets.append(self.adaptive_lbl)
        for w in widgets:
            bar.addWidget(w)
            bar.addSpacing(18)
        bar.addStretch(1)
        bar.addWidget(self.marker_lbl)
        root.addLayout(bar)

        # --- main split: probe map | traces ---
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(split, 1)

        # probe map
        self.probe_glw = pg.GraphicsLayoutWidget()
        self.probe_vb = self.probe_glw.addViewBox(lockAspect=True)
        self.probe_vb.invertX(True)
        self.probe_vb.setMenuEnabled(False)
        split.addWidget(self.probe_glw)

        self.channel_lines = []  # pg.PlotDataItem per pair
        self.channel_scatter = None

        # right side: traces + activity/peak-detection panel + event log
        right = QtWidgets.QWidget()
        right_l = QtWidgets.QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        hint = QtWidgets.QLabel("Click channels on the probe map to plot them  ·  red = HbO, blue = HbR")
        hint.setObjectName("hint")
        trace_controls = QtWidgets.QHBoxLayout()
        trace_controls.addWidget(hint, 1)
        trace_controls.addWidget(QtWidgets.QLabel("Time span:"))
        self.time_span_spin = QtWidgets.QDoubleSpinBox()
        self.time_span_spin.setRange(5.0, MAX_DISPLAY_WINDOW_SEC)
        self.time_span_spin.setValue(self.display_window_sec)
        self.time_span_spin.setSuffix(" s")
        self.time_span_spin.setSingleStep(10.0)
        self.time_span_spin.setToolTip("Horizontal scale shared by all time-series plots")
        self.time_span_spin.valueChanged.connect(self._set_display_window)
        trace_controls.addWidget(self.time_span_spin)
        trace_controls.addWidget(QtWidgets.QLabel("mouse wheel over a plot = vertical scale"))
        right_l.addLayout(trace_controls)

        vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_l.addWidget(vsplit, 1)

        self.trace_plot = pg.PlotWidget()
        self.trace_plot.showGrid(x=True, y=False, alpha=0.25)
        self.trace_plot.setLabel("bottom", "time", units="s")
        self.trace_plot.getAxis("left").setStyle(showValues=False)
        self.trace_plot.getPlotItem().setMenuEnabled(False)
        vsplit.addWidget(self.trace_plot)

        activity_container = QtWidgets.QWidget()
        activity_l = QtWidgets.QVBoxLayout(activity_container)
        activity_l.setContentsMargins(0, 6, 0, 0)
        activity_l.setSpacing(2)
        activity_hint = QtWidgets.QLabel(
            "Activity = −mean HbO across channels  ·  ▲ peak  ▼ trough  "
            "(red = valid/high, blue = invalid/low)  ·  orange dashed = 75th-pct threshold  ·  "
            "gold/red/purple bands = standard / activity / peak-trough iTBS triggers"
        )
        activity_hint.setObjectName("hint")
        activity_hint.setWordWrap(True)
        activity_l.addWidget(activity_hint)

        self.activity_plot = pg.PlotWidget()
        self.activity_plot.showGrid(x=True, y=True, alpha=0.25)
        self.activity_plot.setLabel("bottom", "time", units="s")
        self.activity_plot.setLabel("left", "activity", units="µM")
        self.activity_plot.getPlotItem().setMenuEnabled(False)
        activity_l.addWidget(self.activity_plot, 1)
        vsplit.addWidget(activity_container)

        self.phase_plot = pg.PlotWidget()
        self.phase_plot.showGrid(x=True, y=True, alpha=0.25)
        self.phase_plot.setLabel("bottom", "time", units="s")
        self.phase_plot.setLabel("left", "estimated phase", units="rad")
        self.phase_plot.getAxis("left").setTicks(
            [[(-np.pi, "-π"), (-np.pi / 2, "-π/2"), (0, "0"), (np.pi / 2, "π/2"), (np.pi, "π")]]
        )
        self.phase_plot.setYRange(-np.pi, np.pi, padding=0.05)
        self.phase_plot.getPlotItem().setMenuEnabled(False)
        self.phase_plot.setVisible(not self.legacy_reactive)
        vsplit.addWidget(self.phase_plot)
        self.phase_curve = self.phase_plot.plot(pen=pg.mkPen(ACCENT, width=1.8))
        vsplit.setSizes([380, 340, 180])

        self.activity_curve = self.activity_plot.plot(pen=pg.mkPen(FG, width=2))
        self.threshold_line = pg.InfiniteLine(
            angle=0, pen=pg.mkPen("#e8b84b", width=1.5, style=QtCore.Qt.DashLine)
        )
        self.threshold_line.setVisible(False)
        self.activity_plot.addItem(self.threshold_line)

        # Adaptive timing pair: gold is the proposed TMS time, green is the
        # later predicted extremum, and the band makes their relationship explicit.
        self.pred_line = pg.InfiniteLine(
            angle=90, pen=pg.mkPen(PREDICTED_LINE_COLOR, width=1.5, style=QtCore.Qt.DashLine),
            label=f"predicted {self.target_phase or 'extremum'}",
            labelOpts={"position": 0.92, "color": PREDICTED_LINE_COLOR},
        )
        self.scheduled_line = pg.InfiniteLine(
            angle=90, pen=pg.mkPen(SCHEDULED_LINE_COLOR, width=2), label="TMS trigger",
            labelOpts={"position": 0.78, "color": SCHEDULED_LINE_COLOR},
        )
        self.prediction_pair_band = pg.LinearRegionItem(
            brush=pg.mkBrush(*PREDICTION_PAIR_BRUSH), movable=False
        )
        self.prediction_pair_band.setZValue(-24)
        self.shift_band = pg.LinearRegionItem(brush=pg.mkBrush(*SHIFT_BAND_BRUSH), movable=False)
        self.shift_band.setZValue(-25)
        for item in (self.shift_band, self.prediction_pair_band, self.pred_line, self.scheduled_line):
            item.setVisible(False)
            self.activity_plot.addItem(item)

        self.trace_pred_line = pg.InfiniteLine(
            angle=90, pen=pg.mkPen(PREDICTED_LINE_COLOR, width=1.5, style=QtCore.Qt.DashLine),
            label=f"predicted {self.target_phase or 'extremum'}",
            labelOpts={"position": 0.92, "color": PREDICTED_LINE_COLOR},
        )
        self.trace_scheduled_line = pg.InfiniteLine(
            angle=90, pen=pg.mkPen(SCHEDULED_LINE_COLOR, width=2), label="TMS trigger",
            labelOpts={"position": 0.78, "color": SCHEDULED_LINE_COLOR},
        )
        self.trace_prediction_pair_band = pg.LinearRegionItem(
            brush=pg.mkBrush(*PREDICTION_PAIR_BRUSH), movable=False
        )
        self.trace_prediction_pair_band.setZValue(-24)
        self._add_trace_adaptive_overlays()

        self.event_list = QtWidgets.QListWidget()
        self.event_list.setObjectName("events")
        self.event_list.setMaximumHeight(140)
        right_l.addWidget(QtWidgets.QLabel("Recent events"))
        right_l.addWidget(self.event_list, 1)

        split.addWidget(right)
        split.setSizes([560, 900])

        self.trace_curves: dict[str, dict[str, pg.PlotDataItem]] = {}
        self.marker_lines: list[pg.InfiniteLine] = []

    def _set_display_window(self, seconds: float):
        """Persist the requested horizontal scale across live redraws."""
        self.display_window_sec = float(seconds)

    def _apply_theme(self):
        pg.setConfigOption("background", BG)
        pg.setConfigOption("foreground", FG)
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background-color: {BG}; color: {FG}; font-family: 'Segoe UI', sans-serif; }}
            QLabel#status {{ font-size: 13px; font-weight: 600; color: {ACCENT}; }}
            QLabel#marker {{ font-size: 13px; font-weight: 600; color: {HBO_COLOR}; }}
            QLabel#hint {{ color: {MUTED}; font-size: 11px; padding: 2px 0 4px 2px; }}
            QListWidget#events {{ background-color: #10131a; border: 1px solid {GRID}; font-size: 11px; }}
            QSplitter::handle {{ background-color: {GRID}; }}
        """)

    # --------------------------------------------------------- connection --
    def _try_connect(self, amp_name, marker_name):
        if self.amp_inlet is not None and self.marker_inlet is not None:
            self.connect_timer.stop()
            return

        if self.amp_inlet is None:
            found = resolve_streams(timeout=0.2, name=amp_name)
            if found:
                self.amp_inlet = StreamInlet(found[0])
                self.amp_inlet.open_stream()
                sinfo = self.amp_inlet.get_sinfo()
                self._setup_channels(sinfo)
                self.sfreq = sinfo.sfreq
                self.analyzer = ActivityAnalyzer(self.sfreq)
                if not self.legacy_reactive:
                    self.predictor = PhasePredictor(
                        self.sfreq, center_period_sec=self.center_period_sec, period_lo_sec=self.period_lo_sec,
                        period_hi_sec=self.period_hi_sec, buffer_sec=self.buffer_sec, pad_sec=self.pad_sec,
                        ar_order=self.ar_order, freq_window_sec=self.freq_window_sec,
                        recompute_hz=self.recompute_hz, target_phase=self.target_phase,
                    )
                    self.controller = AdaptiveTriggerController(
                        STANDARD_BURST_DURATION_SEC, STANDARD_BURST_INTERVAL_SEC - STANDARD_BURST_DURATION_SEC,
                        STANDARD_BURST_INTERVAL_SEC, self.lead_sec, self.max_shift_sec, self._on_adaptive_fire,
                        require_prediction=self.require_prediction, n_bursts=self.n_bursts,
                        burst_gap_sec=self.burst_gap_sec, target_phase=self.target_phase,
                    )
                self.status_lbl.setText(
                    f"● connected — {sinfo.n_channels} ch @ {self.sfreq:.2f} Hz — analysis: {self.analysis_channels_desc}"
                )

        if self.marker_inlet is None:
            found = resolve_streams(timeout=0.2, name=marker_name)
            if found:
                self.marker_inlet = StreamInlet(found[0])
                self.marker_inlet.open_stream()

    def _setup_channels(self, sinfo):
        metas = parse_channels(sinfo)
        self.n_channels = len(metas)

        by_pair: dict[tuple, dict[float, int]] = {}
        pos_by_pair = {}
        for m in metas:
            key = (m.source, m.detector)
            by_pair.setdefault(key, {})[m.wavelength] = m.index
            pos_by_pair[key] = (m.src_pos, m.det_pos, m.distance_m)

        self.pairs = []
        for (src, det), idx_by_wl in by_pair.items():
            src_pos, det_pos, dist = pos_by_pair[(src, det)]
            pair = ChannelPair(src, det, idx_by_wl, src_pos, det_pos, dist)
            pair.set_extinction(self.ext)
            self.pairs.append(pair)

        try:
            self.analysis_pairs = select_pairs(self.pairs, self.channels)
        except ValueError as exc:
            raise SystemExit(str(exc))
        if len(self.analysis_pairs) == len(self.pairs):
            self.analysis_channels_desc = f"grand avg ({len(self.pairs)} ch)"
        elif len(self.analysis_pairs) == 1:
            self.analysis_channels_desc = self.analysis_pairs[0].label
        else:
            self.analysis_channels_desc = "avg(" + ",".join(p.label for p in self.analysis_pairs) + ")"
        print(f"Analysis channels: {self.analysis_channels_desc}")

        self.baseline_deadline = None  # set once first sample arrives
        self._draw_probe_map(metas)

    def _draw_probe_map(self, metas):
        self.probe_vb.clear()

        srcs = {m.source: m.src_pos for m in metas}
        dets = {m.detector: m.det_pos for m in metas}

        self.channel_lines = []
        for pair in self.pairs:
            x = [pair.src_pos[0], pair.det_pos[0]]
            y = [pair.src_pos[2], pair.det_pos[2]]  # use x/z (left-right / up-down) as flattened probe view
            line = pg.PlotDataItem(x, y, pen=pg.mkPen(MUTED, width=3))
            self.probe_vb.addItem(line)
            self.channel_lines.append((pair, line))

        src_xy = np.array([[p[0], p[2]] for p in srcs.values()])
        det_xy = np.array([[p[0], p[2]] for p in dets.values()])

        src_scatter = pg.ScatterPlotItem(
            pos=src_xy, symbol="s", size=16,
            brush=pg.mkBrush("#e8b84b"), pen=pg.mkPen(BG, width=1),
        )
        det_scatter = pg.ScatterPlotItem(
            pos=det_xy, symbol="o", size=14,
            brush=pg.mkBrush("#5c8fe8"), pen=pg.mkPen(BG, width=1),
        )
        self.probe_vb.addItem(src_scatter)
        self.probe_vb.addItem(det_scatter)

        for name, pos in srcs.items():
            t = pg.TextItem(name, color=MUTED, anchor=(0.5, 1.3))
            t.setPos(pos[0], pos[2])
            self.probe_vb.addItem(t)
        for name, pos in dets.items():
            t = pg.TextItem(name, color=MUTED, anchor=(0.5, -0.3))
            t.setPos(pos[0], pos[2])
            self.probe_vb.addItem(t)

        # clickable midpoints to select channels for the trace panel
        mids = np.array([p.mid[[0, 2]] for p in self.pairs])
        self.channel_scatter = pg.ScatterPlotItem(
            pos=mids, size=14, symbol="d",
            brush=pg.mkBrush(MUTED), pen=pg.mkPen(BG, width=1),
            hoverable=True, tip=lambda x, y, data: data,
        )
        self.channel_scatter.setData(pos=mids, data=[p.label for p in self.pairs])
        self.channel_scatter.sigClicked.connect(self._on_channel_clicked)
        self.probe_vb.addItem(self.channel_scatter)
        self.probe_vb.enableAutoRange()

    def _on_channel_clicked(self, scatter, points):
        if not points:
            return
        label = points[0].data()
        if label in self.selected:
            self.selected.discard(label)
        else:
            self.selected.add(label)
        self._rebuild_trace_curves()

    def _rebuild_trace_curves(self):
        self.trace_plot.clear()
        self._add_trace_adaptive_overlays()
        self.trace_curves = {}
        self.marker_lines = []
        n = len(self.selected)
        for i, label in enumerate(sorted(self.selected)):
            offset = (n - 1 - i) * 12.0  # stack top-to-bottom
            hbo = self.trace_plot.plot(pen=pg.mkPen(HBO_COLOR, width=1.6))
            hbr = self.trace_plot.plot(pen=pg.mkPen(HBR_COLOR, width=1.6))
            txt = pg.TextItem(label, color=FG, anchor=(0, 0.5))
            txt.setPos(0, offset + 3)
            self.trace_plot.addItem(txt)
            self.trace_curves[label] = {"hbo": hbo, "hbr": hbr, "offset": offset, "label_item": txt}

    def _add_trace_adaptive_overlays(self):
        for item in (self.trace_prediction_pair_band, self.trace_pred_line, self.trace_scheduled_line):
            item.setVisible(False)
            self.trace_plot.addItem(item)

    # -------------------------------------------------------------- tick --
    def _on_tick(self):
        self._pull_markers()
        self._pull_amplitude()
        self._update_status()

    def _pull_markers(self):
        if self.marker_inlet is None:
            return
        samples, timestamps = self.marker_inlet.pull_chunk(timeout=0.0, max_samples=64)
        for s, ts in zip(samples, timestamps):
            desc = s[0]
            self.recent_events.append(desc)
            self.marker_lbl.setText(f"▶ event {desc}")
            self.event_list.insertItem(0, f"{time.strftime('%H:%M:%S')}   event {desc}")
            while self.event_list.count() > 50:
                self.event_list.takeItem(self.event_list.count() - 1)
            if self.n_samples_seen:
                t = self.n_samples_seen / self.sfreq
                line = pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen(ACCENT, width=1, style=QtCore.Qt.DashLine))
                self.trace_plot.addItem(line)
                self.marker_lines.append((t, line))

    def _pull_amplitude(self):
        if self.amp_inlet is None:
            return
        samples, timestamps = self.amp_inlet.pull_chunk(timeout=0.0, max_samples=2048)
        if len(samples) == 0:
            return

        if self.t0 is None:
            self.t0 = timestamps[0]
            self.baseline_deadline = BASELINE_SEC

        # recomputed once per tick (not per sample): np.percentile over a ~300s buffer
        # is too costly to run at every incoming sample, especially at high playback speeds.
        threshold = self.analyzer.percentile_threshold() if self.analyzer else None

        for sample in samples:
            self.n_samples_seen += 1
            t = self.n_samples_seen / self.sfreq if self.sfreq else 0.0

            if not self.baseline_ready:
                for pair in self.pairs:
                    pair.add_baseline_sample(sample)
                if t >= self.baseline_deadline:
                    for pair in self.pairs:
                        pair.finalize_baseline()
                    self.baseline_ready = True
                    if self.controller is not None:
                        self.controller.start(t)
                continue

            for pair in self.pairs:
                dhbo, dhbr = pair.update(sample)
                if pair.label not in self.trace_curves:
                    continue
                store = self.trace_curves[pair.label].setdefault("_buf", ([], [], []))
                tbuf, hbo_buf, hbr_buf = store
                tbuf.append(t)
                hbo_buf.append(dhbo + self.trace_curves[pair.label]["offset"])
                hbr_buf.append(dhbr + self.trace_curves[pair.label]["offset"])

            activity_val = -float(np.mean([p.last_hbo for p in self.analysis_pairs]))
            self.activity_t.append(t)
            self.activity_v.append(activity_val)
            self._update_standard_triggers(t)
            if threshold is not None:
                self._update_threshold_region(t, activity_val, threshold)
            for ev in self.analyzer.push(t, activity_val):
                self._handle_activity_event(ev)

            if self.predictor is not None:
                self.last_pred = self.predictor.update(t, activity_val)
                if self.last_pred is not None and self.last_pred["t_now"] != self._last_phase_t:
                    self._last_phase_t = self.last_pred["t_now"]
                    self.phase_t.append(self.last_pred["t_now"])
                    self.phase_v.append(self.last_pred["phase_now"])
                if self.controller is not None:
                    self.controller.step(t, self.last_pred)

        # trim to window and redraw
        if self.trace_curves:
            t_now = self.n_samples_seen / self.sfreq if self.sfreq else 0.0
            t_min = t_now - self.display_window_sec
            t_keep = t_now - MAX_DISPLAY_WINDOW_SEC
            for label, curves in self.trace_curves.items():
                store = curves.get("_buf")
                if not store:
                    continue
                tbuf, hbo_buf, hbr_buf = store
                while tbuf and tbuf[0] < t_keep:
                    tbuf.pop(0); hbo_buf.pop(0); hbr_buf.pop(0)
                curves["hbo"].setData(tbuf, hbo_buf)
                curves["hbr"].setData(tbuf, hbr_buf)
                curves["label_item"].setPos(max(t_min, tbuf[0] if tbuf else t_min), curves["offset"] + 3)
            self.trace_plot.setXRange(t_min, t_now, padding=0)

            for t_ev, line in list(self.marker_lines):
                if t_ev < t_keep:
                    self.trace_plot.removeItem(line)
                    self.marker_lines.remove((t_ev, line))

        if self.activity_t:
            t_now = self.n_samples_seen / self.sfreq if self.sfreq else 0.0
            t_min_a = t_now - self.display_window_sec
            t_keep_a = t_now - MAX_DISPLAY_WINDOW_SEC
            while self.activity_t and self.activity_t[0] < t_keep_a:
                self.activity_t.pop(0)
                self.activity_v.pop(0)
            self.activity_curve.setData(self.activity_t, self.activity_v)
            self.activity_plot.setXRange(t_min_a, t_now, padding=0)
            self._trim_activity_items(t_keep_a)
            while self.phase_t and self.phase_t[0] < t_keep_a:
                self.phase_t.pop(0)
                self.phase_v.pop(0)
            self.phase_curve.setData(self.phase_t, self.phase_v, connect="finite")
            self.phase_plot.setXRange(t_min_a, t_now, padding=0)
            if threshold is not None:
                self.threshold_line.setPos(threshold)
                self.threshold_line.setVisible(True)
            if self.controller is not None:
                self._update_adaptive_overlays()

        self._update_probe_colors()

    def _update_probe_colors(self):
        if not self.baseline_ready:
            return
        cmap = pg.colormap.get("CET-D1")
        for pair, line in self.channel_lines:
            norm = float(np.clip(pair.last_hbo / 2.0 + 0.5, 0.0, 1.0))
            color = cmap.mapToQColor(norm)
            line.setPen(pg.mkPen(color, width=3))

    # --------------------------------------------- activity / peak-trough --
    def _register_activity_item(self, t_end: float, item):
        self.activity_plot.addItem(item)
        self.activity_items.append((t_end, item))

    def _trim_activity_items(self, t_min: float):
        keep = []
        for t_end, item in self.activity_items:
            if t_end < t_min:
                self.activity_plot.removeItem(item)
            else:
                keep.append((t_end, item))
        self.activity_items = keep

    def _update_standard_triggers(self, t: float):
        while self.next_standard_trigger_t <= t:
            start = self.next_standard_trigger_t
            end = start + STANDARD_BURST_DURATION_SEC
            region = pg.LinearRegionItem(values=[start, end], brush=pg.mkBrush(*STANDARD_TRIGGER_BRUSH), movable=False)
            region.setZValue(-20)
            self._register_activity_item(end, region)
            self.next_standard_trigger_t += STANDARD_BURST_INTERVAL_SEC

    def _update_threshold_region(self, t: float, v: float, thr: float):
        above = v > thr
        if above and not self.was_above_threshold:
            self.pending_threshold_start = t
        if (not above) and self.was_above_threshold and self.pending_threshold_start is not None:
            region = pg.LinearRegionItem(
                values=[self.pending_threshold_start, t], brush=pg.mkBrush(*ACTIVITY_TRIGGER_BRUSH), movable=False
            )
            region.setZValue(-15)
            self._register_activity_item(t, region)
            self.pending_threshold_start = None
        self.was_above_threshold = above

    def _handle_activity_event(self, ev: dict):
        kind, valid, t, v = ev["kind"], ev["valid"], ev["t"], ev["v"]
        color = HBO_COLOR if v > 0 else HBR_COLOR
        symbol = "t1" if kind == "peak" else "t"  # pyqtgraph renders 't1' pointing up, 't' pointing down

        marker = pg.ScatterPlotItem(
            pos=[(t, v)], size=16, symbol=symbol, brush=pg.mkBrush(color), pen=pg.mkPen(BG, width=1)
        )
        self._register_activity_item(t, marker)

        region = pg.LinearRegionItem(
            values=[t - PEAK_TRIGGER_HALF_WIDTH_SEC, t + PEAK_TRIGGER_HALF_WIDTH_SEC],
            brush=pg.mkBrush(*PEAKTROUGH_TRIGGER_BRUSH), movable=False,
        )
        region.setZValue(-10)
        self._register_activity_item(t + PEAK_TRIGGER_HALF_WIDTH_SEC, region)

        if kind == "peak":
            self.n_peaks += 1
        else:
            self.n_troughs += 1
        self.counts_lbl.setText(f"peaks {self.n_peaks}  ·  troughs {self.n_troughs}")

        arrow = "▲" if kind == "peak" else "▼"
        tag = "valid" if valid else "invalid"
        self.event_list.insertItem(0, f"{time.strftime('%H:%M:%S')}   {arrow} {kind} ({tag}) @ {t:0.1f}s")
        while self.event_list.count() > 50:
            self.event_list.takeItem(self.event_list.count() - 1)

    def _on_adaptive_fire(self, t: float, info: dict):
        """Display-only mirror of what AdaptiveTriggerController would fire on
        in itbs_trigger.py -- no hardware is touched from the viewer."""
        pred = info["prediction"]
        tag = "fallback, still warming up" if pred is None else f"{pred['predicted_kind']}, shift {info['shift']:+.1f}s"
        burst_tag = f" [{info['burst_index']}/{info['n_bursts']}]" if info["n_bursts"] > 1 else ""
        self.event_list.insertItem(0, f"{time.strftime('%H:%M:%S')}   ⚡ ADAPTIVE FIRE{burst_tag} ({tag}) @ {t:0.1f}s")
        while self.event_list.count() > 50:
            self.event_list.takeItem(self.event_list.count() - 1)
    def _update_adaptive_overlays(self):
        ctrl = self.controller
        pred = self.last_pred

        if pred is not None:
            self.pred_line.setPos(pred["predicted_t"])
            self.pred_line.setVisible(True)
            self.trace_pred_line.setPos(pred["predicted_t"])
            self.trace_pred_line.setVisible(True)
            period_str = f"{1.0 / pred['freq_inst_hz']:.1f}s"
        else:
            self.pred_line.setVisible(False)
            self.trace_pred_line.setVisible(False)
            period_str = "warming up"

        shift = 0.0
        if ctrl.nominal_t is not None:
            self.shift_band.setRegion((ctrl.nominal_t - self.max_shift_sec, ctrl.nominal_t + self.max_shift_sec))
            self.shift_band.setVisible(True)
            if ctrl.scheduled_t is not None:
                self.scheduled_line.setPos(ctrl.scheduled_t)
                self.scheduled_line.setVisible(True)
                self.trace_scheduled_line.setPos(ctrl.scheduled_t)
                self.trace_scheduled_line.setVisible(True)
                if pred is not None:
                    pair = (ctrl.scheduled_t, pred["predicted_t"])
                    self.prediction_pair_band.setRegion(pair)
                    self.prediction_pair_band.setVisible(True)
                    self.trace_prediction_pair_band.setRegion(pair)
                    self.trace_prediction_pair_band.setVisible(True)
                else:
                    self.prediction_pair_band.setVisible(False)
                    self.trace_prediction_pair_band.setVisible(False)
                shift = ctrl.scheduled_t - ctrl.nominal_t
            else:
                self.scheduled_line.setVisible(False)  # require_prediction: this slot has nothing scheduled yet
                self.trace_scheduled_line.setVisible(False)
                self.prediction_pair_band.setVisible(False)
                self.trace_prediction_pair_band.setVisible(False)
            if self._last_nominal_drawn != ctrl.nominal_t:
                tick = pg.InfiniteLine(
                    pos=ctrl.nominal_t, angle=90, pen=pg.mkPen(NOMINAL_TICK_COLOR, width=1, style=QtCore.Qt.DotLine)
                )
                self._register_activity_item(ctrl.nominal_t + self.max_shift_sec, tick)
                self._last_nominal_drawn = ctrl.nominal_t
        else:
            self.shift_band.setVisible(False)
            self.scheduled_line.setVisible(False)
            self.trace_scheduled_line.setVisible(False)
            self.prediction_pair_band.setVisible(False)
            self.trace_prediction_pair_band.setVisible(False)

        self.adaptive_lbl.setText(f"period~{period_str}  ·  shift {shift:+.1f}s (±{self.max_shift_sec:.1f}s)")

    def _update_status(self):
        if self.amp_inlet is None:
            self.status_lbl.setText("● searching for LSL streams...")
            return
        if not self.baseline_ready:
            remaining = max(0.0, (self.baseline_deadline or BASELINE_SEC) - self.n_samples_seen / max(self.sfreq, 1e-9))
            self.status_lbl.setText(f"● connected — collecting baseline ({remaining:0.1f}s)...")
        else:
            self.status_lbl.setText(f"● connected — live")

        t_now = self.n_samples_seen / self.sfreq if self.sfreq else 0.0
        m, s = divmod(int(t_now), 60)
        h, m = divmod(m, 60)
        self.elapsed_lbl.setText(f"{h:02d}:{m:02d}:{s:02d}")

        now, count0 = self._last_rate_check
        elapsed = time.perf_counter() - now
        if elapsed >= 1.0:
            rate = (self.n_samples_seen - count0) / elapsed
            self._sample_rate_est = rate
            self._last_rate_check = (time.perf_counter(), self.n_samples_seen)
        self.rate_lbl.setText(f"{self._sample_rate_est:0.1f} Hz")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--amp-name", default="SNIRF_Amplitude")
    p.add_argument("--marker-name", default="SNIRF_Markers")
    p.add_argument("--channels", nargs="+", default=[GRAND_AVERAGE], metavar="LABEL",
                    help="which source-detector pair(s) feed activity_val. One label (e.g. S1_D2) -> that "
                         "channel only; several labels -> their average; 'grandaverage' (default) -> average "
                         "across every available pair. Mirrors itbs_trigger.py's --channels.")
    p.add_argument("--legacy-reactive", action="store_true",
                    help="don't mirror the adaptive phase-predictive overlays -- just show the old peak/trough "
                         "detector and trigger-window strategies, matching --legacy-reactive in itbs_trigger.py")
    p.add_argument("--lead-sec", type=float, default=DEFAULT_LEAD_SEC)
    p.add_argument("--max-shift-sec", type=float, default=DEFAULT_MAX_SHIFT_SEC)
    p.add_argument("--center-period-sec", type=float, default=DEFAULT_CENTER_PERIOD_SEC)
    p.add_argument("--period-lo-sec", type=float, default=DEFAULT_PERIOD_LO_SEC)
    p.add_argument("--period-hi-sec", type=float, default=DEFAULT_PERIOD_HI_SEC)
    p.add_argument("--buffer-sec", type=float, default=DEFAULT_BUFFER_SEC)
    p.add_argument("--pad-sec", type=float, default=DEFAULT_PAD_SEC)
    p.add_argument("--ar-order", type=int, default=DEFAULT_AR_ORDER)
    p.add_argument("--freq-window-sec", type=float, default=DEFAULT_FREQ_WINDOW_SEC)
    p.add_argument("--recompute-hz", type=float, default=DEFAULT_RECOMPUTE_HZ)
    p.add_argument("--target-phase", choices=("peak", "trough"), default=None,
                    help="which extremum to commit to targeting for the whole run. Required unless "
                         "--legacy-reactive is used -- 'either' is not supported, mirroring itbs_trigger.py "
                         "(its true scheduling offset is too large to recalibrate away).")
    p.add_argument("--require-prediction", action="store_true",
                    help="mirror itbs_trigger.py's no-fallback mode: nothing scheduled/fired without a real "
                         "prediction.")
    p.add_argument("--n-bursts", type=int, default=1,
                    help="mirror itbs_trigger.py's multi-burst grouping (default: 1)")
    p.add_argument("--burst-gap-sec", type=float, default=0.0)
    args = p.parse_args()

    if not args.legacy_reactive and args.target_phase is None:
        raise SystemExit("--target-phase peak or trough is required (not the removed 'either') unless --legacy-reactive is used")

    app = QtWidgets.QApplication(sys.argv)
    win = Viewer(
        args.amp_name, args.marker_name, legacy_reactive=args.legacy_reactive, lead_sec=args.lead_sec,
        max_shift_sec=args.max_shift_sec, center_period_sec=args.center_period_sec,
        period_lo_sec=args.period_lo_sec, period_hi_sec=args.period_hi_sec, buffer_sec=args.buffer_sec,
        pad_sec=args.pad_sec, ar_order=args.ar_order, freq_window_sec=args.freq_window_sec,
        recompute_hz=args.recompute_hz, target_phase=args.target_phase,
        require_prediction=args.require_prediction, n_bursts=args.n_bursts, burst_gap_sec=args.burst_gap_sec,
        channels=args.channels,
    )
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
