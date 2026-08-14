"""Slick real-time viewer for an LSL fNIRS stream produced by stream_snirf_lsl.py.

Connects to the 'SNIRF_Amplitude' and 'SNIRF_Markers' LSL streams, converts
raw dual-wavelength intensity to HbO/HbR concentration changes on the fly
(modified Beer-Lambert law, using the OMLC/Prahl hemoglobin extinction
spectrum bundled as extinction_coef_600_1000nm.csv, same source table used by
MNE-Python), and displays:

  - a live probe map (sources/detectors/channels) colour-coded by current
    HbO signal, click a channel to add/remove it from the scrolling traces
  - a scrolling multi-channel HbO/HbR time series with TMS/iTBS event markers
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
    EXTINCTION_CSV,
    PEAK_TRIGGER_HALF_WIDTH_SEC,
    STANDARD_BURST_DURATION_SEC,
    STANDARD_BURST_INTERVAL_SEC,
    ActivityAnalyzer,
    ChannelPair,
    ExtinctionTable,
    parse_channels,
)

WINDOW_SEC = 30.0
ACTIVITY_DISPLAY_WINDOW_SEC = 60.0
FLASH_MS = 550

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


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class Viewer(QtWidgets.QMainWindow):
    def __init__(self, amp_name: str, marker_name: str):
        super().__init__()
        self.setWindowTitle("fNIRS Real-Time Viewer  —  LSL")
        self.resize(1500, 900)

        self.amp_inlet: StreamInlet | None = None
        self.marker_inlet: StreamInlet | None = None
        self.pairs: list[ChannelPair] = []
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
        self.activity_items: list[tuple[float, object]] = []  # (t_end, graphics item) on activity_plot
        self.n_peaks = 0
        self.n_troughs = 0
        self.next_standard_trigger_t = 0.0
        self.was_above_threshold = False
        self.pending_threshold_start: float | None = None

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
        self.marker_lbl = QtWidgets.QLabel("")
        self.marker_lbl.setObjectName("marker")
        for w in (self.status_lbl, self.elapsed_lbl, self.rate_lbl, self.counts_lbl):
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
        right_l.addWidget(hint)

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
        vsplit.setSizes([420, 420])

        self.activity_curve = self.activity_plot.plot(pen=pg.mkPen(FG, width=2))
        self.threshold_line = pg.InfiniteLine(
            angle=0, pen=pg.mkPen("#e8b84b", width=1.5, style=QtCore.Qt.DashLine)
        )
        self.threshold_line.setVisible(False)
        self.activity_plot.addItem(self.threshold_line)

        # big flash overlay, shown briefly the instant a peak/trough is confirmed
        self.flash_label = QtWidgets.QLabel(self.activity_plot)
        self.flash_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.flash_label.hide()

        self.event_list = QtWidgets.QListWidget()
        self.event_list.setObjectName("events")
        self.event_list.setMaximumHeight(140)
        right_l.addWidget(QtWidgets.QLabel("Recent events"))
        right_l.addWidget(self.event_list, 1)

        split.addWidget(right)
        split.setSizes([560, 900])

        self.trace_curves: dict[str, dict[str, pg.PlotDataItem]] = {}
        self.marker_lines: list[pg.InfiniteLine] = []

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
                self.status_lbl.setText(f"● connected — {sinfo.n_channels} ch @ {self.sfreq:.2f} Hz")

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

            activity_val = -float(np.mean([p.last_hbo for p in self.pairs]))
            self.activity_t.append(t)
            self.activity_v.append(activity_val)
            self._update_standard_triggers(t)
            if threshold is not None:
                self._update_threshold_region(t, activity_val, threshold)
            for ev in self.analyzer.push(t, activity_val):
                self._handle_activity_event(ev)

        # trim to window and redraw
        if self.trace_curves:
            t_now = self.n_samples_seen / self.sfreq if self.sfreq else 0.0
            t_min = t_now - WINDOW_SEC
            for label, curves in self.trace_curves.items():
                store = curves.get("_buf")
                if not store:
                    continue
                tbuf, hbo_buf, hbr_buf = store
                while tbuf and tbuf[0] < t_min:
                    tbuf.pop(0); hbo_buf.pop(0); hbr_buf.pop(0)
                curves["hbo"].setData(tbuf, hbo_buf)
                curves["hbr"].setData(tbuf, hbr_buf)
                curves["label_item"].setPos(max(t_min, tbuf[0] if tbuf else t_min), curves["offset"] + 3)
            self.trace_plot.setXRange(t_min, t_now, padding=0)

            for t_ev, line in list(self.marker_lines):
                if t_ev < t_min:
                    self.trace_plot.removeItem(line)
                    self.marker_lines.remove((t_ev, line))

        if self.activity_t:
            t_now = self.n_samples_seen / self.sfreq if self.sfreq else 0.0
            t_min_a = t_now - ACTIVITY_DISPLAY_WINDOW_SEC
            while self.activity_t and self.activity_t[0] < t_min_a:
                self.activity_t.pop(0)
                self.activity_v.pop(0)
            self.activity_curve.setData(self.activity_t, self.activity_v)
            self.activity_plot.setXRange(t_min_a, t_now, padding=0)
            self._trim_activity_items(t_min_a)
            if threshold is not None:
                self.threshold_line.setPos(threshold)
                self.threshold_line.setVisible(True)

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

        self._flash(kind, color)

    def _flash(self, kind: str, color: str):
        self.activity_plot.setBackground(color)
        QtCore.QTimer.singleShot(FLASH_MS, lambda: self.activity_plot.setBackground(BG))

        text_color = BG if kind == "peak" else "#f5f7fa"
        self.flash_label.setText(("▲ PEAK DETECTED" if kind == "peak" else "▼ TROUGH DETECTED"))
        self.flash_label.setStyleSheet(
            f"background: transparent; color: {text_color}; font-size: 28px; font-weight: 800; letter-spacing: 2px;"
        )
        self.flash_label.adjustSize()
        pw, ph = self.activity_plot.width(), self.activity_plot.height()
        self.flash_label.move((pw - self.flash_label.width()) // 2, (ph - self.flash_label.height()) // 2)
        self.flash_label.show()
        self.flash_label.raise_()
        QtCore.QTimer.singleShot(FLASH_MS, self.flash_label.hide)

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
    args = p.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = Viewer(args.amp_name, args.marker_name)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
