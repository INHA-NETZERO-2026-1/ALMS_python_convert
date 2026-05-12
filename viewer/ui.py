"""ALMS BIN File Viewer — UI 레이어.
module/ 와 scipy 를 직접 import 하지 않는다. 모든 데이터 작업은
viewer.workers (백그라운드) 또는 viewer.services (동기) 를 거친다."""

import sys
import os
import tempfile
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QGroupBox,
    QTextEdit, QSpinBox, QDoubleSpinBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QStatusBar, QGridLayout, QMessageBox, QProgressBar,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from viewer.widgets import MplCanvas
from viewer.workers import ParseWorker, STFTWorker
from viewer import services, plots


def _ch_display_name(ch, idx: int) -> str:
    name = (getattr(ch, "name", None) or getattr(ch, "ch_name", None) or f"CH{idx}")
    return name.strip() or f"CH{idx}"


class ALMSViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.alms_data = None
        self.stft_results = []
        self.current_ch = 0
        self._tmp_dir = tempfile.mkdtemp(prefix="alms_viewer_")

        self._setup_ui()
        self._apply_dark_theme()

    # ── UI 구성 ───────────────────────────────────────────────────

    def _setup_ui(self):
        self.setWindowTitle("ALMS BIN File Viewer  |  (주)리얼게인 NIMS")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)

        main_layout.addWidget(self._build_left_panel(), stretch=0)
        main_layout.addWidget(self._build_right_panel(), stretch=1)

        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("BIN 파일을 열어주세요.")

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(290)
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        # 파일 그룹
        file_group = QGroupBox("파일")
        fl = QVBoxLayout(file_group)

        self.btn_open = QPushButton("📂  BIN 파일 열기")
        self.btn_open.setFixedHeight(40)
        self.btn_open.clicked.connect(self._open_file)
        fl.addWidget(self.btn_open)

        self.btn_export_csv = QPushButton("💾  CSV 내보내기")
        self.btn_export_csv.setFixedHeight(36)
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.clicked.connect(self._export_csv)
        fl.addWidget(self.btn_export_csv)

        self.lbl_filename = QLabel("파일 없음")
        self.lbl_filename.setWordWrap(True)
        self.lbl_filename.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        fl.addWidget(self.lbl_filename)
        layout.addWidget(file_group)

        # 헤더 정보
        info_group = QGroupBox("파일 정보")
        info_layout = QGridLayout(info_group)
        info_layout.setSpacing(4)

        self._info_labels = {}
        rows = [
            ("SITE",          "site"),
            ("이벤트 채널",   "event_ch"),
            ("총 채널수",     "total_ch"),
            ("샘플링 레이트", "sr"),
            ("이벤트 시간",   "duration"),
            ("알람 상태",     "alarm"),
            ("신호 종류",     "sig_type"),
            ("이벤트 날짜",   "date"),
        ]
        for i, (label, key) in enumerate(rows):
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
            val = QLabel("-")
            val.setStyleSheet("color: #cdd6f4; font-size: 11px; font-weight: bold;")
            info_layout.addWidget(lbl, i, 0)
            info_layout.addWidget(val, i, 1)
            self._info_labels[key] = val
        layout.addWidget(info_group)

        # 채널 선택
        ch_group = QGroupBox("채널 선택")
        ch_layout = QVBoxLayout(ch_group)
        self.combo_ch = QComboBox()
        self.combo_ch.currentIndexChanged.connect(self._on_channel_changed)
        ch_layout.addWidget(self.combo_ch)
        layout.addWidget(ch_group)

        # STFT 파라미터
        stft_group = QGroupBox("STFT 파라미터")
        sg = QGridLayout(stft_group)
        sg.setSpacing(6)

        sg.addWidget(QLabel("Window 크기"), 0, 0)
        self.spin_window = QComboBox()
        self.spin_window.addItems(["256", "512", "1024", "2048", "4096"])
        self.spin_window.setCurrentText("1024")
        sg.addWidget(self.spin_window, 0, 1)

        sg.addWidget(QLabel("Overlap (%)"), 1, 0)
        self.spin_overlap = QSpinBox()
        self.spin_overlap.setRange(0, 95)
        self.spin_overlap.setValue(75)
        self.spin_overlap.setSingleStep(5)
        sg.addWidget(self.spin_overlap, 1, 1)

        sg.addWidget(QLabel("Window 함수"), 2, 0)
        self.combo_window = QComboBox()
        self.combo_window.addItems(["hann", "hamming", "blackman", "bartlett", "boxcar"])
        sg.addWidget(self.combo_window, 2, 1)

        sg.addWidget(QLabel("주파수 상한 (kHz)"), 3, 0)
        self.spin_fmax = QSpinBox()
        self.spin_fmax.setRange(10, 2000)
        self.spin_fmax.setValue(600)
        sg.addWidget(self.spin_fmax, 3, 1)

        self.btn_analyze = QPushButton("▶  STFT 분석 실행")
        self.btn_analyze.setFixedHeight(38)
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self._run_stft_interactive)
        sg.addWidget(self.btn_analyze, 4, 0, 1, 2)

        self.btn_all_stft = QPushButton("⚡  전채널 STFT + 피처 추출")
        self.btn_all_stft.setFixedHeight(38)
        self.btn_all_stft.setEnabled(False)
        self.btn_all_stft.clicked.connect(self._run_all_stft)
        sg.addWidget(self.btn_all_stft, 5, 0, 1, 2)

        layout.addWidget(stft_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)
        layout.addWidget(self.progress_bar)

        layout.addStretch()
        return panel

    def _build_right_panel(self) -> QWidget:
        self.tabs = QTabWidget()

        tab_raw = QWidget()
        raw_layout = QVBoxLayout(tab_raw)
        self.canvas_raw = MplCanvas(width=12, height=5)
        raw_layout.addWidget(NavigationToolbar(self.canvas_raw, tab_raw))
        raw_layout.addWidget(self.canvas_raw)
        self.tabs.addTab(tab_raw, "📈  Raw Signal")

        tab_stft = QWidget()
        stft_layout = QVBoxLayout(tab_stft)
        self.canvas_stft = MplCanvas(width=12, height=8)
        stft_layout.addWidget(NavigationToolbar(self.canvas_stft, tab_stft))
        stft_layout.addWidget(self.canvas_stft)
        self.tabs.addTab(tab_stft, "🔥  STFT Spectrogram")

        tab_features = QWidget()
        feat_layout = QVBoxLayout(tab_features)

        self.table_features = QTableWidget()
        self.table_features.setColumnCount(7)
        self.table_features.setHorizontalHeaderLabels([
            "채널명", "RMS (V)", "Max Val", "Max Freq (Hz)",
            "Peak@60kHz", "Peak@100kHz", "이벤트 채널",
        ])
        self.table_features.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_features.setAlternatingRowColors(True)
        self.table_features.setEditTriggers(QTableWidget.NoEditTriggers)
        feat_layout.addWidget(self.table_features)

        self.canvas_rms = MplCanvas(width=12, height=4)
        feat_layout.addWidget(self.canvas_rms)
        self.tabs.addTab(tab_features, "📊  피처 분석 (Phase 3)")

        tab_overview = QWidget()
        ov_layout = QVBoxLayout(tab_overview)
        self.canvas_overview = MplCanvas(width=12, height=8)
        ov_layout.addWidget(NavigationToolbar(self.canvas_overview, tab_overview))
        ov_layout.addWidget(self.canvas_overview)
        self.tabs.addTab(tab_overview, "🗂️  전채널 개요")

        return self.tabs

    # ── 파일 열기 / Phase 1 ──────────────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "ALMS BIN 파일 선택", "", "BIN Files (*.bin);;All Files (*)"
        )
        if not path:
            return

        self.stft_results = []
        self.statusbar.showMessage(f"📂 Phase 1: 파싱 중…  {os.path.basename(path)}")
        self.btn_open.setEnabled(False)
        self.progress_bar.setVisible(True)

        self._parse_worker = ParseWorker(path)
        self._parse_worker.finished.connect(self._on_parse_done)
        self._parse_worker.error.connect(self._on_parse_error)
        self._parse_worker.start()

    def _on_parse_done(self, data):
        self.alms_data = data
        self.progress_bar.setVisible(False)
        self.btn_open.setEnabled(True)
        self.btn_export_csv.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.btn_all_stft.setEnabled(True)

        h = data.header
        fname = os.path.basename(data.file_path)
        self.lbl_filename.setText(fname)
        self.statusbar.showMessage(
            f"✅ Phase 1 완료: {fname}  |  채널: {h.total_ch}  |  "
            f"샘플: {h.n_samples:,}  |  {h.sampling_rate:,} Hz"
        )

        self._info_labels["site"].setText(h.site or "-")
        self._info_labels["event_ch"].setText(str(h.event_ch))
        self._info_labels["total_ch"].setText(str(h.total_ch))
        self._info_labels["sr"].setText(f"{h.sampling_rate:,} Hz")
        self._info_labels["duration"].setText(f"{h.event_duration} ms")
        self._info_labels["alarm"].setText(getattr(h, "alarm_result_str", "-"))
        self._info_labels["sig_type"].setText(getattr(h, "signal_type_str", "-"))
        ev_date = getattr(h, "event_date", "") or "-"
        self._info_labels["date"].setText(ev_date[:19] if len(ev_date) > 19 else ev_date)

        self.combo_ch.blockSignals(True)
        self.combo_ch.clear()
        for i, ch in enumerate(data.channels):
            marker = "  ★" if i == h.event_ch else ""
            self.combo_ch.addItem(f"[{i}] {_ch_display_name(ch, i)}{marker}")
        self.combo_ch.blockSignals(False)
        self.combo_ch.setCurrentIndex(h.event_ch)

        self.current_ch = h.event_ch
        self._plot_raw()
        self._plot_overview()

    def _on_parse_error(self, msg: str):
        self.progress_bar.setVisible(False)
        self.btn_open.setEnabled(True)
        self.statusbar.showMessage(f"❌ 파싱 오류: {msg}")
        QMessageBox.critical(self, "파싱 오류", msg)

    # ── 채널 변경 ─────────────────────────────────────────────────

    def _on_channel_changed(self, idx: int):
        self.current_ch = idx
        if self.alms_data:
            self._plot_raw()

    # ── Raw Signal 플롯 ───────────────────────────────────────────

    def _plot_raw(self):
        if not self.alms_data:
            return
        ch = self.alms_data.channels[self.current_ch]
        plots.plot_raw(
            fig=self.canvas_raw.fig,
            raw=ch.raw_data,
            fs=self.alms_data.header.sampling_rate,
            dur_ms=self.alms_data.header.event_duration,
            ch_name=_ch_display_name(ch, self.current_ch),
        )
        self.canvas_raw.draw()

    # ── 인터랙티브 단일 채널 STFT ─────────────────────────────────

    def _run_stft_interactive(self):
        if not self.alms_data:
            return
        self.tabs.setCurrentIndex(1)
        self._plot_stft_interactive()

    def _plot_stft_interactive(self):
        ch = self.alms_data.channels[self.current_ch]
        sr = self.alms_data.header.sampling_rate
        dur = self.alms_data.header.event_duration

        nperseg = int(self.spin_window.currentText())
        overlap_pct = self.spin_overlap.value()
        noverlap = int(nperseg * overlap_pct / 100)
        win_func = self.combo_window.currentText()
        fmax_khz = self.spin_fmax.value()
        ch_name = _ch_display_name(ch, self.current_ch)

        freqs, times, power_db = services.compute_interactive_stft(
            raw=ch.raw_data,
            fs=sr,
            nperseg=nperseg,
            noverlap=noverlap,
            window=win_func,
            fmax_hz=fmax_khz * 1000,
        )

        plots.plot_spectrogram(
            fig=self.canvas_stft.fig,
            freqs=freqs,
            times=times,
            power_db=power_db,
            raw=ch.raw_data,
            dur_ms=dur,
            ch_name=ch_name,
            fmax_khz=fmax_khz,
            nperseg=nperseg,
            overlap_pct=overlap_pct,
            win_func=win_func,
        )
        self.canvas_stft.draw()

        freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 0
        time_res = (times[1] - times[0]) * 1000 if len(times) > 1 else 0
        self.statusbar.showMessage(
            f"✅ STFT 완료  |  {ch_name}  |  "
            f"주파수 해상도: {freq_res:.1f} Hz  |  시간 해상도: {time_res:.2f} ms"
        )

    # ── 전채널 STFT + Phase 3 피처 ───────────────────────────────

    def _run_all_stft(self):
        if not self.alms_data:
            return

        self.btn_all_stft.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.statusbar.showMessage("⚡ Phase 2~3: 전채널 STFT + 피처 추출 중…")

        self._stft_worker = STFTWorker(self.alms_data, self._tmp_dir)
        self._stft_worker.finished.connect(self._on_all_stft_done)
        self._stft_worker.error.connect(self._on_all_stft_error)
        self._stft_worker.start()

    def _on_all_stft_done(self, results: list):
        self.stft_results = results
        self.progress_bar.setVisible(False)
        self.btn_all_stft.setEnabled(True)
        self.btn_analyze.setEnabled(True)

        h = self.alms_data.header
        services.get_event_ch_result(results, h.event_ch)
        best_result = services.get_max_ch_result(results)

        self.statusbar.showMessage(
            f"✅ Phase 2~3 완료  |  전채널 {len(results)}개  |  "
            f"이벤트 CH={h.event_ch}  |  "
            f"MaxVal CH={best_result['ch_index'] if best_result else '-'}"
        )

        self._update_features_table(results, h.event_ch)
        self.tabs.setCurrentIndex(2)

    def _on_all_stft_error(self, msg: str):
        self.progress_bar.setVisible(False)
        self.btn_all_stft.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.statusbar.showMessage(f"❌ STFT 오류: {msg}")
        QMessageBox.critical(self, "STFT 오류", msg)

    # ── 피처 테이블 ───────────────────────────────────────────────

    def _update_features_table(self, results: list, event_ch: int):
        self.table_features.setRowCount(len(results))

        for row, r in enumerate(results):
            is_event = (r["ch_index"] == event_ch)

            items = [
                r.get("ch_name", f"CH{r['ch_index']}"),
                f"{r.get('rms', 0):.6f}",
                f"{r.get('max_val', 0):.4f}",
                f"{r.get('max_freq', 0):.1f}",
                f"{r.get('peak_60k', 0):.4f}",
                f"{r.get('peak_100k', 0):.4f}",
                "★ Event" if is_event else "",
            ]

            rms = r.get("rms", 0)
            if rms > 0.01:
                fg = QColor('#f38ba8')
            elif rms > 0.001:
                fg = QColor('#fab387')
            else:
                fg = QColor('#a6e3a1')

            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setForeground(fg)
                if is_event:
                    item.setBackground(QColor('#2a2a3e'))
                self.table_features.setItem(row, col, item)

        plots.plot_rms_bar(self.canvas_rms.fig, results, event_ch)
        self.canvas_rms.draw()

    # ── 전채널 Raw 개요 ───────────────────────────────────────────

    def _plot_overview(self):
        if not self.alms_data:
            return
        plots.plot_overview(
            fig=self.canvas_overview.fig,
            channels=self.alms_data.channels,
            dur_ms=self.alms_data.header.event_duration,
            event_ch=self.alms_data.header.event_ch,
        )
        self.canvas_overview.draw()

    # ── CSV 내보내기 ──────────────────────────────────────────────

    def _export_csv(self):
        if not self.alms_data:
            return
        path, _ = QFileDialog.getSaveFileName(self, "CSV 저장", "", "CSV Files (*.csv)")
        if path:
            try:
                services.export_csv(self.alms_data.file_path, path)
                self.statusbar.showMessage(f"✅ CSV 저장 완료: {path}")
            except Exception as e:
                QMessageBox.warning(self, "CSV 오류", str(e))

    # ── 다크 테마 (Catppuccin Mocha) ─────────────────────────────

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
                font-weight: bold;
                color: #89b4fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 6px 12px;
                font-size: 12px;
            }
            QPushButton:hover   { background-color: #45475a; border-color: #89b4fa; }
            QPushButton:pressed { background-color: #89b4fa; color: #1e1e2e; }
            QPushButton:disabled{ background-color: #181825; color: #45475a; border-color: #313244; }
            QComboBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QComboBox::drop-down { border: none; }
            QSpinBox, QDoubleSpinBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 3px 6px;
            }
            QTabWidget::pane    { border: 1px solid #45475a; background-color: #1e1e2e; }
            QTabBar::tab {
                background-color: #181825;
                color: #a6adc8;
                padding: 8px 16px;
                border: 1px solid #313244;
                border-bottom: none;
                border-radius: 4px 4px 0 0;
            }
            QTabBar::tab:selected { background-color: #313244; color: #cdd6f4; border-color: #45475a; }
            QTableWidget {
                background-color: #181825;
                alternate-background-color: #1e1e2e;
                color: #cdd6f4;
                gridline-color: #313244;
                border: 1px solid #45475a;
            }
            QHeaderView::section {
                background-color: #313244;
                color: #89b4fa;
                padding: 6px;
                border: 1px solid #45475a;
                font-weight: bold;
            }
            QStatusBar { background-color: #181825; color: #a6adc8; border-top: 1px solid #313244; }
            QLabel      { color: #cdd6f4; }
            QProgressBar {
                background-color: #181825;
                border: 1px solid #45475a;
                border-radius: 4px;
                text-align: center;
                color: #cdd6f4;
            }
            QProgressBar::chunk { background-color: #89b4fa; border-radius: 3px; }
        """)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ALMS BIN Viewer")
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    viewer = ALMSViewer()
    viewer.show()
    sys.exit(app.exec_())
