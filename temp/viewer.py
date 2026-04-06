"""
ALMS BIN File Viewer
- BIN 파일 파싱 (Phase 1)
- STFT Spectrogram 시각화 (Phase 2)
- 특징값 추출 및 표시 (Phase 3)
"""

import sys
import os
import numpy as np
from scipy import signal as scipy_signal

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QGroupBox,
    QSplitter, QTextEdit, QSlider, QSpinBox, QDoubleSpinBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QStatusBar, QFrame, QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from parser import parse_alms_bin, ALMSData, export_to_csv, extract_features


# ──────────────────────────────────────────────
# matplotlib 캔버스 위젯
# ──────────────────────────────────────────────

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#1e1e2e')
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


# ──────────────────────────────────────────────
# 파싱 워커 스레드
# ──────────────────────────────────────────────

class ParseWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            data = parse_alms_bin(self.filepath)
            self.finished.emit(data)
        except Exception as e:
            self.error.emit(str(e))


# ──────────────────────────────────────────────
# 메인 윈도우
# ──────────────────────────────────────────────

class ALMSViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.alms_data: ALMSData = None
        self.current_ch = 0
        self._setup_ui()
        self._apply_dark_theme()

    # ── UI 구성 ───────────────────────────────

    def _setup_ui(self):
        self.setWindowTitle("ALMS BIN File Viewer  |  (주)리얼게인 NIMS")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # ── 좌측 패널 ─────────────────────────
        left_panel = self._build_left_panel()
        main_layout.addWidget(left_panel, stretch=0)

        # ── 우측 패널 (탭) ────────────────────
        right_panel = self._build_right_panel()
        main_layout.addWidget(right_panel, stretch=1)

        # 상태바
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("BIN 파일을 열어주세요.")

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(280)
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        # 파일 열기
        file_group = QGroupBox("파일")
        file_layout = QVBoxLayout(file_group)

        self.btn_open = QPushButton("📂  BIN 파일 열기")
        self.btn_open.setFixedHeight(40)
        self.btn_open.clicked.connect(self._open_file)
        file_layout.addWidget(self.btn_open)

        self.btn_export_csv = QPushButton("💾  CSV 내보내기")
        self.btn_export_csv.setFixedHeight(36)
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.clicked.connect(self._export_csv)
        file_layout.addWidget(self.btn_export_csv)

        self.lbl_filename = QLabel("파일 없음")
        self.lbl_filename.setWordWrap(True)
        self.lbl_filename.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        file_layout.addWidget(self.lbl_filename)
        layout.addWidget(file_group)

        # 헤더 정보
        info_group = QGroupBox("파일 정보")
        info_layout = QGridLayout(info_group)
        info_layout.setSpacing(4)

        def make_info_row(label):
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
            val = QLabel("-")
            val.setStyleSheet("color: #cdd6f4; font-size: 11px; font-weight: bold;")
            return lbl, val

        self._info_labels = {}
        rows = [
            ("SITE", "site"), ("이벤트 채널", "event_ch"),
            ("총 채널수", "total_ch"), ("샘플링레이트", "sr"),
            ("이벤트 시간", "duration"), ("알람 상태", "alarm"),
            ("신호 종류", "sig_type"), ("이벤트 날짜", "date"),
        ]
        for i, (label, key) in enumerate(rows):
            lbl, val = make_info_row(label)
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
        stft_layout = QGridLayout(stft_group)
        stft_layout.setSpacing(6)

        stft_layout.addWidget(QLabel("Window 크기"), 0, 0)
        self.spin_window = QComboBox()
        self.spin_window.addItems(["256", "512", "1024", "2048", "4096"])
        self.spin_window.setCurrentText("1024")
        stft_layout.addWidget(self.spin_window, 0, 1)

        stft_layout.addWidget(QLabel("Overlap (%)"), 1, 0)
        self.spin_overlap = QSpinBox()
        self.spin_overlap.setRange(0, 95)
        self.spin_overlap.setValue(75)
        self.spin_overlap.setSingleStep(5)
        stft_layout.addWidget(self.spin_overlap, 1, 1)

        stft_layout.addWidget(QLabel("Window 함수"), 2, 0)
        self.combo_window = QComboBox()
        self.combo_window.addItems(["hann", "hamming", "blackman", "bartlett", "boxcar"])
        stft_layout.addWidget(self.combo_window, 2, 1)

        stft_layout.addWidget(QLabel("주파수 상한 (kHz)"), 3, 0)
        self.spin_fmax = QSpinBox()
        self.spin_fmax.setRange(10, 2000)
        self.spin_fmax.setValue(600)
        stft_layout.addWidget(self.spin_fmax, 3, 1)

        self.btn_analyze = QPushButton("▶  STFT 분석 실행")
        self.btn_analyze.setFixedHeight(38)
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self._run_stft)
        stft_layout.addWidget(self.btn_analyze, 4, 0, 1, 2)

        layout.addWidget(stft_group)
        layout.addStretch()

        return panel

    def _build_right_panel(self) -> QWidget:
        self.tabs = QTabWidget()

        # ── 탭 1: Raw Signal ─────────────────
        self.tab_raw = QWidget()
        raw_layout = QVBoxLayout(self.tab_raw)
        self.canvas_raw = MplCanvas(width=12, height=5)
        toolbar_raw = NavigationToolbar(self.canvas_raw, self.tab_raw)
        raw_layout.addWidget(toolbar_raw)
        raw_layout.addWidget(self.canvas_raw)
        self.tabs.addTab(self.tab_raw, "📈  Raw Signal")

        # ── 탭 2: STFT Spectrogram ───────────
        self.tab_stft = QWidget()
        stft_layout = QVBoxLayout(self.tab_stft)
        self.canvas_stft = MplCanvas(width=12, height=8)
        toolbar_stft = NavigationToolbar(self.canvas_stft, self.tab_stft)
        stft_layout.addWidget(toolbar_stft)
        stft_layout.addWidget(self.canvas_stft)
        self.tabs.addTab(self.tab_stft, "🔥  STFT Spectrogram")

        # ── 탭 3: 특징값 ─────────────────────
        self.tab_features = QWidget()
        feat_layout = QVBoxLayout(self.tab_features)

        self.table_features = QTableWidget()
        self.table_features.setColumnCount(5)
        self.table_features.setHorizontalHeaderLabels([
            "채널명", "RMS", "Peak@60kHz", "Peak@100kHz", "Peak@5~7kHz"
        ])
        self.table_features.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_features.setAlternatingRowColors(True)
        self.table_features.setEditTriggers(QTableWidget.NoEditTriggers)
        feat_layout.addWidget(self.table_features)

        # RMS 바 차트
        self.canvas_rms = MplCanvas(width=12, height=4)
        feat_layout.addWidget(self.canvas_rms)
        self.tabs.addTab(self.tab_features, "📊  특징값 분석")

        # ── 탭 4: 전채널 개요 ────────────────
        self.tab_overview = QWidget()
        ov_layout = QVBoxLayout(self.tab_overview)
        self.canvas_overview = MplCanvas(width=12, height=8)
        toolbar_ov = NavigationToolbar(self.canvas_overview, self.tab_overview)
        ov_layout.addWidget(toolbar_ov)
        ov_layout.addWidget(self.canvas_overview)
        self.tabs.addTab(self.tab_overview, "🗂️  전채널 개요")

        return self.tabs

    # ── 파일 열기 ─────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "ALMS BIN 파일 선택", "", "BIN Files (*.bin);;All Files (*)"
        )
        if not path:
            return

        self.statusbar.showMessage(f"파일 로딩 중: {os.path.basename(path)}")
        self.btn_open.setEnabled(False)

        self.worker = ParseWorker(path)
        self.worker.finished.connect(self._on_parse_done)
        self.worker.error.connect(self._on_parse_error)
        self.worker.start()

    def _on_parse_done(self, data: ALMSData):
        self.alms_data = data
        self.btn_open.setEnabled(True)
        self.btn_export_csv.setEnabled(True)
        self.btn_analyze.setEnabled(True)

        h = data.header
        fname = os.path.basename(data.file_path)
        self.lbl_filename.setText(fname)
        self.statusbar.showMessage(f"✅ 로딩 완료: {fname}  |  채널수: {h.total_ch}  |  샘플: {h.data_length:,}개")

        # 헤더 정보 업데이트
        self._info_labels["site"].setText(h.site)
        self._info_labels["event_ch"].setText(str(h.event_ch))
        self._info_labels["total_ch"].setText(str(h.total_ch))
        self._info_labels["sr"].setText(f"{h.sampling_rate:,} Hz")
        self._info_labels["duration"].setText(f"{h.event_duration} ms")
        self._info_labels["alarm"].setText(h.alarm_result_str)
        self._info_labels["sig_type"].setText(h.signal_type_str)
        self._info_labels["date"].setText(h.event_date[:19] if h.event_date else "-")

        # 채널 콤보박스 업데이트
        self.combo_ch.blockSignals(True)
        self.combo_ch.clear()
        for i, ch in enumerate(data.channels):
            name = ch.ch_name.strip() or f"CH{ch.ch_no}"
            self.combo_ch.addItem(f"[{i+1}] {name}")
        self.combo_ch.blockSignals(False)
        self.combo_ch.setCurrentIndex(0)

        # 자동으로 첫 채널 플롯 + 특징값 계산
        self.current_ch = 0
        self._plot_raw()
        self._update_features_table()
        self._plot_overview()

    def _on_parse_error(self, msg: str):
        self.btn_open.setEnabled(True)
        self.statusbar.showMessage(f"❌ 오류: {msg}")
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(self, "파싱 오류", msg)

    # ── 채널 변경 ─────────────────────────────

    def _on_channel_changed(self, idx: int):
        self.current_ch = idx
        if self.alms_data:
            self._plot_raw()

    # ── Raw Signal 플롯 ───────────────────────

    def _plot_raw(self):
        if not self.alms_data:
            return

        fig = self.canvas_raw.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor('#181825')
        fig.patch.set_facecolor('#1e1e2e')

        ch = self.alms_data.channels[self.current_ch]
        data = ch.raw_data.astype(np.float64)
        t = self.alms_data.header.time_axis

        if len(t) != len(data):
            t = np.linspace(0, self.alms_data.header.event_duration / 1000, len(data))

        ax.plot(t, data, color='#89b4fa', linewidth=0.6, alpha=0.9)
        ax.axhline(0, color='#45475a', linewidth=0.5)

        ch_name = ch.ch_name.strip() or f"CH{ch.ch_no}"
        rms_val = np.sqrt(np.mean(data**2))

        ax.set_title(f"Raw Signal  |  {ch_name}  |  RMS = {rms_val:.5f} V",
                     color='#cdd6f4', fontsize=12)
        ax.set_xlabel("Time (sec)", color='#a6adc8')
        ax.set_ylabel("Amplitude (V)", color='#a6adc8')
        ax.tick_params(colors='#a6adc8')
        ax.spines[['top', 'right']].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color('#45475a')

        fig.tight_layout()
        self.canvas_raw.draw()

    # ── STFT 실행 ─────────────────────────────

    def _run_stft(self):
        if not self.alms_data:
            return

        self.tabs.setCurrentWidget(self.tab_stft)
        self._plot_stft()

    def _plot_stft(self):
        ch = self.alms_data.channels[self.current_ch]
        data = ch.raw_data.astype(np.float64)
        sr = self.alms_data.header.sampling_rate

        nperseg = int(self.spin_window.currentText())
        overlap_pct = self.spin_overlap.value() / 100
        noverlap = int(nperseg * overlap_pct)
        win_func = self.combo_window.currentText()
        f_max_khz = self.spin_fmax.value()

        # STFT 계산
        freqs, times, Zxx = scipy_signal.stft(
            data, fs=sr, window=win_func,
            nperseg=nperseg, noverlap=noverlap
        )

        # 파워 스펙트럼 (dB)
        power_db = 20 * np.log10(np.abs(Zxx) + 1e-12)

        # 주파수 범위 제한
        f_max_hz = f_max_khz * 1000
        freq_mask = freqs <= f_max_hz
        freqs_plot = freqs[freq_mask] / 1000  # kHz 변환
        power_plot = power_db[freq_mask, :]

        fig = self.canvas_stft.fig
        fig.clear()
        fig.patch.set_facecolor('#1e1e2e')

        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)
        ax_spec = fig.add_subplot(gs[0])
        ax_raw = fig.add_subplot(gs[1])

        # Spectrogram
        ax_spec.set_facecolor('#181825')
        im = ax_spec.pcolormesh(
            times, freqs_plot, power_plot,
            shading='gouraud', cmap='inferno'
        )
        cbar = fig.colorbar(im, ax=ax_spec, pad=0.01)
        cbar.set_label('Power (dB)', color='#a6adc8')
        cbar.ax.yaxis.set_tick_params(color='#a6adc8')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#a6adc8')

        ch_name = ch.ch_name.strip() or f"CH{ch.ch_no}"
        ax_spec.set_title(
            f"STFT Spectrogram  |  {ch_name}  |  Window={nperseg}  Overlap={self.spin_overlap.value()}%  [{win_func}]",
            color='#cdd6f4', fontsize=11
        )
        ax_spec.set_ylabel("Frequency (kHz)", color='#a6adc8')
        ax_spec.tick_params(colors='#a6adc8')
        for spine in ax_spec.spines.values():
            spine.set_color('#45475a')

        # 주요 주파수 라인 표시
        for f_khz, label, color in [(60, '60 kHz (AE)', '#a6e3a1'), (100, '100 kHz (AE)', '#f38ba8')]:
            if f_khz <= f_max_khz:
                ax_spec.axhline(f_khz, color=color, linewidth=0.8, linestyle='--', alpha=0.7, label=label)
        ax_spec.legend(loc='upper right', fontsize=8, facecolor='#313244', labelcolor='#cdd6f4')

        # Raw signal (아래)
        t = self.alms_data.header.time_axis
        if len(t) != len(data):
            t = np.linspace(0, self.alms_data.header.event_duration / 1000, len(data))

        ax_raw.set_facecolor('#181825')
        ax_raw.plot(t, data, color='#89b4fa', linewidth=0.5)
        ax_raw.set_xlabel("Time (sec)", color='#a6adc8')
        ax_raw.set_ylabel("Amplitude (V)", color='#a6adc8')
        ax_raw.tick_params(colors='#a6adc8')
        for spine in ax_raw.spines.values():
            spine.set_color('#45475a')
        ax_raw.set_xlim([times[0], times[-1]])

        self.canvas_stft.draw()
        self.statusbar.showMessage(
            f"✅ STFT 완료  |  {ch_name}  |  주파수 해상도: {freqs[1]-freqs[0]:.1f} Hz  |  시간 해상도: {(times[1]-times[0])*1000:.2f} ms"
        )

    # ── 특징값 테이블 ─────────────────────────

    def _update_features_table(self):
        if not self.alms_data:
            return

        features = extract_features(self.alms_data)
        ch_names = list(features.keys())

        self.table_features.setRowCount(len(ch_names))

        for row, (ch_name, feat) in enumerate(features.items()):
            self.table_features.setItem(row, 0, QTableWidgetItem(ch_name))
            self.table_features.setItem(row, 1, QTableWidgetItem(f"{feat['rms']:.6f}"))
            self.table_features.setItem(row, 2, QTableWidgetItem(f"{feat['peak_60kHz']:.4f}"))
            self.table_features.setItem(row, 3, QTableWidgetItem(f"{feat['peak_100kHz']:.4f}"))
            self.table_features.setItem(row, 4, QTableWidgetItem(f"{feat['peak_5_7kHz']:.4f}"))

            # RMS 기반 색상
            rms = feat['rms']
            if rms > 0.01:
                color = QColor('#f38ba8')  # 빨강 (높음)
            elif rms > 0.001:
                color = QColor('#fab387')  # 주황
            else:
                color = QColor('#a6e3a1')  # 초록 (낮음)

            for col in range(5):
                item = self.table_features.item(row, col)
                if item:
                    item.setForeground(color)

        # RMS 바 차트
        self._plot_rms_bar(features)

    def _plot_rms_bar(self, features: dict):
        fig = self.canvas_rms.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor('#181825')
        fig.patch.set_facecolor('#1e1e2e')

        names = list(features.keys())
        rms_values = [features[n]['rms'] for n in names]

        colors = ['#f38ba8' if r > 0.01 else '#fab387' if r > 0.001 else '#89b4fa'
                  for r in rms_values]

        bars = ax.bar(range(len(names)), rms_values, color=colors, edgecolor='#45475a', linewidth=0.5)

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8, color='#a6adc8')
        ax.set_ylabel("RMS (V)", color='#a6adc8')
        ax.set_title("채널별 RMS 레벨", color='#cdd6f4')
        ax.tick_params(colors='#a6adc8')
        for spine in ax.spines.values():
            spine.set_color('#45475a')

        fig.tight_layout()
        self.canvas_rms.draw()

    # ── 전채널 개요 ───────────────────────────

    def _plot_overview(self):
        if not self.alms_data:
            return

        n_ch = len(self.alms_data.channels)
        if n_ch == 0:
            return

        cols = 4
        rows = (n_ch + cols - 1) // cols

        fig = self.canvas_overview.fig
        fig.clear()
        fig.patch.set_facecolor('#1e1e2e')

        t = self.alms_data.header.time_axis

        for i, ch in enumerate(self.alms_data.channels):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_facecolor('#181825')

            data = ch.raw_data.astype(np.float64)
            if len(t) != len(data):
                t_plot = np.linspace(0, self.alms_data.header.event_duration / 1000, len(data))
            else:
                t_plot = t

            ax.plot(t_plot, data, color='#89b4fa', linewidth=0.4)
            ch_name = ch.ch_name.strip() or f"CH{ch.ch_no}"
            rms = np.sqrt(np.mean(data**2)) if len(data) > 0 else 0
            ax.set_title(f"{ch_name}\nRMS={rms:.4f}", fontsize=7, color='#cdd6f4')
            ax.tick_params(labelsize=6, colors='#6c7086')
            for spine in ax.spines.values():
                spine.set_color('#313244')

        fig.suptitle("전채널 Raw Signal 개요", color='#cdd6f4', fontsize=12)
        fig.tight_layout()
        self.canvas_overview.draw()

    # ── CSV 내보내기 ──────────────────────────

    def _export_csv(self):
        if not self.alms_data:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "CSV 저장", "", "CSV Files (*.csv)"
        )
        if path:
            export_to_csv(self.alms_data, path)
            self.statusbar.showMessage(f"✅ CSV 저장 완료: {path}")

    # ── 다크 테마 ─────────────────────────────

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
            QPushButton:hover {
                background-color: #45475a;
                border-color: #89b4fa;
            }
            QPushButton:pressed {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QPushButton:disabled {
                background-color: #181825;
                color: #45475a;
                border-color: #313244;
            }
            QComboBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 3px 6px;
            }
            QTabWidget::pane {
                border: 1px solid #45475a;
                background-color: #1e1e2e;
            }
            QTabBar::tab {
                background-color: #181825;
                color: #a6adc8;
                padding: 8px 16px;
                border: 1px solid #313244;
                border-bottom: none;
                border-radius: 4px 4px 0 0;
            }
            QTabBar::tab:selected {
                background-color: #313244;
                color: #cdd6f4;
                border-color: #45475a;
            }
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
            QStatusBar {
                background-color: #181825;
                color: #a6adc8;
                border-top: 1px solid #313244;
            }
            QLabel {
                color: #cdd6f4;
            }
        """)


# ──────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ALMS BIN Viewer")

    # 고DPI 지원
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    viewer = ALMSViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()