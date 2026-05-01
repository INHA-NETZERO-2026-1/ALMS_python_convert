"""
ALMS BIN File Viewer  (Phase 1 ~ 3 통합)
=========================================
Phase 1 : BIN 파싱        (parser.py)
Phase 2 : STFT             (stft.py + fourier.py)
Phase 3 : 피처 추출        (stft.py  perform_stft_all_ch)

의존 모듈: parser.py, stft.py, fourier.py
PyPI    : PyQt5, matplotlib, numpy, scipy
"""

import sys
import os
import tempfile
import numpy as np
from scipy import signal as scipy_signal

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QGroupBox,
    QSplitter, QTextEdit, QSlider, QSpinBox, QDoubleSpinBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QStatusBar, QFrame, QGridLayout, QSizePolicy, QMessageBox,
    QProgressBar,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

# ── 한글 폰트 설정 ──────────────────────────────────────────────
def _set_korean_font():
    """OS별 한글 폰트를 matplotlib에 자동 적용"""
    # 우선순위 순서로 후보 폰트 탐색
    candidates = [
        "Malgun Gothic",    # Windows
        "AppleGothic",      # macOS
        "NanumGothic",      # Linux (나눔고딕 설치 시)
        "NanumBarunGothic",
        "Noto Sans CJK KR", # Linux (Noto 설치 시)
        "Noto Sans KR",
        "DejaVu Sans",      # 최후 fallback (깨지지는 않지만 한글 미지원)
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((f for f in candidates if f in available), None)

    if chosen:
        matplotlib.rc('font', family=chosen)
    matplotlib.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지

_set_korean_font()

# ── parser.py API (test_pipeline.py 기준) ──────────────────────
from module.parser import (
    parse_alms_bin,
    readEventCh,
    getEventData,
    getAllData,
    getEventDate,
    exportCSV,
)

# ── stft.py API ─────────────────────────────────────────────────
import module.stft as stft_module


# ────────────────────────────────────────────────────────────────
# matplotlib 캔버스 위젯
# ────────────────────────────────────────────────────────────────

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#1e1e2e')
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


# ────────────────────────────────────────────────────────────────
# Phase 1 파싱 워커
# ────────────────────────────────────────────────────────────────

class ParseWorker(QThread):
    finished = pyqtSignal(object)   # ALMSData
    error    = pyqtSignal(str)

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            data = parse_alms_bin(self.filepath)
            self.finished.emit(data)
        except Exception as e:
            self.error.emit(str(e))


# ────────────────────────────────────────────────────────────────
# Phase 2~3  STFT + 피처 워커
# ────────────────────────────────────────────────────────────────

class STFTWorker(QThread):
    """perform_stft_all_ch 를 백그라운드 스레드에서 실행"""
    finished = pyqtSignal(list)     # results list
    error    = pyqtSignal(str)
    progress = pyqtSignal(int)      # 0~100

    def __init__(self, alms_data, output_dir: str):
        super().__init__()
        self.alms_data  = alms_data
        self.output_dir = output_dir

    def run(self):
        try:
            results = stft_module.perform_stft_all_ch(
                output_dir = self.output_dir,
                alms_data  = self.alms_data,
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


# ────────────────────────────────────────────────────────────────
# 메인 윈도우
# ────────────────────────────────────────────────────────────────

class ALMSViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.alms_data   = None
        self.stft_results = []          # perform_stft_all_ch 결과
        self.current_ch  = 0
        self._tmp_dir    = tempfile.mkdtemp(prefix="alms_viewer_")

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

    # ── 좌측 패널 ─────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(290)
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        # 파일 열기 ───────────────────────────────────────────────
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

        # 헤더 정보 ───────────────────────────────────────────────
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

        # 채널 선택 ───────────────────────────────────────────────
        ch_group = QGroupBox("채널 선택")
        ch_layout = QVBoxLayout(ch_group)
        self.combo_ch = QComboBox()
        self.combo_ch.currentIndexChanged.connect(self._on_channel_changed)
        ch_layout.addWidget(self.combo_ch)
        layout.addWidget(ch_group)

        # STFT 파라미터 ───────────────────────────────────────────
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

        # 전채널 일괄 STFT (Phase 2~3)
        self.btn_all_stft = QPushButton("⚡  전채널 STFT + 피처 추출")
        self.btn_all_stft.setFixedHeight(38)
        self.btn_all_stft.setEnabled(False)
        self.btn_all_stft.clicked.connect(self._run_all_stft)
        sg.addWidget(self.btn_all_stft, 5, 0, 1, 2)

        layout.addWidget(stft_group)

        # 진행 표시 ───────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)   # indeterminate
        layout.addWidget(self.progress_bar)

        layout.addStretch()
        return panel

    # ── 우측 패널 (탭) ────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        self.tabs = QTabWidget()

        # 탭 1: Raw Signal ────────────────────────────────────────
        tab_raw = QWidget()
        raw_layout = QVBoxLayout(tab_raw)
        self.canvas_raw = MplCanvas(width=12, height=5)
        raw_layout.addWidget(NavigationToolbar(self.canvas_raw, tab_raw))
        raw_layout.addWidget(self.canvas_raw)
        self.tabs.addTab(tab_raw, "📈  Raw Signal")

        # 탭 2: STFT Spectrogram ──────────────────────────────────
        tab_stft = QWidget()
        stft_layout = QVBoxLayout(tab_stft)
        self.canvas_stft = MplCanvas(width=12, height=8)
        stft_layout.addWidget(NavigationToolbar(self.canvas_stft, tab_stft))
        stft_layout.addWidget(self.canvas_stft)
        self.tabs.addTab(tab_stft, "🔥  STFT Spectrogram")

        # 탭 3: 피처 분석 (Phase 3) ───────────────────────────────
        tab_features = QWidget()
        feat_layout  = QVBoxLayout(tab_features)

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

        # 탭 4: 전채널 개요 ───────────────────────────────────────
        tab_overview = QWidget()
        ov_layout    = QVBoxLayout(tab_overview)
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

        # 헤더 정보 업데이트
        self._info_labels["site"].setText(h.site or "-")
        self._info_labels["event_ch"].setText(str(h.event_ch))
        self._info_labels["total_ch"].setText(str(h.total_ch))
        self._info_labels["sr"].setText(f"{h.sampling_rate:,} Hz")
        self._info_labels["duration"].setText(f"{h.event_duration} ms")
        self._info_labels["alarm"].setText(getattr(h, "alarm_result_str", "-"))
        self._info_labels["sig_type"].setText(getattr(h, "signal_type_str", "-"))
        ev_date = getattr(h, "event_date", "") or "-"
        self._info_labels["date"].setText(ev_date[:19] if len(ev_date) > 19 else ev_date)

        # 채널 콤보박스
        self.combo_ch.blockSignals(True)
        self.combo_ch.clear()
        for i, ch in enumerate(data.channels):
            name = (ch.name or ch.ch_name or f"CH{i}").strip() or f"CH{i}"
            marker = "  ★" if i == h.event_ch else ""
            self.combo_ch.addItem(f"[{i}] {name}{marker}")
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

        fig = self.canvas_raw.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor('#181825')
        fig.patch.set_facecolor('#1e1e2e')

        ch   = self.alms_data.channels[self.current_ch]
        raw  = np.array(ch.raw_data, dtype=np.float64)
        fs   = self.alms_data.header.sampling_rate
        dur  = self.alms_data.header.event_duration
        t    = np.linspace(0, dur / 1000, len(raw))

        ax.plot(t, raw, color='#89b4fa', linewidth=0.6, alpha=0.9)
        ax.axhline(0, color='#45475a', linewidth=0.5)

        ch_name = (getattr(ch, "name", None) or getattr(ch, "ch_name", None) or f"CH{self.current_ch}").strip()
        rms_val = float(np.sqrt(np.mean(raw ** 2))) if len(raw) > 0 else 0.0

        ax.set_title(
            f"Raw Signal  |  {ch_name}  |  RMS = {rms_val:.5f} V  |  {len(raw):,} samples",
            color='#cdd6f4', fontsize=12,
        )
        ax.set_xlabel("Time (sec)", color='#a6adc8')
        ax.set_ylabel("Amplitude (V)", color='#a6adc8')
        ax.tick_params(colors='#a6adc8')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for sp in ax.spines.values():
            sp.set_color('#45475a')

        fig.tight_layout()
        self.canvas_raw.draw()

    # ── 인터랙티브 단일 채널 STFT ─────────────────────────────────

    def _run_stft_interactive(self):
        if not self.alms_data:
            return
        self.tabs.setCurrentIndex(1)
        self._plot_stft_interactive()

    def _plot_stft_interactive(self):
        ch   = self.alms_data.channels[self.current_ch]
        data = np.array(ch.raw_data, dtype=np.float64)
        sr   = self.alms_data.header.sampling_rate
        dur  = self.alms_data.header.event_duration

        nperseg    = int(self.spin_window.currentText())
        overlap_pct = self.spin_overlap.value() / 100
        noverlap   = int(nperseg * overlap_pct)
        win_func   = self.combo_window.currentText()
        f_max_khz  = self.spin_fmax.value()

        freqs, times, Zxx = scipy_signal.stft(
            data, fs=sr, window=win_func, nperseg=nperseg, noverlap=noverlap
        )
        power_db  = 20 * np.log10(np.abs(Zxx) + 1e-12)
        freq_mask = freqs <= f_max_khz * 1000
        f_plot    = freqs[freq_mask] / 1000   # kHz
        p_plot    = power_db[freq_mask, :]

        fig = self.canvas_stft.fig
        fig.clear()
        fig.patch.set_facecolor('#1e1e2e')
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)

        # Spectrogram
        ax_spec = fig.add_subplot(gs[0])
        ax_spec.set_facecolor('#181825')
        im = ax_spec.pcolormesh(times, f_plot, p_plot, shading='gouraud', cmap='inferno')

        cbar = fig.colorbar(im, ax=ax_spec, pad=0.01)
        cbar.set_label('Power (dB)', color='#a6adc8')
        cbar.ax.yaxis.set_tick_params(color='#a6adc8')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#a6adc8')

        ch_name = (getattr(ch, "name", None) or getattr(ch, "ch_name", None) or f"CH{self.current_ch}").strip()
        ax_spec.set_title(
            f"STFT Spectrogram  |  {ch_name}  |  "
            f"Window={nperseg}  Overlap={self.spin_overlap.value()}%  [{win_func}]",
            color='#cdd6f4', fontsize=11,
        )
        ax_spec.set_ylabel("Frequency (kHz)", color='#a6adc8')
        ax_spec.tick_params(colors='#a6adc8')
        for sp in ax_spec.spines.values():
            sp.set_color('#45475a')

        for f_khz, label, clr in [(60, '60 kHz (AE)', '#a6e3a1'), (100, '100 kHz (AE)', '#f38ba8')]:
            if f_khz <= f_max_khz:
                ax_spec.axhline(f_khz, color=clr, linewidth=0.8, linestyle='--', alpha=0.7, label=label)
        ax_spec.legend(loc='upper right', fontsize=8, facecolor='#313244', labelcolor='#cdd6f4')

        # Raw signal (아래)
        t = np.linspace(0, dur / 1000, len(data))
        ax_raw = fig.add_subplot(gs[1])
        ax_raw.set_facecolor('#181825')
        ax_raw.plot(t, data, color='#89b4fa', linewidth=0.5)
        ax_raw.set_xlabel("Time (sec)", color='#a6adc8')
        ax_raw.set_ylabel("Amplitude (V)", color='#a6adc8')
        ax_raw.tick_params(colors='#a6adc8')
        for sp in ax_raw.spines.values():
            sp.set_color('#45475a')
        ax_raw.set_xlim([times[0], times[-1]])

        self.canvas_stft.draw()
        freq_res  = freqs[1] - freqs[0] if len(freqs) > 1 else 0
        time_res  = (times[1] - times[0]) * 1000 if len(times) > 1 else 0
        self.statusbar.showMessage(
            f"✅ STFT 완료  |  {ch_name}  |  "
            f"주파수 해상도: {freq_res:.1f} Hz  |  시간 해상도: {time_res:.2f} ms"
        )

    # ── 전채널 STFT + Phase 3 피처 (stft.py 활용) ────────────────

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
        ev_result   = stft_module.get_event_ch_result(results, h.event_ch)
        best_result = stft_module.get_max_ch_result(results)

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

    # ── 피처 테이블 업데이트 ──────────────────────────────────────

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
                fg = QColor('#f38ba8')    # 빨강 (높음)
            elif rms > 0.001:
                fg = QColor('#fab387')    # 주황
            else:
                fg = QColor('#a6e3a1')    # 초록 (낮음)

            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setForeground(fg)
                if is_event:
                    item.setBackground(QColor('#2a2a3e'))
                self.table_features.setItem(row, col, item)

        self._plot_rms_bar(results, event_ch)

    # ── RMS 바 차트 ───────────────────────────────────────────────

    def _plot_rms_bar(self, results: list, event_ch: int):
        fig = self.canvas_rms.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor('#181825')
        fig.patch.set_facecolor('#1e1e2e')

        names  = [r.get("ch_name", f"CH{r['ch_index']}") for r in results]
        rms_v  = [r.get("rms", 0) for r in results]
        colors = []
        for i, r in enumerate(results):
            if r["ch_index"] == event_ch:
                colors.append('#f9e2af')   # 노랑 – 이벤트 채널
            elif r.get("rms", 0) > 0.01:
                colors.append('#f38ba8')   # 빨강
            elif r.get("rms", 0) > 0.001:
                colors.append('#fab387')   # 주황
            else:
                colors.append('#89b4fa')   # 파랑

        ax.bar(range(len(names)), rms_v, color=colors, edgecolor='#45475a', linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8, color='#a6adc8')
        ax.set_ylabel("RMS (V)", color='#a6adc8')
        ax.set_title("채널별 RMS 레벨  (★ 노랑 = 이벤트 채널)", color='#cdd6f4')
        ax.tick_params(colors='#a6adc8')
        for sp in ax.spines.values():
            sp.set_color('#45475a')

        fig.tight_layout()
        self.canvas_rms.draw()

    # ── 전채널 Raw 개요 ───────────────────────────────────────────

    def _plot_overview(self):
        if not self.alms_data:
            return

        channels = self.alms_data.channels
        n_ch = len(channels)
        if n_ch == 0:
            return

        cols = 4
        rows = (n_ch + cols - 1) // cols
        dur  = self.alms_data.header.event_duration
        ev_ch = self.alms_data.header.event_ch

        fig = self.canvas_overview.fig
        fig.clear()
        fig.patch.set_facecolor('#1e1e2e')

        for i, ch in enumerate(channels):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_facecolor('#181825')

            raw = np.array(ch.raw_data, dtype=np.float64)
            t   = np.linspace(0, dur / 1000, len(raw))
            clr = '#f9e2af' if i == ev_ch else '#89b4fa'
            ax.plot(t, raw, color=clr, linewidth=0.4)

            ch_name = (getattr(ch, "name", None) or getattr(ch, "ch_name", None) or f"CH{i}").strip()
            rms = float(np.sqrt(np.mean(raw ** 2))) if len(raw) > 0 else 0.0
            marker = "★ " if i == ev_ch else ""
            ax.set_title(f"{marker}{ch_name}\nRMS={rms:.4f}", fontsize=7, color='#cdd6f4')
            ax.tick_params(labelsize=6, colors='#6c7086')
            for sp in ax.spines.values():
                sp.set_color('#313244')

        fig.suptitle("전채널 Raw Signal 개요  (★ 노랑 = 이벤트 채널)", color='#cdd6f4', fontsize=12)
        fig.tight_layout()
        self.canvas_overview.draw()

    # ── CSV 내보내기 ──────────────────────────────────────────────

    def _export_csv(self):
        if not self.alms_data:
            return
        path, _ = QFileDialog.getSaveFileName(self, "CSV 저장", "", "CSV Files (*.csv)")
        if path:
            try:
                exportCSV(self.alms_data.file_path, path)
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


# ────────────────────────────────────────────────────────────────
# 진입점
# ────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ALMS BIN Viewer")
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    viewer = ALMSViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
