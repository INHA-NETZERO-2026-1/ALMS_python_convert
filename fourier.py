"""
ALMS fourier_lib.py  —  LPMS fourier_lib.py 기반, ALMS용 수정본

주요 변경:
  ub   : 25,000 Hz → 600,000 Hz  (누설음 AE 대역)
  nFFT : 128       → 1024         (주파수 해상도 확보)
  hop  : 1(고정)   → nFFT//4      (75% overlap, 메모리 절약)
  focus(): FrameSize=None → 전체 구간 (누설은 연속 신호)
  stft(): p0/p1 제거 → stft(val) 단순 호출 (scipy 버전 호환)
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm


class Fourier_obj:

    def __init__(self, val, dt: float, fs: int, nFFT: int, hop: int = None):
        """
        val  : Raw 신호 데이터 (list 또는 np.ndarray)
        dt   : 샘플링 간격 (sec)
        fs   : 샘플링 주파수 (Hz)
        nFFT : Window 크기
        hop  : STFT hop 크기. None → nFFT//4 (75% overlap)
        """
        self.N    = len(val)
        self.val  = np.array(val, dtype=np.float64)
        self.dt   = dt
        self.fs   = fs
        self.nFFT = nFFT
        self.hop  = hop if hop is not None else max(1, nFFT // 4)

    # ── private ──────────────────────────────────────────────────

    def __stft(self, ub: int = 600000):
        cur_val = self.focused_val if hasattr(self, 'focused_val') else self.val

        win        = signal.windows.hann(self.nFFT)
        self.__SFT = signal.ShortTimeFFT(
            win=win, hop=self.hop, fs=self.fs,
            scale_to='magnitude', phase_shift=0
        )
        # scipy ShortTimeFFT: stft(x) — p0/p1 없이 호출
        zData     = self.__SFT.stft(cur_val)
        absZ      = np.abs(zData)

        n_cols     = absZ.shape[1]
        self.xData = np.arange(n_cols) * self.__SFT.delta_t
        yData      = self.__SFT.f
        self.yData = yData[yData <= ub]
        self.absZ  = absZ[:self.yData.shape[0], :]

    def __colormap_config(self):
        if not hasattr(self, 'absZ'):
            raise RuntimeError("__stft()를 먼저 실행하세요.")
        colorMax, colorMin = self.absZ.max(), self.absZ.min()
        colors = ['#000000','#c20078','#0343df','#00ffff',
                  '#15b01a','#ffff14','#fe420f','#e50000','#ffffff']
        self.norm = plt.Normalize(colorMin, colorMax)
        self.cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
        self.levels = [colorMin]
        for i in range(1, 8):
            tmp = colorMin + i * (colorMax - colorMin) / 9
            if self.levels[-1] != tmp: self.levels.append(tmp)
        if self.levels[-1] != colorMax: self.levels.append(colorMax)
        if self.levels[0]  != 0:        self.levels.insert(0, 0)
        if len(self.levels) == 1:       self.levels.append(1)
        self.colormapping = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    # ── public ───────────────────────────────────────────────────

    def focus(self, FrameSize: int = None):
        """
        분석 구간 설정 후 STFT 수행

        FrameSize=None  → 전체 구간 (ALMS 권장: 누설은 연속 신호)
        FrameSize=int   → 피크 주변 구간 (LPMS 호환)

        Returns: absZ, yData(Hz), xData(sec)
        """
        if FrameSize is None:
            self.FrameSize = self.N
            if hasattr(self, 'focused_val'):
                del self.focused_val
        else:
            self.FrameSize  = FrameSize
            max_abs_idx     = np.argmax(np.abs(self.val))
            start_idx       = max_abs_idx - FrameSize // 2
            end_idx         = max_abs_idx + FrameSize // 2
            if start_idx < 0:
                end_idx -= start_idx;  start_idx = 0
            elif end_idx >= self.N:
                start_idx -= end_idx - (self.N - 1);  end_idx = self.N - 1
            if start_idx < 0:
                raise Exception("input data indexing error")
            self.focused_val = self.val[start_idx:end_idx]

        self.__stft()
        return self.absZ, self.yData, self.xData

    def analyze_1(self, stft_file_name: str = "", ub: int = 600000):
        """축 없는 Spectrogram 이미지 저장 (LPMS analyze_1 동일)"""
        if not hasattr(self, 'absZ'):   self.__stft(ub)
        if not hasattr(self, 'levels'): self.__colormap_config()
        fig, ax = plt.subplots()
        ax.contourf(self.xData, self.yData, self.absZ,
                    levels=self.levels, cmap=self.cmap, norm=self.norm)
        ax.axis("off")
        fig.set_figheight(9);  fig.set_figwidth(9)
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.margins(0);  fig.patch.set_visible(False)
        if stft_file_name:
            plt.savefig(f'{stft_file_name}_origin.png', dpi=400)
        else:
            plt.show()
        plt.close()

    def analyze(self, stft_file_name: str = "", ub: int = 600000):
        """Spectrogram + 주파수축 + Raw파형 저장 (LPMS analyze 동일)"""
        if not hasattr(self, 'absZ'):   self.__stft(ub)
        if not hasattr(self, 'levels'): self.__colormap_config()
        cur_val = self.focused_val if hasattr(self, 'focused_val') else self.val

        margin = 0.09;  sz = 9
        fig, ax = plt.subplots(2, 3,
            gridspec_kw={"width_ratios": [0.9, 9, 0.1], "height_ratios": [9, 1]})

        # 주파수 누적 파워 (LPMS와 동일 방식)
        xD      = self.absZ.sum(axis=1)
        xD_plot = np.zeros(self.nFFT)
        n       = min(len(xD), self.nFFT)
        xD_plot[:n] = xD[:n]
        yD = np.arange(self.nFFT) * self.__SFT.delta_f
        ax[0,0].plot(xD_plot, yD, color='green')
        ax[0,0].set_ylim([0, ub])
        ax[0,0].set_ylabel('Frequency(Hz)')

        contour = ax[0,1].contourf(self.xData, self.yData, self.absZ,
                                   levels=self.levels, cmap=self.cmap, norm=self.norm)
        ax[0,1].axis("off")

        t = np.linspace(0, len(cur_val) * self.dt, len(cur_val))
        ax[1,1].plot(t, cur_val, color='green')
        ax[1,1].set_xlabel('Time(Sec)')

        fig.delaxes(ax[1,0]);  fig.delaxes(ax[1,2])
        fig.set_figheight(sz); fig.set_figwidth(sz)
        fig.colorbar(contour, cax=ax[0,2], orientation='vertical', spacing='proportional')
        plt.subplots_adjust(wspace=margin, hspace=margin)
        plt.tight_layout()
        if stft_file_name:
            plt.savefig(f'{stft_file_name}.png', dpi=400)
        else:
            plt.show()
        plt.close()

    def get_peak_at(self, center_hz: int, bandwidth_hz: int = 5000) -> float:
        """특정 주파수 대역 STFT 피크값 (논문 특징 추출용)"""
        if not hasattr(self, 'absZ'):
            raise RuntimeError("focus()를 먼저 실행하세요.")
        lo   = center_hz - bandwidth_hz
        hi   = center_hz + bandwidth_hz
        mask = (self.yData >= lo) & (self.yData <= hi)
        return float(self.absZ[mask, :].max()) if mask.any() else 0.0

    def get_max_info(self):
        """STFT 최대값 위치 반환 → (max_val, max_freq_Hz, max_time_sec)"""
        if not hasattr(self, 'absZ'):
            raise RuntimeError("focus()를 먼저 실행하세요.")
        max_idx = np.argmax(self.absZ)
        fi, ti  = np.unravel_index(max_idx, self.absZ.shape)
        return float(self.absZ[fi, ti]), float(self.yData[fi]), float(self.xData[ti])