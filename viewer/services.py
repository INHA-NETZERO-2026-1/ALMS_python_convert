"""UI 스레드에서 호출하는 동기 데이터 래퍼.
module/ 와 scipy 의존성을 이 파일에 격리."""

import numpy as np
from scipy import signal as scipy_signal

from module.parser import exportCSV
import module.stft as _stft


def compute_interactive_stft(raw, fs: int, nperseg: int, noverlap: int,
                             window: str, fmax_hz: int):
    """대화형 단일 채널 STFT 계산. (freqs, times, power_db) 반환.
    fmax_hz는 호출자 정보로만 사용되며 잘라내기는 plots 쪽에서 처리."""
    data = np.asarray(raw, dtype=np.float64)
    freqs, times, Zxx = scipy_signal.stft(
        data, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap
    )
    power_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
    return freqs, times, power_db


def export_csv(bin_path: str, out_path: str) -> None:
    exportCSV(bin_path, out_path)


def get_event_ch_result(results: list, event_ch: int):
    return _stft.get_event_ch_result(results, event_ch)


def get_max_ch_result(results: list):
    return _stft.get_max_ch_result(results)
