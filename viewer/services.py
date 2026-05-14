"""UI 스레드에서 호출하는 동기 데이터 래퍼.
module/ 와 scipy 의존성을 이 파일에 격리."""

import numpy as np
from scipy import signal as scipy_signal

from module.parser import exportCSV
import module.stft as _stft


def compute_interactive_stft(raw, fs: int, nperseg: int, noverlap: int,
                             window: str, fmax_hz: int):
    """대화형 단일 채널 STFT 계산. (freqs, times, magnitude) 반환.
    fmax_hz는 호출자 정보로만 사용되며 잘라내기는 plots 쪽에서 처리."""
    data = np.asarray(raw, dtype=np.float64)

    # =====================================================================
    # 남정우 수정 (1/2) : DC 제거 추가
    # ---------------------------------------------------------------------
    # [수정 전] DC 제거 없음 → 신호의 평균값(직류 성분)이 남아있어
    #           저주파 에너지가 스펙트로그램 전체를 오염시킴
    #
    # [수정 후] data -= mean(data) 로 DC 성분 제거
    #           → 저주파 노이즈 억제, 실제 AE 신호 대역만 강조됨
    # =====================================================================
    data = data - np.mean(data)

    freqs, times, Zxx = scipy_signal.stft(
        data, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap
    )

    # =====================================================================
    # 남정우 수정 (2/2) : dB 스케일 → 선형(Linear) Magnitude 스케일 전환
    # ---------------------------------------------------------------------
    # [수정 전] power_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
    #           → log 함수가 노이즈 플로어(-120dB)까지 밝게 표현
    #           → 스펙트로그램 전체가 붉게 보여 신호 구분 불가
    #
    # [수정 후] magnitude = np.abs(Zxx)  (선형 magnitude)
    #           → 노이즈는 0에 가까워 어둡게, 실제 신호만 밝게 표시
    # =====================================================================
    magnitude = np.abs(Zxx)
    return freqs, times, magnitude


def export_csv(bin_path: str, out_path: str) -> None:
    exportCSV(bin_path, out_path)


def get_event_ch_result(results: list, event_ch: int):
    return _stft.get_event_ch_result(results, event_ch)


def get_max_ch_result(results: list):
    return _stft.get_max_ch_result(results)
