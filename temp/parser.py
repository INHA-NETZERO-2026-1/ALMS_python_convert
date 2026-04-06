"""
ALMS BIN 파일 파서
기존 C# 분석 SW 코드(bin_load_Code.txt) 기반으로 Python으로 재구현
"""

import struct
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class ALMSChannelConfig:
    """채널별 설정값"""
    ch_no: int = 0
    ch_name: str = ""
    channel_bypass: int = 0
    alarm_inhibit: int = 0
    hit_detection: int = 0
    hit_rise_time: int = 0
    hit_duration_time: int = 0
    crack_detection: int = 0
    pdt: float = 0.0
    hdt: float = 0.0
    hlt: float = 0.0
    attenuation: float = 0.0
    low_signal_setpoint: float = 0.0
    low_signal_time: float = 0.0
    leak_alarm_setpoint: float = 0.0
    leak_alarm_time: float = 0.0
    leak_warning_time: float = 0.0
    hit_amplitude_min: float = 0.0
    hit_amplitude_max: float = 0.0
    min_hit_energy: float = 0.0
    max_hit_energy: float = 0.0
    hit_rise_time_min: float = 0.0
    hit_rise_time_max: float = 0.0
    hit_duration_time_min: float = 0.0
    hit_duration_time_max: float = 0.0
    fixed_setpoint: float = 0.0
    max_cracks_since_midnight: float = 0.0
    crack_amplitude_result: float = 0.0
    crack_hit_energy_result: float = 0.0
    crack_hit_rise_time_result: float = 0.0
    crack_duration_time_result: float = 0.0
    asl_result: float = 0.0
    crack_setpoint: float = 0.0
    raw_data: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ALMSHeader:
    """ALMS BIN 파일 헤더"""
    site: str = ""
    system_type: int = 0          # 1 = ALMS
    event_ch: int = 0
    total_ch: int = 0
    event_date: str = ""
    alarm_result: int = 0         # 0=Normal, 1=Leak Warning, 2=Leak Alarm, 3=Crack Alarm, 4=Hi Alarm, 5=Hi-Hi Alarm
    sampling_rate: int = 0        # Hz
    user_id: int = 0              # 0=Monitoring, 1=Operator, 2=Admin
    event_duration: int = 0       # ms
    signal_type: int = 0          # 0=BG Noise, 1=Event, 2=PST, ...

    @property
    def alarm_result_str(self) -> str:
        labels = ["Normal", "Leak Warning", "Leak Alarm", "Crack Alarm", "Hi Alarm", "Hi-Hi Alarm"]
        return labels[self.alarm_result] if 0 <= self.alarm_result < len(labels) else "Unknown"

    @property
    def signal_type_str(self) -> str:
        labels = ["Background Noise", "Event", "PST", "Air Injection Test", "Pencil Break Test", "LPMS Trigger"]
        return labels[self.signal_type] if 0 <= self.signal_type < len(labels) else "Unknown"

    @property
    def data_length(self) -> int:
        """채널당 샘플 수"""
        return self.sampling_rate * self.event_duration // 1000

    @property
    def time_axis(self) -> np.ndarray:
        """시간축 배열 (초)"""
        return np.linspace(0, self.event_duration / 1000, self.data_length)


@dataclass
class ALMSData:
    """ALMS BIN 파일 전체 데이터"""
    header: ALMSHeader = field(default_factory=ALMSHeader)
    channels: List[ALMSChannelConfig] = field(default_factory=list)
    file_path: str = ""

    def get_channel_names(self) -> List[str]:
        return [ch.ch_name.strip() or f"CH{ch.ch_no}" for ch in self.channels]

    def get_raw_data(self, ch_index: int) -> np.ndarray:
        if 0 <= ch_index < len(self.channels):
            return self.channels[ch_index].raw_data
        return np.array([])

    def get_rms(self, ch_index: int) -> float:
        data = self.get_raw_data(ch_index)
        if len(data) == 0:
            return 0.0
        return float(np.sqrt(np.mean(data ** 2)))

    def get_all_rms(self) -> List[float]:
        return [self.get_rms(i) for i in range(len(self.channels))]


def _read_string(data: bytes) -> str:
    """바이트 배열에서 null-terminated 문자열 읽기"""
    try:
        null_pos = data.find(b'\x00')
        if null_pos >= 0:
            data = data[:null_pos]
        return data.decode('cp949', errors='replace').strip()
    except Exception:
        return data.decode('latin-1', errors='replace').strip()


def _read_short(data: bytes) -> int:
    return struct.unpack('<h', data)[0]


def _read_int(data: bytes) -> int:
    return struct.unpack('<i', data)[0]


def _read_float(data: bytes) -> float:
    return struct.unpack('<f', data)[0]


def parse_alms_bin(filepath: str) -> Optional[ALMSData]:
    """
    ALMS BIN 파일 파싱
    C# 코드(bin_load_Code.txt) 구조를 그대로 Python으로 재구현
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")

    result = ALMSData(file_path=filepath)
    header = ALMSHeader()

    with open(filepath, 'rb') as f:

        # ── 헤더 파싱 ──────────────────────────────────────
        # SITE (8 bytes)
        header.site = _read_string(f.read(8))

        # System type (2 bytes) - 1이어야 ALMS
        header.system_type = _read_short(f.read(2))
        if header.system_type != 1:
            raise ValueError(f"ALMS 파일이 아닙니다. System type: {header.system_type}")

        # Event Channel (2 bytes)
        header.event_ch = _read_short(f.read(2))

        # Total Channel (2 bytes)
        header.total_ch = _read_short(f.read(2))

        # Skip 2 bytes (Event Type)
        f.read(2)

        # Event Date (24 bytes)
        header.event_date = _read_string(f.read(24))

        # Alarm Result (2 bytes)
        header.alarm_result = _read_short(f.read(2))

        # Skip 6 bytes
        f.read(6)

        # Sampling Rate (4 bytes, int)
        header.sampling_rate = _read_int(f.read(4))

        # User ID (2 bytes)
        header.user_id = _read_short(f.read(2))

        # Skip 2 bytes
        f.read(2)

        # Event Duration (4 bytes, int) - milliseconds
        header.event_duration = _read_int(f.read(4))

        # Signal Type (2 bytes)
        header.signal_type = _read_short(f.read(2))

        # Skip 2 bytes
        f.read(2)

        # g/V (4 bytes, float) - skip
        f.read(4)

        # mils/V (4 bytes, float) - skip
        f.read(4)

        # Skip 8*3 = 24 bytes
        f.read(24)

        result.header = header

        # ── 채널별 데이터 파싱 ────────────────────────────
        n_samples = header.data_length

        for _ in range(header.total_ch):
            ch = ALMSChannelConfig()

            ch.ch_no = struct.unpack('<h', f.read(2))[0]
            f.read(6)  # skip

            ch.ch_name = _read_string(f.read(16))

            ch.channel_bypass = struct.unpack('<h', f.read(2))[0]
            ch.alarm_inhibit = struct.unpack('<h', f.read(2))[0]
            ch.hit_detection = struct.unpack('<h', f.read(2))[0]
            ch.hit_rise_time = struct.unpack('<h', f.read(2))[0]
            ch.hit_duration_time = struct.unpack('<h', f.read(2))[0]
            ch.crack_detection = struct.unpack('<h', f.read(2))[0]

            ch.pdt = struct.unpack('<f', f.read(4))[0]
            ch.hdt = struct.unpack('<f', f.read(4))[0]
            ch.hlt = struct.unpack('<f', f.read(4))[0]
            ch.attenuation = struct.unpack('<f', f.read(4))[0]
            ch.low_signal_setpoint = struct.unpack('<f', f.read(4))[0]
            ch.low_signal_time = struct.unpack('<f', f.read(4))[0]
            ch.leak_alarm_setpoint = struct.unpack('<f', f.read(4))[0]
            ch.leak_alarm_time = struct.unpack('<f', f.read(4))[0]
            ch.leak_warning_time = struct.unpack('<f', f.read(4))[0]
            ch.hit_amplitude_min = struct.unpack('<f', f.read(4))[0]
            ch.hit_amplitude_max = struct.unpack('<f', f.read(4))[0]
            ch.min_hit_energy = struct.unpack('<f', f.read(4))[0]
            ch.max_hit_energy = struct.unpack('<f', f.read(4))[0]
            ch.hit_rise_time_min = struct.unpack('<f', f.read(4))[0]
            ch.hit_rise_time_max = struct.unpack('<f', f.read(4))[0]
            ch.hit_duration_time_min = struct.unpack('<f', f.read(4))[0]
            ch.hit_duration_time_max = struct.unpack('<f', f.read(4))[0]
            ch.fixed_setpoint = struct.unpack('<f', f.read(4))[0]
            ch.max_cracks_since_midnight = struct.unpack('<f', f.read(4))[0]
            ch.crack_amplitude_result = struct.unpack('<f', f.read(4))[0]
            ch.crack_hit_energy_result = struct.unpack('<f', f.read(4))[0]
            ch.crack_hit_rise_time_result = struct.unpack('<f', f.read(4))[0]
            ch.crack_duration_time_result = struct.unpack('<f', f.read(4))[0]
            ch.asl_result = struct.unpack('<f', f.read(4))[0]
            ch.crack_setpoint = struct.unpack('<f', f.read(4))[0]

            f.read(16)  # skip

            # Raw Data (float32 × n_samples)
            raw_bytes = f.read(4 * n_samples)
            ch.raw_data = np.frombuffer(raw_bytes, dtype=np.float32).copy()

            result.channels.append(ch)

    return result


def export_to_csv(alms_data: ALMSData, output_path: str):
    """파싱 결과를 CSV로 내보내기 (기존 SW 결과와 비교용)"""
    import csv

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)

        # 헤더 정보
        h = alms_data.header
        writer.writerow(["SITE :", h.site])
        writer.writerow(["System :", "ALMS"])
        writer.writerow(["Event Ch :", h.event_ch])
        writer.writerow(["Total Ch No :", h.total_ch])
        writer.writerow(["Event Date :", h.event_date])
        writer.writerow(["Alarm Result :", h.alarm_result_str])
        writer.writerow(["Sampling Rate :", h.sampling_rate])
        writer.writerow(["Event Duration :", h.event_duration])
        writer.writerow(["Signal Type :", h.signal_type_str])
        writer.writerow([])

        # Raw Data
        writer.writerow(["ALMS Raw Data :"])
        ch_names = alms_data.get_channel_names()
        writer.writerow(["idx"] + ch_names)

        n = alms_data.header.data_length
        for k in range(n):
            row = [k]
            for ch in alms_data.channels:
                row.append(ch.raw_data[k] if k < len(ch.raw_data) else "")
            writer.writerow(row)

    print(f"CSV 저장 완료: {output_path}")


def extract_features(alms_data: ALMSData) -> dict:
    """
    논문 기반 특징 추출
    - AE 신호: RMS, Peak@60kHz, Peak@100kHz
    - 가속도 신호: RMS, Peak@5~7kHz (채널 구분 필요)
    """
    from scipy import signal as scipy_signal

    features = {}
    sr = alms_data.header.sampling_rate

    for i, ch in enumerate(alms_data.channels):
        if len(ch.raw_data) == 0:
            continue

        data = ch.raw_data.astype(np.float64)
        ch_name = ch.ch_name.strip() or f"CH{ch.ch_no}"

        # RMS
        rms = np.sqrt(np.mean(data ** 2))

        # FFT
        freqs = np.fft.rfftfreq(len(data), d=1.0 / sr)
        fft_mag = np.abs(np.fft.rfft(data))

        # Peak at 60 kHz (±5 kHz 범위)
        mask_60k = (freqs >= 55000) & (freqs <= 65000)
        peak_60k = float(np.max(fft_mag[mask_60k])) if mask_60k.any() else 0.0

        # Peak at 100 kHz (±5 kHz 범위)
        mask_100k = (freqs >= 95000) & (freqs <= 105000)
        peak_100k = float(np.max(fft_mag[mask_100k])) if mask_100k.any() else 0.0

        # Peak at 5~7 kHz (가속도 센서용)
        mask_5_7k = (freqs >= 5000) & (freqs <= 7000)
        peak_5_7k = float(np.max(fft_mag[mask_5_7k])) if mask_5_7k.any() else 0.0

        features[ch_name] = {
            "rms": float(rms),
            "peak_60kHz": peak_60k,
            "peak_100kHz": peak_100k,
            "peak_5_7kHz": peak_5_7k,
        }

    return features