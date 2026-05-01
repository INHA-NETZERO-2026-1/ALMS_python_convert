"""
ALMS bin_reader.py
==================
기존 LPMS bin_reader.py를 ALMS용으로 완전 대체하는 파일입니다.

원본 참조:
  - C# 코드: 2_bin_load_Code.txt (tsCSVExport_Click)
  - LPMS:    bin_reader.py (기존 LPMS 전용, 구조 다름)

LPMS vs ALMS 구조 차이:
  LPMS - 헤더 512 bytes 고정, 18채널 고정, 20000샘플 고정
  ALMS - 헤더 가변, 채널 수 헤더에서 읽음, 샘플 수 = SamplingRate × Duration / 1000

외부 인터페이스:
  기존 LPMS bin_reader.py와 동일한 함수명을 유지하여
  imageGenerator.py, reportCreator.py 등 상위 코드 수정을 최소화합니다.

  getEventData(path)  → 이벤트 채널 Raw Data 반환 (list)
  getAllData(path)     → 전 채널 Raw Data 반환 (list of list)
  getEventDate(path)  → 이벤트 날짜 문자열 반환
  readEventCh(file)   → 이벤트 채널 번호 반환 (int)
"""

import struct
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ────────────────────────────────────────────────────────────────
# 데이터 클래스
# ────────────────────────────────────────────────────────────────

@dataclass
class ALMSHeader:
    """ALMS BIN 파일 글로벌 헤더"""
    site: str = ""
    system_type: int = 0        # 0=LPMS, 1=ALMS, 2=RCPVMS, 3=IVMS
    event_ch: int = 0           # 이벤트 발생 채널 번호 (0-based)
    total_ch: int = 0           # 전체 채널 수
    event_date: str = ""        # 이벤트 발생 날짜/시간 문자열
    alarm_result: int = 0       # 0=Normal, 1=Leak Warning, 2=Leak Alarm, 3=Crack Alarm, 4=Hi Alarm, 5=Hi-Hi Alarm
    sampling_rate: int = 0      # 샘플링레이트 (Hz)
    user_id: int = 0            # 0=Monitoring, 1=Operator, 2=Admin
    event_duration: int = 0     # 이벤트 지속 시간 (ms)
    signal_type: int = 0        # 0=BG Noise, 1=Event, 2=PST, 3=Air Injection, 4=Pencil Break, 5=LPMS Trigger

    @property
    def n_samples(self) -> int:
        """채널당 샘플 수 = SamplingRate × EventDuration / 1000"""
        return self.sampling_rate * self.event_duration // 1000

    @property
    def time_axis(self) -> np.ndarray:
        """이벤트 구간의 시간축(sec)"""
        return np.linspace(0, self.event_duration / 1000.0, self.n_samples)

    @property
    def alarm_result_str(self) -> str:
        labels = ["Normal", "Leak Warning", "Leak Alarm", "Crack Alarm", "Hi Alarm", "Hi-Hi Alarm"]
        return labels[self.alarm_result] if 0 <= self.alarm_result < len(labels) else "Unknown"

    @property
    def signal_type_str(self) -> str:
        labels = ["Background Noise", "Event", "PST", "Air Injection Test", "Pencil Break Test", "LPMS Trigger"]
        return labels[self.signal_type] if 0 <= self.signal_type < len(labels) else "Unknown"


@dataclass
class ALMSChannel:
    """채널별 설정값 + Raw Data"""
    # ── 채널 식별 ──────────────────────────
    ch_no: int = 0
    ch_name: str = ""

    # ── 채널 설정 (Int16) ──────────────────
    channel_bypass: int = 0
    alarm_inhibit: int = 0
    hit_detection: int = 0
    hit_rise_time: int = 0
    hit_duration_time: int = 0
    crack_detection: int = 0

    # ── 알람 파라미터 (Single/float) ───────
    pdt: float = 0.0                      # Peak Definition Time (μs)
    hdt: float = 0.0                      # Hit Definition Time (μs)
    hlt: float = 0.0                      # Hit Lockout Time (μs)
    attenuation: float = 0.0             # Gain (dB)
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

    # ── Raw Data ────────────────────────────
    raw_data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    @property
    def name(self) -> str:
        """정리된 채널 이름 반환"""
        return self.ch_name.strip() or f"CH{self.ch_no}"

    @property
    def rms(self) -> float:
        """RMS 값 계산"""
        if len(self.raw_data) == 0:
            return 0.0
        return float(np.sqrt(np.mean(self.raw_data.astype(np.float64) ** 2)))


@dataclass
class ALMSData:
    """ALMS BIN 파일 전체 파싱 결과"""
    header: ALMSHeader = field(default_factory=ALMSHeader)
    channels: List[ALMSChannel] = field(default_factory=list)
    file_path: str = ""

    # ── 데이터 접근 헬퍼 ──────────────────

    def get_channel_names(self) -> List[str]:
        return [ch.name for ch in self.channels]

    def get_raw_data(self, ch_index: int) -> np.ndarray:
        """인덱스로 채널 Raw Data 반환"""
        if 0 <= ch_index < len(self.channels):
            return self.channels[ch_index].raw_data
        return np.array([], dtype=np.float32)

    def get_event_channel_data(self) -> np.ndarray:
        """이벤트 채널(header.event_ch) Raw Data 반환"""
        return self.get_raw_data(self.header.event_ch)

    def get_all_raw_data(self) -> List[np.ndarray]:
        """전 채널 Raw Data 리스트 반환"""
        return [ch.raw_data for ch in self.channels]

    def get_all_raw_data_as_list(self) -> List[List[float]]:
        """전 채널 Raw Data를 list of list로 반환 (LPMS 호환용)"""
        return [ch.raw_data.tolist() for ch in self.channels]

    def get_rms(self, ch_index: int) -> float:
        return self.channels[ch_index].rms if 0 <= ch_index < len(self.channels) else 0.0

    def get_all_rms(self) -> List[float]:
        return [ch.rms for ch in self.channels]


# ────────────────────────────────────────────────────────────────
# 내부 읽기 헬퍼 함수 (C# BinaryReader 1:1 대응)
# ────────────────────────────────────────────────────────────────

def _read_bytes(f, n: int) -> bytes:
    """n 바이트 읽기"""
    return f.read(n)

def _skip(f, n: int):
    """n 바이트 건너뛰기 (C#: br.ReadBytes(n) 후 버리는 것과 동일)"""
    f.read(n)

def _read_int16(f) -> int:
    """C#: br.ReadInt16() → signed short (2 bytes, little-endian)"""
    return struct.unpack('<h', f.read(2))[0]

def _read_int32(f) -> int:
    """C#: br.ReadInt() → signed int (4 bytes, little-endian)"""
    return struct.unpack('<i', f.read(4))[0]

def _read_single(f) -> float:
    """C#: br.ReadSingle() → float32 (4 bytes, little-endian)"""
    return struct.unpack('<f', f.read(4))[0]

def _read_string(data: bytes) -> str:
    """바이트 배열 → null-terminated 문자열 (C#: getCharFromByte)"""
    try:
        null_pos = data.find(b'\x00')
        raw = data[:null_pos] if null_pos >= 0 else data
        return raw.decode('cp949', errors='replace').strip()
    except Exception:
        return data.decode('latin-1', errors='replace').strip()


def _read_fixed_string(data: bytes) -> str:
    """고정폭 문자열에서 NUL padding만 제거"""
    try:
        return data.replace(b'\x00', b'').decode('cp949', errors='replace').strip()
    except Exception:
        return data.replace(b'\x00', b'').decode('latin-1', errors='replace').strip()


# ────────────────────────────────────────────────────────────────
# 핵심 파서: C# tsCSVExport_Click 1:1 Python 변환
# ────────────────────────────────────────────────────────────────

def parse_alms_bin(bin_file_path: str) -> ALMSData:
    """
    ALMS BIN 파일을 파싱하여 ALMSData 반환

    C# tsCSVExport_Click 코드를 Python으로 1:1 변환
    읽기 순서, 바이트 크기, skip 위치 모두 C# 코드와 동일

    Parameters
    ----------
    bin_file_path : str
        ALMS BIN 파일 경로

    Returns
    -------
    ALMSData
        파싱 결과 전체

    Raises
    ------
    FileNotFoundError
        파일이 존재하지 않을 때
    ValueError
        ALMS 파일이 아닐 때 (system_type != 1)
    """
    import os
    if not os.path.exists(bin_file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {bin_file_path}")

    result = ALMSData(file_path=bin_file_path)
    header = ALMSHeader()

    with open(bin_file_path, 'rb') as f:

        # ── 글로벌 헤더 ──────────────────────────────────────────
        # C#: br.ReadBytes(8) → getCharFromByte → SITE
        header.site = _read_fixed_string(f.read(8))

        # C#: br.ReadBytes(2) → readshort → System Type
        # strSystem = { "LPMS", "ALMS", "RCPVMS", "IVMS" }
        # buffint != 1 → "ALMS 파일이 아닙니다." MessageBox
        header.system_type = _read_int16(f)
        if header.system_type != 1:
            system_names = {0: "LPMS", 2: "RCPVMS", 3: "IVMS"}
            name = system_names.get(header.system_type, f"Unknown({header.system_type})")
            raise ValueError(f"ALMS 파일이 아닙니다. 이 파일의 System Type: {name}")

        # C#: br.ReadBytes(2) → readshort → Event Ch
        header.event_ch = _read_int16(f)

        # C#: br.ReadBytes(2) → readshort → Total Ch No
        header.total_ch = _read_int16(f)
        total_ch = header.total_ch

        # C#: br.ReadBytes(2) → (Event Type, 주석 처리됨 → skip)
        _skip(f, 2)

        # C#: br.ReadBytes(24) → getCharFromByte → Event Date
        header.event_date = _read_string(f.read(24))

        # C#: br.ReadBytes(2) → readshort → Alarm Result
        # strAlarmResult = { "Normal", "Leak Warning", "Leak Alarm", "Crack Alarm", "Hi Alarm", "Hi-Hi Alarm" }
        header.alarm_result = _read_int16(f)

        # C#: br.ReadBytes(6) → skip
        _skip(f, 6)

        # C#: br.ReadBytes(4) → readInt → Sampling Rate
        header.sampling_rate = _read_int32(f)
        n_sampling_rate = header.sampling_rate

        # C#: br.ReadBytes(2) → readshort → User ID
        # strUSERID = { "Monitoring", "Operator", "Admin" }
        header.user_id = _read_int16(f)

        # C#: br.ReadBytes(2) → skip
        _skip(f, 2)

        # C#: br.ReadBytes(4) → readInt → Event Duration (ms)
        header.event_duration = _read_int32(f)
        n_event_duration = header.event_duration

        # C#: br.ReadBytes(2) → readshort → Signal Type
        # strSignalType = { "Background Noise", "Event", "PST", "Air Injection Test", "Pencil Break Test", "LPMS Trigger" }
        header.signal_type = _read_int16(f)

        # C#: br.ReadBytes(2) → skip
        _skip(f, 2)

        # C#: br.ReadBytes(4) → g/V (주석 처리됨 → skip)
        _skip(f, 4)

        # C#: br.ReadBytes(4) → mils/V (주석 처리됨 → skip)
        _skip(f, 4)

        # C#: br.ReadBytes(8 * 3) → skip (24 bytes)
        _skip(f, 8 * 3)

        result.header = header

        # ── 채널별 데이터 ─────────────────────────────────────────
        # C#: float[,] ALMSraw = new float[Total_CH, nSamplingRate * nEventDuration / 1000]
        n_samples = n_sampling_rate * n_event_duration // 1000

        for j in range(total_ch):
            ch = ALMSChannel()

            # C#: fw.WriteLine("Ch No. : ," + br.ReadInt16())
            ch.ch_no = _read_int16(f)

            # C#: br.ReadBytes(6) → skip
            _skip(f, 6)

            # C#: br.ReadBytes(16) → getCharFromByte → Ch Name
            ch.ch_name = _read_string(f.read(16))

            # C#: br.ReadInt16() × 6 → 채널 설정값
            ch.channel_bypass    = _read_int16(f)  # Channel Bypass
            ch.alarm_inhibit     = _read_int16(f)  # Alarm Inhibit
            ch.hit_detection     = _read_int16(f)  # Hit Detection
            ch.hit_rise_time     = _read_int16(f)  # Hit Rise Time
            ch.hit_duration_time = _read_int16(f)  # Hit Duration Time
            ch.crack_detection   = _read_int16(f)  # Crack Detection

            # C#: br.ReadSingle() × 26 → 알람 파라미터
            ch.pdt                      = _read_single(f)  # PDT
            ch.hdt                      = _read_single(f)  # HDT
            ch.hlt                      = _read_single(f)  # HLT
            ch.attenuation              = _read_single(f)  # Attenuation (Gain)
            ch.low_signal_setpoint      = _read_single(f)  # Low Signal Setpoint
            ch.low_signal_time          = _read_single(f)  # Low Signal Time
            ch.leak_alarm_setpoint      = _read_single(f)  # Leak Alarm Setpoint
            ch.leak_alarm_time          = _read_single(f)  # Leak Alarm Time
            ch.leak_warning_time        = _read_single(f)  # Leak Warning Time
            ch.hit_amplitude_min        = _read_single(f)  # Hit Amplitude Min
            ch.hit_amplitude_max        = _read_single(f)  # Hit Amplitude Max
            ch.min_hit_energy           = _read_single(f)  # Min Hit Energy
            ch.max_hit_energy           = _read_single(f)  # Max Hit Energy
            ch.hit_rise_time_min        = _read_single(f)  # Hit Rise Time Min
            ch.hit_rise_time_max        = _read_single(f)  # Hit Rise Time Max
            ch.hit_duration_time_min    = _read_single(f)  # Hit Duration Time Min
            ch.hit_duration_time_max    = _read_single(f)  # Hit Duration Time Max
            ch.fixed_setpoint           = _read_single(f)  # Fixed Setpoint
            ch.max_cracks_since_midnight = _read_single(f) # Max Cracks Since Midnight
            ch.crack_amplitude_result   = _read_single(f)  # Crack Amplitude Result
            ch.crack_hit_energy_result  = _read_single(f)  # Crack Hit Energy Result
            ch.crack_hit_rise_time_result = _read_single(f) # Crack Hit Rise Time Result
            ch.crack_duration_time_result = _read_single(f) # Crack Duration Time Result
            ch.asl_result               = _read_single(f)  # ASL Result
            ch.crack_setpoint           = _read_single(f)  # Crack Setpoint

            # C#: br.ReadBytes(16) → skip
            _skip(f, 16)

            # C#: for (int k = 0; k < (nSamplingRate * nEventDuration) / 1000; k++)
            #         ALMSraw[j, k] = br.ReadSingle()
            raw_bytes = f.read(4 * n_samples)
            ch.raw_data = np.frombuffer(raw_bytes, dtype=np.float32).copy()

            result.channels.append(ch)

    return result


# ────────────────────────────────────────────────────────────────
# 공개 인터페이스 함수
# LPMS bin_reader.py와 동일한 함수명으로 상위 코드 호환성 유지
# ────────────────────────────────────────────────────────────────

def readEventCh(file) -> int:
    """
    이벤트 채널 번호 읽기
    LPMS: file.seek(10) → 2 bytes
    ALMS: 동일 (byte 10 = Event Ch)
    """
    file.seek(10)
    return struct.unpack('<h', file.read(2))[0]


def getEventData(bin_file_path: str) -> List[float]:
    """
    이벤트 채널 Raw Data 반환 (list of float)
    LPMS bin_reader.getEventData()와 동일한 인터페이스

    Returns
    -------
    list[float]
        이벤트 채널의 Raw Data (float 리스트)
    """
    try:
        data = parse_alms_bin(bin_file_path)
        return data.get_event_channel_data().tolist()
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return []


def getAllData(bin_file_path: str) -> List[List[float]]:
    """
    전 채널 Raw Data 반환 (list of list)
    LPMS bin_reader.getAllData()와 동일한 인터페이스

    Returns
    -------
    list[list[float]]
        [채널0 데이터, 채널1 데이터, ..., 채널N 데이터]
    """
    try:
        data = parse_alms_bin(bin_file_path)
        return data.get_all_raw_data_as_list()
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return []


def getEventDate(bin_file_path: str) -> str:
    """
    이벤트 날짜 문자열 반환
    LPMS bin_reader.getEventDate()와 동일한 인터페이스

    Returns
    -------
    str
        이벤트 날짜 문자열 (예: "2024-09-10 12:30:00.000")
    """
    try:
        data = parse_alms_bin(bin_file_path)
        return data.header.event_date
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return ""


def getHeader(bin_file_path: str) -> Optional[ALMSHeader]:
    """
    헤더 정보 반환 (ALMS 추가 함수 - LPMS에는 없음)

    Returns
    -------
    ALMSHeader or None
    """
    try:
        data = parse_alms_bin(bin_file_path)
        return data.header
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None


def getALMSData(bin_file_path: str) -> Optional[ALMSData]:
    """
    파싱 결과 전체 반환 (ALMS 추가 함수 - LPMS에는 없음)
    채널 설정값, RMS, 헤더 등 모든 정보가 필요할 때 사용

    Returns
    -------
    ALMSData or None
    """
    try:
        return parse_alms_bin(bin_file_path)
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None


def exportCSV(bin_file_path: str, csv_output_path: str):
    """
    CSV 내보내기 (C# tsCSVExport_Click 동작과 동일한 결과물 생성)
    기존 SW 결과와 비교 검증용

    Parameters
    ----------
    bin_file_path : str
        ALMS BIN 파일 경로
    csv_output_path : str
        저장할 CSV 파일 경로
    """
    data = parse_alms_bin(bin_file_path)
    export_to_csv(data, csv_output_path)


def export_to_csv(alms_data: ALMSData, out_path: str) -> None:
    """
    ALMSData 객체에서 직접 CSV 내보내기 (viewer용)
    기존 exportCSV와 동일한 CSV 형식을 사용합니다.
    """
    import csv

    data = alms_data
    h = data.header

    system_names = ["LPMS", "ALMS", "RCPVMS", "IVMS"]
    alarm_labels = ["Normal", "Leak Warning", "Leak Alarm", "Crack Alarm", "Hi Alarm", "Hi-Hi Alarm"]
    userid_labels = ["Monitoring", "Operator", "Admin"]
    signal_labels = ["Background Noise", "Event", "PST", "Air Injection Test", "Pencil Break Test", "LPMS Trigger"]

    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)

        # ── 헤더 정보 (C# fw.WriteLine 순서와 동일) ──
        w.writerow(["SITE : ", h.site])
        w.writerow(["System : ", system_names[h.system_type] if 0 <= h.system_type < len(system_names) else "Unknown"])
        w.writerow(["Event Ch : ", h.event_ch])
        w.writerow(["Total Ch No : ", h.total_ch])
        w.writerow(["Event Date : ", h.event_date])
        w.writerow(["Alarm Result : ", alarm_labels[h.alarm_result] if 0 <= h.alarm_result < len(alarm_labels) else "Unknown"])
        w.writerow(["Sampling Rate : ", h.sampling_rate])
        if 0 <= h.user_id < len(userid_labels):
            w.writerow(["User ID : ", userid_labels[h.user_id]])
        else:
            w.writerow(["User ID : ", "ERR"])
        w.writerow(["Event Duration : ", h.event_duration])
        w.writerow(["Signal Type : ", signal_labels[h.signal_type] if 0 <= h.signal_type < len(signal_labels) else "Unknown"])

        # ── 채널별 설정값 ──
        for ch in data.channels:
            w.writerow(["Ch No. : ", ch.ch_no])
            w.writerow(["Ch Name : ", ch.ch_name])
            w.writerow(["Channel Bypass : ", ch.channel_bypass])
            w.writerow(["Alarm Inhibit : ", ch.alarm_inhibit])
            w.writerow(["Hit Detection : ", ch.hit_detection])
            w.writerow(["Hit Rise Time : ", ch.hit_rise_time])
            w.writerow(["Hit Duration Time : ", ch.hit_duration_time])
            w.writerow(["Crack Detection : ", ch.crack_detection])
            w.writerow(["PDT : ", ch.pdt])
            w.writerow(["HDT : ", ch.hdt])
            w.writerow(["HLT : ", ch.hlt])
            w.writerow(["Attenuation (Gain) : ", ch.attenuation])
            w.writerow(["Low Signal Setpoint : ", ch.low_signal_setpoint])
            w.writerow(["Low Signal Time : ", ch.low_signal_time])
            w.writerow(["Leak Alarm Setpoint : ", ch.leak_alarm_setpoint])
            w.writerow(["Leak Alarm Time : ", ch.leak_alarm_time])
            w.writerow(["Leak Warning Time : ", ch.leak_warning_time])
            w.writerow(["Hit Amplitude Min : ", ch.hit_amplitude_min])
            w.writerow(["Hit Amplitude Max : ", ch.hit_amplitude_max])
            w.writerow(["Min Hit Energy : ", ch.min_hit_energy])
            w.writerow(["Max Hit Energy : ", ch.max_hit_energy])
            w.writerow(["Hit Rise Time Min : ", ch.hit_rise_time_min])
            w.writerow(["Hit Rise Time Max : ", ch.hit_rise_time_max])
            w.writerow(["Hit Duration Time Min : ", ch.hit_duration_time_min])
            w.writerow(["Hit Duration Time Max : ", ch.hit_duration_time_max])
            w.writerow(["Fixed Setpoint : ", ch.fixed_setpoint])
            w.writerow(["Max Cracks Since Midnight : ", ch.max_cracks_since_midnight])
            w.writerow(["Crack Amplitude Result : ", ch.crack_amplitude_result])
            w.writerow(["Crack Hit Energy Result : ", ch.crack_hit_energy_result])
            w.writerow(["Crack Hit Rise Time Result : ", ch.crack_hit_rise_time_result])
            w.writerow(["Crack Duration Time Result : ", ch.crack_duration_time_result])
            w.writerow(["ASL Result : ", ch.asl_result])
            w.writerow(["Crack Setpoint : ", ch.crack_setpoint])

        # ── Raw Data (C# 출력 순서와 동일) ──
        w.writerow(["ALMS Raw Data :"])
        header_row = ["idx"] + [ch.name for ch in data.channels]
        w.writerow(header_row)

        n = h.n_samples
        for k in range(n):
            row = [k]
            for ch in data.channels:
                row.append(ch.raw_data[k] if k < len(ch.raw_data) else "")
            w.writerow(row)

    print(f"CSV 저장 완료: {out_path}")


# ────────────────────────────────────────────────────────────────
# 단독 실행 테스트
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    # 테스트 BIN 파일 생성 후 검증
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    test_file = "../test_data/2-2.Sample Data_ALMS_TRIGGER_250905054513824.bin"

    if not os.path.exists(test_file):
        print("테스트 파일 없음 → generate_test_bin.py 먼저 실행하세요.")
        sys.exit(1)

    print("=" * 55)
    print("  ALMS bin_reader.py 동작 검증")
    print("=" * 55)

    # ── 1. 전체 파싱 ──
    data = parse_alms_bin(test_file)
    h = data.header

    print(f"\n[헤더]")
    print(f"  SITE          : {h.site}")
    print(f"  System Type   : {h.system_type} (ALMS)")
    print(f"  Event Ch      : {h.event_ch}")
    print(f"  Total Ch      : {h.total_ch}")
    print(f"  Event Date    : {h.event_date}")
    print(f"  Alarm Result  : {h.alarm_result_str}")
    print(f"  Sampling Rate : {h.sampling_rate:,} Hz")
    print(f"  Event Duration: {h.event_duration} ms")
    print(f"  Signal Type   : {h.signal_type_str}")
    print(f"  N Samples     : {h.n_samples:,} per channel")

    print(f"\n[채널 데이터]")
    for i, ch in enumerate(data.channels):
        print(f"  [{ch.name}]  샘플수={len(ch.raw_data):,}  RMS={ch.rms:.6f} V")

    # ── 2. LPMS 호환 함수 검증 ──
    print(f"\n[LPMS 호환 함수 검증]")

    event_ch = readEventCh(open(test_file, 'rb'))
    print(f"  readEventCh()  = {event_ch}")

    event_data = getEventData(test_file)
    print(f"  getEventData() = {len(event_data):,}개  (첫 5개: {event_data[:5]})")

    all_data = getAllData(test_file)
    print(f"  getAllData()    = {len(all_data)}채널 × {len(all_data[0]):,}샘플")

    event_date = getEventDate(test_file)
    print(f"  getEventDate() = {event_date}")

    # ── 3. CSV 내보내기 ──
    os.makedirs("../test_data", exist_ok=True)
    exportCSV(test_file, "../test_data/output_check.csv")

    print(f"\n✅ 모든 검증 완료!")
