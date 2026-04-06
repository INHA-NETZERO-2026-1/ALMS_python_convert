"""
테스트용 ALMS BIN 파일 생성기
실제 BIN 파일이 없을 때 파서/뷰어 동작 검증용
"""

import struct
import numpy as np
import os


def write_string_padded(f, s: str, length: int):
    encoded = s.encode('cp949')[:length]
    f.write(encoded.ljust(length, b'\x00'))


def write_short(f, v: int):
    f.write(struct.pack('<h', v))


def write_int(f, v: int):
    f.write(struct.pack('<i', v))


def write_float(f, v: float):
    f.write(struct.pack('<f', v))


def generate_dummy_bin(filepath: str, n_channels: int = 4, sampling_rate: int = 500000, duration_ms: int = 100):
    """
    ALMS BIN 더미 파일 생성
    - sampling_rate: Hz (기본 500kHz)
    - duration_ms: 이벤트 지속시간 (ms)
    """
    n_samples = sampling_rate * duration_ms // 1000

    with open(filepath, 'wb') as f:

        # ── 헤더 ──────────────────────────────
        write_string_padded(f, "TEST_NPP", 8)  # SITE
        write_short(f, 1)  # System = ALMS
        write_short(f, 1)  # Event Channel
        write_short(f, n_channels)  # Total CH
        write_short(f, 1)  # Event Type (skip)
        write_string_padded(f, "2024-09-10 12:30:00.000", 24)  # Event Date
        write_short(f, 1)  # Alarm Result = Leak Warning
        f.write(b'\x00' * 6)  # skip 6
        write_int(f, sampling_rate)  # Sampling Rate
        write_short(f, 0)  # User ID = Monitoring
        f.write(b'\x00' * 2)  # skip 2
        write_int(f, duration_ms)  # Event Duration (ms)
        write_short(f, 1)  # Signal Type = Event
        f.write(b'\x00' * 2)  # skip 2
        write_float(f, 1.0)  # g/V
        write_float(f, 1.0)  # mils/V
        f.write(b'\x00' * 24)  # skip 8*3

        # ── 채널별 데이터 ──────────────────────
        ch_names = [f"U-10{i + 1}" for i in range(n_channels)]
        t = np.linspace(0, duration_ms / 1000, n_samples)

        for i in range(n_channels):
            write_short(f, i + 1)  # Ch No
            f.write(b'\x00' * 6)  # skip
            write_string_padded(f, ch_names[i], 16)  # Ch Name

            write_short(f, 0)  # Channel Bypass
            write_short(f, 0)  # Alarm Inhibit
            write_short(f, 1)  # Hit Detection
            write_short(f, 0)  # Hit Rise Time
            write_short(f, 0)  # Hit Duration Time
            write_short(f, 1)  # Crack Detection

            write_float(f, 10000.0)  # PDT
            write_float(f, 20000.0)  # HDT
            write_float(f, 20000.0)  # HLT
            write_float(f, 40.0)  # Attenuation
            write_float(f, 0.1)  # Low Signal Setpoint
            write_float(f, 1000.0)  # Low Signal Time
            write_float(f, 1.0)  # Leak Alarm Setpoint
            write_float(f, 1000.0)  # Leak Alarm Time
            write_float(f, 500.0)  # Leak Warning Time
            write_float(f, 0.0)  # Hit Amplitude Min
            write_float(f, 10.0)  # Hit Amplitude Max
            write_float(f, 0.0)  # Min Hit Energy
            write_float(f, 0.0)  # Max Hit Energy
            write_float(f, 0.0)  # Hit Rise Time Min
            write_float(f, 70000.0)  # Hit Rise Time Max
            write_float(f, 0.0)  # Hit Duration Min
            write_float(f, 70000.0)  # Hit Duration Max
            write_float(f, 1.0)  # Fixed Setpoint
            write_float(f, 10000.0)  # Max Cracks Since Midnight
            write_float(f, 0.0)  # Crack Amplitude Result
            write_float(f, 0.0)  # Crack Hit Energy Result
            write_float(f, 0.0)  # Crack Hit Rise Time Result
            write_float(f, 0.0)  # Crack Duration Time Result
            write_float(f, 0.5)  # ASL Result
            write_float(f, 1.0)  # Crack Setpoint
            f.write(b'\x00' * 16)  # skip

            # Raw Data - 누설 신호 시뮬레이션
            # AE 신호: 60kHz + 100kHz 성분 + 노이즈
            noise = np.random.normal(0, 0.001, n_samples).astype(np.float32)

            if i == 0:
                # 이벤트 채널: 강한 누설 신호
                signal = (
                        0.05 * np.sin(2 * np.pi * 60000 * t) * np.exp(-t * 50) +
                        0.03 * np.sin(2 * np.pi * 100000 * t) * np.exp(-t * 80) +
                        noise
                ).astype(np.float32)
            else:
                # 다른 채널: 약한 신호
                amp = 0.005 / (i + 1)
                signal = (
                        amp * np.sin(2 * np.pi * 60000 * t) * np.exp(-t * 100) +
                        noise * 0.5
                ).astype(np.float32)

            f.write(signal.tobytes())

    print(f"더미 BIN 파일 생성 완료: {filepath}")
    print(f"  - 채널수: {n_channels}")
    print(f"  - 샘플링레이트: {sampling_rate:,} Hz")
    print(f"  - 이벤트 시간: {duration_ms} ms")
    print(f"  - 채널당 샘플수: {n_samples:,}")
    print(f"  - 파일 크기: {os.path.getsize(filepath):,} bytes")


if __name__ == "__main__":
    os.makedirs("../test_data", exist_ok=True)

    # 더미 파일 생성
    generate_dummy_bin(
        filepath="../test_data/ALMS_TEST_EVENT_20240910_123000.bin",
        n_channels=4,
        sampling_rate=500000,
        duration_ms=100
    )

    # 파서 검증
    import sys

    sys.path.insert(0, '..')
    from parser import parse_alms_bin, extract_features

    data = parse_alms_bin("../test_data/ALMS_TEST_EVENT_20240910_123000.bin")
    print("\n── 파싱 결과 ──────────────────────────")
    print(f"SITE         : {data.header.site}")
    print(f"채널수        : {data.header.total_ch}")
    print(f"샘플링레이트   : {data.header.sampling_rate:,} Hz")
    print(f"이벤트 시간    : {data.header.event_duration} ms")
    print(f"알람 상태      : {data.header.alarm_result_str}")
    print(f"신호 종류      : {data.header.signal_type_str}")
    print(f"채널당 샘플수  : {data.header.data_length:,}")

    print("\n── 채널별 RMS ──────────────────────────")
    for i, ch in enumerate(data.channels):
        name = ch.ch_name.strip()
        rms = data.get_rms(i)
        print(f"  [{name}] RMS = {rms:.6f} V  |  샘플수: {len(ch.raw_data)}")

    print("\n── 특징값 추출 ─────────────────────────")
    features = extract_features(data)
    for ch_name, feat in features.items():
        print(f"  [{ch_name}]")
        print(f"    RMS         = {feat['rms']:.6f}")
        print(f"    Peak@60kHz  = {feat['peak_60kHz']:.4f}")
        print(f"    Peak@100kHz = {feat['peak_100kHz']:.4f}")
        print(f"    Peak@5-7kHz = {feat['peak_5_7kHz']:.4f}")

    print("\n✅ 파서 검증 완료!")