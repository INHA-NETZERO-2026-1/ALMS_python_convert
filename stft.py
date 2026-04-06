"""
ALMS stft.py - LPMS stft.py 구조 유지, ALMS 파라미터로 수정
"""
import os
import numpy as np
import fourier as lib
from parser import ALMSData

ALMS_UB        = 600000
ALMS_NFFT      = 1024
ALMS_FRAMESIZE = None

def perform_stft(stft_file_name, stft_file_name_origin, eventChData, fs,
                 ub=ALMS_UB, nFFT=ALMS_NFFT, frame_size=None):
    val = list(eventChData) if not isinstance(eventChData, list) else eventChData
    if len(val) == 0:
        raise ValueError("eventChData가 비어 있습니다.")
    dt = 1.0 / fs
    obj = lib.Fourier_obj(val=val, dt=dt, fs=fs, nFFT=nFFT)
    stft_result, freqs, times = obj.focus(FrameSize=frame_size)
    obj.analyze(stft_file_name=stft_file_name, ub=ub)
    obj.analyze_1(stft_file_name=stft_file_name_origin, ub=ub)
    max_val, max_freq, max_time = obj.get_max_info()
    return round(max_val, 2), round(max_freq, 2), round(max_time * 1000, 2)

def perform_stft_all_ch(output_dir, alms_data, ub=ALMS_UB, nFFT=ALMS_NFFT, frame_size=None):
    h = alms_data.header
    fs = h.sampling_rate
    dt = 1.0 / fs
    img_dir    = os.path.join(output_dir, 'image')
    origin_dir = os.path.join(output_dir, 'origin_image')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(origin_dir, exist_ok=True)
    file_stem = os.path.splitext(os.path.basename(alms_data.file_path))[0]
    results = []
    for i, ch in enumerate(alms_data.channels):
        val = ch.raw_data.tolist()
        if len(val) == 0:
            continue
        ch_tag = ch.name.replace(' ', '_').replace('/', '_')
        img_name         = f"{file_stem}_{ch_tag}"
        stft_path        = os.path.join(img_dir,    img_name)
        stft_path_origin = os.path.join(origin_dir, img_name)
        obj = lib.Fourier_obj(val=val, dt=dt, fs=fs, nFFT=nFFT)
        stft_result, freqs, times = obj.focus(FrameSize=frame_size)
        obj.analyze(stft_file_name=stft_path, ub=ub)
        obj.analyze_1(stft_file_name=stft_path_origin, ub=ub)
        max_val, max_freq, max_time = obj.get_max_info()
        peak_60k  = obj.get_peak_at(center_hz=60000,  bandwidth_hz=5000)
        peak_100k = obj.get_peak_at(center_hz=100000, bandwidth_hz=5000)
        results.append({
            "ch_name":         ch.name,
            "ch_index":        i,
            "max_val":         round(max_val, 4),
            "max_freq":        round(max_freq, 2),
            "max_time":        round(max_time * 1000, 2),
            "rms":             round(ch.rms, 6),
            "peak_60k":        round(peak_60k, 4),
            "peak_100k":       round(peak_100k, 4),
            "img_path":        stft_path + ".png",
            "img_path_origin": stft_path_origin + "_origin.png",
        })
        print(f"  [{ch.name}] max_val={max_val:.4f}  "
              f"max_freq={max_freq:.0f}Hz  RMS={ch.rms:.6f}  "
              f"Peak60k={peak_60k:.4f}  Peak100k={peak_100k:.4f}")
    return results

def get_event_ch_result(all_ch_results, event_ch_index):
    for r in all_ch_results:
        if r["ch_index"] == event_ch_index:
            return r
    return None

def get_max_ch_result(all_ch_results):
    if not all_ch_results:
        return None
    return max(all_ch_results, key=lambda r: r["max_val"])

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parser import parse_alms_bin

    test_file = "test_data/test.bin"
    if not os.path.exists(test_file):
        print("테스트 파일 없음 → generate_test_bin.py 먼저 실행하세요.")
        sys.exit(1)

    print("=" * 60)
    print("  ALMS stft.py 동작 검증")
    print("=" * 60)

    alms_data = parse_alms_bin(test_file)
    h = alms_data.header
    print(f"\n샘플링레이트: {h.sampling_rate:,} Hz  |  채널수: {h.total_ch}  |  샘플수: {h.n_samples:,}")

    # 단일 채널 (LPMS 호환)
    print(f"\n[단일 채널 STFT - 이벤트 채널 {h.event_ch}]")
    event_data = alms_data.get_event_channel_data().tolist()
    os.makedirs("test_data/image", exist_ok=True)
    os.makedirs("test_data/origin_image", exist_ok=True)
    max_val, max_freq, max_time = perform_stft(
        stft_file_name        = "test_data/image/test_event_ch",
        stft_file_name_origin = "test_data/origin_image/test_event_ch",
        eventChData           = event_data,
        fs                    = h.sampling_rate,
    )
    print(f"  max_val={max_val}  max_freq={max_freq}Hz  max_time={max_time}ms")

    # 전 채널
    print(f"\n[전 채널 STFT]")
    results = perform_stft_all_ch(output_dir="test_data", alms_data=alms_data)

    best  = get_max_ch_result(results)
    event = get_event_ch_result(results, h.event_ch)
    print(f"\n[STFT 최대값 채널]: {best['ch_name']}  (max_val={best['max_val']})")
    print(f"[이벤트 채널]:      {event['ch_name']}  (max_val={event['max_val']})")
    print(f"\n✅ stft.py 검증 완료!")