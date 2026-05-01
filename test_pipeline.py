"""
ALMS Phase 1~3 파이프라인 테스트
=================================
테스트 대상 파일: test_data/2-2.Sample Data_ALMS_TRIGGER_250905054513824.bin

Phase 1: BIN 파싱    (parser.py)
Phase 2: STFT        (stft.py + fourier.py)
Phase 3: 피처 추출   (stft.py perform_stft_all_ch)
"""

import os
import sys

BIN_FILE   = "test_data/2-2.Sample Data_ALMS_TRIGGER_250905054513824.bin"
OUTPUT_DIR = "test_data/pipeline_out"

PASS = "[PASS]"
FAIL = "[FAIL]"

errors = []

def check(label, condition, detail=""):
    if condition:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label}" + (f"  → {detail}" if detail else ""))
        errors.append(label)

# ────────────────────────────────────────────────────────────────
# 사전 확인
# ────────────────────────────────────────────────────────────────

if not os.path.exists(BIN_FILE):
    print(f"테스트 파일 없음: {BIN_FILE}")
    print("→ test_data 디렉토리에 샘플 BIN 파일을 추가하세요.")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────
# Phase 1: BIN 파싱
# ────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Phase 1: BIN 파싱 (parser.py)")
print("=" * 60)

from module.parser import (
    parse_alms_bin,
    readEventCh,
    getEventData,
    getAllData,
    getEventDate,
    exportCSV,
)

alms_data = parse_alms_bin(BIN_FILE)
h = alms_data.header

print(f"\n  SITE          : {h.site}")
print(f"  System Type   : {h.system_type} (ALMS=1)")
print(f"  Event Ch      : {h.event_ch}")
print(f"  Total Ch      : {h.total_ch}")
print(f"  Event Date    : {h.event_date}")
print(f"  Alarm Result  : {h.alarm_result_str}")
print(f"  Sampling Rate : {h.sampling_rate:,} Hz")
print(f"  Duration      : {h.event_duration} ms")
print(f"  Signal Type   : {h.signal_type_str}")
print(f"  N Samples     : {h.n_samples:,} per channel")

print()
check("system_type == 1 (ALMS)",    h.system_type == 1)
check("total_ch >= 1",              h.total_ch >= 1)
check("sampling_rate > 0",          h.sampling_rate > 0)
check("event_duration > 0",         h.event_duration > 0)
check("n_samples == fs * dur / 1000",
      h.n_samples == h.sampling_rate * h.event_duration // 1000)
check("channels 수 == total_ch",    len(alms_data.channels) == h.total_ch)

for i, ch in enumerate(alms_data.channels):
    n = len(ch.raw_data)
    check(f"  CH{i} ({ch.name}) 샘플수 == {h.n_samples}",
          n == h.n_samples, f"실제={n}")

# LPMS 호환 함수
print("\n  [LPMS 호환 함수]")
with open(BIN_FILE, 'rb') as f:
    ev_ch = readEventCh(f)
check("readEventCh() == header.event_ch",  ev_ch == h.event_ch, f"{ev_ch} vs {h.event_ch}")

ev_data = getEventData(BIN_FILE)
check("getEventData() 길이 == n_samples",  len(ev_data) == h.n_samples, f"{len(ev_data)}")

all_data = getAllData(BIN_FILE)
check("getAllData() 채널수 == total_ch",   len(all_data) == h.total_ch, f"{len(all_data)}")

ev_date = getEventDate(BIN_FILE)
check("getEventDate() 비어있지 않음",      len(ev_date) > 0, f"'{ev_date}'")

# CSV 내보내기
csv_path = os.path.join(OUTPUT_DIR, "output.csv")
exportCSV(BIN_FILE, csv_path)
check("exportCSV() 파일 생성됨",           os.path.exists(csv_path))

# 이벤트 채널 RMS (신호 채널이 배경 채널보다 커야 함)
ev_rms   = alms_data.channels[h.event_ch].rms
other_rms = [alms_data.channels[i].rms
             for i in range(h.total_ch) if i != h.event_ch]
if other_rms:
    check("이벤트 채널 RMS > 평균 배경 RMS",
          ev_rms > sum(other_rms) / len(other_rms),
          f"event={ev_rms:.6f}, bg_avg={sum(other_rms)/len(other_rms):.6f}")

# ────────────────────────────────────────────────────────────────
# Phase 2~3: STFT + 피처 추출
# ────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("  Phase 2~3: STFT + 피처 추출 (stft.py)")
print("=" * 60)

from module import stft

# 단일 채널 (LPMS 호환 인터페이스)
print(f"\n  [단일 채널 STFT - 이벤트 채널 {h.event_ch}]")
img_dir    = os.path.join(OUTPUT_DIR, "image")
origin_dir = os.path.join(OUTPUT_DIR, "origin_image")
os.makedirs(img_dir,    exist_ok=True)
os.makedirs(origin_dir, exist_ok=True)

max_val, max_freq, max_time = stft.perform_stft(
    stft_file_name        = os.path.join(img_dir,    "event_ch"),
    stft_file_name_origin = os.path.join(origin_dir, "event_ch"),
    eventChData           = ev_data,
    fs                    = h.sampling_rate,
)
print(f"  max_val={max_val}  max_freq={max_freq} Hz  max_time={max_time} ms")

check("단일채널 max_val > 0",     max_val > 0)
check("단일채널 max_freq > 0",    max_freq > 0)
check("단일채널 Spectrogram 생성됨",
      os.path.exists(os.path.join(img_dir, "event_ch.png")))

# 전 채널 STFT
print(f"\n  [전 채널 STFT]")
results = stft.perform_stft_all_ch(
    output_dir = OUTPUT_DIR,
    alms_data  = alms_data,
)

check("전채널 결과 수 == total_ch", len(results) == h.total_ch, f"{len(results)}")

for r in results:
    ch_label = f"CH{r['ch_index']} ({r['ch_name']})"
    check(f"  {ch_label} max_val > 0",    r["max_val"] > 0,  f"{r['max_val']}")
    check(f"  {ch_label} rms > 0",        r["rms"] > 0,      f"{r['rms']}")
    check(f"  {ch_label} peak_60k >= 0",  r["peak_60k"] >= 0)
    check(f"  {ch_label} peak_100k >= 0", r["peak_100k"] >= 0)
    check(f"  {ch_label} 이미지 생성됨",  os.path.exists(r["img_path"]), r["img_path"])

# 이벤트 채널 피처 추출
ev_result   = stft.get_event_ch_result(results, h.event_ch)
best_result = stft.get_max_ch_result(results)

print(f"\n  [피처 추출 요약]")
print(f"  {'채널':<20} {'RMS':>10} {'peak_60k':>10} {'peak_100k':>10} {'max_freq':>12}")
print(f"  {'-'*65}")
for r in results:
    marker = " ← Event" if r["ch_index"] == h.event_ch else ""
    print(f"  {r['ch_name']:<20} {r['rms']:>10.6f} {r['peak_60k']:>10.4f} "
          f"{r['peak_100k']:>10.4f} {r['max_freq']:>10.1f} Hz{marker}")

check("get_event_ch_result() 반환됨",   ev_result is not None)
check("get_max_ch_result() 반환됨",     best_result is not None)

if ev_result and best_result:
    check("이벤트 채널이 max_val 최대 채널",
          ev_result["ch_index"] == best_result["ch_index"],
          f"event_ch={ev_result['ch_index']}, best_ch={best_result['ch_index']}")

# ────────────────────────────────────────────────────────────────
# 최종 결과
# ────────────────────────────────────────────────────────────────

print()
print("=" * 60)
if errors:
    print(f"  FAIL — {len(errors)}개 실패:")
    for e in errors:
        print(f"    - {e}")
else:
    print(f"  PASS — 모든 Phase 1~3 검증 통과")
print(f"  출력 파일: {OUTPUT_DIR}/")
print("=" * 60)
