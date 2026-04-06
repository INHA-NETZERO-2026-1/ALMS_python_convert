# ALMS BIN Viewer

원자력 발전소 음향누설감시계통(ALMS) BIN 파일 분석 도구  
탄소중립 6기 / **Phase 1~3 구현 완료**

---

## 개요

기존 C# Viewer SW는 BIN 파일을 CSV로만 내보내고 누설 유무(RMS 임계값 초과)만 판단했다.  
이 프로젝트는 **누설 크기 / 모양 / 압력까지 자동 분류**하는 지능형 분석 SW를 목표로 한다.

```
Phase 1  BIN 파일 파싱          ✅ 완성  (parser.py)
Phase 2  STFT Viewer (PyQt5)   ✅ 완성  (temp/viewer.py)
Phase 3  특징값 추출             ✅ 완성  (stft.py)
Phase 4  데이터 수집 및 라벨링   ⬜ 미착수  (실제 BIN 파일 필요)
Phase 5  ANN 모델 학습           ⬜ 미착수
Phase 6  Viewer에 AI 통합        ⬜ 미착수
```

---

## 파일 구성

```
Carbon1/
├── parser.py              # Phase 1: ALMS BIN 파서 (C# 코드 1:1 Python 변환)
├── fourier.py             # Phase 2: STFT 핵심 엔진
├── stft.py                # Phase 3: STFT 래퍼 + 전채널 분석 + 피처 추출
├── test_pipeline.py       # Phase 1~3 통합 파이프라인 테스트
├── temp/
│   ├── viewer.py          # PyQt5 STFT Viewer GUI
│   ├── run.py             # Viewer 실행 진입점
│   └── generate_test_bin.py  # 더미 BIN 파일 생성기
└── test_data/
    └── test.bin           # 실제 ALMS BIN 파일 (SKR#1, 23채널, fs=1MHz)
```

---

## 설치

```bash
pip install PyQt5 numpy scipy matplotlib
```

---

## 실행

```bash
# Phase 1~3 파이프라인 테스트 (터미널)
python test_pipeline.py

# Viewer 실행 (GUI)
python temp/run.py
```

---

## Phase 1: BIN 파싱 (`parser.py`)

C# `tsCSVExport_Click` 코드를 Python으로 1:1 변환.

### BIN 파일 구조 (글로벌 헤더)

| 바이트 | 크기 | 내용 |
|--------|------|------|
| 0 | 8 | SITE 이름 |
| 8 | 2 | System Type (1=ALMS) |
| 10 | 2 | Event Channel 번호 |
| 12 | 2 | Total Channel 수 |
| 14 | 2 | (skip) |
| 16 | 24 | Event Date |
| 40 | 2 | Alarm Result |
| 42 | 6 | (skip) |
| 48 | 4 | Sampling Rate (Hz, int32) |
| 52 | 2 | User ID |
| 54 | 2 | (skip) |
| 56 | 4 | Event Duration (ms, int32) |
| 60 | 2 | Signal Type |
| 62 | 18 | (skip) |
| 80+ | — | 채널별 데이터 × Total_CH |

채널 1개당: 헤더 172 bytes + Raw Data (`float32 × SamplingRate × Duration / 1000`)

### 주요 API

```python
from parser import parse_alms_bin, getEventData, getAllData, getEventDate, readEventCh

# 전체 파싱
alms_data = parse_alms_bin("test_data/test.bin")
h = alms_data.header
# h.site, h.event_ch, h.total_ch, h.sampling_rate, h.event_duration, h.n_samples

# LPMS 호환 인터페이스 (기존 imageGenerator.py 등과 동일)
event_data = getEventData("test_data/test.bin")   # list[float]
all_data   = getAllData("test_data/test.bin")      # list[list[float]]
event_date = getEventDate("test_data/test.bin")   # str
event_ch   = readEventCh(open("test_data/test.bin", "rb"))  # int

# CSV 내보내기 (C# SW 출력과 동일한 포맷)
from parser import exportCSV
exportCSV("test_data/test.bin", "output.csv")
```

### 데이터 클래스

```python
ALMSHeader   # 글로벌 헤더
ALMSChannel  # 채널 설정값 + raw_data (np.ndarray, float32) + rms (property)
ALMSData     # header + channels (list)
```

---

## Phase 2~3: STFT + 피처 추출 (`stft.py`, `fourier.py`)

### STFT 파라미터

| 파라미터 | 값 | 이유 |
|----------|-----|------|
| `ub` (주파수 상한) | 600,000 Hz | AE 센서 대역 100~600 kHz |
| `nFFT` | 1024 | 주파수 해상도 ≈ 488 Hz/bin (60/100 kHz 피크 분리 가능) |
| `hop` | nFFT // 4 | 75% overlap — hop=1은 4 GB 초과 |
| `focus()` | FrameSize=None | 전구간 (누설은 충격이 아닌 연속 신호) |

### 주요 API

```python
import stft
from parser import parse_alms_bin

alms_data = parse_alms_bin("test_data/test.bin")
h = alms_data.header

# 단일 채널 STFT (LPMS 호환)
max_val, max_freq, max_time = stft.perform_stft(
    stft_file_name        = "out/image/event_ch",
    stft_file_name_origin = "out/origin_image/event_ch",
    eventChData           = alms_data.get_event_channel_data().tolist(),
    fs                    = h.sampling_rate,   # 헤더에서 읽어야 함, 하드코딩 금지
)
# max_time 단위: ms

# 전채널 STFT + 피처 추출
results = stft.perform_stft_all_ch(output_dir="out", alms_data=alms_data)
# results: list[dict]
# 키: ch_name, ch_index, max_val, max_freq, max_time, rms, peak_60k, peak_100k,
#     img_path, img_path_origin

# 채널 선택
event_result = stft.get_event_ch_result(results, h.event_ch)
best_result  = stft.get_max_ch_result(results)
```

### 논문 기반 피처 (Phase 5 ANN 입력)

| 피처 | 코드 | 센서 |
|------|------|------|
| AE RMS | `ch.rms` | AE |
| Peak @ 60 kHz | `obj.get_peak_at(60000)` | AE |
| Peak @ 100 kHz | `obj.get_peak_at(100000)` | AE |
| 가속도 RMS | `acc_ch.rms` | 가속도 |
| Peak @ 5~7 kHz | `obj.get_peak_at(6000)` | 가속도 |

---

## Phase 2: Viewer (`temp/viewer.py`)

PyQt5 기반 데스크탑 앱. 탭 4개:

| 탭 | 내용 |
|----|------|
| Raw Signal | 선택 채널 파형 + RMS |
| STFT Spectrogram | Window/Overlap/주파수 상한 설정, 60/100 kHz 라인 표시 |
| 특징값 분석 | 채널별 RMS/Peak 테이블 + 바 차트 |
| 전채널 개요 | 전 채널 파형 한눈에 |

---

## 파이프라인 테스트 결과 (`test_pipeline.py`)

실제 BIN 파일 기준: SKR#1 / 23채널 / fs=1,000,000 Hz / 200 ms

```
Phase 1: BIN 파싱
  [PASS] system_type == 1 (ALMS)
  [PASS] 23채널 모두 샘플수 == 200,000
  [PASS] LPMS 호환 함수 4종 정상 동작
  [PASS] exportCSV() 정상 출력

Phase 2~3: STFT + 피처 추출
  [PASS] 전채널 STFT 이미지 23개 생성
  [PASS] 전채널 RMS / peak_60k / peak_100k 추출
  [PASS] get_event_ch_result() / get_max_ch_result() 정상 반환
```

---

## 주의사항

- **`fs` 하드코딩 금지**: 반드시 `alms_data.header.sampling_rate`에서 읽어서 전달
- **메모리**: `hop=1`은 200,000 샘플에서 메모리 초과 발생. `hop = nFFT // 4` 유지
- **System Type 검증**: `parse_alms_bin()`은 System Type != 1이면 `ValueError` 발생 (정상 동작)

---

## 다음 단계 (Phase 4~6)

### Phase 4 — 데이터 수집
```bash
# 실제 BIN 파일 입수 후
python -c "from parser import exportCSV; exportCSV('real.bin', 'check.csv')"
# → C# SW 출력과 비교 검증
```

### Phase 5 — ANN 학습
```bash
pip install scikit-learn torch torchvision
# 데이터 적음: sklearn MLPClassifier (논문 기반 5 피처 입력)
# 데이터 많음: CNN (Spectrogram 이미지 입력)
```

ANN 출력 (분류 대상):
- 누설 크기: 1.0 / 1.4 / 1.7 / 2.0 mm (hole), 7.8 / 15.4 / 22.7 / 31.4 mm (slit)
- 누설 모양: hole / slit
- 압력: 4 / 5 / 6 atm

### Phase 6 — Viewer 통합
- `predictor.py` 작성 (모델 추론)
- `temp/viewer.py`에 분류 결과 탭 추가

---

## 참고

논문: 김영훈 외, "음향방출기법을 이용한 원전 고온 고압 배관의 누설 특성 평가에 관한 연구",  
비파괴검사학회지 Vol.29 No.5 (2009)
