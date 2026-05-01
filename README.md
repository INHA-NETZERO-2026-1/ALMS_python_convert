# ALMS BIN Viewer

원자력 발전소 음향누설감시계통(ALMS) BIN 파일 분석 도구입니다.  
현재 범위는 **Phase 1~3: BIN 파싱, STFT 시각화, 특징값 추출**입니다.

---

## 개요

기존 C# Viewer SW의 BIN 로딩/CSV 내보내기 흐름을 Python으로 옮기고, ALMS 이벤트 데이터를 채널별로 분석할 수 있도록 구성했습니다.

| 단계 | 상태 | 구현 내용 |
|------|------|-----------|
| Phase 1 | 완료 | ALMS BIN 파싱, 헤더/채널 데이터 추출, CSV 내보내기 |
| Phase 2 | 완료 | 단일/전채널 STFT 이미지 생성, PyQt5 Viewer |
| Phase 3 | 완료 | RMS, 60 kHz, 100 kHz 피크 등 특징값 추출 |
| Phase 4 | 예정 | 실제 데이터 수집 및 라벨링 |
| Phase 5 | 예정 | ANN/CNN 모델 학습 |
| Phase 6 | 예정 | Viewer에 AI 추론 결과 통합 |

---

## 파일 구성

```text
Carbon1/
├── module/
│   ├── parser.py          # ALMS BIN 파서 및 LPMS 호환 API
│   ├── fourier.py         # STFT/Fourier 처리 엔진
│   └── stft.py            # 단일/전채널 STFT, 특징값 추출
├── viewer.py              # PyQt5 기반 ALMS BIN Viewer
├── test_pipeline.py       # Phase 1~3 통합 검증 스크립트
├── CLAUDE.md              # 개발 메모
├── README.md
└── test_data/             # 샘플/출력 데이터 디렉터리(.gitignore 대상)
```

참고 자료:

- `1.DataFormat_ALMS이벤트 파일구조_R4_2024_1216.xlsx`: ALMS 이벤트 파일 구조 문서
- `test_data/2-2.Sample Data_ALMS_TRIGGER_250905054513824.bin`: 현재 테스트 파이프라인에서 사용하는 샘플 BIN 파일

---

## 설치

```bash
pip install PyQt5 numpy scipy matplotlib
```

Python 3.10 이상 환경을 권장합니다.

---

## 실행

### 통합 파이프라인 검증

```bash
python test_pipeline.py
```

`test_pipeline.py`는 아래 샘플 파일을 기준으로 파싱, CSV 출력, 단일 채널 STFT, 전채널 STFT, 특징값 추출을 검증합니다.

```text
test_data/2-2.Sample Data_ALMS_TRIGGER_250905054513824.bin
```

출력 파일은 기본적으로 `test_data/pipeline_out/` 아래에 생성됩니다.

### Viewer 실행

```bash
python viewer.py
```

Viewer 기능:

| 탭 | 내용 |
|----|------|
| Raw Signal | 선택 채널 파형 및 RMS 확인 |
| STFT Spectrogram | Window, overlap, window 함수, 주파수 상한 설정 후 단일 채널 STFT 표시 |
| 피처 분석 | 전채널 RMS, Max Val, Max Freq, Peak@60kHz, Peak@100kHz 테이블 및 RMS 바 차트 |
| 전채널 개요 | 전체 채널 파형 개요 |

---

## Phase 1: BIN 파싱

구현 파일: `module/parser.py`

C# `tsCSVExport_Click` 로직을 기준으로 ALMS BIN 파일의 글로벌 헤더와 채널별 헤더/Raw Data를 읽습니다.

### BIN 글로벌 헤더

| 바이트 | 크기 | 내용 |
|--------|------|------|
| 0 | 8 | SITE 이름 |
| 8 | 2 | System Type (`1 = ALMS`) |
| 10 | 2 | Event Channel 번호 |
| 12 | 2 | Total Channel 수 |
| 14 | 2 | Skip |
| 16 | 24 | Event Date |
| 40 | 2 | Alarm Result |
| 42 | 6 | Skip |
| 48 | 4 | Sampling Rate (Hz, int32) |
| 52 | 2 | User ID |
| 54 | 2 | Skip |
| 56 | 4 | Event Duration (ms, int32) |
| 60 | 2 | Signal Type |
| 62 | 18 | Skip |
| 80+ | - | 채널별 데이터 |

채널 1개는 `172 bytes` 채널 헤더와 `float32 * n_samples` Raw Data로 구성됩니다.

```python
from module.parser import (
    parse_alms_bin,
    getEventData,
    getAllData,
    getEventDate,
    readEventCh,
    exportCSV,
)

bin_path = "test_data/2-2.Sample Data_ALMS_TRIGGER_250905054513824.bin"

alms_data = parse_alms_bin(bin_path)
h = alms_data.header

print(h.site, h.event_ch, h.total_ch, h.sampling_rate, h.event_duration, h.n_samples)

event_data = getEventData(bin_path)       # list[float]
all_data = getAllData(bin_path)           # list[list[float]]
event_date = getEventDate(bin_path)       # str

with open(bin_path, "rb") as f:
    event_ch = readEventCh(f)             # int

exportCSV(bin_path, "test_data/output_check.csv")
```

### 주요 데이터 클래스

| 클래스 | 역할 |
|--------|------|
| `ALMSHeader` | 사이트, 이벤트 채널, 채널 수, 샘플링레이트, 이벤트 시간 등 글로벌 헤더 |
| `ALMSChannel` | 채널 설정값, Raw Data, RMS 계산 프로퍼티 |
| `ALMSData` | 전체 헤더/채널 목록과 채널 접근 헬퍼 |

---

## Phase 2~3: STFT 및 특징값 추출

구현 파일: `module/stft.py`, `module/fourier.py`

### 기본 STFT 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `ub` | `600000` Hz | AE 센서 분석 주파수 상한 |
| `nFFT` | `1024` | STFT FFT 크기 |
| `frame_size` | `None` | 전체 이벤트 구간 분석 |

```python
from module import stft
from module.parser import parse_alms_bin

bin_path = "test_data/2-2.Sample Data_ALMS_TRIGGER_250905054513824.bin"
alms_data = parse_alms_bin(bin_path)
h = alms_data.header

# 단일 이벤트 채널 STFT
max_val, max_freq, max_time = stft.perform_stft(
    stft_file_name="test_data/pipeline_out/image/event_ch",
    stft_file_name_origin="test_data/pipeline_out/origin_image/event_ch",
    eventChData=alms_data.get_event_channel_data().tolist(),
    fs=h.sampling_rate,
)

# 전채널 STFT + 특징값 추출
results = stft.perform_stft_all_ch(
    output_dir="test_data/pipeline_out",
    alms_data=alms_data,
)

event_result = stft.get_event_ch_result(results, h.event_ch)
best_result = stft.get_max_ch_result(results)
features = stft.extract_features(alms_data)
```

`perform_stft_all_ch()` 결과는 채널별로 아래 값을 포함합니다.

| 키 | 설명 |
|----|------|
| `ch_name`, `ch_index` | 채널명 및 채널 인덱스 |
| `max_val`, `max_freq`, `max_time` | STFT 최대값, 주파수, 시간(ms) |
| `rms` | 채널 Raw Data RMS |
| `peak_60k`, `peak_100k` | 60 kHz, 100 kHz 주변 피크 |
| `img_path`, `img_path_origin` | 생성된 STFT 이미지 경로 |

---

## 테스트 기준

`test_pipeline.py`는 다음 항목을 확인합니다. 스크립트는 모든 항목을 출력한 뒤 결과 요약을 표시합니다.

- System Type이 ALMS(`1`)인지 확인
- 전체 채널 수와 채널별 샘플 수 검증
- `readEventCh()`, `getEventData()`, `getAllData()`, `getEventDate()` 동작 확인
- `exportCSV()` 출력 파일 생성 확인
- 단일 이벤트 채널 STFT 이미지 생성 확인
- 전채널 STFT 결과와 RMS/피크 특징값 생성 확인
- 이벤트 채널 결과와 최대 STFT 채널 결과 조회 확인

현재 샘플 파일(`2-2.Sample Data_ALMS_TRIGGER_250905054513824.bin`) 기준으로는 파싱과 전채널 STFT 생성은 정상 동작하지만, 이벤트 채널(`CH0`)이 RMS/Max Val 최대 채널이라는 가정은 맞지 않습니다. 따라서 아래 검증은 실패할 수 있습니다.

- 이벤트 채널 RMS > 평균 배경 RMS
- 단일 이벤트 채널 `max_val > 0`
- 단일 이벤트 채널 `max_freq > 0`
- 이벤트 채널이 `max_val` 최대 채널

---

## 주의사항

- `fs`는 하드코딩하지 말고 반드시 `alms_data.header.sampling_rate` 값을 사용합니다.
- 샘플 BIN과 STFT 출력물은 용량이 커질 수 있으므로 `test_data/`는 Git 추적 대상에서 제외되어 있습니다.
- `parse_alms_bin()`은 System Type이 `1(ALMS)`가 아니면 `ValueError`를 발생시킵니다.
- `module/` 아래 파일은 현재 import 기준 경로입니다. 예전 `temp/` 실행 경로는 사용하지 않습니다.

---

## 다음 단계

### Phase 4: 데이터 수집 및 라벨링

- 실제 ALMS BIN 파일 추가 확보
- C# Viewer CSV 출력과 `exportCSV()` 결과 비교
- 누설 크기, 모양, 압력 라벨 체계 확정

### Phase 5: 모델 학습

- 소량 데이터: RMS/주파수 피크 기반 `sklearn` MLPClassifier 검토
- 충분한 이미지 데이터: Spectrogram 기반 CNN 검토

예상 분류 대상:

- 누설 크기: hole/slit별 크기
- 누설 모양: `hole`, `slit`
- 압력: 실험 조건별 pressure class

### Phase 6: Viewer 통합

- `predictor.py` 또는 동등한 추론 모듈 추가
- Viewer에 AI 분류 결과 탭 추가
- 분석 결과 저장/리포트 기능 검토
