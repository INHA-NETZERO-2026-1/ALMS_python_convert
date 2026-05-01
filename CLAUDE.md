# ALMS BIN Viewer — CLAUDE.md

원자력 발전소 배관 누설 감시 시스템(ALMS) BIN 파일 분석 SW.  
탄소중립 6기 프로젝트. Phase 1~3 완성, Phase 4~6 미착수.

---

## 프로젝트 구조

```
Carbon1/
├── bin_reader.py          # ALMS BIN 파서 (C# 1:1 변환)
├── fourier_lib.py         # STFT 핵심 엔진
├── stft.py                # STFT 래퍼 + 전채널 분석 + 피처 추출
├── alms_viewer.py         # PyQt5 STFT Viewer GUI
├── alms_parser.py         # 초기 파서 (참고용, bin_reader.py가 대체)
├── run.py                 # Viewer 실행 진입점
├── temp/
│   ├── generate_test_bin.py  # 더미 BIN 파일 생성기
│   ├── viewer.py             # (작업용 임시)
│   └── parser.py             # (작업용 임시)
└── test_data/             # 테스트 BIN 파일 저장 위치
```

---

## 개발 로드맵

| Phase | 내용 | 상태 |
|-------|------|------|
| 1 | BIN 파일 파싱 | 완성 |
| 2 | STFT Viewer (PyQt5) | 완성 |
| 3 | 특징값 추출 | 완성 |
| 4 | 데이터 수집 및 라벨링 | 미착수 (실제 BIN 파일 필요) |
| 5 | ANN 모델 학습 | 미착수 |
| 6 | Viewer에 AI 통합 | 미착수 |

---

## 핵심 상수 (절대 하드코딩 금지)

```python
ALMS_UB        = 600000  # 주파수 상한 Hz (AE 센서 범위)
ALMS_NFFT      = 1024    # Window 크기 (주파수 해상도 488 Hz/bin)
ALMS_FRAMESIZE = None    # None = 전구간 분석 (누설은 연속 신호)
```

`fs`(샘플링레이트)는 반드시 `alms_data.header.sampling_rate`에서 읽어야 한다. 절대 하드코딩 금지.

---

## 핵심 데이터 클래스 (`bin_reader.py`)

```python
ALMSHeader   # 글로벌 헤더: site, event_ch, total_ch, sampling_rate, event_duration 등
ALMSChannel  # 채널: ch_no, ch_name, raw_data (np.ndarray, float32)
ALMSData     # 전체: header + channels (list)
```

---

## 주요 API

### `bin_reader.py`

```python
# LPMS 호환 인터페이스 (imageGenerator.py, reportCreator.py 수정 최소화)
readEventCh(file)        → int
getEventData(bin_path)   → list[float]
getAllData(bin_path)      → list[list]
getEventDate(bin_path)   → str

# ALMS 전용
getHeader(bin_path)      → ALMSHeader
getALMSData(bin_path)    → ALMSData
exportCSV(bin_path, out) → None
parse_alms_bin(bin_path) → ALMSData   # System Type != 1이면 ValueError
```

### `stft.py`

```python
# 단일 채널 (LPMS 호환)
max_val, max_freq, max_time = perform_stft(
    stft_file_name, stft_file_name_origin,
    eventChData, fs,
    ub=600000, nFFT=1024, frame_size=None
)
# max_time 단위: ms (밀리초)

# 전채널 분석 (ALMS 전용)
results = perform_stft_all_ch(output_dir, alms_data, ub, nFFT, frame_size)
# results: list[dict], 키: ch_name, ch_index, max_val, max_freq, max_time,
#          rms, peak_60k, peak_100k, img_path, img_path_origin

get_event_ch_result(results, event_ch_index)  # 이벤트 채널 결과
get_max_ch_result(results)                    # max_val 최대 채널 결과
```

### `fourier_lib.py`

```python
obj = Fourier_obj(val, dt, fs, nFFT, hop=None)  # hop 미지정 시 nFFT//4
stft_result, freqs, times = obj.focus(FrameSize=None)  # 전구간
obj.analyze(stft_file_name, ub)        # Spectrogram + 파형 + 주파수축
obj.analyze_1(stft_file_name, ub)      # 축 없는 Spectrogram (AI 학습용)
max_val, max_freq, max_time = obj.get_max_info()
peak = obj.get_peak_at(center_hz=60000, bandwidth_hz=5000)
```

---

## 주의사항

**메모리:** `hop=1`은 50,000샘플에서 4GB 초과. `hop = nFFT // 4` (75% overlap) 유지.

**System Type 검증:** `parse_alms_bin()`은 System Type != 1이면 `ValueError`. LPMS(0)/RCPVMS(2)/IVMS(3) 파일을 넣으면 오류가 나는 것이 정상.

**LPMS 호환:** `getEventData()`, `getAllData()`, `getEventDate()`, `readEventCh()`는 LPMS와 동일한 인터페이스. `imageGenerator.py` 수정 시 이 함수들을 우선 활용.

---

## LPMS vs ALMS 파라미터 차이

| 항목 | LPMS | ALMS |
|------|------|------|
| `ub` | 25,000 Hz | **600,000 Hz** |
| `nFFT` | 128 | **1024** |
| `hop` | 1 | **nFFT // 4** |
| `focus()` | FrameSize=2000 | **FrameSize=None** |
| `fs` | 200,000 Hz 하드코딩 | **헤더에서 읽음** |
| 채널 수 | 18채널 고정 | **헤더에서 읽음** |
| 샘플 수 | 20,000개 고정 | **SamplingRate × Duration / 1000** |

---

## Phase 4~6 구현 가이드

### Phase 4 — 데이터 수집

실제 ALMS BIN 파일 입수 후:
1. `exportCSV()`로 C# SW 출력과 파싱 결과 비교 검증
2. 누설 조건별 분류 (크기/모양/압력)
3. `perform_stft_all_ch()`로 전채널 피처 일괄 추출 → CSV

### Phase 5 — ANN 모델

논문 기반 입력 피처 (5개):
```python
features = {
    "ae_rms":        ch.rms,
    "ae_peak_60k":   obj.get_peak_at(60000),
    "ae_peak_100k":  obj.get_peak_at(100000),
    "acc_rms":       acc_ch.rms,
    "acc_peak_5_7k": obj.get_peak_at(6000),
}
```

출력 (3개 분류):
- 누설 크기: 1.0 / 1.4 / 1.7 / 2.0 mm (hole), 7.8 / 15.4 / 22.7 / 31.4 mm (slit)
- 누설 모양: hole / slit
- 압력: 4 / 5 / 6 atm

데이터 수량에 따라:
- 적으면 → 논문 기반 ANN (sklearn MLPClassifier 또는 PyTorch + Levenberg-Marquardt)
- 많으면 → CNN (Spectrogram 이미지 입력)

### Phase 6 — Viewer 통합

- `predictor.py`: ALMS용 모델 추론
- `imageGenerator.py`: `getEventData()` → `getALMSData()` + `perform_stft_all_ch()`로 교체, `fs`는 헤더에서 읽기
- `reportCreator.py`: 18채널 고정 루프 → `range(alms_data.header.total_ch)`로 교체

---

## 실행 순서 (테스트)

```bash
# 1. 테스트 BIN 생성 + 파서 검증
python module/generate_test_bin.py

# 2. bin_reader.py 단독 검증
python bin_reader.py

# 3. stft.py 단독 검증 (이미지 생성)
python stft.py

# 4. Viewer 실행
python run.py
```

---

## 의존성

```bash
pip install PyQt5 numpy scipy matplotlib
# Phase 5 이후
pip install scikit-learn torch torchvision
```

---

## 참고

- 논문: 김영훈 외, "음향방출기법을 이용한 원전 고온 고압 배관의 누설 특성 평가에 관한 연구", 비파괴검사학회지 Vol.29 No.5 (2009)
- 원본 LPMS SW 코드가 별도 보관되어 있음. `imageGenerator.py`, `predictor.py`, `reportCreator.py`는 LPMS 원본을 기반으로 ALMS용 수정 필요.
