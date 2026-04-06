#!/usr/bin/env python3
"""
ALMS BIN Viewer 실행 스크립트
"""
import sys
import os

# 현재 디렉토리를 패스에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 더미 파일이 없으면 생성
test_dir = os.path.join(os.path.dirname(__file__), "../test_data")
test_file = os.path.join(test_dir, "ALMS_TEST_EVENT_20240910_123000.bin")

if not os.path.exists(test_file):
    print("테스트 BIN 파일 생성 중...")
    os.makedirs(test_dir, exist_ok=True)
    from generate_test_bin import generate_dummy_bin
    generate_dummy_bin(test_file, n_channels=6, sampling_rate=500000, duration_ms=100)
    print(f"테스트 파일 생성 완료: {test_file}\n")

from viewer import main
main()