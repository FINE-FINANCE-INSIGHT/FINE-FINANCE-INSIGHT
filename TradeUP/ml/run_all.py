"""
run_all.py — 데이터 전처리 + 모델 학습 실행 파일

이 파일 하나만 실행하면 아래 순서대로 자동 실행됩니다.

1) data_preprocessing.py  → final.csv 생성
2) train_model.py         → 모델 학습 및 결과 JSON 생성
"""

import os
import sys
import subprocess

# 현재 run_all.py 기준 경로
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))      # .../TradeUP/ml
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                   # .../TradeUP
ML_DIR = os.path.join(PROJECT_ROOT, "ml")

# 파일 경로 설정
PREPROCESS_PATH = os.path.join(ML_DIR, "data_preprocessing.py")
TRAIN_PATH = os.path.join(ML_DIR, "train_model.py")

def run_script(script_path):
    """Python 파일을 서브프로세스로 실행하는 함수"""
    print("\n------------------------------------------------------")
    print(f"실행 중: {os.path.basename(script_path)}")
    print("------------------------------------------------------\n")

    result = subprocess.run([sys.executable, script_path], text=True)

    if result.returncode != 0:
        print(f"오류 발생: {os.path.basename(script_path)} 실행 실패")
        sys.exit(1)
    else:
        print(f"완료: {os.path.basename(script_path)}\n")


def main():
    print("------------------------------------------------------")
    print("전처리 → 모델학습 전체 실행 시작")
    print("------------------------------------------------------")

    # 1) 데이터 전처리
    run_script(PREPROCESS_PATH)

    # 2) 모델 학습
    run_script(TRAIN_PATH)

    print("------------------------------------------------------")
    print("전처리 + 모델학습 전체 작업 완료")
    print("------------------------------------------------------\n")


if __name__ == "__main__":
    main()
