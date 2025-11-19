"""
===========================================================
📌 config.py  — ML 파트 공통 설정 파일 (FINAL)
===========================================================
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
from datetime import datetime

# ============================================================
# 1️⃣ 기본 경로 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # ml/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                    # TradeUP/
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ============================================================
# 2️⃣ data/ 하위 폴더 자동 생성
# ============================================================
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RESULT_DIR = os.path.join(DATA_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================
# 3️⃣ 공통 파일 경로
# ============================================================
FINAL_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "final.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

def RESULT_JSON_PATH():
    today = datetime.today().strftime("%Y-%m-%d")
    return os.path.join(RESULT_DIR, f"{today}.json")

# ============================================================
# 4️⃣ 외부 API Key (실제 키 적용된 버전)
# ============================================================
# 반드시 타이핑으로 입력하면 오류 없음!

FRED_API_KEY = "437f7530f06e8e0320f082cca21941d9"   # ✔ FRED API KEY
ECOS_API_KEY = "M0EEUL0QP8TTA3RAM4Q9"               # ✔ ECOS API KEY

# ============================================================
# 5️⃣ 공통 로그 함수
# ============================================================
def log(msg: str):
    print(f"[ML-CONFIG] {msg}")
