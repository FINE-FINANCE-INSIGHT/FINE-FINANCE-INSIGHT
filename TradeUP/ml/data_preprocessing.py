"""
===============================================================
📌 data_preprocessing.py — 데이터 수집 + 병합 + 전처리 (FINAL)
===============================================================
"""

import os
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime

# ================================================================
# Python Module Path 문제 해결 (VSCode 실행 지원)
# ================================================================
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 이제 ml.config import가 어디서 실행해도 됨
from ml.config import (
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FINAL_CSV_PATH,
    FRED_API_KEY,
    ECOS_API_KEY,
    log
)


# ================================================================
# 날짜 범위 설정
# ================================================================
START_DATE = "2019-12-31"
TODAY = datetime.today().strftime("%Y-%m-%d")

DATE_RANGE = pd.date_range("2020-01-01", TODAY, freq="D")
DATE_FRAME = pd.DataFrame({"date": DATE_RANGE})
DATE_FRAME["date"] = DATE_FRAME["date"].dt.strftime("%Y-%m-%d")


# ================================================================
# 1) FRED API
# ================================================================
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_fred_series(series_id):
    """
    FRED 단일 시계열 수집 함수.
    TradeUP에서는 문자열 줄바꿈 시 파라미터 깨짐이 발생하므로,
    한 줄로 깔끔하게 연결되는 방식으로 구성해야 한다.
    """

    url = (
        f"{BASE_URL}"
        f"?series_id={series_id}"
        f"&api_key={FRED_API_KEY}"
        f"&file_type=json"
        f"&observation_start={START_DATE}"
        f"&observation_end={TODAY}"
    )

    response = requests.get(url)
    data = response.json()

    if "observations" not in data:
        print("[DEBUG FRED ERROR RESPONSE]", data)
        log(f"❌ FRED 수집 실패: {series_id}")
        return None

    df = pd.DataFrame(data["observations"])[["date", "value"]]
    df.columns = ["date", series_id]
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df


def load_fred_data():
    log("📡 [FRED] 데이터 수집 시작")

    series = {
        "WTI": "DCOILWTICO",
        "US_Upper": "DFEDTARU",
        "US_Lower": "DFEDTARL",
        "CPIAUCSL": "CPIAUCSL",
        "CPILFESL": "CPILFESL",
        "PPIACO": "PPIACO",
        "PALLFNFINDEXQ": "PALLFNFINDEXQ"
    }

    dfs = []
    for name, sid in series.items():
        df = fetch_fred_series(sid)
        if df is not None:
            df.columns = ["date", name]
            dfs.append(df)

    if len(dfs) == 0:
        raise RuntimeError("FRED 데이터 수집 실패 (API Key 또는 URL 문제)")

    final = dfs[0]
    for df in dfs[1:]:
        final = final.merge(df, on="date", how="outer")

    final["US_Avg"] = (final["US_Upper"] + final["US_Lower"]) / 2

    # 날짜 보정 + 결측치 보완
    final = pd.merge(DATE_FRAME, final, on="date", how="left").sort_values("date").ffill()

    log("✅ [FRED] 수집 완료")
    return final


# ================================================================
# 2) ECOS 한국 데이터
# ================================================================
def load_ecos_data():
    log("📡 [ECOS] 한국 데이터 수집 시작")

    START_YM = "202001"
    END_YM = datetime.today().strftime("%Y%m")

    series = {
        "KR_Rate": {"stat": "722Y001", "item": "0101000"},
        "KOR_CPI": {"stat": "901Y009", "item": "0"},
    }

    dfs = []
    for key, info in series.items():
        ecos_url = (
            f"https://ecos.bok.or.kr/api/StatisticSearch/{ECOS_API_KEY}/json/kr/1/1000/"
            f"{info['stat']}/M/{START_YM}/{END_YM}/{info['item']}"
        )
        res = requests.get(ecos_url).json()

        rows = res["StatisticSearch"]["row"]
        df = pd.DataFrame(rows)[["TIME", "DATA_VALUE"]]
        df.columns = ["date", key]
        df["date"] = pd.to_datetime(df["date"], format="%Y%m").dt.strftime("%Y-%m-%d")
        df[key] = pd.to_numeric(df[key], errors="coerce")
        dfs.append(df)

    final = dfs[0]
    for df in dfs[1:]:
        final = final.merge(df, on="date", how="outer")

    final = pd.merge(DATE_FRAME, final, on="date", how="left").sort_values("date").ffill()

    log("✅ [ECOS] 수집 완료")
    return final


# ================================================================
# 3) Yahoo Finance
# ================================================================
def load_yahoo_data():
    log("📡 [Yahoo] 금융시장 데이터 수집 시작")

    raw = yf.download("USDKRW=X", start=START_DATE, end=TODAY, progress=False)[["Close"]].reset_index()
    raw.columns = ["date", "USD/KRW"]
    raw["date"] = pd.to_datetime(raw["date"])

    df = pd.DataFrame({"date": DATE_RANGE})
    df = df.merge(raw, on="date", how="left")
    df["USD/KRW"] = df["USD/KRW"].ffill()

    df["MA_7"] = df["USD/KRW"].rolling(7).mean()
    df["MA_30"] = df["USD/KRW"].rolling(30).mean()
    df["Change(%)"] = df["USD/KRW"].pct_change() * 100
    df["Change(%)"] = df["Change(%)"].fillna(0)

    indices = {
        "KOSPI": "^KS11",
        "SP500": "^GSPC",
        "VIX": "^VIX",
        "DXY": "DX-Y.NYB",
    }

    for name, ticker in indices.items():
        tdf = yf.download(ticker, start=START_DATE, end=TODAY, progress=False)[["Close"]].reset_index()
        tdf.columns = ["date", name]
        tdf["date"] = pd.to_datetime(tdf["date"])
        df = df.merge(tdf, on="date", how="left")
        df[name] = df[name].ffill().bfill()

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    log("✅ [Yahoo] 수집 완료")
    return df


# ================================================================
# 4) 병합 & 파생변수 생성
# ================================================================
def build_final_csv():
    log("[FINAL] 최종 데이터 병합 및 파생변수 생성 시작")

    df_fred = load_fred_data()
    df_ecos = load_ecos_data()
    df_yahoo = load_yahoo_data()

    merged = df_yahoo.merge(df_ecos, on="date").merge(df_fred, on="date")
    merged = merged.sort_values("date")

    merged["RateDiff(US-KR)"] = merged["US_Avg"] - merged["KR_Rate"]
    merged["CPI_DIFF(KR-US)"] = merged["KOR_CPI"] - merged["CPIAUCSL"]
    merged["vol_30d"] = merged["USD/KRW"].pct_change().rolling(30).std() * np.sqrt(252)
    merged["mom_90d"] = merged["USD/KRW"].pct_change(90) * 100

    merged.to_csv(FINAL_CSV_PATH, index=False, encoding="utf-8-sig")
    log(f"📁 final.csv 저장 완료 → {FINAL_CSV_PATH}")

    return merged


# ================================================================
# 5) 오늘 예측용 X_today 생성
# ================================================================
def generate_today_features():
    """
    final.csv 기반으로 오늘 예측용 1-row feature(X_today)를 생성.
    train_model.py에서 사용한 feature engineering(특히 lag)과 동일하게 처리해야 함.
    """
    if not os.path.exists(FINAL_CSV_PATH):
        build_final_csv()

    df = pd.read_csv(FINAL_CSV_PATH)

    # -----------------------------
    # train_model.py와 동일한 처리
    # -----------------------------
    df["log_usdkrw"] = np.log(df["USD/KRW"])
    df["log_diff"] = df["log_usdkrw"].diff()

    # lag feature 생성 (중요)
    for lag in [1, 2, 3]:
        df[f"USD/KRW_lag{lag}"] = df["USD/KRW"].shift(lag)

    df = df.dropna().reset_index(drop=True)

    # 오늘 데이터 1행
    X_today = df.iloc[-1:].drop(columns=["date", "log_usdkrw", "log_diff"])

    return X_today



# ================================================================
# 6) VSCode 직접 실행 지원
# ================================================================
if __name__ == "__main__":
    print("🚀 data_preprocessing.py 실행 시작!")

    df = build_final_csv()
    print("✅ final.csv 생성 완료:", df.shape)

    X_today = generate_today_features()
    print("\n🎯 오늘 예측용 X_today:")
    print(X_today)

    print("\n🎉 data_preprocessing.py 실행 완료")
