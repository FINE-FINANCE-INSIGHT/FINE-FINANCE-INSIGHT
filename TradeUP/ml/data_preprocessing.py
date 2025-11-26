"""
### 최종 데이터 셋 코드 입니다 ###

- FRED (미국 경제지표)
- ECOS (한국 경제지표) 
- Yahoo Finance (환율 + 주요지수)

세 가지 데이터를 병합하고 -> final.csv 로 저장한 뒤
 파생변수를 생성합니다. -> final.csv 로 저장합니다.

 """

import pandas as pd
import requests
import yfinance as yf
from datetime import datetime # 자동 반영 추가
import os

# =========================================================
# 공통 설정
# =========================================================
START_DATE = "2019-10-01"  # 날짜 수정했음 기존 2019-12-31

END_DATE = datetime.today().strftime("%Y-%m-%d")

DATE_RANGE = pd.date_range("2019-10-01", END_DATE, freq="D")
DATE_FRAME = pd.DataFrame({"date": DATE_RANGE})
DATE_FRAME["date"] = DATE_FRAME["date"].dt.strftime("%Y-%m-%d")

# data 폴더 경로 자동 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(BASE_DIR, "data")
os.makedirs(data_dir, exist_ok=True)

# =========================================================
# 공통 FILL 정책 함수
# =========================================================

def safe_ffill(df, col):
    df[col] = df[col].ffill()
    return df

def monthly_safe_ffill(df, col):
    df = df.sort_values("date").copy()
    last_valid_date = df[df[col].notna()]["date"].max()
    mask = df["date"] <= last_valid_date
    df.loc[mask, col] = df.loc[mask, col].ffill()
    return df

def quarterly_safe_ffill(df, col):
    df = df.sort_values("date").copy()
    last_valid_date = df[df[col].notna()]["date"].max()
    mask = df["date"] <= last_valid_date
    df.loc[mask, col] = df.loc[mask, col].ffill()
    return df

def irregular_ffill(df, col):
    df = df.sort_values("date").copy()
    df[col] = df[col].ffill()
    return df

def monthly_update_fill(df, col):
    df = df.sort_values("date").copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_dates = df[df[col].notna()]["date"].tolist()
    if not valid_dates:
        return df
    
    for i, cur_date in enumerate(valid_dates):
        cur_val = df.loc[df["date"] == cur_date, col].values[0]
        cur_month = pd.to_datetime(cur_date).strftime("%Y-%m")

        mask = df["date"].str.startswith(cur_month)
        df.loc[mask, col] = cur_val

        if i > 0:
            prev_date = valid_dates[i - 1]
            prev_val = df.loc[df["date"] == prev_date, col].values[0]
            prev_end = pd.to_datetime(cur_date) - pd.Timedelta(days=1)
            between_mask = (df["date"] > prev_date) & (df["date"] <= prev_end.strftime("%Y-%m-%d"))
            df.loc[between_mask, col] = prev_val

    df[col] = df[col].ffill()
    return df
    

# =========================================================
# 1. FRED (미국 데이터)
# =========================================================
print("\n[1] 미국 FRED 데이터 수집 중...")

FRED_API_KEY = "437f7530f06e8e0320f082cca21941d9"
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

series = { 
    "WTI": "DCOILWTICO",
    "US_Upper": "DFEDTARU",
    "US_Lower": "DFEDTARL",
    "CPIAUCSL": "CPIAUCSL",
    "CPILFESL": "CPILFESL",
    "PPIACO": "PPIACO",
    "PALLFNFINDEXQ": "PALLFNFINDEXQ",
    "DFF" : "DFF"
}

def fetch_fred_series(series_id):
    url = (
        f"{BASE_URL}?series_id={series_id}"
        f"&api_key={FRED_API_KEY}"
        f"&file_type=json"
        f"&observation_start={START_DATE}"
        f"&observation_end={END_DATE}"
    )
    res = requests.get(url)
    data = res.json()

    if "observations" not in data:
        print(f"[ERROR] {series_id} 수집 실패:", data)
        return None

    df = pd.DataFrame(data["observations"])[["date", "value"]]
    df.columns = ["date", series_id]
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df

fred_dfs = []
for name, sid in series.items():
    print(f"- {name} ({sid}) 다운로드 중...")
    df = fetch_fred_series(sid)
    if df is not None:
        df.columns = ["date", name]
        fred_dfs.append(df)

df_fred = fred_dfs[0]
for df in fred_dfs[1:]:
    df_fred = df_fred.merge(df, on="date", how="outer")

df_fred["US_Avg"] = (df_fred["US_Upper"] + df_fred["US_Lower"]) / 2

# 2020-01-01 WTI 보정
if pd.isna(df_fred.loc[df_fred["date"] == "2020-01-01", "WTI"]).any():
    print("2020-01-01 WTI 누락 → 2019-12-31값으로 보정")
    prev_url = (
        f"{BASE_URL}?series_id=DCOILWTICO"
        f"&api_key={FRED_API_KEY}"
        f"&file_type=json"
        f"&observation_start=2019-12-31"
        f"&observation_end=2019-12-31"
    )
    prev_res = requests.get(prev_url).json()
    if "observations" in prev_res and len(prev_res["observations"]) > 0:
        prev_val = float(prev_res["observations"][0]["value"])
        df_fred.loc[df_fred["date"] == "2020-01-01", "WTI"] = prev_val
        print(f"보정 완료: 2019-12-31({prev_val}) → 2020-01-01")
    else:
        print("2019-12-31 WTI 조회 실패")

daily_cols_fred = ["WTI", "US_Upper", "US_Lower", "DFF", "US_Avg"]
monthly_cols_fred = ["CPIAUCSL", "CPILFESL", "PPIACO"]
quarterly_cols_fred = ["PALLFNFINDEXQ"]

for col in daily_cols_fred:
    df_fred = safe_ffill(df_fred, col)
for col in monthly_cols_fred:
    df_fred = monthly_safe_ffill(df_fred, col)
for col in quarterly_cols_fred:
    df_fred = quarterly_safe_ffill(df_fred, col)

df_fred_full = pd.merge(DATE_FRAME, df_fred, on="date", how="left").sort_values("date")

print("[FRED] 데이터 완료")

# =========================================================
# 2. ECOS (한국 기준금리 + CPI)
# =========================================================
print("\n[2] ECOS 한국 데이터 수집 중...")

API_KEY = "M0EEUL0QP8TTA3RAM4Q9"
START_YM = "202001"
END_YM = datetime.today().strftime("%Y%m")

series_ecos = {
    "KR_Rate": {"stat": "722Y001", "item": "0101000", "desc": "한국 기준금리"},
    "KOR_CPI": {"stat": "901Y009", "item": "0", "desc": "한국 CPI"}
}

macro_dfs = []
for key, info in series_ecos.items():
    print(f"- {info['desc']} 수집 중...")
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/1000/"
        f"{info['stat']}/M/{START_YM}/{END_YM}/{info['item']}"
    )
    res = requests.get(url)
    data = res.json()
    if "StatisticSearch" not in data:
        print(f"[ERROR] {info['desc']} 실패")
        continue

    rows = data["StatisticSearch"]["row"]
    df = pd.DataFrame(rows)[["TIME", "DATA_VALUE"]]
    df.columns = ["date", key]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m").dt.strftime("%Y-%m-%d")
    df[key] = pd.to_numeric(df[key], errors="coerce")
    macro_dfs.append(df)

df_ecos = macro_dfs[0]
for df_t in macro_dfs[1:]:
    df_ecos = df_ecos.merge(df_t, on="date", how="outer")

df_ecos_full = (
    DATE_FRAME
    .merge(df_ecos, on="date", how="left")
    .sort_values("date")
)

df_ecos_full = monthly_update_fill(df_ecos_full, "KOR_CPI")
df_ecos_full = irregular_ffill(df_ecos_full, "KR_Rate")
df_ecos_full["KOR_CPI"] = df_ecos_full["KOR_CPI"].ffill()
df_ecos_full["KR_Rate"] = df_ecos_full["KR_Rate"].ffill()

print("[ECOS] 데이터 완료")

# =========================================================
# 3. Yahoo Finance
# =========================================================
print("\n[3] Yahoo Finance 데이터 수집 중...")

raw = yf.download("USDKRW=X", start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)[["Close"]].reset_index()
raw.columns = ["date", "USD/KRW"]
raw["date"] = pd.to_datetime(raw["date"])

df_yahoo = pd.DataFrame({"date": DATE_RANGE})
df_yahoo["date"] = pd.to_datetime(df_yahoo["date"])
df_yahoo = df_yahoo.merge(raw, on="date", how="left")

if pd.isna(df_yahoo.loc[0, "USD/KRW"]):
    prev_day = raw[raw["date"] == pd.Timestamp("2019-12-31")]["USD/KRW"].values
    if len(prev_day) > 0:
        df_yahoo.loc[0, "USD/KRW"] = prev_day[0]

df_yahoo["USD/KRW"] = df_yahoo["USD/KRW"].ffill()

df_yahoo["MA_7"] = df_yahoo["USD/KRW"].rolling(7, min_periods=1).mean()
df_yahoo["MA_30"] = df_yahoo["USD/KRW"].rolling(30, min_periods=1).mean()
df_yahoo["Change(%)"] = df_yahoo["USD/KRW"].pct_change() * 100
df_yahoo["Change(%)"] = df_yahoo["Change(%)"].fillna(0)

indices = {
    "KOSPI": "^KS11", 
    "SP500": "^GSPC", 
    "VIX": "^VIX", 
    "DXY": "DX-Y.NYB"
}

for name, ticker in indices.items():
    print(f"- {name} ({ticker}) 다운로드 중...")
    temp = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)[["Close"]].reset_index()
    temp.columns = ["date", name]
    temp["date"] = pd.to_datetime(temp["date"])
    df_yahoo = df_yahoo.merge(temp, on="date", how="left")

    prev_val = temp[temp["date"] == pd.Timestamp("2019-12-31")][name].values
    if len(prev_val) > 0 and pd.isna(df_yahoo.loc[0, name]):
        df_yahoo.loc[0, name] = prev_val[0]

    df_yahoo[name] = df_yahoo[name].ffill().bfill()

df_yahoo["date"] = df_yahoo["date"].dt.strftime("%Y-%m-%d")

print("[Yahoo] 데이터 완료")

# =========================================================
# 4. 최종 병합
# =========================================================
print("\n[4] 세 데이터셋 병합 중...")

merged = (
    df_yahoo
    .merge(df_ecos_full, on="date", how="left")
    .merge(df_fred_full, on="date", how="left")
)
merged = merged.sort_values("date")

merged_path = os.path.join(data_dir, "final.csv")
merged.to_csv(merged_path, index=False, encoding="utf-8-sig")

print(f"최종 병합 완료: {merged_path}")
print(merged.head(10))
print(merged.tail(10))

# =========================================================
# 5. 파생변수 생성 
# =========================================================
print("\n[5] 파생변수 생성 중...")

merged = pd.read_csv(merged_path)

merged["RateDiff(US-KR)"] = merged["US_Avg"] - merged["KR_Rate"]
merged["CPI_DIFF(KR-US)"] = merged["KOR_CPI"] - merged["CPIAUCSL"]
merged["US_PolicyGap"] = merged["DFF"] - merged["US_Avg"]
merged["vol_30d"] = merged["USD/KRW"].pct_change().rolling(30).std() * (252 ** 0.5)
merged["mom_90d"] = merged["USD/KRW"].pct_change(90) * 100

merged = merged[merged["date"] >= "2020-01-01"]

cols = [
    "date",
    "US_Upper", "US_Lower", "US_Avg", "DFF", "US_PolicyGap",
    "KR_Rate", "RateDiff(US-KR)",
    "WTI", "KOSPI", "SP500", "VIX", "DXY",
    "KOR_CPI", "CPIAUCSL", "CPILFESL", "CPI_DIFF(KR-US)",
    "PPIACO", "PALLFNFINDEXQ",
    "USD/KRW", "MA_7", "MA_30", "Change(%)",
    "vol_30d", "mom_90d"
]
merged = merged[cols]

# =========================================================
# 6. Lag 변수 생성 및 최종 저장
# =========================================================

print("\n[6] Lag 변수 생성 중 ...")

lag_features =  ["USD/KRW", "DXY", "KOSPI", "SP500", "VIX"]
for col in lag_features:
    for lag in [1, 3, 7]:
        merged[f"{col}_lag{lag}"] = merged[col].shift(lag)

merged["Change(%)_lag1"] = merged["Change(%)"].shift(1)
merged["vol_30d_lag1"] = merged["vol_30d"].shift(1)
merged["mom_90d_lag1"] = merged["mom_90d"].shift(1)

merged["date"] = pd.to_datetime(merged["date"])
merged["year"] = merged["date"].dt.year
merged["month"] = merged["date"].dt.month
merged["day"] = merged["date"].dt.day
merged["weekday"] = merged["date"].dt.weekday

merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")

merged = merged.ffill().bfill()

merged = merged[merged["date"] >= "2020-01-01"].reset_index(drop=True)

final_path = os.path.join(data_dir, "final.csv")
merged.to_csv(final_path, index=False, encoding="utf-8-sig")

print(f"\n최종 데이터 저장 완료: {final_path}")
print(merged.head(10))
print(merged.tail(10))
