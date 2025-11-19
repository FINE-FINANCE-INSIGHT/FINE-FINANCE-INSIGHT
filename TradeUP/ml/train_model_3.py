import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber
import joblib

# ======================================================
# 1️⃣ 경로 설정
# ======================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "final.csv")

ML_DIR = os.path.join(BASE_DIR, "ml")
MODEL_DIR = os.path.join(ML_DIR, "artifacts")
OUTPUT_DIR = os.path.join(ML_DIR, "output")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
HISTORY_PATH = os.path.join(MODEL_DIR, "lstm_history.pkl")

print(f"Data path: {DATA_PATH}")

# ======================================================
# 2️⃣ 데이터 로드
# ======================================================
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape}")

# ======================================================
# 3️⃣ diff(log_diff) 생성
# ======================================================
df["USD/KRW_diff"] = np.log(df["USD/KRW"]).diff()
df = df.dropna().reset_index(drop=True)

y_all = df["USD/KRW_diff"].values.astype("float32")

# ======================================================
# 4️⃣ lag + feature 생성 (존재하는 것만 자동 사용)
# ======================================================
# 자동 lag 생성
def add_lag(df, col, lags):
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

df = add_lag(df, "USD/KRW", [1, 3, 7])
df = add_lag(df, "Change(%)", [1])
df = add_lag(df, "VIX", [1, 7])
df = add_lag(df, "DXY", [1, 7])
df = add_lag(df, "KOSPI", [1])
df = add_lag(df, "SP500", [1])
df = add_lag(df, "vol_30d", [1])

# 날짜 계절성
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["weekday"] = df["date"].dt.weekday

df = df.dropna().reset_index(drop=True)

# 모델 입력 feature (존재하는 것만 선택)
feature_candidates = [
    "Change(%)","USD/KRW_lag1","Change(%)_lag1","mom_90d","MA_7","vol_30d",
    "VIX_lag7","USD/KRW_lag7","VIX_lag1","vol_30d_lag1","VIX","USD/KRW_lag3",
    "DXY_lag7","DXY","DXY_lag1","WTI","KOSPI","SP500","SP500_lag1",
    "KOSPI_lag1","weekday","month","RateDiff(US-KR)","CPI_DIFF(KR-US)"
]

features = [f for f in feature_candidates if f in df.columns]
print("Features:", len(features))

# ======================================================
# 5️⃣ 스케일링
# ======================================================
scaler = StandardScaler()
X_all = scaler.fit_transform(df[features].astype("float32"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "lstm_scaler.pkl"))

# ======================================================
# 6️⃣ 시퀀스 생성
# ======================================================
def create_sequences(X, y, seq_len=30):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

SEQ_LEN = 30
X_seq, y_seq = create_sequences(X_all, y_all, seq_len=SEQ_LEN)

# ======================================================
# 7️⃣ train/val/test split
# ======================================================
n = len(X_seq)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

X_train, y_train = X_seq[:train_end], y_seq[:train_end]
X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
X_test, y_test = X_seq[val_end:], y_seq[val_end:]

# ======================================================
# 8️⃣ LSTM 모델 구성
# ======================================================
def build_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(32, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        LSTM(16),
        Dense(8, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss=Huber(delta=0.01))
    return model

model = build_model((SEQ_LEN, X_train.shape[2]))

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True),
]

print("Training LSTM...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

joblib.dump(history.history, HISTORY_PATH)

# ======================================================
# 9️⃣ 평가
# ======================================================
y_pred = model.predict(X_test).ravel()
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R2 :", r2)

# ======================================================
# 🔟 JSON 저장 (명세 준수)
# ======================================================
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

last_pred_diff = float(y_pred[-1])
predicted_direction = "UP" if last_pred_diff > 0 else "DOWN"

confidence = max(0.5, min(_sigmoid(abs(last_pred_diff) * 200), 0.99))

# 상관계수 기반 top5
top_features = []
corr_list = []
for name in features:
    try:
        corr = np.corrcoef(df[name].values[-len(y_all):], y_all)[0, 1]
    except:
        corr = 0.0
    corr_list.append((name, corr))

corr_sorted = sorted(corr_list, key=lambda x: abs(x[1]), reverse=True)[:5]

for name, corr in corr_sorted:
    top_features.append({"name": name, "impact": f"{corr:+.3f}"})

date_str = str(df["date"].iloc[-1].date())

prediction_payload = {
    "date": date_str,
    "predicted_direction": predicted_direction,
    "confidence": float(round(confidence, 3)),
    "top_features": top_features,
}

with open(os.path.join(OUTPUT_DIR, "prediction_model3.json"), "w", encoding="utf-8") as f:
    json.dump(prediction_payload, f, indent=2, ensure_ascii=False)

print("Saved prediction JSON.")
