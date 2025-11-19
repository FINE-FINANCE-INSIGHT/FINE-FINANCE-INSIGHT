"""
==========================================
USD/KRW 환율 예측 - LightGBM 모델 학습 코드 (train_model_2)
==========================================

목적:
- 경제지표 기반으로 USD/KRW 환율의 비선형 패턴을 학습하여 예측

입력: ../data/final.csv
출력:
    - ../ml/artifacts/lgbm_model.pkl
    - ../ml/artifacts/feature_importance.csv
    - ../ml/output/metrics_model2.json
    - ../ml/output/prediction_model2.json
"""

# ===============================
# 1) 라이브러리 임포트
# ===============================
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# SHAP optional
try:
    import shap
    USE_SHAP = True
except Exception:
    USE_SHAP = False


# ===============================
# 2) 경로 설정 & 데이터 로드
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "final.csv")

ML_DIR = os.path.join(BASE_DIR, "ml")
ARTIFACT_DIR = os.path.join(ML_DIR, "artifacts")
OUTPUT_DIR = os.path.join(ML_DIR, "output")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Data path:", DATA_PATH)

df = pd.read_csv(DATA_PATH)

# ===============================
# 3) 타깃 생성
# ===============================
df["log_return"] = np.log(df["USD/KRW"] / df["USD/KRW"].shift(1))
df = df.dropna().reset_index(drop=True)

target = "USD/KRW"
exclude_cols = ["date", target]

features = [
    "USD/KRW_lag1", "USD/KRW_lag3", "USD/KRW_lag7",
    "MA_7", "Change(%)", "Change(%)_lag1",
    "VIX", "VIX_lag1", "VIX_lag7",
    "DXY", "DXY_lag1", "DXY_lag7",
    "vol_30d", "mom_90d", "vol_30d_lag1",
    "KOSPI", "KOSPI_lag1", "SP500", "SP500_lag1", "WTI",
    "RateDiff(US-KR)", "US_PolicyGap", "CPI_DIFF(KR-US)",
    "month", "weekday"
]

# 실제 존재하는 컬럼만 사용
features = [f for f in features if f in df.columns]

X = df[features]
y = df[target]

# ===============================
# 4) 시계열 분할
# ===============================
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print("Train:", X_train.shape, "Test:", X_test.shape)

# ===============================
# 5) LightGBM 설정
# ===============================
model = lgb.LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.01,
    num_leaves=40,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

callbacks = [
    lgb.early_stopping(stopping_rounds=100),
    lgb.log_evaluation(period=100),
]

# ===============================
# 6) 학습
# ===============================
print("\nTraining model...")
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="mae",
    callbacks=callbacks,
)

# ===============================
# 7) 평가
# ===============================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", round(mae, 5))
print("R2:", round(r2, 3))

# ===============================
# 8) 중요도 저장
# ===============================
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

top_features_df = importance.head(30)
imp_path = os.path.join(ARTIFACT_DIR, "feature_importance.csv")
top_features_df.to_csv(imp_path, index=False, encoding="utf-8-sig")
print("Saved feature importance ->", imp_path)

# ===============================
# 9) 시각화 (필요 시만)
# ===============================
plt.figure(figsize=(8, 5))
plt.barh(importance["feature"], importance["importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance (LightGBM)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(y_test.reset_index(drop=True).values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ===============================
# 10) 모델 저장
# ===============================
model_path = os.path.join(ARTIFACT_DIR, "lgbm_model.pkl")
joblib.dump(model, model_path)
print("\nSaved model ->", model_path)

# ===============================
# 11) metrics_model2.json 저장
# ===============================
metrics = {
    "version": "lightgbm_model2",
    "model": "LightGBM",
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "metrics": {
        "MAE": float(mae),
        "R2": float(r2),
    },
    "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
}
metrics_path = os.path.join(OUTPUT_DIR, "metrics_model2.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)
print("Saved metrics JSON ->", metrics_path)

# ===============================
# 12) prediction_model2.json 저장
# ===============================
def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))

last_actual = float(y_test.iloc[-1])
last_pred = float(y_pred[-1])

predicted_direction = "UP" if last_pred > last_actual else "DOWN"

change_ratio = (last_pred - last_actual) / max(1e-6, abs(last_actual))
conf_raw = abs(change_ratio) * 50
confidence = _sigmoid(conf_raw)
confidence = max(0.5, min(confidence, 0.99))

top_features = []
try:
    if USE_SHAP:
        print("Using SHAP for top_features...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        last_shap = shap_values[-1]
        abs_vals = np.abs(last_shap)
        idx_sorted = np.argsort(-abs_vals)[:5]

        feat_arr = np.array(features)
        for idx in idx_sorted:
            name = str(feat_arr[idx])
            impact_str = f"{last_shap[idx]:+.3f}"
            top_features.append({"name": name, "impact": impact_str})
    else:
        raise RuntimeError("SHAP not available")
except Exception:
    importances = model.feature_importances_
    feat_arr = np.array(features)
    idx_sorted = np.argsort(-importances)[:5]

    df_test = df.iloc[split_idx:]
    y_test_arr = y_test.values

    for idx in idx_sorted:
        name = str(feat_arr[idx])
        xi = df_test[name].values
        try:
            corr = np.corrcoef(xi, y_test_arr)[0, 1]
        except Exception:
            corr = 0.0

        sign = 1.0 if corr >= 0 else -1.0
        imp_norm = importances[idx] / (importances[idx_sorted].sum() + 1e-8)
        impact_val = sign * imp_norm
        impact_str = f"{impact_val:+.3f}"
        top_features.append({"name": name, "impact": impact_str})

date_str = str(df["date"].iloc[-1]) if "date" in df.columns else ""

prediction_payload = {
    "date": date_str,
    "predicted_direction": predicted_direction,
    "confidence": float(round(confidence, 3)),
    "top_features": top_features,
}

pred_path = os.path.join(OUTPUT_DIR, "prediction_model2.json")
with open(pred_path, "w", encoding="utf-8") as f:
    json.dump(prediction_payload, f, indent=2, ensure_ascii=False)
print("Saved prediction JSON ->", pred_path)

print("\nFeature ranking:")
print(importance.to_string(index=False))
