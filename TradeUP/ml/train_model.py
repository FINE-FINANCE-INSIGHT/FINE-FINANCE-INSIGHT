"""
====================================================================
💱 train_model.py — USD/KRW 방향성 예측 모델 학습 (v17 Ensemble)
====================================================================

[역할 요약]
- data_preprocessing.py 가 만들어둔 final.csv 를 불러온다.
- 추가 Feature(로그/차분/환율 랙)와 타깃(내일 방향성)을 생성한다.
- XGBoost + LightGBM + CatBoost 3개 분류모델을 학습한다.
- 세 모델의 예측확률을 평균(Soft Blending)하여 최종 예측에 사용한다.
- ROC-Youden 지표로 최적 Cutoff를 구해 방향성(UP/DOWN)을 이진 분류한다.
- 학습된 모델과 스케일러, Feature 목록, Cutoff 값을 하나의 dict로 저장한다.
- 평가 지표(metrics_v17.json)를 저장한다.
- 🔥 추가: 학습이 끝난 뒤, 마지막 Test 시점 기준으로
  prediction_v17.json 을 생성한다.

[JSON 명세 예시]

{
  "date": "2025-09-30",
  "predicted_direction": "UP",
  "confidence": 0.723,
  "top_features": [
    {"name": "DXY", "impact": "+0.214"},
    {"name": "RateDiff(US-KR)", "impact": "+0.186"},
    {"name": "KOSPI", "impact": "-0.142"},
    {"name": "VIX", "impact": "+0.101"},
    {"name": "WTI", "impact": "-0.078"}
  ]
}

※ 이 스크립트 하나만 실행하면
   - 모델 학습/저장(model.pkl)
   - metrics_v17.json
   - prediction_v17.json
  이 모두 생성된다. (predict.py 불필요)
====================================================================
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import joblib

# ============================================================
# 🔧 VSCode/직접 실행에서도 ml.config 임포트 되게 경로 보정
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../TradeUP/ml
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)               # .../TradeUP
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 이제 어디서 실행하든 아래 임포트가 동작함
from ml.config import FINAL_CSV_PATH, MODEL_PATH, PROCESSED_DATA_DIR, log

# ============================================================
# 📁 공통 경로 설정 (output 폴더 등)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SHAP은 있으면 사용, 없으면 fallback
try:
    import shap
    USE_SHAP = True
except Exception:
    USE_SHAP = False


# ====================================================================
# 1️⃣ 학습 데이터 로드 + Feature/Target 생성
# ====================================================================
def load_and_prepare_data():
    """
    final.csv 를 불러와서 모델 학습에 필요한 Feature / Target을 생성한다.

    [입력 데이터] (final.csv)
        - date
        - USD/KRW, MA_7, MA_30, Change(%)
        - KOSPI, SP500, VIX, DXY
        - KR_Rate, KOR_CPI
        - WTI, US_Upper, US_Lower, US_Avg
        - CPIAUCSL, CPILFESL, PPIACO, PALLFNFINDEXQ
        - RateDiff(US-KR), CPI_DIFF(KR-US), vol_30d, mom_90d

    [추가 생성]
        - log_usdkrw : log(USD/KRW)
        - log_diff   : log_usdkrw의 1일 차분
        - USD/KRW_lag1, lag2, lag3 : 환율 랙(이전 1~3일 값)
        - target_cls : 다음날 log_diff > 0 → 1(상승) / 0(하락)
        - target_reg : 다음날 log_diff (실수값)

    [반환]
        X        : 학습/검증에 사용할 Feature DataFrame
        y_cls    : 분류 타깃(0/1, 방향성)
        y_reg    : 회귀 타깃(다음날 로그 차분)
        feature_cols : Feature 컬럼명 리스트 (모델 저장용)
        df       : 전처리 후 전체 DataFrame (date 포함, dropna 이후)
    """
    log(f"📥 학습 데이터 로드 중... ({FINAL_CSV_PATH})")
    df = pd.read_csv(FINAL_CSV_PATH)
    log(f"✅ 데이터 로드 완료: {df.shape}")

    # -----------------------------
    # 1) 로그 변환 + 차분
    # -----------------------------
    df["log_usdkrw"] = np.log(df["USD/KRW"])
    df["log_diff"] = df["log_usdkrw"].diff()

    # -----------------------------
    # 2) 환율 랙(이전 1~3일 값) 생성
    # -----------------------------
    for lag in [1, 2, 3]:
        df[f"USD/KRW_lag{lag}"] = df["USD/KRW"].shift(lag)

    # -----------------------------
    # 3) 타깃 생성 (다음날 기준)
    # -----------------------------
    # 다음날 로그 차분이 양수면 "상승(1)", 아니면 "하락(0)"
    df["target_cls"] = (df["log_diff"].shift(-1) > 0).astype(int)
    df["target_reg"] = df["log_diff"].shift(-1)

    # NaN이 생기는 구간(앞부분 diff, 랙 / 마지막 1일 target)을 제거
    df = df.dropna().reset_index(drop=True)

    # -----------------------------
    # 4) Feature / Target 분리
    # -----------------------------
    drop_cols = [
        "date",
        "USD/KRW",
        "log_usdkrw",
        "log_diff",
        "target_cls",
        "target_reg",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y_cls = df["target_cls"]
    y_reg = df["target_reg"]

    log(f"📊 Feature 수: {len(feature_cols)}, 데이터 수: {len(df)}")

    return X, y_cls, y_reg, feature_cols, df


# ====================================================================
# 2️⃣ Train / Test 분리 + 스케일링 + SMOTE
# ====================================================================
def train_test_split_and_scale(X, y_cls, y_reg, train_ratio=0.8):
    """
    시계열 특성을 고려하여 앞 80%를 Train, 뒤 20%를 Test로 사용한다.
    (shuffle 하지 않음)

    이후 StandardScaler로 스케일링하고,
    분류용 타깃(y_cls)에 대해서만 SMOTE 오버샘플링을 적용한다.

    [반환]
        X_train_sm, y_train_sm : SMOTE 적용된 분류 학습 데이터
        X_train_scaled, X_test_scaled : 스케일링된 Feature (회귀용 포함)
        y_train_cls, y_test_cls       : 분류 타깃
        y_train_reg, y_test_reg       : 회귀 타깃
        scaler                        : 학습된 StandardScaler 객체
    """
    n = len(X)
    train_size = int(n * train_ratio)

    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train_cls, y_test_cls = y_cls.iloc[:train_size], y_cls.iloc[train_size:]
    y_train_reg, y_test_reg = y_reg.iloc[:train_size], y_reg.iloc[train_size:]

    log(f"✂️ Train: {len(X_train)} / Test: {len(X_test)} (비율 {train_ratio*100:.1f}%)")

    # -----------------------------
    # 스케일링 (StandardScaler)
    # -----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------
    # SMOTE (분류용 데이터 불균형 보정)
    # -----------------------------
    log("⚖️ SMOTE 오버샘플링 중...")
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train_scaled, y_train_cls)
    log(f"✅ SMOTE 완료 → Train 샘플: {X_train_sm.shape[0]}")

    return (
        X_train_sm,
        y_train_sm,
        X_train_scaled,
        X_test_scaled,
        y_train_cls,
        y_test_cls,
        y_train_reg,
        y_test_reg,
        scaler,
    )


# ====================================================================
# 3️⃣ Ensemble 분류 모델 (XGB + LGBM + CatBoost)
# ====================================================================
def train_ensemble_classifier(X_train_sm, y_train_sm):
    """
    V17에서 사용했던 3개 분류 모델을 학습한다.

    - XGBoostClassifier
    - LGBMClassifier
    - CatBoostClassifier

    반환값으로 세 모델 객체(xgb, lgbm, cat)를 돌려준다.
    """

    log("🚀 Ensemble 분류 모델 학습 시작 (XGB + LGBM + CatBoost)")

    # XGBoost 분류기
    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=1.5,  # 불균형 데이터 가중치
        random_state=42,
        eval_metric="logloss",
        tree_method="hist",
    )

    # LightGBM 분류기
    lgbm_model = LGBMClassifier(
        n_estimators=400,
        max_depth=-1,
        learning_rate=0.025,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=42,
    )

    # CatBoost 분류기 (터미널 출력 줄이기 위해 verbose=False)
    cat_model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.04,
        depth=6,
        loss_function="Logloss",
        verbose=False,
        random_seed=42,
    )

    # 실제 학습
    xgb_model.fit(X_train_sm, y_train_sm)
    lgbm_model.fit(X_train_sm, y_train_sm)
    cat_model.fit(X_train_sm, y_train_sm)

    log("✅ Ensemble 분류 모델 학습 완료")

    return xgb_model, lgbm_model, cat_model


def evaluate_classifier_ensemble(
    xgb_model, lgbm_model, cat_model, X_test_scaled, y_test_cls
):
    """
    Test 구간에서 3개 모델의 예측 확률을 평균 내어(Soft Blending)
    최종 분류 성능(Accuracy, F1, AUC, Best Cutoff)을 계산한다.

    [반환]
        metrics_cls : dict(accuracy, f1, auc, cutoff, ...)
        y_pred      : 최종 이진 예측 (0/1)
        final_proba : 최종 예측 확률 (0~1)
        best_thr    : ROC-Youden 기준 최적 cutoff 값
    """
    # 각 모델의 양성(1) 클래스에 대한 예측 확률
    xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    lgbm_proba = lgbm_model.predict_proba(X_test_scaled)[:, 1]
    cat_proba = cat_model.predict_proba(X_test_scaled)[:, 1]

    # Soft Blending (3개 확률의 단순 평균)
    final_proba = (xgb_proba + lgbm_proba + cat_proba) / 3.0

    # ROC Curve 기반으로 최적 Cutoff 탐색 (Youden's J = TPR - FPR 최대)
    fpr, tpr, thresholds = roc_curve(y_test_cls, final_proba)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_thr = thresholds[best_idx]

    # 최적 Cutoff 기준으로 최종 이진 예측 생성
    y_pred = (final_proba >= best_thr).astype(int)

    # 평가 지표 계산
    acc = accuracy_score(y_test_cls, y_pred)
    f1 = f1_score(y_test_cls, y_pred)
    auc = roc_auc_score(y_test_cls, final_proba)

    log(
        f"🎯 방향성 분류 결과 (Cutoff={best_thr:.3f}) "
        f"Accuracy={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
    )

    metrics_cls = {
        "accuracy": float(acc),
        "f1": float(f1),
        "auc": float(auc),
        "cutoff": float(best_thr),
    }

    return metrics_cls, y_pred, final_proba, best_thr


# ====================================================================
# 4️⃣ 회귀 모델 (다음날 로그 차분 예측 — 선택적)
# ====================================================================
def train_regressor(X_train_scaled, y_train_reg, X_test_scaled, y_test_reg):
    """
    XGBoostRegressor 하나로 회귀 모델을 학습한다.
    (v17에서 사용했던 구조를 그대로 가져온 것)

    [반환]
        reg_model : 학습된 회귀 모델
        metrics_reg : 회귀 평가 지표 dict(MSE, RMSE, MAE, R2)
    """
    log("📐 Log-Δ환율 회귀 모델(XGBRegressor) 학습 시작")

    reg_model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="rmse",
        tree_method="hist",
    )

    reg_model.fit(X_train_scaled, y_train_reg)
    y_pred_reg = reg_model.predict(X_test_scaled)

    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)

    log(
        f"📈 Log-Δ환율 회귀 성능: "
        f"MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, R2={r2:.4f}"
    )

    metrics_reg = {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
    }

    return reg_model, metrics_reg


# ====================================================================
# 5️⃣ 모델/지표 저장
# ====================================================================
def save_model_and_metrics(
    xgb_model,
    lgbm_model,
    cat_model,
    reg_model,
    scaler,
    feature_cols,
    metrics_cls,
    metrics_reg,
    cutoff,
):
    """
    - 모델/스케일러/Feature 목록/최적 Cutoff 를 하나의 dict로 묶어 MODEL_PATH에 저장
      (config.MODEL_PATH → 보통 TradeUP/ml/model.pkl)
    - 분류/회귀 평가 지표를 metrics_v17.json 으로 저장
    """
    final_model = {
        "xgb_model": xgb_model,
        "lgbm_model": lgbm_model,
        "cat_model": cat_model,
        "reg_model": reg_model,
        "scaler": scaler,
        "feature_names": feature_cols,
        "cutoff": float(cutoff),
        "version": "v17_ensemble",
    }

    # 1) 모델 저장 (joblib)
    joblib.dump(final_model, MODEL_PATH)
    log(f"💾 모델 저장 완료: {MODEL_PATH}")

    # 2) 지표 저장 (JSON)
    metrics = {
        "version": "v17_ensemble",
        "classifier": metrics_cls,
        "regressor": metrics_reg,
    }

    metrics_path = os.path.join(OUTPUT_DIR, "metrics_v17.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    log(f"📊 성능 지표 저장 완료: {metrics_path}")


# ====================================================================
# 6️⃣ prediction_v17.json 저장 (새로 추가)
# ====================================================================
def save_prediction_v17(
    df_processed,
    feature_cols,
    xgb_model,
    final_proba,
    cutoff,
    X_test_scaled,
    y_test_cls,
):
    """
    - 마지막 Test 시점 기준으로 방향성 예측을 수행하고
      prediction_v17.json 을 생성한다.

    JSON 형식:
    {
      "date": "YYYY-MM-DD",
      "predicted_direction": "UP" or "DOWN",
      "confidence": 0.723,
      "top_features": [
        {"name": "DXY", "impact": "+0.214"},
        ...
      ]
    }
    """
    log("🧾 prediction_v17.json 생성 중...")

    # 1) 마지막 Test 시점 기준 확률/타깃
    proba_last = float(final_proba[-1])
    y_last = int(y_test_cls.iloc[-1])

    # 안정화를 위해 clip
    proba_last = min(max(proba_last, 1e-6), 1 - 1e-6)

    # cutoff 기준 방향성
    if proba_last >= cutoff:
        predicted_direction = "UP"
        confidence = proba_last
    else:
        predicted_direction = "DOWN"
        confidence = 1.0 - proba_last

    # confidence 0.5 ~ 0.99 사이로 제한
    confidence = max(0.5, min(float(confidence), 0.99))

    # 2) top_features 계산
    top_features = []
    try:
        if USE_SHAP:
            log("🧠 SHAP 기반 top_features 계산 중...")
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_test_scaled[-1:])

            # 이진분류일 경우 shap_values 가 list 일 수 있음
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            shap_last = shap_values[0]  # (n_features,)
            abs_vals = np.abs(shap_last)
            idx_sorted = np.argsort(-abs_vals)[:5]

            feat_arr = np.array(feature_cols)
            for idx in idx_sorted:
                name = str(feat_arr[idx])
                impact_val = float(shap_last[idx])
                impact_str = f"{impact_val:+.3f}"  # "+0.214" 형식
                top_features.append({"name": name, "impact": impact_str})
        else:
            raise RuntimeError("SHAP not available")
    except Exception as e:
        log(f"⚠️ SHAP 사용 불가, feature_importances 기반으로 대체: {e}")
        importances = xgb_model.feature_importances_
        feat_arr = np.array(feature_cols)
        idx_sorted = np.argsort(-importances)[:5]

        X_test_scaled_arr = np.asarray(X_test_scaled)
        y_arr = y_test_cls.values

        for idx in idx_sorted:
            name = str(feat_arr[idx])
            xi = X_test_scaled_arr[:, idx]
            try:
                corr = float(np.corrcoef(xi, y_arr)[0, 1])
            except Exception:
                corr = 0.0
            sign = 1.0 if corr >= 0 else -1.0
            imp_norm = importances[idx] / (importances[idx_sorted].sum() + 1e-8)
            impact_val = float(sign * imp_norm)
            impact_str = f"{impact_val:+.3f}"
            top_features.append({"name": name, "impact": impact_str})

    # 3) date (마지막 행 날짜)
    if "date" in df_processed.columns:
        date_str = str(df_processed["date"].iloc[-1])
    else:
        date_str = ""

    prediction_payload = {
        "date": date_str,
        "predicted_direction": predicted_direction,
        "confidence": float(round(confidence, 3)),
        "top_features": top_features,
    }

    pred_path = os.path.join(OUTPUT_DIR, "prediction_v17.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(prediction_payload, f, indent=2, ensure_ascii=False)

    log(f"📤 prediction_v17.json 저장 완료: {pred_path}")


# ====================================================================
# 7️⃣ 메인 실행 함수
# ====================================================================
def main():
    log("===================================================")
    log("🚀 [STEP 3] V17 Ensemble 모델 학습 시작")
    log("===================================================")

    # 1) 데이터 로드 + Feature/Target 생성
    X, y_cls, y_reg, feature_cols, df_processed = load_and_prepare_data()

    # 2) Train/Test Split + Scaling + SMOTE
    (
        X_train_sm,
        y_train_sm,
        X_train_scaled,
        X_test_scaled,
        y_train_cls,
        y_test_cls,
        y_train_reg,
        y_test_reg,
        scaler,
    ) = train_test_split_and_scale(X, y_cls, y_reg)

    # 3) Ensemble 분류 모델 학습
    xgb_model, lgbm_model, cat_model = train_ensemble_classifier(
        X_train_sm, y_train_sm
    )

    # 4) 분류 모델 평가
    metrics_cls, y_pred_cls, final_proba, best_thr = evaluate_classifier_ensemble(
        xgb_model, lgbm_model, cat_model, X_test_scaled, y_test_cls
    )

    # 5) 회귀 모델 학습/평가
    reg_model, metrics_reg = train_regressor(
        X_train_scaled, y_train_reg, X_test_scaled, y_test_reg
    )

    # 6) 모델 및 성능 지표 저장
    save_model_and_metrics(
        xgb_model,
        lgbm_model,
        cat_model,
        reg_model,
        scaler,
        feature_cols,
        metrics_cls,
        metrics_reg,
        cutoff=best_thr,
    )

    # 7) 🔥 prediction_v17.json 생성 (마지막 Test 시점 기준)
    save_prediction_v17(
        df_processed=df_processed,
        feature_cols=feature_cols,
        xgb_model=xgb_model,
        final_proba=final_proba,
        cutoff=best_thr,
        X_test_scaled=X_test_scaled,
        y_test_cls=y_test_cls,
    )

    log("===================================================")
    log("✅ [STEP 3 완료] V17 Ensemble 모델 학습 및 예측 JSON 생성 완료")
    log("===================================================")


# ====================================================================
# 8️⃣ 직접 실행 지원 (VSCode에서 F5 눌러 실행해도 동작)
# ====================================================================
if __name__ == "__main__":
    main()
