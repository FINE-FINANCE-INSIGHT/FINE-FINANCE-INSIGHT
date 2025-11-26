"""
====================================================================
💱 train_model.py — USD/KRW 방향성 예측 모델 학습 (v17 Ensemble)
====================================================================
"""

import os
import sys
import json
import warnings
from datetime import datetime

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

# config import
from ml.config import FINAL_CSV_PATH, MODEL_PATH, RESULT_JSON_PATH, log

# SHAP import
try:
    import shap
    USE_SHAP = True
except Exception:
    USE_SHAP = False


# ====================================================================
# 1) load_and_prepare_data
# ====================================================================
def load_and_prepare_data():
    log(f"📥 학습 데이터 로드 중... ({FINAL_CSV_PATH})")
    df = pd.read_csv(FINAL_CSV_PATH)
    log(f"✅ 데이터 로드 완료: {df.shape}")

    df["log_usdkrw"] = np.log(df["USD/KRW"])
    df["log_diff"] = df["log_usdkrw"].diff()

    # Lag Feature
    for lag in [1, 2, 3]:
        df[f"USD/KRW_lag{lag}"] = df["USD/KRW"].shift(lag)

    df["target_cls"] = (df["log_diff"].shift(-1) > 0).astype(int)
    df["target_reg"] = df["log_diff"].shift(-1)

    df = df.dropna().reset_index(drop=True)

    # Feature 선택
    drop_cols = ["date", "USD/KRW", "log_usdkrw", "log_diff", "target_cls", "target_reg"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y_cls = df["target_cls"]
    y_reg = df["target_reg"]

    log(f"📊 Feature 수: {len(feature_cols)}, 데이터 수: {len(df)}")

    return X, y_cls, y_reg, feature_cols, df


# ====================================================================
# 2) train_test_split_and_scale
# ====================================================================
def train_test_split_and_scale(X, y_cls, y_reg, train_ratio=0.8):
    n = len(X)
    train_size = int(n * train_ratio)

    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train_cls, y_test_cls = y_cls.iloc[:train_size], y_cls.iloc[train_size:]
    y_train_reg, y_test_reg = y_reg.iloc[:train_size], y_reg.iloc[train_size:]

    log(f"✂️ Train: {len(X_train)} / Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log("⚖️ SMOTE 오버샘플링 중...")
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train_scaled, y_train_cls)
    log(f"✅ SMOTE 완료 → {X_train_sm.shape[0]} rows")

    return (
        X_train_sm, y_train_sm,
        X_train_scaled, X_test_scaled,
        y_train_cls, y_test_cls,
        y_train_reg, y_test_reg,
        scaler
    )


# ====================================================================
# 3) Ensemble Classifier
# ====================================================================
def train_ensemble_classifier(X_train_sm, y_train_sm):
    log("🚀 Ensemble 분류 모델 학습 시작")

    xgb_model = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=1.5, random_state=42,
        eval_metric="logloss", tree_method="hist"
    )

    lgbm_model = LGBMClassifier(
        n_estimators=400, learning_rate=0.025,
        subsample=0.9, colsample_bytree=0.9,
        class_weight="balanced", random_state=42
    )

    cat_model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.04,
        loss_function="Logloss", verbose=False, random_seed=42
    )

    xgb_model.fit(X_train_sm, y_train_sm)
    lgbm_model.fit(X_train_sm, y_train_sm)
    cat_model.fit(X_train_sm, y_train_sm)

    log("✅ Ensemble 분류 모델 학습 완료")
    return xgb_model, lgbm_model, cat_model


# ====================================================================
# 4) Evaluate Classifier
# ====================================================================
def evaluate_classifier_ensemble(
    xgb_model, lgbm_model, cat_model,
    X_test_scaled, y_test_cls
):
    xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    lgbm_proba = lgbm_model.predict_proba(X_test_scaled)[:, 1]
    cat_proba = cat_model.predict_proba(X_test_scaled)[:, 1]

    final_proba = (xgb_proba + lgbm_proba + cat_proba) / 3

    fpr, tpr, thresholds = roc_curve(y_test_cls, final_proba)
    j = tpr - fpr
    best_idx = np.argmax(j)
    best_thr = thresholds[best_idx]

    y_pred = (final_proba >= best_thr).astype(int)

    acc = accuracy_score(y_test_cls, y_pred)
    f1 = f1_score(y_test_cls, y_pred)
    auc = roc_auc_score(y_test_cls, final_proba)

    log(f"🎯 Accuracy={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    return (
        {"accuracy": acc, "f1": f1, "auc": auc, "cutoff": float(best_thr)},
        y_pred, final_proba, best_thr
    )


# ====================================================================
# 5) train_regressor
# ====================================================================
def train_regressor(X_train_scaled, y_train_reg, X_test_scaled, y_test_reg):
    reg = XGBRegressor(
        n_estimators=400, learning_rate=0.02, max_depth=4,
        subsample=0.9, colsample_bytree=0.8,
        random_state=42, eval_metric="rmse"
    )

    reg.fit(X_train_scaled, y_train_reg)
    pred = reg.predict(X_test_scaled)

    metrics = {
        "MSE": mean_squared_error(y_test_reg, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test_reg, pred))),
        "MAE": mean_absolute_error(y_test_reg, pred),
        "R2": r2_score(y_test_reg, pred)
    }

    log(f"📈 회귀 성능: MAE={metrics['MAE']:.6f}, R2={metrics['R2']:.4f}")
    return reg, metrics


# ====================================================================
# 6) save_model_and_metrics
# ====================================================================
def save_model_and_metrics(
    xgb_model, lgbm_model, cat_model,
    reg_model, scaler, feature_cols,
    metrics_cls, metrics_reg, cutoff
):

    obj = {
        "xgb": xgb_model,
        "lgbm": lgbm_model,
        "cat": cat_model,
        "reg": reg_model,
        "scaler": scaler,
        "feature_names": feature_cols,
        "cutoff": float(cutoff)
    }

    joblib.dump(obj, MODEL_PATH)
    log(f"💾 모델 저장 완료: {MODEL_PATH}")

    # metrics_v17.json → ml/output/
    output_dir = os.path.join(CURRENT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, "metrics_v17.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {"classifier": metrics_cls, "regressor": metrics_reg},
            f, indent=4, ensure_ascii=False
        )

    log(f"📊 성능 지표 저장 완료: {metrics_path}")


# ====================================================================
# 7) save_prediction_json (날짜 = 오늘 날짜)
# ====================================================================
def save_prediction_json(
    df_processed, feature_cols,
    xgb_model, final_proba, cutoff,
    X_test_scaled, y_test_cls
):
    log("🧾 최종 prediction JSON 생성 중...")

    # 확률
    p = float(final_proba[-1])
    p = min(max(p, 1e-6), 1 - 1e-6)

    predicted = "UP" if p >= cutoff else "DOWN"
    confidence = max(0.5, min(float(max(p, 1 - p)), 0.99))

    # SHAP Top5
    top_features = []
    try:
        if USE_SHAP:
            explainer = shap.TreeExplainer(xgb_model)
            vals = explainer.shap_values(X_test_scaled[-1:])
            if isinstance(vals, list):
                vals = vals[0]
            s = vals[0]
            idxs = np.argsort(-np.abs(s))[:5]
            for i in idxs:
                top_features.append({
                    "name": feature_cols[i],
                    "impact": f"{float(s[i]):+.3f}"
                })
    except Exception:
        importances = xgb_model.feature_importances_
        idxs = np.argsort(-importances)[:5]
        for i in idxs:
            top_features.append({
                "name": feature_cols[i],
                "impact": f"{float(importances[i]):.3f}"
            })

    # 🔥 오늘 날짜로 강제 적용
    date_str = datetime.today().strftime("%Y-%m-%d")

    payload = {
        "date": date_str,
        "predicted_direction": predicted,
        "confidence": confidence,
        "top_features": top_features
    }

    save_path = RESULT_JSON_PATH()
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log(f"📤 저장 완료: {save_path}")


# ====================================================================
# MAIN
# ====================================================================
def main():
    X, y_cls, y_reg, feature_cols, df_processed = load_and_prepare_data()

    (
        X_train_sm, y_train_sm,
        X_train_scaled, X_test_scaled,
        y_train_cls, y_test_cls,
        y_train_reg, y_test_reg,
        scaler
    ) = train_test_split_and_scale(X, y_cls, y_reg)

    xgb_m, lgbm_m, cat_m = train_ensemble_classifier(X_train_sm, y_train_sm)

    metrics_cls, y_pred, final_proba, best_thr = evaluate_classifier_ensemble(
        xgb_m, lgbm_m, cat_m, X_test_scaled, y_test_cls
    )

    reg_m, metrics_reg = train_regressor(
        X_train_scaled, y_train_reg, X_test_scaled, y_test_reg
    )

    save_model_and_metrics(
        xgb_m, lgbm_m, cat_m,
        reg_m, scaler, feature_cols,
        metrics_cls, metrics_reg, cutoff=best_thr
    )

    save_prediction_json(
        df_processed, feature_cols,
        xgb_m, final_proba, best_thr,
        X_test_scaled, y_test_cls
    )

    log("✅ 모든 작업 완료")


if __name__ == "__main__":
    main()
