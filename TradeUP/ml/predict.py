"""
====================================================================
📌 predict.py — 오늘 환율 방향성 예측 + SHAP Top5 + JSON 생성
====================================================================

FastAPI 서버는 이 파일의 run_prediction() 함수만 호출하면 됨.

[전체 흐름]
1) data_preprocessing.generate_today_features() → X_today(1행)
2) ml/model.pkl 로드
3) 스케일링 적용
4) 세 모델(XGB/LGBM/CatBoost) soft blending → 확률 계산
5) cutoff 기준으로 up/down 판별
6) shap_analysis.compute_shap_top5() → top 5 변수 영향력
7) JSON 생성 후, data/results/YYYY-MM-DD.json 저장

====================================================================
"""

import os
import json
import numpy as np
from datetime import datetime
import joblib

# ============================================================
# 🔧 VSCode/파일 직접 실행에서도 ml.config 인식되게 경로 보정
# ============================================================
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../TradeUP/ml
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)               # .../TradeUP
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 이제 ml.config import 정상 동작
from ml.config import MODEL_PATH, RESULT_JSON_PATH, log
from ml.data_preprocessing import generate_today_features
from ml.shap_analysis import compute_shap_top5


# ==================================================================
# 1️⃣ 예측 실행 함수 — FastAPI에서 이 함수만 호출하면 된다
# ==================================================================
def run_prediction():
    log("🚀 [PREDICT] 오늘 예측 시작")

    # -----------------------------
    # 1) 오늘 날짜 데이터(X_today) 로드
    # -----------------------------
    X_today = generate_today_features()  # shape: (1, n_features)
    feature_names = X_today.columns.tolist()

    log(f"📄 오늘 데이터 컬럼 수: {len(feature_names)}")

    # -----------------------------
    # 2) 모델 로드
    # -----------------------------
    model_dict = joblib.load(MODEL_PATH)
    xgb_model = model_dict["xgb_model"]
    lgbm_model = model_dict["lgbm_model"]
    cat_model = model_dict["cat_model"]
    scaler = model_dict["scaler"]
    cutoff = model_dict["cutoff"]
    feature_names_model = model_dict["feature_names"]

    # 필드 순서 보장
    X_today = X_today[feature_names_model]

    # -----------------------------
    # 3) 스케일링 적용
    # -----------------------------
    X_scaled = scaler.transform(X_today)

    # -----------------------------
    # 4) 3개 모델 Soft Blending 수행
    # -----------------------------
    xgb_proba = xgb_model.predict_proba(X_scaled)[:, 1]
    lgbm_proba = lgbm_model.predict_proba(X_scaled)[:, 1]
    cat_proba = cat_model.predict_proba(X_scaled)[:, 1]

    final_proba = float((xgb_proba + lgbm_proba + cat_proba) / 3)

    predicted_direction = "up" if final_proba >= cutoff else "down"

    log(f"🔮 예측 방향: {predicted_direction} (p={final_proba:.3f}) cutoff={cutoff:.3f}")

    # -----------------------------
    # 5) SHAP Top5 계산 (XGBoost 기준)
    # -----------------------------
    top_features = compute_shap_top5(
        model=xgb_model,
        X_today=X_scaled,
        feature_names=feature_names_model
    )

    # -----------------------------
    # 6) JSON 결과 구성
    # -----------------------------
    result = {
        "date": datetime.today().strftime("%Y-%m-%d"),
        "predicted_direction": predicted_direction,
        "confidence": round(final_proba, 3),
        "top_features": top_features
    }

    # -----------------------------
    # 7) data/results/YYYY-MM-DD.json 저장
    # -----------------------------
    result_path = RESULT_JSON_PATH()
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    log(f"💾 JSON 저장 완료 → {result_path}")

    return result


# ==================================================================
# 2️⃣ VSCode에서 직접 실행해도 동작
# ==================================================================
if __name__ == "__main__":
    print(run_prediction())
