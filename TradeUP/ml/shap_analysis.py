"""
===============================================================
📘 shap_analysis.py — SHAP 기반 Top5 Feature 중요도 계산 모듈
===============================================================

※ predict.py에서 import 하여 사용함
※ 모델마다 Feature 영향력을 5개 추출하는 역할
"""

import shap
import numpy as np
import pandas as pd


def compute_shap_top5(model, X_today, feature_names):
    """
    모델(XGB/LGBM/CatBoost 중 랜덤하게 하나 골라서) SHAP 기반 top5 feature 추출

    X_today : 오늘 날짜 1행 (DataFrame)
    feature_names : feature 리스트
    """

    # SHAP TreeExplainer (XGBoost, LightGBM, CatBoost 전부 지원)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_today)

    # SHAP 값이 (1, n_features) 형태이므로 flatten
    shap_values = shap_values.reshape(-1)

    df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values,
        "abs_value": np.abs(shap_values)
    })

    top5 = (
        df.sort_values("abs_value", ascending=False)
          .head(5)[["feature", "shap_value"]]
    )

    # 서버 명세에 맞게 변환
    results = []
    for _, row in top5.iterrows():
        impact = f"{row['shap_value']:+.3f}"
        results.append({
            "name": row["feature"],
            "impact": impact
        })

    return results
