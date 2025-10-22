
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import json
import numpy as np
import pandas as pd
from collections import Counter

COLS: Dict[str, str] = {
    "employee_id": "employee_id",
    "position": "position",
    "total_experience_years": "total_experience_years",
    "role_experience_years": "role_experience_years",
    "tenure_years": "tenure_years",
    "education": "education",
    "certifications_count": "certifications_count",
    "location": "location",
    "performance_rating": "performance_rating",
    "kpi_score": "kpi_score",
    "sql": "sql",
    "power_bi": "power_bi",
    "excel": "excel",
    "python": "python",
}

SKILL_KEYS: List[str] = ["sql", "power_bi", "excel", "python"]

def check_required_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> None:
    missing = [v for v in mapping.values() if v not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}\n\nPresent columns: {sorted(df.columns.tolist())}")

def safe_mode(series: pd.Series) -> Optional[str]:
    series = series.dropna()
    if series.empty:
        return None
    try:
        return series.mode(dropna=True).iloc[0]
    except Exception:
        from collections import Counter
        counts = Counter(series)
        return counts.most_common(1)[0][0] if counts else None

def percent_with_certifications(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors='coerce').fillna(0)
    return 100.0 * (series > 0).mean()

def bin_range_str(value: float, step: float, unit: str = "Years", left: float = 0, right: float | None = None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    lo = max(left, step * np.floor(value / step))
    hi = (lo + step)
    if right is not None:
        hi = min(hi, right)
    return f"{int(lo)}-{int(hi)} {unit}" if unit else f"{lo}-{hi}"

def rating_range_str(value: float, step: float = 0.5, max_val: float = 5.0) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    lo = step * np.floor(value / step)
    hi = min(lo + step, max_val)
    return f"{lo:.1f}-{hi:.0f}" if float(hi).is_integer() else f"{lo:.1f}-{hi:.1f}"

def score_range_str(value: float, step: float = 5.0, max_val: float = 100.0) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    lo = step * np.floor(value / step)
    hi = min(lo + step, max_val)
    return f"{int(lo)}-{int(hi)}"

def build_icp_json(*, position: str, counts: Dict[str, int], thresholds_used: Dict[str, float], role_snapshot: Dict[str, Any], skills_means_1_to_5: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "position": position,
        "counts": {
            "total_employees": int(counts.get("total_employees", 0)),
            "top_performers": int(counts.get("top_performers", 0)),
        },
        "thresholds_used": {
            "performance_rating_min": thresholds_used.get("performance_rating_min"),
            "kpi_top_quantile": thresholds_used.get("kpi_top_quantile"),
        },
        "role_snapshot": {
            "avg_total_experience_years": role_snapshot.get("avg_total_experience_years"),
            "avg_role_experience_years": role_snapshot.get("avg_role_experience_years"),
            "avg_tenure_years": role_snapshot.get("avg_tenure_years"),
            "education_mode": role_snapshot.get("education_mode"),
            "percent_with_certifications": role_snapshot.get("percent_with_certifications"),
            "top_location": role_snapshot.get("top_location"),
            "avg_performance_rating": role_snapshot.get("avg_performance_rating"),
            "avg_kpi_score": role_snapshot.get("avg_kpi_score"),
        },
        "skills_means_1_to_5": skills_means_1_to_5,
    }

def _compute_role_snapshot(df_role: pd.DataFrame, cols: Dict[str, str]) -> Dict[str, Any]:
    perf_col = cols["performance_rating"]
    kpi_col = cols["kpi_score"]

    tot_exp = pd.to_numeric(df_role[cols["total_experience_years"]], errors='coerce').mean()
    role_exp = pd.to_numeric(df_role[cols["role_experience_years"]], errors='coerce').mean()
    tenure = pd.to_numeric(df_role[cols["tenure_years"]], errors='coerce').mean()

    edu_mode = safe_mode(df_role[cols["education"]])
    cert_pct = percent_with_certifications(df_role[cols["certifications_count"]])
    loc_mode = safe_mode(df_role[cols["location"]])

    avg_perf = pd.to_numeric(df_role[perf_col], errors='coerce').mean()
    avg_kpi = pd.to_numeric(df_role[kpi_col], errors='coerce').mean()

    return {
        "avg_total_experience_years": bin_range_str(tot_exp, step=2, unit="Years"),
        "avg_role_experience_years": bin_range_str(role_exp, step=2, unit="Years"),
        "avg_tenure_years": bin_range_str(tenure, step=2, unit="Years"),
        "education_mode": edu_mode if edu_mode is not None else "N/A",
        "percent_with_certifications": f">={int(round(cert_pct))}",
        "top_location": loc_mode if loc_mode is not None else "N/A",
        "avg_performance_rating": rating_range_str(avg_perf, step=0.5, max_val=5.0),
        "avg_kpi_score": score_range_str(avg_kpi, step=5.0, max_val=100.0),
    }

def _skills_means(df_skills_base: pd.DataFrame, cols: Dict[str, str], skill_keys: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in skill_keys:
        col = cols[key]
        mean_val = pd.to_numeric(df_skills_base[col], errors='coerce').mean()
        if pd.isna(mean_val):
            out[key] = "N/A"
        else:
            lo = max(1.0, round(mean_val - 0.5, 1))
            hi = min(5.0, round(mean_val + 0.5, 1))
            if abs(hi - lo) < 1e-6:
                hi = min(5.0, lo + 0.5)
            out[key] = f"{lo}-{hi}"
    return out

def compute_icp(df: pd.DataFrame, position_name: str, perf_min: float = 4.0, kpi_top_quantile: float = 0.75, cols: Dict[str, str] | None = None, skill_keys: List[str] | None = None, use_top_performers_only: bool = True):
    cols = cols or COLS
    skill_keys = skill_keys or SKILL_KEYS

    check_required_columns(df, cols)

    if cols.get("position") in df.columns:
        df_role = df[df[cols["position"]] == position_name].copy()
        if df_role.empty:
            raise ValueError(f"No rows found for POSITION_NAME='{position_name}'. Check your POSITION_NAME or the '{cols['position']}' column values.")
    else:
        df_role = df.copy()

    perf_col = cols["performance_rating"]
    kpi_col = cols["kpi_score"]

    perf = pd.to_numeric(df_role[perf_col], errors='coerce')
    kpi = pd.to_numeric(df_role[kpi_col], errors='coerce')

    kpi_threshold = float(np.nanquantile(kpi, kpi_top_quantile))
    is_top = (perf >= perf_min) & (kpi >= kpi_threshold)

    total_employees = int(len(df_role))
    top_performers = int(is_top.sum())

    role_snapshot = _compute_role_snapshot(df_role, cols)
    df_skills_base = df_role[is_top].copy() if (use_top_performers_only and top_performers > 0) else df_role.copy()
    skills_means = _skills_means(df_skills_base, cols, skill_keys)

    payload = build_icp_json(
        position=position_name,
        counts={"total_employees": total_employees, "top_performers": top_performers},
        thresholds_used={"performance_rating_min": float(perf_min), "kpi_top_quantile": float(kpi_top_quantile)},
        role_snapshot=role_snapshot,
        skills_means_1_to_5=skills_means,
    )
    return payload, df_role

def optional_xgboost_report(df_role: pd.DataFrame, cols: Dict[str, str], skill_keys: List[str]) -> str:
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        from xgboost import XGBClassifier
    except Exception as e:
        return f"XGBoost training skipped: {e}"

    perf = pd.to_numeric(df_role[cols['performance_rating']], errors='coerce')
    kpi = pd.to_numeric(df_role[cols['kpi_score']], errors='coerce')
    kpi_thr = float(np.nanquantile(kpi, 0.75))
    is_top = (perf >= 4.0) & (kpi >= kpi_thr)

    feat_cols = [
        cols["total_experience_years"], cols["role_experience_years"], cols["tenure_years"],
        *[cols[k] for k in skill_keys],
        cols["kpi_score"],
    ]
    X = df_role[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    y = is_top.astype(int).values

    import numpy as np
    if len(np.unique(y)) < 2:
        return "XGBoost training skipped: Need at least two classes in the target."

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return classification_report(y_test, y_pred, digits=3)
