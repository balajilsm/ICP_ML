
import streamlit as st
import pandas as pd
import json
from io import BytesIO
from streamlit_icp_tool.icp_core import COLS, SKILL_KEYS, compute_icp, optional_xgboost_report

st.set_page_config(page_title="ICP JSON Generator", page_icon="🎯", layout="centered")
st.title("🎯 Ideal Candidate Profile (ICP) — Streamlit App")
st.caption("Upload your employee CSV, select a position, and generate an ICP JSON.")

with st.expander("⚙️ Configuration (optional)", expanded=False):
    perf_min = st.slider("Minimum performance rating (perf_min)", 1.0, 5.0, 4.0, 0.1)
    kpi_q = st.slider("KPI top quantile (kpi_top_quantile)", 0.50, 0.95, 0.75, 0.01)
    use_top_only = st.checkbox("Use only top performers to compute skill means", value=True)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    if COLS.get("position") in df.columns:
        positions = sorted(df[COLS["position"]].dropna().unique().tolist())
        if positions:
            position_name = st.selectbox("Select position", positions, index=0)
        else:
            position_name = st.text_input("Position name (no positions found in column):", "HR Analyst")
    else:
        st.info("No 'position' column found; computing ICP over the entire dataset.")
        position_name = "All"

    if st.button("🚀 Generate ICP JSON"):
        try:
            payload, df_role = compute_icp(df, position_name, perf_min, kpi_q, COLS, SKILL_KEYS, use_top_only)
        except Exception as e:
            st.error(f"Failed to compute ICP: {e}")
            st.stop()

        st.success("ICP JSON generated.")
        st.subheader("Preview")
        st.json(payload)

        buf = BytesIO(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
        st.download_button("📥 Download icp.json", buf, file_name="icp.json", mime="application/json")

        with st.expander("🤖 Optional: Run a quick XGBoost training report"):
            if st.button("Run XGBoost demo"):
                report = optional_xgboost_report(df_role, COLS, SKILL_KEYS)
                st.code(report, language="text")
else:
    st.info("Choose a CSV file to begin.")
