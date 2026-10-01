"""Streamlit dashboard: streamlit run app.py"""
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from features import add_features

st.set_page_config(page_title="AI Crop Yield Prediction", page_icon="🌾", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("models/crop_yield_model.joblib")


@st.cache_data
def load_data():
    return pd.read_csv("data/crop_yield_clean.csv")


@st.cache_data
def load_meta():
    return json.load(open("models/meta.json"))


model, df, meta = load_model(), load_data(), load_meta()

st.title("🌾 AI-Based Crop Yield Prediction")
st.caption("P_267 · Python · Pandas · XGBoost · Streamlit")

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Data Explorer", "📈 Model Performance", "ℹ️ About"])

# ---------------------------------------------------------------- PREDICT
with tab1:
    st.subheader("Fasal ki yield predict karo")
    c1, c2, c3 = st.columns(3)
    crop = c1.selectbox("Crop", meta["crops"])
    state = c2.selectbox("State", meta["states"])
    season = c3.selectbox("Season", meta["seasons"])

    c4, c5, c6 = st.columns(3)
    year = c4.number_input("Year", 1990, 2050, 2020)
    area = c5.number_input("Area (hectares)", 1.0, 5_000_000.0, 10_000.0, step=500.0)
    rain = c6.number_input("Annual rainfall (mm)", 100.0, 4000.0, 1000.0, step=50.0)

    c7, c8 = st.columns(2)
    fert_ha = c7.slider("Fertilizer (kg per hectare)", 0, 400, 110)
    pest_ha = c8.slider("Pesticide (kg per hectare)", 0.0, 3.0, 0.45, step=0.05)

    if st.button("Predict Yield", type="primary"):
        row = pd.DataFrame([{
            "Crop": crop, "Season": season, "State": state, "Crop_Year": year,
            "Area": area, "Annual_Rainfall": rain,
            "Fertilizer": fert_ha * area, "Pesticide": pest_ha * area,
        }])
        pred = float(model.predict(add_features(row)[model.feature_names_in_])[0])
        pred = max(pred, 0.0)
        m1, m2 = st.columns(2)
        m1.metric("Predicted yield", f"{pred:,.2f} tonnes/ha")
        m2.metric("Estimated total production", f"{pred * area:,.0f} tonnes")

        hist = df[df["Crop"] == crop]["Yield"]
        st.info(f"Historical average for **{crop}**: {hist.mean():.2f} t/ha "
                f"(range {hist.quantile(.05):.2f} – {hist.quantile(.95):.2f})")

# ---------------------------------------------------------------- EXPLORER
with tab2:
    st.subheader("Dataset overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{len(df):,}")
    k2.metric("Crops", df["Crop"].nunique())
    k3.metric("States", df["State"].nunique())
    k4.metric("Years", f"{df['Crop_Year'].min()}–{df['Crop_Year'].max()}")

    f1, f2 = st.columns(2)
    sel_crops = f1.multiselect("Filter crops", meta["crops"], default=meta["crops"][:4])
    sel_states = f2.multiselect("Filter states", meta["states"], default=meta["states"])
    fdf = df[df["Crop"].isin(sel_crops) & df["State"].isin(sel_states)]

    if fdf.empty:
        st.warning("Koi data nahi mila, filters change karo.")
    else:
        a, b = st.columns(2)
        a.plotly_chart(px.line(fdf.groupby(["Crop_Year", "Crop"], as_index=False)["Yield"].mean(),
                               x="Crop_Year", y="Yield", color="Crop", title="Yield trend"),
                       use_container_width=True)
        b.plotly_chart(px.box(fdf, x="Crop", y="Yield", color="Crop", log_y=True,
                              title="Yield distribution"), use_container_width=True)
        c, d = st.columns(2)
        c.plotly_chart(px.scatter(fdf, x="Annual_Rainfall", y="Yield", color="Crop", opacity=.6,
                                  title="Rainfall vs Yield"), use_container_width=True)
        d.plotly_chart(px.scatter(fdf, x="Fertilizer_per_ha", y="Yield", color="Crop", opacity=.6,
                                  title="Fertilizer/ha vs Yield"), use_container_width=True)
        st.plotly_chart(px.bar(fdf.groupby("State", as_index=False)["Yield"].mean()
                               .sort_values("Yield"), x="Yield", y="State", orientation="h",
                               title="Average yield by state"), use_container_width=True)
        with st.expander("Raw data dekho"):
            st.dataframe(fdf.head(500), use_container_width=True)

# ---------------------------------------------------------------- PERFORMANCE
with tab3:
    st.subheader(f"Final model: {meta['model']}")
    perf = pd.DataFrame(meta["metrics"]).T.round(4)
    st.dataframe(perf, use_container_width=True)
    st.write(f"**5-fold CV R²:** {meta['cv_r2_mean']:.4f} ± {meta['cv_r2_std']:.4f}")
    p1, p2 = st.columns(2)
    p1.image("outputs/plots/actual_vs_pred.png", caption="Actual vs Predicted")
    try:
        p2.image("outputs/plots/feature_importance.png", caption="Feature importance")
    except Exception:
        pass

# ---------------------------------------------------------------- ABOUT
with tab4:
    st.markdown("""
**Pipeline:** Data load → Cleaning (duplicates, missing values, outliers) → EDA →
Feature engineering (per-hectare inputs, log area, rainfall×fertilizer, rainfall category) →
Model training (Ridge, Random Forest, XGBoost) → Evaluation (R², RMSE, MAE, 5-fold CV) → Streamlit dashboard.

**Note:** `Production` ko feature nahi banaya, kyunki Yield = Production / Area hota hai (data leakage).
""")
    st.write("**Cleaning report:**", meta["cleaning_report"])
