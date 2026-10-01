"""End-to-end pipeline: load -> clean -> EDA -> feature engineering -> train -> evaluate -> save."""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CAT_COLS, FEATURE_COLS, NUM_COLS, add_features

warnings.filterwarnings("ignore")
PLOTS = "outputs/plots"
sns.set_theme(style="whitegrid")

try:
    from xgboost import XGBRegressor
    XGB_OK = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    XGB_OK = False
    print("WARNING: xgboost installed nahi hai -> temporary HistGradientBoosting use ho raha hai. "
          "'pip install xgboost' karke dobara chalao.")

# ------------------------------------------------------------------ 1. LOAD
df = pd.read_csv("data/crop_yield.csv")
print("Raw shape:", df.shape)

# ------------------------------------------------------------------ 2. CLEANING
report = {"raw_rows": len(df)}
for c in ["Crop", "Season", "State"]:
    df[c] = df[c].astype(str).str.strip()
report["duplicates_removed"] = int(df.duplicated().sum())
df = df.drop_duplicates()

report["missing_before"] = df.isna().sum()[df.isna().sum() > 0].to_dict()
for c in ["Annual_Rainfall", "Fertilizer", "Pesticide"]:
    df[c] = df.groupby("Crop")[c].transform(lambda s: s.fillna(s.median()))
df = df.dropna()

df = df[(df["Area"] > 0) & (df["Yield"] > 0)]
# Outliers: har crop ke andar IQR method se hatao
def iqr_mask(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    i = q3 - q1
    return s.between(q1 - 3 * i, q3 + 3 * i)
before = len(df)
df = df[df.groupby("Crop")["Yield"].transform(iqr_mask)]
report["outliers_removed"] = before - len(df)
report["clean_rows"] = len(df)
print("Cleaning report:", report)

# ------------------------------------------------------------------ 3. EDA / VISUALIZATION
fig, ax = plt.subplots(figsize=(9, 4.5))
order = df.groupby("Crop")["Yield"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="Crop", y="Yield", order=order, ax=ax)
ax.set_yscale("log"); ax.set_title("Yield distribution by crop (log scale)")
plt.xticks(rotation=40, ha="right"); plt.tight_layout()
plt.savefig(f"{PLOTS}/yield_by_crop.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(8, 4))
df.groupby("Crop_Year")["Yield"].mean().plot(ax=ax, marker="o")
ax.set_title("Average yield over years"); ax.set_ylabel("Yield (t/ha)")
plt.tight_layout(); plt.savefig(f"{PLOTS}/yield_trend.png", dpi=130); plt.close()

num_eda = df[["Area", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"]]
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(num_eda.corr(), annot=True, fmt=".2f", cmap="YlGn", ax=ax)
ax.set_title("Correlation heatmap"); plt.tight_layout()
plt.savefig(f"{PLOTS}/correlation.png", dpi=130); plt.close()

# ------------------------------------------------------------------ 4. FEATURE ENGINEERING
df = add_features(df)
df.to_csv("data/crop_yield_clean.csv", index=False)

X, y = df[FEATURE_COLS], df["Yield"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS + ["Rainfall_Category"]),
    ("num", StandardScaler(), NUM_COLS),
])

# ------------------------------------------------------------------ 5. MODELS
if XGB_OK:
    boosted = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                           subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1)
    boosted_name = "XGBoost"
else:
    boosted = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=42)
    boosted_name = "GradientBoosting (fallback)"

models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
    boosted_name: boosted,
}

results, fitted = [], {}
for name, model in models.items():
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    results.append({
        "Model": name,
        "R2": r2_score(y_te, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
        "MAE": mean_absolute_error(y_te, pred),
    })
    fitted[name] = (pipe, pred)
    print(f"{name:32s} R2={results[-1]['R2']:.4f}  RMSE={results[-1]['RMSE']:.3f}  MAE={results[-1]['MAE']:.3f}")

res_df = pd.DataFrame(results).sort_values("R2", ascending=False)
res_df.to_csv("outputs/model_comparison.csv", index=False)
best_name = boosted_name  # final model boosted wala hi rakhte hain (project requirement: XGBoost)
best_pipe, best_pred = fitted[best_name]

# ------------------------------------------------------------------ 6. EVALUATION
cv = cross_val_score(Pipeline([("pre", pre), ("model", boosted)]), X, y,
                     cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
print(f"5-fold CV R2: {cv.mean():.4f} +/- {cv.std():.4f}")

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(y_te, best_pred, s=8, alpha=0.4)
lim = [0, max(y_te.max(), best_pred.max())]
ax.plot(lim, lim, "r--"); ax.set_xscale("symlog"); ax.set_yscale("symlog")
ax.set_xlabel("Actual yield"); ax.set_ylabel("Predicted yield")
ax.set_title(f"{best_name}: Actual vs Predicted"); plt.tight_layout()
plt.savefig(f"{PLOTS}/actual_vs_pred.png", dpi=130); plt.close()

# Feature importance (XGBoost/tree models)
try:
    names = best_pipe.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(best_pipe.named_steps["model"].feature_importances_, index=names)
    imp.index = [i.split("__", 1)[1] for i in imp.index]
    top = imp.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    top.plot.barh(ax=ax, color="seagreen"); ax.set_title("Top 15 feature importances")
    plt.tight_layout(); plt.savefig(f"{PLOTS}/feature_importance.png", dpi=130); plt.close()
    imp.sort_values(ascending=False).to_csv("outputs/feature_importance.csv")
except AttributeError:
    pass

# ------------------------------------------------------------------ 7. SAVE
joblib.dump(best_pipe, "models/crop_yield_model.joblib")
meta = {
    "model": best_name,
    "metrics": res_df.set_index("Model").to_dict("index"),
    "cv_r2_mean": float(cv.mean()), "cv_r2_std": float(cv.std()),
    "cleaning_report": report,
    "crops": sorted(df["Crop"].unique()), "states": sorted(df["State"].unique()),
    "seasons": sorted(df["Season"].unique()),
    "year_range": [int(df["Crop_Year"].min()), int(df["Crop_Year"].max())],
}
json.dump(meta, open("models/meta.json", "w"), indent=2)
print("Saved model + metadata. Ab chalao: streamlit run app.py")
