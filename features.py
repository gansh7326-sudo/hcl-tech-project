"""Shared feature engineering (training aur Streamlit app dono yahi use karte hain)."""
import numpy as np
import pandas as pd

CAT_COLS = ["Crop", "Season", "State"]
NUM_BASE = ["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
ENGINEERED = [
    "Fertilizer_per_ha", "Pesticide_per_ha", "Log_Area",
    "Rainfall_x_Fert_per_ha", "Years_Since_Start", "Rainfall_Category",
]
NUM_COLS = NUM_BASE + [c for c in ENGINEERED if c != "Rainfall_Category"]
FEATURE_COLS = CAT_COLS + ["Rainfall_Category"] + NUM_COLS
BASE_YEAR = 1997


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw columns se naye features banata hai. NOTE: 'Production' ko feature nahi
    banaya kyunki Yield = Production / Area hota hai (target leakage)."""
    df = df.copy()
    area = df["Area"].clip(lower=1e-6)
    df["Fertilizer_per_ha"] = df["Fertilizer"] / area
    df["Pesticide_per_ha"] = df["Pesticide"] / area
    df["Log_Area"] = np.log1p(df["Area"])
    df["Rainfall_x_Fert_per_ha"] = df["Annual_Rainfall"] * df["Fertilizer_per_ha"] / 1000
    df["Years_Since_Start"] = df["Crop_Year"] - BASE_YEAR
    df["Rainfall_Category"] = pd.cut(
        df["Annual_Rainfall"], bins=[-np.inf, 700, 1200, 2000, np.inf],
        labels=["Low", "Medium", "High", "Very High"],
    ).astype(str)
    return df
