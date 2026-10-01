"""Synthetic dataset (Indian crop-yield Kaggle dataset jaisa schema).
Agar aapke paas real dataset hai to isko chalane ki zaroorat nahi -
bas data/crop_yield.csv rakh do.
Columns: Crop, Crop_Year, Season, State, Area, Production, Annual_Rainfall,
         Fertilizer, Pesticide, Yield
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

crops = {  # base yield (t/ha), preferred season, rain sweet-spot (mm)
    "Rice": (2.4, "Kharif", 1400), "Wheat": (3.0, "Rabi", 800),
    "Maize": (2.7, "Kharif", 1000), "Cotton(lint)": (0.5, "Kharif", 900),
    "Sugarcane": (65.0, "Whole Year", 1500), "Potato": (20.0, "Rabi", 700),
    "Groundnut": (1.3, "Kharif", 900), "Soybean": (1.1, "Kharif", 1000),
    "Arhar/Tur": (0.8, "Kharif", 900), "Bajra": (1.1, "Kharif", 600),
}
states = {  # state: (avg rainfall, fertility multiplier)
    "Uttar Pradesh": (1000, 1.10), "Punjab": (650, 1.35), "Haryana": (600, 1.25),
    "Maharashtra": (1100, 0.90), "Karnataka": (1150, 0.95), "Tamil Nadu": (950, 1.05),
    "West Bengal": (1750, 1.10), "Bihar": (1200, 1.00), "Madhya Pradesh": (1050, 0.90),
    "Gujarat": (800, 0.95), "Rajasthan": (550, 0.75), "Andhra Pradesh": (950, 1.00),
}

rows = []
for year in range(1997, 2021):
    for state, (s_rain, fert_mult) in states.items():
        for crop, (base, season, sweet) in crops.items():
            if rng.random() < 0.25:      # har combination har saal nahi hota
                continue
            area = float(np.exp(rng.normal(9.5, 1.3)))              # hectares
            rain = max(150, rng.normal(s_rain, s_rain * 0.22))      # mm
            fert_ha = max(20, rng.normal(110, 40) * fert_mult)      # kg/ha
            pest_ha = max(0.1, rng.normal(0.45, 0.2))               # kg/ha
            rain_eff = np.exp(-((rain - sweet) / (0.6 * sweet)) ** 2)
            fert_eff = 1 + 0.45 * (1 - np.exp(-fert_ha / 90))
            pest_eff = 1 + 0.08 * (1 - np.exp(-pest_ha / 0.4))
            trend = 1 + 0.012 * (year - 1997)
            y = base * (0.55 + 0.6 * rain_eff) * fert_eff * pest_eff * trend * fert_mult
            y *= rng.lognormal(0, 0.10)
            rows.append(dict(
                Crop=crop, Crop_Year=year, Season=season, State=state,
                Area=round(area, 1), Production=round(y * area, 1),
                Annual_Rainfall=round(rain, 1),
                Fertilizer=round(fert_ha * area, 1), Pesticide=round(pest_ha * area, 1),
                Yield=round(y, 4),
            ))

df = pd.DataFrame(rows)

# Data cleaning demo ke liye thodi "gandagi" daalte hain
n = len(df)
for col in ["Annual_Rainfall", "Fertilizer", "Pesticide"]:
    df.loc[rng.choice(n, int(0.01 * n), replace=False), col] = np.nan
df.loc[rng.choice(n, 25, replace=False), "Yield"] *= 6       # outliers
df["Crop"] = df["Crop"].where(rng.random(n) > 0.02, df["Crop"] + " ")  # trailing spaces
df = pd.concat([df, df.sample(40, random_state=1)], ignore_index=True)  # duplicates

df.to_csv("data/crop_yield.csv", index=False)
print("Saved data/crop_yield.csv ->", df.shape)
