# P_267 – AI-Based Crop Yield Prediction

Machine-learning project jo Indian crop data (crop, state, season, rainfall, fertilizer, pesticide, area) se
**yield (tonnes/hectare)** predict karta hai, aur Streamlit dashboard me dikhata hai.

## Folder structure
```
crop_yield_project/
├── data/                  # crop_yield.csv (raw), crop_yield_clean.csv (cleaned + features)
├── models/                # trained model (.joblib) + meta.json
├── outputs/               # plots, model_comparison.csv
├── features.py            # shared feature engineering
├── generate_sample_data.py# synthetic dataset banata hai (sirf demo ke liye)
├── train_model.py         # cleaning -> EDA -> features -> training -> evaluation
├── app.py                 # Streamlit dashboard
└── requirements.txt
```

## Run karne ke steps
```bash
pip install -r requirements.txt
python generate_sample_data.py   # SKIP karo agar real dataset use kar rahe ho
python train_model.py            # XGBoost model train + plots save
streamlit run app.py
```

## Real dataset use karna (recommended)
Kaggle par "Crop Production in India" / "crop_yield.csv" search karo. File ko `data/crop_yield.csv` naam se rakho.
Required columns: `Crop, Crop_Year, Season, State, Area, Production, Annual_Rainfall, Fertilizer, Pesticide, Yield`.
Phir sirf `python train_model.py` chalao.

## Pipeline
1. **Data cleaning** - spaces strip, duplicates hatana, missing values (crop-wise median), invalid area/yield, outliers (IQR).
2. **Visualization** - yield by crop, yearly trend, correlation heatmap (`outputs/plots/`).
3. **Feature engineering** - Fertilizer/ha, Pesticide/ha, log(area), rainfall x fertilizer, years since 1997, rainfall category.
4. **Models** - Ridge (baseline), Random Forest, **XGBoost (final)**.
5. **Evaluation** - R2, RMSE, MAE, 5-fold cross-validation, actual-vs-predicted, feature importance.
6. **Dashboard** - Predict tab, Data Explorer, Model Performance, About.

## Important note (viva ke liye)
`Production` ko feature nahi banaya, kyunki `Yield = Production / Area` hota hai. Use karte to **data leakage** hoti
aur accuracy fake high aati.

## Future scope
Real-time weather API, soil data (NPK, pH), satellite/NDVI data, hyperparameter tuning (Optuna), SHAP explainability.
