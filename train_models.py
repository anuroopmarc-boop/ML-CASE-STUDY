"""
train_models.py  –  Fit & save the best models for the Streamlit app.

Run once:
    python train_models.py

Outputs (saved to ./models/):
    reg_pipeline.pkl   – Gradient Boosting Regressor pipeline (used-car price)
    clf_pipeline.pkl   – Random Forest Classifier pipeline (employee promotion)
    reg_meta.pkl       – dict with column lists, caps, dept_means etc.
    clf_meta.pkl       – dict with column lists, caps, dept_means etc.
"""

import os, warnings, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. REGRESSION  –  Used Car Price
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("REGRESSION: Used-car price prediction")
print("=" * 60)

REG_DATA = os.path.join("DATASET", "archive (2)", "used_cars.csv")
df_reg = pd.read_csv(REG_DATA)
df_reg = df_reg.drop(columns=["Unnamed: 0"], errors="ignore")

# strip stray spaces
for col in df_reg.select_dtypes(include="object").columns:
    df_reg[col] = df_reg[col].astype(str).str.strip()

TARGET_REG = "price"
REF_YEAR = 2020

# --- cleaning ---
df_reg = df_reg.drop_duplicates().reset_index(drop=True)
df_reg = df_reg[df_reg["year"] <= 2020].reset_index(drop=True)
hidden = (df_reg["engineSize"] == 0) & (df_reg["fuelType"] != "Electric")
df_reg.loc[hidden, "engineSize"] = np.nan

# --- engineered features ---
df_reg["car_age"] = REF_YEAR - df_reg["year"]
df_reg["mileage_per_year"] = df_reg["mileage"] / (df_reg["car_age"] + 1)

cat_cols_reg = ["model", "transmission", "fuelType", "Make"]
num_cols_reg = ["mileage", "tax", "mpg", "engineSize", "car_age", "mileage_per_year"]

X_reg = df_reg[cat_cols_reg + num_cols_reg]
y_reg = df_reg[TARGET_REG]

price_bins = pd.qcut(y_reg, q=5, labels=False)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE, stratify=price_bins
)
X_tr, X_te = X_tr.copy(), X_te.copy()

# impute engineSize
model_median = X_tr.groupby("model")["engineSize"].median()
global_median = X_tr["engineSize"].median()
for part in (X_tr, X_te):
    part["engineSize"] = (
        part["engineSize"].fillna(part["model"].map(model_median)).fillna(global_median)
    )

# outlier caps (train-derived)
caps_reg = {}
for col in ["mileage", "tax", "mpg", "engineSize", "mileage_per_year"]:
    caps_reg[col] = (X_tr[col].quantile(0.01), X_tr[col].quantile(0.99))
    X_tr[col] = X_tr[col].clip(*caps_reg[col])
    X_te[col] = X_te[col].clip(*caps_reg[col])

preprocessor_reg = ColumnTransformer([
    ("num", StandardScaler(), num_cols_reg),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols_reg),
])

pipeline_reg = Pipeline([
    ("preprocessor", preprocessor_reg),
    ("model", GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=5,
        subsample=0.8, random_state=RANDOM_STATE
    )),
])

print("Fitting Gradient Boosting Regressor …")
pipeline_reg.fit(X_tr, y_tr)
y_pred_reg = pipeline_reg.predict(X_te)
r2 = r2_score(y_te, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_te, y_pred_reg))
mae = mean_absolute_error(y_te, y_pred_reg)
print(f"  R² = {r2:.4f}  RMSE = {rmse:.0f}  MAE = {mae:.0f}")

# Save unique values for UI dropdowns (from training data)
makes = sorted(X_tr["Make"].dropna().unique().tolist())
models_per_make = (
    X_tr[["Make", "model"]].drop_duplicates()
    .groupby("Make")["model"].apply(sorted).apply(list).to_dict()
)
transmissions = sorted(X_tr["transmission"].dropna().unique().tolist())
fuel_types = sorted(X_tr["fuelType"].dropna().unique().tolist())

reg_meta = {
    "cat_cols": cat_cols_reg,
    "num_cols": num_cols_reg,
    "caps": caps_reg,
    "model_median": model_median.to_dict(),
    "global_median": global_median,
    "ref_year": REF_YEAR,
    "makes": makes,
    "models_per_make": models_per_make,
    "transmissions": transmissions,
    "fuel_types": fuel_types,
    "year_min": int(X_tr["car_age"].apply(lambda a: REF_YEAR - a).min()),
    "year_max": int(X_tr["car_age"].apply(lambda a: REF_YEAR - a).max()),
    "metrics": {"R2": round(r2, 4), "RMSE": round(rmse, 0), "MAE": round(mae, 0)},
}

joblib.dump(pipeline_reg, "models/reg_pipeline.pkl")
joblib.dump(reg_meta, "models/reg_meta.pkl")
print("  Saved: models/reg_pipeline.pkl  models/reg_meta.pkl")


# ─────────────────────────────────────────────────────────────
# 2. CLASSIFICATION  –  Employee Promotion
# ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("CLASSIFICATION: Employee promotion prediction")
print("=" * 60)

CLF_DATA = os.path.join("DATASET", "archive (1)", "train.csv")
df_clf = pd.read_csv(CLF_DATA)
df_clf = df_clf.drop(columns=["employee_id"], errors="ignore")
for col in df_clf.select_dtypes(include=["object", "string"]).columns:
    df_clf[col] = df_clf[col].astype("object").where(
        df_clf[col].isna(), df_clf[col].astype(str).str.strip()
    )

TARGET_CLF = "is_promoted"

# cleaning
df_clf = df_clf.drop_duplicates().reset_index(drop=True)

# handle missing previous_year_rating – flag + fill 0
df_clf["no_previous_rating"] = df_clf["previous_year_rating"].isnull().astype(int)
df_clf["previous_year_rating"] = df_clf["previous_year_rating"].fillna(0)

# handle missing education
df_clf["education"] = df_clf["education"].fillna("Unknown")

# engineered features
df_clf["service_ratio"] = df_clf["length_of_service"] / (df_clf["age"] - 18).clip(lower=1)
df_clf["trainings_per_year"] = df_clf["no_of_trainings"] / df_clf["length_of_service"]
df_clf["performance_score"] = df_clf["previous_year_rating"] + 2 * df_clf["awards_won?"]

cat_cols_clf = ["department", "region", "education", "gender", "recruitment_channel"]
num_cols_clf = [
    "no_of_trainings", "age", "previous_year_rating", "length_of_service",
    "awards_won?", "avg_training_score",
    "no_previous_rating", "service_ratio", "trainings_per_year",
    "performance_score", "score_vs_dept_avg",
]

X_clf = df_clf[cat_cols_clf + [c for c in num_cols_clf if c != "score_vs_dept_avg"]]
y_clf = df_clf[TARGET_CLF]

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
)
X_tr2, X_te2 = X_tr2.copy(), X_te2.copy()

# score_vs_dept_avg (train statistics only)
dept_mean = X_tr2.groupby("department")["avg_training_score"].mean()
overall_mean = X_tr2["avg_training_score"].mean()
for part in (X_tr2, X_te2):
    part["score_vs_dept_avg"] = (
        part["avg_training_score"] - part["department"].map(dept_mean).fillna(overall_mean)
    )

# outlier caps
caps_clf = {}
for col in ["no_of_trainings", "age", "length_of_service", "service_ratio", "trainings_per_year"]:
    caps_clf[col] = (X_tr2[col].quantile(0.01), X_tr2[col].quantile(0.99))
    X_tr2[col] = X_tr2[col].clip(*caps_clf[col])
    X_te2[col] = X_te2[col].clip(*caps_clf[col])

all_num_clf = [c for c in num_cols_clf if c != "score_vs_dept_avg"] + ["score_vs_dept_avg"]

preprocessor_clf = ColumnTransformer([
    ("num", StandardScaler(), all_num_clf),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols_clf),
])

pipeline_clf = Pipeline([
    ("preprocessor", preprocessor_clf),
    ("model", RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )),
])

print("Fitting Random Forest Classifier …")
pipeline_clf.fit(X_tr2, y_tr2)
y_pred_clf = pipeline_clf.predict(X_te2)
proba_clf = pipeline_clf.predict_proba(X_te2)[:, 1]
acc = accuracy_score(y_te2, y_pred_clf)
f1 = f1_score(y_te2, y_pred_clf, average="weighted")
auc = roc_auc_score(y_te2, proba_clf)
print(f"  Accuracy = {acc:.4f}  F1 (weighted) = {f1:.4f}  ROC-AUC = {auc:.4f}")

departments = sorted(X_tr2["department"].dropna().unique().tolist())
regions = sorted(X_tr2["region"].dropna().unique().tolist())
educations = sorted(df_clf["education"].unique().tolist())
genders = sorted(X_tr2["gender"].dropna().unique().tolist())
rec_channels = sorted(X_tr2["recruitment_channel"].dropna().unique().tolist())

clf_meta = {
    "cat_cols": cat_cols_clf,
    "num_cols": all_num_clf,
    "caps": caps_clf,
    "dept_mean": dept_mean.to_dict(),
    "overall_mean": overall_mean,
    "departments": departments,
    "regions": regions,
    "educations": educations,
    "genders": genders,
    "rec_channels": rec_channels,
    "metrics": {"Accuracy": round(acc, 4), "F1_weighted": round(f1, 4), "ROC_AUC": round(auc, 4)},
}

joblib.dump(pipeline_clf, "models/clf_pipeline.pkl")
joblib.dump(clf_meta, "models/clf_meta.pkl")
print("  Saved: models/clf_pipeline.pkl  models/clf_meta.pkl")
print()
print("Done! [OK]  All models saved to ./models/")
