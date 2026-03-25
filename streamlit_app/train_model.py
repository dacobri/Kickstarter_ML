"""
Train the Random Forest model and save all artifacts needed by the Streamlit app.
Run once: python train_model.py
Reads from ../notebooks/data/ (parquet files produced by notebooks 01-04).
Saves model + lookup tables to artifacts/
"""

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
NOTEBOOK_DATA = os.path.join(os.path.dirname(__file__), "..", "notebooks", "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── load data ────────────────────────────────────────────────────────────────
print("Loading training data...")
train_df = pd.read_parquet(os.path.join(NOTEBOOK_DATA, "train_df.parquet"))
val_df = pd.read_parquet(os.path.join(NOTEBOOK_DATA, "val_df.parquet"))

with open(os.path.join(NOTEBOOK_DATA, "feature_cols.json")) as f:
    FEATURE_COLS = json.load(f)

with open(os.path.join(NOTEBOOK_DATA, "tfidf_word_list.json")) as f:
    tfidf_words = json.load(f)

print(f"  Train: {len(train_df)} rows, Val: {len(val_df)} rows")
print(f"  Features: {len(FEATURE_COLS)}")

# ── build lookup artifacts from training data ────────────────────────────────

goal = pd.to_numeric(train_df["goal"], errors="coerce").fillna(0)
global_mean_success = float(train_df["success"].mean())
SMOOTHING = 10


def build_target_encoder(df, col, target="success", smoothing=10):
    global_mean = df[target].mean()
    stats = df.groupby(col)[target].agg(["mean", "count"])
    encoder = {}
    for cat, row in stats.iterrows():
        n = row["count"]
        encoder[cat] = (n * row["mean"] + smoothing * global_mean) / (n + smoothing)
    encoder["__global_mean__"] = global_mean
    return encoder


cat_name_encoder = build_target_encoder(train_df, "cat_name")
cat_parent_encoder = build_target_encoder(train_df, "cat_parent_name")

cat_medians = goal.groupby(train_df["cat_name"]).median().to_dict()
global_median_goal = float(goal.median())

# Median goal of SUCCESSFUL campaigns per category (for advice)
successful = train_df[train_df["success"] == 1]
goal_success = pd.to_numeric(successful["goal"], errors="coerce").fillna(0)
cat_success_medians = goal_success.groupby(successful["cat_name"]).median().to_dict()

# Countries with >= 500 campaigns
vc = train_df["country"].value_counts()
keep_countries = sorted(vc[vc >= 500].index.tolist())

# Category name -> parent name mapping
cat_to_parent = (
    train_df.drop_duplicates("cat_name")
    .set_index("cat_name")["cat_parent_name"]
    .to_dict()
)

# Prep days 99th percentile (for winsorising)
launched = pd.to_numeric(train_df["launched_at"], errors="coerce")
created = pd.to_numeric(train_df["created_at"], errors="coerce")
prep = ((launched - created) / 86400).clip(lower=0)
prep_99 = float(prep.quantile(0.99))

# Goal 99th percentile (for goal_per_day cap)
goal_99 = float(goal.quantile(0.99))

# Video success lift
has_vid = train_df["video"].notna().astype(int)
video_success_rate = float(train_df[has_vid == 1]["success"].mean())
no_video_success_rate = float(train_df[has_vid == 0]["success"].mean())

# ── feature engineering (replicate notebook 03b) ─────────────────────────────

blurb_terms = [w["term"] for w in tfidf_words["blurb"]]
name_terms = [w["term"] for w in tfidf_words["name"]]


def engineer_features(df, cat_name_enc, cat_parent_enc, cat_meds, keep_ctries):
    """Replicate the exact feature engineering from notebook 03b."""
    out = pd.DataFrame(index=df.index)

    g = pd.to_numeric(df["goal"], errors="coerce").fillna(0).clip(lower=0)
    deadline = pd.to_numeric(df["deadline"], errors="coerce")
    launched_at = pd.to_numeric(df["launched_at"], errors="coerce")
    created_at = pd.to_numeric(df["created_at"], errors="coerce")

    out["log_goal"] = np.log1p(g)
    out["duration_days"] = ((deadline - launched_at) / 86400).clip(0, 90)
    raw_prep = ((launched_at - created_at) / 86400).clip(lower=0)
    out["prep_days"] = raw_prep.clip(upper=prep_99)
    out["has_video"] = df["video"].notna().astype(int)
    out["blurb_length"] = df["blurb"].fillna("").str.len().astype(int)
    out["name_length"] = df["name"].fillna("").str.len().astype(int)
    out["blurb_word_count"] = (
        df["blurb"].fillna("").str.split().str.len().fillna(0).astype(int)
    )
    out["name_number"] = (
        df["name"].fillna("").str.contains(r"\d", regex=True).astype(int)
    )
    out["goal_is_round"] = ((g % 1000) == 0).astype(int)
    out["is_usd"] = (df["currency"] == "USD").astype(int)

    ts = pd.to_datetime(launched_at, unit="s", utc=True)
    out["launched_month"] = ts.dt.month.astype(float)
    out["launched_dayofweek"] = ts.dt.dayofweek.astype(float)

    out["goal_per_day"] = (g / out["duration_days"]).fillna(0).clip(upper=goal_99)

    # log_goal_vs_cat_median
    med_mapped = df["cat_name"].map(cat_meds).fillna(global_median_goal)
    out["log_goal_vs_cat_median"] = np.log1p(g / med_mapped.replace(0, 1))

    # Target encoding
    gm = cat_name_enc.get("__global_mean__", 0.5)
    out["cat_name_encoded"] = df["cat_name"].map(cat_name_enc).fillna(gm)
    gm2 = cat_parent_enc.get("__global_mean__", 0.5)
    out["cat_parent_encoded"] = df["cat_parent_name"].map(cat_parent_enc).fillna(gm2)

    # Country OHE
    country_clean = df["country"].where(df["country"].isin(keep_ctries), "Other")
    for c in sorted(keep_ctries) + ["Other"]:
        col = f"country_{c}"
        out[col] = (country_clean == c).astype(int)

    # TF-IDF binary features
    blurb_text = df["blurb"].fillna("").astype(str).str.lower()
    name_text = df["name"].fillna("").astype(str).str.lower()
    for term in blurb_terms:
        col = "blurb_has_" + term.replace(" ", "_")
        out[col] = blurb_text.str.contains(term, regex=False).astype(int)
    for term in name_terms:
        col = "name_has_" + term.replace(" ", "_")
        out[col] = name_text.str.contains(term, regex=False).astype(int)

    # Ensure all expected columns exist, fill missing with 0
    for c in FEATURE_COLS:
        if c not in out.columns:
            out[c] = 0

    return out[FEATURE_COLS].fillna(0)


print("Engineering features...")
X_train = engineer_features(
    train_df, cat_name_encoder, cat_parent_encoder, cat_medians, keep_countries
)
y_train = train_df["success"].astype(int)

# Quick sanity check against saved parquet
X_train_saved = pd.read_parquet(os.path.join(NOTEBOOK_DATA, "X_train.parquet"))
print(f"  Our X_train shape: {X_train.shape}")
print(f"  Saved X_train shape: {X_train_saved.shape}")
print(f"  Columns match: {list(X_train.columns) == list(X_train_saved.columns)}")

# Check correlation with saved features
sample_idx = X_train.index[:1000]
for col in ["log_goal", "cat_name_encoded", "duration_days", "has_video"]:
    corr = np.corrcoef(
        X_train.loc[sample_idx, col].values,
        X_train_saved.loc[sample_idx, col].values,
    )[0, 1]
    print(f"  {col} correlation: {corr:.6f}")

# ── train Random Forest ──────────────────────────────────────────────────────

print("\nTraining Random Forest (n_estimators=300, max_depth=20)...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    max_features="sqrt",
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

# Validate on val set
X_val = engineer_features(
    val_df, cat_name_encoder, cat_parent_encoder, cat_medians, keep_countries
)
y_val = val_df["success"].astype(int)
from sklearn.metrics import roc_auc_score

val_auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
print(f"  Validation ROC-AUC: {val_auc:.4f}")

# ── save everything ──────────────────────────────────────────────────────────

print("\nSaving artifacts...")

# Model
joblib.dump(rf, os.path.join(ARTIFACTS_DIR, "rf_model.joblib"))

# Feature columns
with open(os.path.join(ARTIFACTS_DIR, "feature_cols.json"), "w") as f:
    json.dump(FEATURE_COLS, f)

# TF-IDF word lists (just the terms)
with open(os.path.join(ARTIFACTS_DIR, "tfidf_terms.json"), "w") as f:
    json.dump({"blurb": blurb_terms, "name": name_terms}, f)

# Lookup tables
lookups = {
    "cat_name_encoder": {k: round(v, 6) for k, v in cat_name_encoder.items()},
    "cat_parent_encoder": {k: round(v, 6) for k, v in cat_parent_encoder.items()},
    "cat_medians": {k: round(v, 2) for k, v in cat_medians.items()},
    "cat_success_medians": {k: round(v, 2) for k, v in cat_success_medians.items()},
    "global_median_goal": round(global_median_goal, 2),
    "global_mean_success": round(global_mean_success, 6),
    "prep_days_99pct": round(prep_99, 2),
    "goal_99pct": round(goal_99, 2),
    "keep_countries": keep_countries,
    "cat_to_parent": cat_to_parent,
    "video_success_rate": round(video_success_rate, 4),
    "no_video_success_rate": round(no_video_success_rate, 4),
}
with open(os.path.join(ARTIFACTS_DIR, "lookups.json"), "w") as f:
    json.dump(lookups, f, indent=2)

# Feature importances (for factor analysis)
importances = dict(zip(FEATURE_COLS, rf.feature_importances_))
with open(os.path.join(ARTIFACTS_DIR, "rf_importances.json"), "w") as f:
    json.dump({k: round(v, 8) for k, v in importances.items()}, f, indent=2)

print(f"\nDone! All artifacts saved to {ARTIFACTS_DIR}/")
print(f"  rf_model.joblib  ({os.path.getsize(os.path.join(ARTIFACTS_DIR, 'rf_model.joblib')) / 1e6:.1f} MB)")
print(f"  feature_cols.json")
print(f"  tfidf_terms.json")
print(f"  lookups.json")
print(f"  rf_importances.json")
