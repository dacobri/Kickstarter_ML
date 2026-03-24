# Kickstarter Campaign Success Prediction

Predicting whether a Kickstarter crowdfunding campaign will reach its funding goal using **only pre-launch information** — a binary classification project built with scikit-learn and XGBoost on 160,000+ real campaigns.

> **Course:** Artificial Intelligence II — ESADE Business School (MSc Business Analytics)
> **Date:** March 2026

---

## Authors

| Name | Role |
|---|---|
| **Brice Da Costa** | Data cleaning, EDA, feature engineering, modelling |
| **Baran Erdogan** | Data cleaning, EDA, feature engineering |
| **Mats Hoffmann** | Critical reflection, fairness analysis, business recommendations |
| **Maria Angelica Mora Zamora** | Report writing, project documentation |
| **Hiroaki Nakano** | Data cleaning, feature engineering |

---

## Overview

Kickstarter uses an all-or-nothing funding model: if a campaign doesn't reach its goal by the deadline, all pledges are returned. Around 37% of campaigns fail. This project asks: **can we predict failure before launch, giving creators a chance to adjust?**

We trained six classifiers on temporal splits of the data (no future leakage) and evaluated them on a held-out test set spanning Sep 2024 – Feb 2026. The best validation AUC is **0.787** (Random Forest), but under distributional shift, the simpler **Logistic Regression generalises best** to the test set (AUC 0.708) — a textbook illustration of the bias-variance tradeoff.

### Key Results

| Model | Val AUC | Test AUC | Val→Test Drop |
|---|---|---|---|
| Random Forest | 0.7868 | 0.6974 | 0.0894 |
| Logistic Regression | 0.7792 | 0.7078 | 0.0714 |
| XGBoost | 0.7760 | 0.6730 | 0.1030 |
| MLP | 0.7753 | 0.6820 | 0.0933 |
| Gradient Boosting | 0.7705 | 0.6629 | 0.1076 |
| Decision Tree | 0.7445 | 0.6390 | 0.1055 |

---

## Project Structure

```
Kickstarter_ML/
├── notebooks/
│   ├── 01_data_cleaning.ipynb          # Load 85 CSV shards, deduplicate, parse JSON, temporal split
│   ├── 02_eda.ipynb                    # EDA, leakage detection (spotlight r=1.0), staff_pick removal
│   ├── 03a_tfidf.ipynb                 # TF-IDF binary features from blurb + name (z-test, p<0.01)
│   ├── 03b_feature_engineering.ipynb   # Target encoding, OHE, 30 structured + 99 text → 129 features
│   ├── 04_modelling.ipynb              # GridSearchCV + TimeSeriesSplit(5) for 6 models, test eval
│   ├── 05_reflection.ipynb             # Construct gap, distributional shift, fairness, Goodhart's Law
│   │
│   └── data/                           # Intermediate outputs (included for reproducibility)
│       ├── *.parquet                   # Cleaned datasets, train/val/test splits, feature matrices
│       ├── *.json                      # Feature lists, TF-IDF terms, leakage columns
│       ├── figures/                    # All saved visualisations (15 PNG files)
│       └── results/                    # Model metrics, feature importances
│
├── data/
│   └── Kickstarter_2026-02-12T03_20_22_018Z/   # 85 raw CSV shards (~1.6 GB, included in repo)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| Source | [WebRobots.io](https://webrobots.io/kickstarter-datasets/) Kickstarter bulk export |
| Snapshot | 2026-02-12 |
| Raw files | 85 CSV shards (~1.6 GB) |
| Unique campaigns | ~209,000 after deduplication |
| Binary subset | ~160,000 (successful + failed only) |
| Time range | Dec 2014 – Feb 2026 |

The raw CSV files (85 shards, ~1.6 GB) are included in this repository under `data/`. They were sourced from [WebRobots.io](https://webrobots.io/kickstarter-datasets/) (snapshot: `Kickstarter_2026-02-12T03_20_22_018Z`).

The intermediate parquet files are also included in `notebooks/data/`, so **notebooks 02–05 can be run without re-executing Notebook 01**.

---

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

Python 3.9+ required.

### Running the Pipeline

Run notebooks in order — each reads outputs from the previous one:

```
01_data_cleaning → 02_eda → 03a_tfidf → 03b_feature_engineering → 04_modelling → 05_reflection
```

All paths are relative. Open notebooks from the `notebooks/` directory.

**Quick start (no raw data needed):** Open `04_modelling.ipynb` or `05_reflection.ipynb` directly — the pre-computed feature matrices and results are included.

---

## Methodology

### Temporal Split

| Split | Period | Campaigns | Success Rate |
|---|---|---|---|
| Train | Dec 2014 – Jan 2023 | 102,887 | 56.9% |
| Validation | Jan 2023 – Sep 2024 | 25,721 | 65.2% |
| Test | Sep 2024 – Feb 2026 | 32,153 | 69.8% |

All splits are strictly chronological — no random shuffling — to prevent temporal leakage and simulate real deployment conditions.

### Features (129 total)

| Group | Examples | Count |
|---|---|---|
| Temporal | `prep_days`, `duration_days`, `launched_month`, `launched_dayofweek` | 4 |
| Financial | `log_goal`, `goal_per_day`, `log_goal_vs_cat_median` | 3 |
| Categorical | `cat_name_encoded` (target-encoded), `cat_parent_encoded`, 14 country OHE | 16 |
| Text metrics | `blurb_length`, `name_length`, `blurb_word_count` | 3 |
| Binary flags | `has_video`, `name_number`, `goal_is_round`, `is_usd` | 4 |
| TF-IDF | Binary term flags from `name` + `blurb`, z-test selected (p<0.01) | 99 |

### Leakage Prevention

- **Post-campaign columns dropped:** `pledged`, `backers_count`, `spotlight` (r=1.0 with target), `usd_pledged`, `converted_pledged_amount`, `percent_funded`, `state_changed_at`, exchange rate columns
- **`staff_pick` removed:** Encodes non-public editorial knowledge — a subtle leakage risk
- **Target encoding fitted on training data only** — prevents test-set success rates from leaking into features
- **TimeSeriesSplit cross-validation** — validation folds always come after training folds chronologically

### Model Training

All models tuned via **GridSearchCV** with **TimeSeriesSplit(n_splits=5)** and constrained hyperparameter grids. `class_weight='balanced'` applied to all applicable classifiers to handle the shifting class imbalance across splits.

---

## Key Findings

**1. Features are the ceiling, not the algorithm.**
All six models cluster within a 4-point AUC range on validation (0.74–0.79). Pre-launch information is the binding constraint — better features (creator history, social signals) would matter more than algorithm choice.

**2. Simpler models generalise better under shift.**
Logistic Regression overtakes Random Forest on the test set (0.708 vs 0.697). The RF's nonlinear patterns from the training era don't transfer across the distributional shift.

**3. The val→test drop is platform-level, not model-specific.**
All models degrade by 0.07–0.11 AUC. The uniform drop confirms distributional shift (success rate: 57% → 70%) rather than individual model overfitting.

**4. TF-IDF features improve generalisation.**
Ablation: removing 99 text features increases the val→test gap from 0.089 to 0.101. Text features add stable signal; the degradation is driven by structural platform changes, not vocabulary drift.

**5. The model predicts execution quality, not idea quality.**
A well-marketed mediocre idea can score higher than a brilliant idea with a badly-set goal. This construct gap is inherent and must be communicated to any user of the model.

---

## Critical Reflection

The reflection notebook (05) covers topics that go beyond standard ML evaluation:

- **Construct validity** — distinguishing what we measure (funding success) from what we care about (creative value)
- **Distributional shift** — quantifying the 13-point rise in success rate and its impact on model calibration
- **Goodhart's Law** — if creators optimise for the model's inputs, those inputs lose their predictive value
- **Fairness** — country/currency features are predictive but structurally biased against non-US creators
- **Deployment recommendations** — relative ranking is more robust than absolute probability estimation

---

## License

This project was developed for academic purposes at ESADE Business School. The Kickstarter data is sourced from [WebRobots.io](https://webrobots.io/kickstarter-datasets/) and is subject to their terms of use.

---

*ESADE Business School — MSc Business Analytics — AI II Final Project — March 2026*
