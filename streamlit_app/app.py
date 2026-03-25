"""
Kickstarter Campaign Success Predictor
A pre-launch tool for campaign creators.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kickstarter Success Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS (dark theme, clean presentation style) ────────────────────────
st.markdown(
    """
<style>
    /* Global */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Headers */
    h1 {
        color: #fafafa !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        margin-bottom: 0.25rem !important;
    }
    h2 {
        color: #c0c0c0 !important;
        font-weight: 400 !important;
        font-size: 1rem !important;
        margin-top: 0 !important;
    }
    h3 {
        color: #fafafa !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        border-bottom: 1px solid #2a2a3a;
        padding-bottom: 0.4rem;
        margin-top: 1.5rem !important;
    }

    /* Score card */
    .score-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        border: 1px solid #2a2a4a;
        margin-bottom: 1.5rem;
    }
    .score-value {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .score-label {
        font-size: 0.9rem;
        color: #aaa;
        margin-top: 0.5rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* Factor cards */
    .factor-positive {
        background: #0d2818;
        border-left: 3px solid #22c55e;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .factor-negative {
        background: #2a1215;
        border-left: 3px solid #ef4444;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .factor-title {
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
    }
    .factor-detail {
        font-size: 0.85rem;
        color: #aaa;
    }

    /* Advice cards */
    .advice-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .advice-title {
        font-weight: 600;
        color: #60a5fa;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    .advice-text {
        font-size: 0.88rem;
        color: #ccc;
        line-height: 1.5;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0a0a12;
        border-right: 1px solid #1a1a2a;
    }
    section[data-testid="stSidebar"] h1 {
        font-size: 1.4rem !important;
    }

    /* Progress bar custom */
    .prob-bar-container {
        background: #1a1a2e;
        border-radius: 8px;
        overflow: hidden;
        height: 28px;
        margin: 0.75rem 0;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 8px;
        display: flex;
        align-items: center;
        padding-left: 12px;
        font-weight: 700;
        font-size: 0.85rem;
        color: #fff;
        transition: width 0.5s ease;
    }

    /* Divider */
    hr {
        border-color: #2a2a3a !important;
    }

    /* Input styling */
    .stSelectbox label, .stNumberInput label, .stCheckbox label,
    .stTextArea label, .stTextInput label, .stSlider label {
        color: #ccc !important;
        font-size: 0.9rem !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── load artifacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    art_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    model = joblib.load(os.path.join(art_dir, "rf_model.joblib"))
    with open(os.path.join(art_dir, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(art_dir, "tfidf_terms.json")) as f:
        tfidf_terms = json.load(f)
    with open(os.path.join(art_dir, "lookups.json")) as f:
        lookups = json.load(f)
    with open(os.path.join(art_dir, "rf_importances.json")) as f:
        importances = json.load(f)
    return model, feature_cols, tfidf_terms, lookups, importances


try:
    model, FEATURE_COLS, tfidf_terms, lookups, importances = load_model()
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Run `python train_model.py` first to train the model."
    )
    st.stop()

# Unpack lookups
cat_name_encoder = lookups["cat_name_encoder"]
cat_parent_encoder = lookups["cat_parent_encoder"]
cat_medians = lookups["cat_medians"]
cat_success_medians = lookups["cat_success_medians"]
keep_countries = lookups["keep_countries"]
cat_to_parent = lookups["cat_to_parent"]
global_median_goal = lookups["global_median_goal"]
prep_99 = lookups["prep_days_99pct"]
goal_99 = lookups["goal_99pct"]

COUNTRY_NAMES = {
    "US": "United States",
    "GB": "United Kingdom",
    "CA": "Canada",
    "AU": "Australia",
    "DE": "Germany",
    "FR": "France",
    "IT": "Italy",
    "ES": "Spain",
    "MX": "Mexico",
    "NL": "Netherlands",
    "SE": "Sweden",
    "HK": "Hong Kong",
    "DK": "Denmark",
}

# Build sorted category lists
all_categories = sorted([c for c in cat_name_encoder if c != "__global_mean__"])
all_parents = sorted([c for c in cat_parent_encoder if c != "__global_mean__"])

# ── sidebar: campaign inputs ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# Campaign Details")
    st.markdown(
        '<p style="color:#888;font-size:0.85rem;margin-top:-0.5rem;">Fill in your pre-launch campaign info</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── things the creator CAN change ──
    st.markdown(
        '<p style="color:#60a5fa;font-weight:600;font-size:0.85rem;letter-spacing:1px;">CAMPAIGN SETUP</p>',
        unsafe_allow_html=True,
    )

    goal = st.number_input(
        "Funding goal (USD)",
        min_value=1,
        max_value=10_000_000,
        value=5000,
        step=500,
        help="How much money you want to raise",
    )

    duration = st.slider(
        "Campaign duration (days)",
        min_value=1,
        max_value=60,
        value=30,
        help="How long your campaign will run",
    )

    has_video = st.checkbox("Campaign has a video", value=True)

    st.markdown(
        '<p style="color:#60a5fa;font-weight:600;font-size:0.85rem;letter-spacing:1px;margin-top:1rem;">CONTENT</p>',
        unsafe_allow_html=True,
    )

    campaign_name = st.text_input(
        "Campaign title",
        value="",
        placeholder="e.g. Handcrafted Ceramic Mugs",
    )

    blurb = st.text_area(
        "Campaign blurb",
        value="",
        placeholder="Short description of your campaign (max ~135 characters)",
        height=80,
    )

    st.markdown(
        '<p style="color:#60a5fa;font-weight:600;font-size:0.85rem;letter-spacing:1px;margin-top:1rem;">TIMING</p>',
        unsafe_allow_html=True,
    )

    MONTHS = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    launch_month = st.selectbox("Launch month", MONTHS, index=2)
    launch_month_num = MONTHS.index(launch_month) + 1

    DAYS = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]
    launch_day = st.selectbox("Launch day of week", DAYS, index=1)
    launch_day_num = DAYS.index(launch_day)

    prep_days = st.number_input(
        "Days spent preparing before launch",
        min_value=0,
        max_value=365,
        value=14,
        help="Time between creating the project and launching it",
    )

    st.markdown("---")

    # ── things the creator CANNOT change ──
    st.markdown(
        '<p style="color:#888;font-weight:600;font-size:0.85rem;letter-spacing:1px;">CATEGORY</p>',
        unsafe_allow_html=True,
    )

    parent_cat = st.selectbox("Parent category", all_parents, index=all_parents.index("Technology") if "Technology" in all_parents else 0)

    # Filter subcategories by parent
    sub_cats_for_parent = sorted(
        [c for c, p in cat_to_parent.items() if p == parent_cat]
    )
    if not sub_cats_for_parent:
        sub_cats_for_parent = all_categories

    sub_cat = st.selectbox("Subcategory", sub_cats_for_parent)

    country_display = sorted(COUNTRY_NAMES.values()) + ["Other"]
    display_to_code = {v: k for k, v in COUNTRY_NAMES.items()}
    display_to_code["Other"] = "Other"
    country_sel = st.selectbox(
        "Country",
        country_display,
        index=country_display.index("United States"),
    )
    country_code = display_to_code.get(country_sel, "Other")

    st.markdown("---")
    predict_btn = st.button("Predict Success", type="primary", use_container_width=True)


# ── feature engineering for a single campaign ────────────────────────────────

def build_features(
    goal_val, duration_val, prep_val, has_vid, blurb_text, name_text,
    month, dayofweek, sub_category, parent_category, country,
):
    """Build feature vector matching the exact 129-feature pipeline."""
    row = {}

    # Numeric features
    row["log_goal"] = np.log1p(goal_val)
    row["duration_days"] = min(max(duration_val, 0), 90)
    row["prep_days"] = min(max(prep_val, 0), prep_99)
    row["has_video"] = int(has_vid)

    b = blurb_text or ""
    n = name_text or ""
    row["blurb_length"] = len(b)
    row["name_length"] = len(n)
    row["blurb_word_count"] = len(b.split()) if b.strip() else 0
    row["name_number"] = 1 if any(c.isdigit() for c in n) else 0
    row["goal_is_round"] = 1 if goal_val % 1000 == 0 else 0
    row["is_usd"] = 1 if country in ["US"] else 0  # Simplified: USD if US

    row["launched_month"] = float(month)
    row["launched_dayofweek"] = float(dayofweek)

    dur = max(duration_val, 1)
    row["goal_per_day"] = min(goal_val / dur, goal_99)

    # Goal vs category median
    cat_med = cat_medians.get(sub_category, global_median_goal)
    cat_med = max(cat_med, 1)
    row["log_goal_vs_cat_median"] = np.log1p(goal_val / cat_med)

    # Target encoding
    gm = cat_name_encoder.get("__global_mean__", 0.5)
    row["cat_name_encoded"] = cat_name_encoder.get(sub_category, gm)
    gm2 = cat_parent_encoder.get("__global_mean__", 0.5)
    row["cat_parent_encoded"] = cat_parent_encoder.get(parent_category, gm2)

    # Country OHE
    ctry = country if country in keep_countries else "Other"
    for c in keep_countries + ["Other"]:
        row[f"country_{c}"] = 1 if ctry == c else 0

    # TF-IDF binary features
    blurb_lower = b.lower()
    name_lower = n.lower()
    for term in tfidf_terms["blurb"]:
        col = "blurb_has_" + term.replace(" ", "_")
        row[col] = 1 if term in blurb_lower else 0
    for term in tfidf_terms["name"]:
        col = "name_has_" + term.replace(" ", "_")
        row[col] = 1 if term in name_lower else 0

    # Build DataFrame with correct column order
    df = pd.DataFrame([row])
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0
    return df[FEATURE_COLS].fillna(0)


# ── factor analysis ──────────────────────────────────────────────────────────

def analyze_factors(features_df, probability):
    """
    Analyze which factors are helping and hurting,
    and generate actionable advice for changeable factors.
    """
    factors = []

    # 1. GOAL AMOUNT (changeable)
    cat_med = cat_medians.get(sub_cat, global_median_goal)
    cat_succ_med = cat_success_medians.get(sub_cat, cat_med)
    goal_ratio = goal / max(cat_med, 1)

    if goal_ratio > 2.0:
        factors.append({
            "name": "Funding Goal",
            "direction": "negative",
            "importance": importances.get("log_goal", 0) + importances.get("log_goal_vs_cat_median", 0) + importances.get("goal_per_day", 0),
            "detail": f"${goal:,.0f} is {goal_ratio:.1f}x the median for {sub_cat} (${cat_med:,.0f})",
            "changeable": True,
            "advice": f"The median goal for successful {sub_cat} campaigns is ${cat_succ_med:,.0f}. "
                       f"Consider whether you can reduce your goal closer to that range, "
                       f"or whether stretch goals could work for your campaign.",
        })
    elif goal_ratio > 1.3:
        factors.append({
            "name": "Funding Goal",
            "direction": "neutral",
            "importance": importances.get("log_goal", 0),
            "detail": f"${goal:,.0f} is {goal_ratio:.1f}x the median for {sub_cat} (${cat_med:,.0f})",
            "changeable": True,
            "advice": f"Your goal is somewhat above the category median. The median for successful {sub_cat} campaigns is ${cat_succ_med:,.0f}.",
        })
    else:
        factors.append({
            "name": "Funding Goal",
            "direction": "positive",
            "importance": importances.get("log_goal", 0),
            "detail": f"${goal:,.0f} is within the typical range for {sub_cat} (median ${cat_med:,.0f})",
            "changeable": True,
        })

    # 2. VIDEO (changeable)
    if not has_video:
        factors.append({
            "name": "No Campaign Video",
            "direction": "negative",
            "importance": importances.get("has_video", 0),
            "detail": f"Campaigns with video succeed at {lookups['video_success_rate']*100:.0f}% vs {lookups['no_video_success_rate']*100:.0f}% without",
            "changeable": True,
            "advice": "Adding a video is one of the most impactful changes you can make. "
                       f"It's associated with a +{(lookups['video_success_rate'] - lookups['no_video_success_rate'])*100:.0f} percentage point lift in success rate.",
        })
    else:
        factors.append({
            "name": "Has Campaign Video",
            "direction": "positive",
            "importance": importances.get("has_video", 0),
            "detail": f"Video campaigns succeed at {lookups['video_success_rate']*100:.0f}% vs {lookups['no_video_success_rate']*100:.0f}% without",
            "changeable": True,
        })

    # 3. DURATION (changeable)
    if duration > 40:
        factors.append({
            "name": "Campaign Duration",
            "direction": "negative",
            "importance": importances.get("duration_days", 0),
            "detail": f"{duration} days is longer than typical successful campaigns",
            "changeable": True,
            "advice": "Campaigns of 30 days or fewer tend to perform better. "
                       "Longer campaigns can signal low confidence and lose momentum. "
                       "Consider 30 days as a sweet spot.",
        })
    elif duration < 15:
        factors.append({
            "name": "Campaign Duration",
            "direction": "neutral",
            "importance": importances.get("duration_days", 0),
            "detail": f"{duration} days is quite short",
            "changeable": True,
            "advice": "Very short campaigns can work but give less time to build momentum. "
                       "15-30 days is generally the optimal range.",
        })
    else:
        factors.append({
            "name": "Campaign Duration",
            "direction": "positive",
            "importance": importances.get("duration_days", 0),
            "detail": f"{duration} days is in the optimal range",
            "changeable": True,
        })

    # 4. PREPARATION TIME (changeable)
    if prep_days < 3:
        factors.append({
            "name": "Preparation Time",
            "direction": "negative",
            "importance": importances.get("prep_days", 0),
            "detail": f"{prep_days} days of preparation is very low",
            "changeable": True,
            "advice": "Preparation time is the second most predictive feature in our model. "
                       "Creators who spend more time preparing (building an audience, refining rewards, "
                       "creating media) tend to succeed more. Consider delaying launch to prepare.",
        })
    elif prep_days < 7:
        factors.append({
            "name": "Preparation Time",
            "direction": "neutral",
            "importance": importances.get("prep_days", 0),
            "detail": f"{prep_days} days of preparation",
            "changeable": True,
            "advice": "More preparation time generally correlates with higher success rates. "
                       "Consider whether there are aspects of your campaign you could refine further.",
        })
    else:
        factors.append({
            "name": "Preparation Time",
            "direction": "positive",
            "importance": importances.get("prep_days", 0),
            "detail": f"{prep_days} days of preparation",
            "changeable": True,
        })

    # 5. CATEGORY (not changeable)
    cat_enc_val = cat_name_encoder.get(sub_cat, lookups["global_mean_success"])
    global_sr = lookups["global_mean_success"]
    if cat_enc_val > global_sr + 0.1:
        factors.append({
            "name": f"Category: {sub_cat}",
            "direction": "positive",
            "importance": importances.get("cat_name_encoded", 0),
            "detail": f"This subcategory has a higher-than-average success rate ({cat_enc_val*100:.0f}%)",
            "changeable": False,
        })
    elif cat_enc_val < global_sr - 0.1:
        factors.append({
            "name": f"Category: {sub_cat}",
            "direction": "negative",
            "importance": importances.get("cat_name_encoded", 0),
            "detail": f"This subcategory has a below-average success rate ({cat_enc_val*100:.0f}%)",
            "changeable": False,
        })
    else:
        factors.append({
            "name": f"Category: {sub_cat}",
            "direction": "neutral",
            "importance": importances.get("cat_name_encoded", 0),
            "detail": f"Category success rate ({cat_enc_val*100:.0f}%) is near the average ({global_sr*100:.0f}%)",
            "changeable": False,
        })

    # 6. BLURB quality (changeable) - check text features
    blurb_lower = (blurb or "").lower()
    name_lower = (campaign_name or "").lower()
    negative_terms = []
    positive_terms = []

    for w in tfidf_terms["blurb"]:
        col = "blurb_has_" + w.replace(" ", "_")
        imp = importances.get(col, 0)
        if w in blurb_lower and imp > 0.0005:
            # Check if this term is associated with success or failure from the word list
            positive_terms.append((w, "blurb", imp))

    for w in tfidf_terms["name"]:
        col = "name_has_" + w.replace(" ", "_")
        imp = importances.get(col, 0)
        if w in name_lower and imp > 0.0005:
            positive_terms.append((w, "name", imp))

    # Blurb length check
    blurb_len = len(blurb or "")
    if blurb_len < 30 and blurb_len > 0:
        factors.append({
            "name": "Blurb Length",
            "direction": "negative",
            "importance": importances.get("blurb_length", 0),
            "detail": f"Your blurb is only {blurb_len} characters",
            "changeable": True,
            "advice": "A more descriptive blurb helps both the algorithm and potential backers understand your project. "
                       "Aim for at least 80-130 characters.",
        })
    elif blurb_len == 0:
        factors.append({
            "name": "No Blurb",
            "direction": "negative",
            "importance": importances.get("blurb_length", 0) + importances.get("blurb_word_count", 0),
            "detail": "No campaign description provided",
            "changeable": True,
            "advice": "Write a compelling short description. This affects both text-based features "
                       "and how backers discover your campaign.",
        })

    # Sort by importance
    factors.sort(key=lambda x: x["importance"], reverse=True)
    return factors


# ── main content ─────────────────────────────────────────────────────────────

# Header
st.markdown("# Kickstarter Success Predictor")
st.markdown("## Pre-launch prediction based on 188K historical campaigns")

if not predict_btn:
    # Landing state
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div style="background:#1a1a2e;border-radius:10px;padding:1.5rem;border:1px solid #2a2a4a;text-align:center;">
                <p style="font-size:2rem;margin:0;">1</p>
                <p style="color:#60a5fa;font-weight:600;margin:0.5rem 0 0.3rem;">Enter Details</p>
                <p style="color:#888;font-size:0.85rem;margin:0;">Fill in your campaign details in the sidebar</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div style="background:#1a1a2e;border-radius:10px;padding:1.5rem;border:1px solid #2a2a4a;text-align:center;">
                <p style="font-size:2rem;margin:0;">2</p>
                <p style="color:#60a5fa;font-weight:600;margin:0.5rem 0 0.3rem;">Get Prediction</p>
                <p style="color:#888;font-size:0.85rem;margin:0;">See your predicted success probability</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div style="background:#1a1a2e;border-radius:10px;padding:1.5rem;border:1px solid #2a2a4a;text-align:center;">
                <p style="font-size:2rem;margin:0;">3</p>
                <p style="color:#60a5fa;font-weight:600;margin:0.5rem 0 0.3rem;">Improve</p>
                <p style="color:#888;font-size:0.85rem;margin:0;">Get actionable advice to boost your chances</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<p style="color:#666;font-size:0.8rem;text-align:center;">'
        "Model: Random Forest (300 trees) trained on 102K campaigns | "
        "Validation ROC-AUC: 0.787 | "
        "Features: 129 pre-launch signals"
        "</p>",
        unsafe_allow_html=True,
    )
    st.stop()


# ── prediction ───────────────────────────────────────────────────────────────

features_df = build_features(
    goal_val=goal,
    duration_val=duration,
    prep_val=prep_days,
    has_vid=has_video,
    blurb_text=blurb,
    name_text=campaign_name,
    month=launch_month_num,
    dayofweek=launch_day_num,
    sub_category=sub_cat,
    parent_category=parent_cat,
    country=country_code,
)

probability = model.predict_proba(features_df)[:, 1][0]
factors = analyze_factors(features_df, probability)

# ── display results ──────────────────────────────────────────────────────────

# Score card
if probability >= 0.65:
    score_color = "#22c55e"
    score_bg = "linear-gradient(135deg, #0d2818 0%, #0a3520 100%)"
    score_border = "#22c55e"
    verdict = "STRONG CHANCE"
elif probability >= 0.45:
    score_color = "#f59e0b"
    score_bg = "linear-gradient(135deg, #2a1f0a 0%, #3a2a0a 100%)"
    score_border = "#f59e0b"
    verdict = "MODERATE CHANCE"
else:
    score_color = "#ef4444"
    score_bg = "linear-gradient(135deg, #2a1215 0%, #3a1520 100%)"
    score_border = "#ef4444"
    verdict = "NEEDS WORK"

st.markdown(
    f"""
    <div style="background:{score_bg};border-radius:12px;padding:2rem;text-align:center;
                border:1px solid {score_border};margin-bottom:1.5rem;">
        <div style="font-size:4rem;font-weight:800;color:{score_color};line-height:1.1;">
            {probability*100:.0f}%
        </div>
        <div style="font-size:0.9rem;color:#aaa;margin-top:0.5rem;letter-spacing:2px;">
            PREDICTED SUCCESS PROBABILITY
        </div>
        <div style="font-size:1rem;color:{score_color};margin-top:0.25rem;font-weight:600;">
            {verdict}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Probability bar
st.markdown(
    f"""
    <div class="prob-bar-container">
        <div class="prob-bar-fill" style="width:{probability*100:.0f}%;background:{score_color};">
            {probability*100:.0f}%
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Two columns: factors + advice ────────────────────────────────────────────

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### Key Factors")

    positive_factors = [f for f in factors if f["direction"] == "positive"]
    negative_factors = [f for f in factors if f["direction"] == "negative"]
    neutral_factors = [f for f in factors if f["direction"] == "neutral"]

    if positive_factors:
        st.markdown(
            '<p style="color:#22c55e;font-weight:600;font-size:0.8rem;letter-spacing:1px;margin-bottom:0.5rem;">WORKING IN YOUR FAVOUR</p>',
            unsafe_allow_html=True,
        )
        for f in positive_factors:
            st.markdown(
                f"""<div class="factor-positive">
                    <div class="factor-title">{f["name"]}</div>
                    <div class="factor-detail">{f.get("detail", "")}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    if negative_factors:
        st.markdown(
            '<p style="color:#ef4444;font-weight:600;font-size:0.8rem;letter-spacing:1px;margin-top:1rem;margin-bottom:0.5rem;">WORKING AGAINST YOU</p>',
            unsafe_allow_html=True,
        )
        for f in negative_factors:
            st.markdown(
                f"""<div class="factor-negative">
                    <div class="factor-title">{f["name"]}</div>
                    <div class="factor-detail">{f.get("detail", "")}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    if neutral_factors:
        st.markdown(
            '<p style="color:#f59e0b;font-weight:600;font-size:0.8rem;letter-spacing:1px;margin-top:1rem;margin-bottom:0.5rem;">COULD BE IMPROVED</p>',
            unsafe_allow_html=True,
        )
        for f in neutral_factors:
            st.markdown(
                f"""<div style="background:#1a1a0a;border-left:3px solid #f59e0b;border-radius:6px;padding:0.75rem 1rem;margin-bottom:0.5rem;">
                    <div class="factor-title">{f["name"]}</div>
                    <div class="factor-detail">{f.get("detail", "")}</div>
                </div>""",
                unsafe_allow_html=True,
            )

with col_right:
    st.markdown("### What You Can Improve")

    advice_factors = [
        f for f in factors
        if f.get("changeable") and f.get("advice") and f["direction"] in ("negative", "neutral")
    ]

    if advice_factors:
        for f in advice_factors:
            st.markdown(
                f"""<div class="advice-card">
                    <div class="advice-title">{f["name"]}</div>
                    <div class="advice-text">{f["advice"]}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """<div class="advice-card">
                <div class="advice-title">Looking good</div>
                <div class="advice-text">
                    No major red flags in the factors you can control.
                    Your campaign setup aligns well with historically successful campaigns.
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="color:#555;font-size:0.75rem;text-align:center;">'
    "AI II Final Project | Group 13 | ESADE MSc Business Analytics 2026<br>"
    "Model: Random Forest (n=300, depth=20) | 129 pre-launch features | "
    "Validation ROC-AUC: 0.787 | Trained on 102,887 campaigns (2009-2014)"
    "</p>",
    unsafe_allow_html=True,
)
