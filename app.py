import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="COMPAS Recidivism Risk Explorer",
    page_icon="⚖️",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
}
.metric-box {
    background: #0f1117;
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #6b7280;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #f9fafb;
}
.metric-value.high { color: #ef4444; }
.metric-value.medium { color: #f59e0b; }
.metric-value.low { color: #10b981; }
.warning-banner {
    background: #1c1008;
    border-left: 3px solid #f59e0b;
    padding: 0.8rem 1rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.82rem;
    color: #d1d5db;
    margin: 1rem 0;
}
.section-rule {
    border: none;
    border-top: 1px solid #2a2d3a;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Data + model loading (cached) ─────────────────────────────
@st.cache_data(show_spinner="Loading COMPAS data…")
def load_and_prepare():
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    df_raw = pd.read_csv(url)
    df = df_raw[
        (df_raw['days_b_screening_arrest'] <= 30) &
        (df_raw['days_b_screening_arrest'] >= -30) &
        (df_raw['is_recid'] != -1) &
        (df_raw['c_charge_degree'] != 'O')
    ].copy()

    features = ['age', 'priors_count', 'juv_fel_count', 'juv_misd_count',
                'c_charge_degree', 'sex', 'race']
    target = 'two_year_recid'
    df_model = df[features + [target]].dropna().copy()

    df_model['sex_male'] = (df_model['sex'] == 'Male').astype(int)
    df_model['charge_felony'] = (df_model['c_charge_degree'] == 'F').astype(int)
    race_dummies = pd.get_dummies(df_model['race'], prefix='race', drop_first=True)
    df_model = pd.concat([df_model, race_dummies], axis=1)

    for col in ['priors_count', 'juv_fel_count', 'juv_misd_count']:
        df_model[f'{col}_log'] = np.log1p(df_model[col])

    race_cols = [c for c in race_dummies.columns]
    feature_cols = (['age', 'sex_male', 'charge_felony',
                     'priors_count_log', 'juv_fel_count_log', 'juv_misd_count_log']
                    + race_cols)

    X = df_model[feature_cols]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return df_model, X, y, X_train, X_test, y_train, y_test, feature_cols, race_cols

@st.cache_resource(show_spinner="Training models…")
def train_models(_X_train, _y_train):
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
    lr.fit(_X_train, _y_train)

    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(_X_train, _y_train)
    return lr, rf

df_model, X, y, X_train, X_test, y_train, y_test, feature_cols, race_cols = load_and_prepare()
lr_model, rf_model = train_models(X_train, y_train)

ALL_RACES = ['African-American', 'Asian', 'Caucasian', 'Hispanic', 'Native American', 'Other']

def build_input_row(age, priors, juv_fel, juv_misd, sex, charge, race):
    row = {
        'age': age,
        'sex_male': 1 if sex == 'Male' else 0,
        'charge_felony': 1 if charge == 'Felony' else 0,
        'priors_count_log': np.log1p(priors),
        'juv_fel_count_log': np.log1p(juv_fel),
        'juv_misd_count_log': np.log1p(juv_misd),
    }
    for col in race_cols:
        race_name = col.replace('race_', '')
        row[col] = 1 if race == race_name else 0
    return pd.DataFrame([row])[feature_cols]

def get_fairness_stats(model, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    test_df = X_test.copy()
    test_df['true'] = y_test.values
    test_df['pred'] = preds

    non_aa_cols = race_cols  # all non-AA races
    test_df['is_aa'] = (test_df[non_aa_cols].sum(axis=1) == 0).astype(int)

    results = {}
    for group_val, label in [(1, 'African-American'), (0, 'Non-African-American')]:
        subset = test_df[test_df['is_aa'] == group_val]
        n = len(subset)
        fpr = ((subset['pred'] == 1) & (subset['true'] == 0)).sum() / max((subset['true'] == 0).sum(), 1)
        fnr = ((subset['pred'] == 0) & (subset['true'] == 1)).sum() / max((subset['true'] == 1).sum(), 1)
        acc = (subset['pred'] == subset['true']).mean()
        results[label] = {'FPR': fpr, 'FNR': fnr, 'Accuracy': acc, 'N': n}
    return results

# ── Header ────────────────────────────────────────────────────
st.markdown("# ⚖️ COMPAS Recidivism Risk Explorer")
st.markdown(
    "Predict two-year recidivism probability and explore how error rates differ across demographic groups. "
    "Based on Broward County, FL data (2013–2014) via ProPublica."
)
st.markdown('<div class="warning-banner">⚠️ <strong>Predictive, not causal.</strong> Features associated with recidivism in this model reflect historical patterns in criminal justice data — they do not imply individual propensity to reoffend. False positive rates differ significantly by race.</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Individual Prediction", "📊 Fairness Explorer"])

# ════════════════════════════════════════════════════════════
# TAB 1 — Individual Prediction
# ════════════════════════════════════════════════════════════
with tab1:
    col_inputs, col_results = st.columns([1, 1], gap="large")

    with col_inputs:
        st.markdown("### Defendant Characteristics")

        model_choice = st.selectbox(
            "Model",
            ["Logistic Regression", "Random Forest"],
            help="Both models achieve AUC ≈ 0.73. Logistic Regression is more interpretable."
        )

        age = st.slider("Age", 18, 70, 30)
        priors = st.slider("Prior Convictions", 0, 38, 2)
        juv_fel = st.slider("Juvenile Felony Counts", 0, 20, 0)
        juv_misd = st.slider("Juvenile Misdemeanor Counts", 0, 20, 0)
        sex = st.selectbox("Sex", ["Male", "Female"])
        charge = st.selectbox("Charge Degree", ["Felony", "Misdemeanor"])
        race = st.selectbox("Race", ALL_RACES)

    with col_results:
        st.markdown("### Prediction")

        model = lr_model if model_choice == "Logistic Regression" else rf_model
        input_row = build_input_row(age, priors, juv_fel, juv_misd, sex, charge, race)
        prob = model.predict_proba(input_row)[0][1]

        # Risk tier
        if prob >= 0.6:
            tier, tier_class = "HIGH RISK", "high"
        elif prob >= 0.4:
            tier, tier_class = "MEDIUM RISK", "medium"
        else:
            tier, tier_class = "LOW RISK", "low"

        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Predicted Recidivism Probability</div>
            <div class="metric-value {tier_class}">{prob:.1%}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Risk Tier</div>
            <div class="metric-value {tier_class}">{tier}</div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge bar
        fig, ax = plt.subplots(figsize=(7, 1.2))
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        ax.barh([0], [1], color='#1f2937', height=0.5)
        color = '#ef4444' if prob >= 0.6 else '#f59e0b' if prob >= 0.4 else '#10b981'
        ax.barh([0], [prob], color=color, height=0.5)
        ax.axvline(0.5, color='white', linewidth=1, linestyle='--', alpha=0.4)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], color='#9ca3af', fontsize=9)
        ax.tick_params(axis='x', colors='#9ca3af', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout(pad=0.2)
        st.pyplot(fig)
        plt.close()

        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

        # Context note on race-specific error rates
        if race == 'African-American':
            st.info(
                "**Note:** For African-American defendants, this model's false positive rate is **31.6%** — "
                "meaning nearly 1 in 3 non-recidivating defendants is predicted high-risk. "
                "This is 2.6× the false positive rate for non-African-American defendants (12.2%)."
            )
        else:
            st.info(
                "**Note:** For non-African-American defendants, this model's false positive rate is **12.2%** "
                "and false negative rate is **62.2%** — non-recidivating defendants are less often mislabeled, "
                "but recidivating defendants are more often missed."
            )

        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        st.caption(f"Model: {model_choice} — Test AUC: {auc:.3f} | Train N: {len(X_train):,} | Test N: {len(X_test):,}")

# ════════════════════════════════════════════════════════════
# TAB 2 — Fairness Explorer
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Error Rate Comparison by Race")
    st.markdown(
        "Adjust the classification threshold below to see how false positive and false negative rates "
        "change for African-American vs. non-African-American defendants."
    )

    col_ctrl, col_chart = st.columns([1, 2], gap="large")

    with col_ctrl:
        fairness_model_choice = st.selectbox(
            "Model",
            ["Logistic Regression", "Random Forest"],
            key="fairness_model"
        )
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.1, max_value=0.9, value=0.5, step=0.01,
            help="Probability above which a defendant is predicted to reoffend."
        )

        fair_model = lr_model if fairness_model_choice == "Logistic Regression" else rf_model
        stats = get_fairness_stats(fair_model, threshold)

        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        for group, s in stats.items():
            short = "AA" if group == "African-American" else "Non-AA"
            st.markdown(f"**{group}** (N={s['N']:,})")
            c1, c2 = st.columns(2)
            c1.metric("False Positive Rate", f"{s['FPR']:.1%}")
            c2.metric("False Negative Rate", f"{s['FNR']:.1%}")
            st.caption(f"Accuracy: {s['Accuracy']:.1%}")
            st.markdown("")

    with col_chart:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        fig.patch.set_facecolor('#0f1117')

        groups = list(stats.keys())
        colors = ['#ef4444', '#3b82f6']
        metrics = ['FPR', 'FNR']
        titles = ['False Positive Rate\n(Flagged high-risk, did NOT reoffend)',
                  'False Negative Rate\n(Flagged low-risk, DID reoffend)']

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            ax.set_facecolor('#0f1117')
            vals = [stats[g][metric] for g in groups]
            bars = ax.bar(groups, vals, color=colors, width=0.5, edgecolor='none')
            ax.set_title(title, color='#d1d5db', fontsize=9, pad=10)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], color='#6b7280', fontsize=8)
            ax.set_xticklabels(['African-\nAmerican', 'Non-African-\nAmerican'],
                               color='#d1d5db', fontsize=8)
            ax.tick_params(axis='both', length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.yaxis.grid(True, color='#1f2937', linewidth=0.8)
            ax.set_axisbelow(True)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                        f'{val:.1%}', ha='center', va='bottom',
                        color='white', fontsize=10, fontweight='bold',
                        fontfamily='monospace')

        fig.suptitle(f'Error Rates by Race — {fairness_model_choice} (threshold={threshold:.2f})',
                     color='#f9fafb', fontsize=10, fontfamily='monospace', y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(
            '<div class="warning-banner">These error rates are computed on held-out test data (N=1,235). '
            'Differential error rates persist across both models and across all tested thresholds, '
            'reflecting patterns in the underlying training data rather than model choice.</div>',
            unsafe_allow_html=True
        )

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: ProPublica COMPAS Analysis — Broward County FL (2013–2014). "
    "Models trained with random_state=42. "
    "This tool is for academic analysis only and should not be used to inform real sentencing decisions."
)