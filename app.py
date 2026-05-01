import streamlit as st
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt
from src.scorecard import compute_scaling_params, probability_to_score

st.set_page_config(
    page_title="Credit Risk Scorecard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main { background-color: #0a0f1e; color: #e0e6f0; }
    .score-display {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 5rem; font-weight: 600;
        text-align: center; padding: 1rem;
        border-radius: 12px; margin: 1rem 0;
    }
    .low-risk { color: #00c896; background: rgba(0,200,150,0.1); border: 1px solid #00c896; }
    .medium-risk { color: #f5a623; background: rgba(245,166,35,0.1); border: 1px solid #f5a623; }
    .high-risk { color: #ff4757; background: rgba(255,71,87,0.1); border: 1px solid #ff4757; }
    .metric-card {
        background: #111827; border: 1px solid #1f2937;
        border-radius: 8px; padding: 1rem;
        text-align: center; margin: 0.5rem 0;
    }
    .metric-label { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.1em; }
    .metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 600; color: #e0e6f0; }
    div[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #e0e6f0; }
    .risk-badge {
        display: inline-block; padding: 0.4rem 1.2rem;
        border-radius: 20px; font-weight: 600;
        font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em;
    }
    .glossary-card {
        background: #111827; border: 1px solid #1f2937;
        border-radius: 8px; padding: 1.2rem; margin: 0.8rem 0;
    }
    .glossary-term { font-family: 'IBM Plex Mono', monospace; color: #00c896; font-size: 1rem; font-weight: 600; }
    .glossary-def { color: #9ca3af; font-size: 0.9rem; margin-top: 0.4rem; line-height: 1.6; }
    .glossary-formula { font-family: 'IBM Plex Mono', monospace; color: #f5a623; font-size: 0.85rem; margin-top: 0.4rem; }
</style>
""", unsafe_allow_html=True)

# --- Load Models ------------------------------------------------------------
@st.cache_resource
def load_models():
    with open("data/processed/logit_model.pkl", "rb") as f:
        logit_model = pickle.load(f)
    with open("data/processed/binning_models.pkl", "rb") as f:
        binning_models = pickle.load(f)
    return logit_model, binning_models

@st.cache_data
def load_reference_data():
    decile_table = pd.read_csv("data/processed/decile_table.csv")
    validation_metrics = pd.read_csv("data/processed/validation_metrics.csv")
    train_scores = np.load("data/processed/train_scores.npy")
    test_scores = np.load("data/processed/test_scores.npy")
    return decile_table, validation_metrics, train_scores, test_scores

logit_model, binning_models = load_models()
decile_table, validation_metrics, train_scores, test_scores = load_reference_data()

A, B = compute_scaling_params(PDO=20, anchor_score=600, anchor_odds=10)
FEATURE_ORDER = [
    'sub_grade', 'term', 'verification_status', 'home_ownership', 'purpose',
    'fico_range_low', 'acc_open_past_24mths', 'dti', 'num_tl_op_past_12m',
    'bc_open_to_buy', 'avg_cur_bal', 'mo_sin_rcnt_tl', 'total_bc_limit',
    'loan_amnt', 'tot_hi_cred_lim', 'tot_cur_bal', 'annual_inc',
    'mths_since_recent_inq', 'mo_sin_rcnt_rev_tl_op', 'percent_bc_gt_75',
    'num_actv_rev_tl', 'num_rev_tl_bal_gt_0', 'inq_last_6mths',
    'mths_since_recent_bc', 'mort_acc', 'revol_util',
    'mo_sin_old_rev_tl_op', 'credit_history_months'
]

# --- Load training data for imputation --------------------------------------
train_selected = pd.read_csv("data/processed/train_selected.csv").drop(columns=["target"])
categorical_cols_app = ["sub_grade", "term", "verification_status", "home_ownership", "purpose"]
numeric_cols_app = [col for col in train_selected.columns if col not in categorical_cols_app]
SUBGRADE_MEDIANS = train_selected.groupby("sub_grade")[numeric_cols_app].median()
POPULATION_MEDIANS = train_selected[numeric_cols_app].median()
TRAIN_MODES = train_selected[categorical_cols_app].mode().iloc[0]

# --- Initialize session state ONCE on first load ----------------------------
# Use a single init flag — never overwrite after first load
if "initialized" not in st.session_state:
    st.session_state["initialized"] = True
    st.session_state["inp_sub_grade"] = "A1"
    st.session_state["inp_term"] = " 36 months"
    st.session_state["inp_loan_amnt"] = 10000
    st.session_state["inp_annual_inc"] = 65000
    st.session_state["inp_dti"] = 15.0
    st.session_state["inp_fico"] = 700
    st.session_state["inp_home"] = "RENT"
    st.session_state["inp_verif"] = "Not Verified"
    st.session_state["inp_purpose"] = "debt_consolidation"
    st.session_state["inp_revol"] = 50.0
    st.session_state["inp_inq"] = 1

# --- Navigation -------------------------------------------------------------
page = st.sidebar.selectbox(
    "NAVIGATION",
    ["🏦 Score Borrower", "📊 Score Breakdown", "📈 Model Performance", "📖 Glossary"]
)

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — SCORE BORROWER
# ═══════════════════════════════════════════════════════════════
if page == "🏦 Score Borrower":
    st.title("🏦 Credit Risk Scorecard")
    st.caption("LendingClub PD Model — Logistic Regression + WoE Features")

    col_form, col_score = st.columns([1, 1])

    with col_form:
        st.subheader("Borrower Application")

        # All widgets use value= from session state, NO key= parameter
        # This decouples widget rendering from session state management
        sub_grade_options = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
        sub_grade = st.selectbox(
            "Sub Grade",
            options=sub_grade_options,
            index=sub_grade_options.index(st.session_state["inp_sub_grade"]),
            help="LendingClub's internal risk grade from A1 (safest) to G5 (riskiest). Derived from FICO score, loan amount, and credit history. This is the strongest predictor in the model (IV = 0.51). See Glossary."
        )

        term_options = [" 36 months", " 60 months"]
        term = st.radio(
            "Loan Term",
            options=term_options,
            index=term_options.index(st.session_state["inp_term"]),
            help="Duration of the loan. 60-month loans have substantially higher default rates than 36-month loans (31.9% vs 13.9%). Second strongest predictor (IV = 0.24) — switching terms moves the score by ~15–20 points."
        )

        loan_amnt = st.slider(
            "Loan Amount ($)", 1000, 40000,
            value=st.session_state["inp_loan_amnt"],
            step=500,
            help="Total loan amount requested by the borrower in USD."
        )

        annual_inc = st.number_input(
            "Annual Income ($)",
            min_value=0, max_value=500000,
            value=st.session_state["inp_annual_inc"],
            step=1000,
            help="Self-reported annual income of the borrower. Higher income generally indicates stronger repayment capacity."
        )

        dti = st.slider(
            "Debt-to-Income Ratio (%)", 0.0, 50.0,
            value=float(st.session_state["inp_dti"]),
            step=0.5,
            help="Total monthly debt payments divided by gross monthly income. Higher DTI = more income committed to existing debt = higher default risk. See Glossary."
        )

        fico_range_low = st.slider(
            "FICO Score", 580, 850,
            value=st.session_state["inp_fico"],
            step=5,
            help="Lower bound of the borrower's FICO credit score range at origination. Range: 300 (worst) to 850 (best). See Glossary."
        )

        home_options = ["ANY", "MORTGAGE", "NONE", "OTHER", "OWN", "RENT"]
        home_ownership = st.selectbox(
            "Home Ownership",
            options=home_options,
            index=home_options.index(st.session_state["inp_home"]),
            help="The borrower's home ownership status. MORTGAGE and OWN typically indicate greater financial stability than RENT."
        )

        verif_options = ["Not Verified", "Source Verified", "Verified"]
        verification_status = st.selectbox(
            "Verification Status",
            options=verif_options,
            index=verif_options.index(st.session_state["inp_verif"]),
            help="Whether LendingClub verified the borrower's income. Counterintuitively, verified borrowers sometimes show higher default rates — those with weaker profiles tend to trigger verification."
        )

        purpose_options = [
            "car", "credit_card", "debt_consolidation", "educational",
            "home_improvement", "house", "major_purchase", "medical",
            "moving", "other", "renewable_energy", "small_business",
            "vacation", "wedding"
        ]
        purpose = st.selectbox(
            "Loan Purpose",
            options=purpose_options,
            index=purpose_options.index(st.session_state["inp_purpose"]),
            help="Stated purpose of the loan. Debt consolidation is most common. Small business loans tend to have higher default rates."
        )

        revol_util = st.slider(
            "Revolving Utilization (%)", 0.0, 100.0,
            value=float(st.session_state["inp_revol"]),
            step=1.0,
            help="Percentage of revolving credit in use (e.g. credit card balance / limit). Higher utilization signals financial stress. See Glossary."
        )

        inq_last_6mths = st.number_input(
            "Inquiries Last 6 Months",
            min_value=0, max_value=10,
            value=st.session_state["inp_inq"],
            help="Number of hard credit inquiries in the last 6 months. More inquiries can indicate the borrower is actively seeking new credit, which may signal financial distress."
        )

        st.caption("⚠️ Fields not shown above are set to Sub Grade-conditioned medians from the training population. See Score Breakdown for details.")

        calculate = st.button("CALCULATE SCORE", use_container_width=True)

    with col_score:
        if calculate:
            # Save all current widget values to session state
            st.session_state["inp_sub_grade"] = sub_grade
            st.session_state["inp_term"] = term
            st.session_state["inp_loan_amnt"] = loan_amnt
            st.session_state["inp_annual_inc"] = annual_inc
            st.session_state["inp_dti"] = dti
            st.session_state["inp_fico"] = fico_range_low
            st.session_state["inp_home"] = home_ownership
            st.session_state["inp_verif"] = verification_status
            st.session_state["inp_purpose"] = purpose
            st.session_state["inp_revol"] = revol_util
            st.session_state["inp_inq"] = inq_last_6mths

            # Build input using sub_grade-conditioned medians
            if sub_grade in SUBGRADE_MEDIANS.index:
                input_data = SUBGRADE_MEDIANS.loc[sub_grade].copy()
            else:
                input_data = POPULATION_MEDIANS.copy()

            for col in categorical_cols_app:
                input_data[col] = TRAIN_MODES[col]

            # Override with user inputs
            input_data["sub_grade"] = sub_grade
            input_data["term"] = term
            input_data["loan_amnt"] = loan_amnt
            input_data["annual_inc"] = annual_inc
            input_data["dti"] = dti
            input_data["fico_range_low"] = fico_range_low
            input_data["home_ownership"] = home_ownership
            input_data["verification_status"] = verification_status
            input_data["purpose"] = purpose
            input_data["revol_util"] = revol_util
            input_data["inq_last_6mths"] = inq_last_6mths

            # WoE transformation
            input_df = pd.DataFrame([input_data])
            woe_row = {}
            for col in FEATURE_ORDER:
                optb = binning_models[col]
                woe_row[col] = optb.transform([input_df[col].values[0]], metric="woe")[0]

            woe_df = pd.DataFrame([woe_row])[FEATURE_ORDER]
            woe_const = sm.add_constant(woe_df, has_constant='add')
            pd_prob = float(logit_model.predict(woe_const).values[0])
            score = float(probability_to_score(pd_prob, A, B))

            if score >= 730:
                risk_class = "low-risk"
                risk_label = "🟢 LOW RISK"
            elif score >= 690:
                risk_class = "medium-risk"
                risk_label = "🟡 MEDIUM RISK"
            else:
                risk_class = "high-risk"
                risk_label = "🔴 HIGH RISK"

            # Save results to session state
            st.session_state['woe_row'] = woe_row
            st.session_state['pd_prob'] = pd_prob
            st.session_state['score'] = score
            st.session_state['risk_class'] = risk_class
            st.session_state['risk_label'] = risk_label
            st.session_state['logit_model'] = logit_model

        # Display results if score exists in session state
        if 'score' in st.session_state:
            score = st.session_state['score']
            pd_prob = st.session_state['pd_prob']
            risk_class = st.session_state['risk_class']
            risk_label = st.session_state['risk_label']

            st.subheader("Score Result")
            st.markdown(f"""
                <div class="score-display {risk_class}">{score:.0f}</div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div style="text-align:center; margin: 0.5rem 0;">
                    <span class="risk-badge" style="background: rgba(255,255,255,0.05);
                    color: #e0e6f0; border: 1px solid #374151;">{risk_label}</span>
                </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Probability of Default</div>
                        <div class="metric-value">{pd_prob*100:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            with c2:
                decile_boundaries = np.percentile(train_scores, np.arange(0, 110, 10))
                borrower_decile = int(np.digitize(score, decile_boundaries)) - 1
                borrower_decile = min(max(borrower_decile, 1), 10)
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Score Decile</div>
                        <div class="metric-value">{borrower_decile} / 10</div>
                    </div>
                """, unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(6, 1.5))
            fig.patch.set_facecolor('#0a0f1e')
            ax.set_facecolor('#0a0f1e')
            score_min, score_max = 635, 805
            ax.barh(0, score_max - score_min, color='#1f2937', height=0.4)
            ax.barh(0, score - score_min,
                color='#00c896' if risk_class == 'low-risk' else '#f5a623' if risk_class == 'medium-risk' else '#ff4757',
                height=0.4)
            ax.set_xlim(0, score_max - score_min)
            ax.set_yticks([])
            ax.set_xticks([0, 55, 95, 170])
            ax.set_xticklabels(['635', '690', '730', '805'], color='#6b7280', fontsize=8)
            ax.spines[:].set_visible(False)
            st.pyplot(fig)
            plt.close()

        else:
            st.markdown("""
                <div style="height: 400px; display: flex; align-items: center;
                justify-content: center; color: #374151; font-family: 'IBM Plex Mono', monospace;
                font-size: 0.9rem; border: 1px dashed #1f2937; border-radius: 8px;">
                    ← Fill in borrower details and click CALCULATE SCORE
                </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — SCORE BREAKDOWN
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Score Breakdown":
    st.title("📊 Score Breakdown")

    if 'woe_row' not in st.session_state:
        st.warning("Please score a borrower first on the Score Borrower page.")
    else:
        woe_row = st.session_state['woe_row']
        logit_model = st.session_state['logit_model']
        score = st.session_state['score']
        pd_prob = st.session_state['pd_prob']

        st.markdown("""
        <div style="background:#111827; border:1px solid #1f2937; border-radius:8px; padding:1.2rem; margin-bottom:1.5rem;">
            <div style="color:#00c896; font-family:'IBM Plex Mono',monospace; font-weight:600; margin-bottom:0.5rem;">
                HOW TO READ THIS PAGE
            </div>
            <div style="color:#9ca3af; font-size:0.9rem; line-height:1.8;">
                This page explains <b style="color:#e0e6f0;">why</b> the borrower received their score.<br><br>
                Each feature in the model contributes a certain number of <b style="color:#e0e6f0;">score points</b> based on which risk bin the borrower falls into.<br><br>
                🟢 <b style="color:#00c896;">Positive points</b> = this feature signals lower risk → pushes the score UP<br>
                🔴 <b style="color:#ff4757;">Negative points</b> = this feature signals higher risk → pushes the score DOWN<br><br>
                The final score is the sum of all feature contributions plus a base score from the model intercept.
                Score range for this model: <b style="color:#e0e6f0;">635 (riskiest) to 805 (safest)</b>.<br><br>
                ⚠️ <b style="color:#f5a623;">Features marked with * in the table below</b>
                were not entered by the user. They are set to the
                <b style="color:#e0e6f0;">median value of borrowers with the same Sub Grade</b>
                from the training population (819,364 LendingClub borrowers, 2010–2015).
                For example, a G5 borrower gets G5-typical credit history, balance, and utilization —
                not the overall population average. See the Glossary entry on
                <b style="color:#e0e6f0;">Default Imputation</b> for full details and limitations.
            </div>
        </div>
        """, unsafe_allow_html=True)

        params = logit_model.params
        contributions = {}
        for col in FEATURE_ORDER:
            coef = params.get(col, 0)
            woe_val = woe_row[col]
            contributions[col] = -B * coef * woe_val

        contrib_df = pd.DataFrame({
            "Feature": list(contributions.keys()),
            "WoE Value": [woe_row[c] for c in contributions.keys()],
            "Score Points": list(contributions.values())
        }).sort_values("Score Points", ascending=False).reset_index(drop=True)

        positive_features = contrib_df[contrib_df["Score Points"] > 0]
        negative_features = contrib_df[contrib_df["Score Points"] < 0]

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Final Score (635–805)</div>
                    <div class="metric-value">{score:.0f}</div>
                </div>""", unsafe_allow_html=True)
        with m2:
            pd_color = '#ff4757' if pd_prob > 0.2 else ('#f5a623' if pd_prob > 0.1 else '#00c896')
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Probability of Default</div>
                    <div class="metric-value" style="color:{pd_color}">{pd_prob*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Features Helping ↑</div>
                    <div class="metric-value" style="color:#00c896">{len(positive_features)}</div>
                </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Features Hurting ↓</div>
                    <div class="metric-value" style="color:#ff4757">{len(negative_features)}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns([1, 1])

        user_entered = ["sub_grade", "term", "loan_amnt", "annual_inc", "dti",
                       "fico_range_low", "home_ownership", "verification_status",
                       "purpose", "revol_util", "inq_last_6mths"]

        with col1:
            st.subheader("Feature Contribution Table")
            st.caption("WoE Value = risk signal for this borrower's bin (positive = safer, negative = riskier). Score Points = how much this feature added or subtracted from the final score. * = set to Sub Grade-conditioned training median, not user input.")

            contrib_df["Risk Signal"] = contrib_df["WoE Value"].apply(
                lambda x: "🟢 Lower Risk" if x > 0.1 else ("🔴 Higher Risk" if x < -0.1 else "🟡 Neutral")
            )
            contrib_df["Source"] = contrib_df["Feature"].apply(
                lambda x: "User Input" if x in user_entered else "* Median"
            )

            st.dataframe(
                contrib_df[["Feature", "Source", "Risk Signal", "WoE Value", "Score Points"]].style
                .format({"WoE Value": "{:.4f}", "Score Points": "{:+.2f}"})
                .applymap(
                    lambda x: "color: #00c896" if isinstance(x, float) and x > 0
                             else ("color: #ff4757" if isinstance(x, float) and x < 0 else ""),
                    subset=["Score Points"]
                ),
                use_container_width=True,
                height=500
            )

        with col2:
            st.subheader("Top Score Drivers")
            st.caption("Top 5 features helping and top 5 hurting the score. Green = pushed score up. Red = pulled score down.")

            top_pos = contrib_df.nlargest(5, "Score Points")
            top_neg = contrib_df.nsmallest(5, "Score Points")
            top_n = pd.concat([top_pos, top_neg]).drop_duplicates()
            top_n = top_n.sort_values("Score Points", ascending=True)

            colors = ['#00c896' if x > 0 else '#ff4757' for x in top_n["Score Points"]]
            max_val = top_n["Score Points"].abs().max()
            x_margin = max_val * 0.30

            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor('#111827')
            ax.set_facecolor('#111827')
            fig.subplots_adjust(left=0.32, right=0.85, top=0.95, bottom=0.12)

            bars = ax.barh(top_n["Feature"], top_n["Score Points"], color=colors, height=0.6)
            ax.axvline(0, color='#374151', linewidth=1)
            ax.set_xlabel("Score Points (+ = helps score, − = hurts score)", color='#6b7280', fontsize=8)
            ax.tick_params(colors='#9ca3af', labelsize=7)
            ax.spines[:].set_color('#1f2937')
            ax.set_xlim(-max_val - x_margin, max_val + x_margin)

            for bar, val in zip(bars, top_n["Score Points"]):
                offset = max_val * 0.05
                ax.text(
                    val + offset if val >= 0 else val - offset,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.1f}",
                    va='center',
                    ha='left' if val >= 0 else 'right',
                    color='#e0e6f0',
                    fontsize=7,
                    fontweight='bold'
                )

            st.pyplot(fig)
            plt.close()

            st.markdown("---")
            st.subheader("Plain English Summary")

            top_helper = contrib_df.iloc[0]
            top_hurter = contrib_df.iloc[-1]

            score_verdict = (
                "comfortably above the approval threshold — low risk profile"
                if score >= 730 else
                "in the borderline zone — moderate risk, neither clearly safe nor clearly risky"
                if score >= 690 else
                "below the typical approval threshold — elevated default risk"
            )

            summary_html = f"""
            <div style="background:#111827; border:1px solid #1f2937; border-radius:8px; padding:1.2rem;">
                <div style="color:#9ca3af; font-size:0.9rem; line-height:2.0;">
                    ✅ <b style="color:#00c896;">Strongest positive signal:</b>
                    <b style="color:#e0e6f0;">{top_helper['Feature']}</b> contributed
                    <b style="color:#00c896;">{top_helper['Score Points']:+.1f} points</b> —
                    this borrower falls into a lower-risk bin for this feature.<br>
                    ❌ <b style="color:#ff4757;">Strongest negative signal:</b>
                    <b style="color:#e0e6f0;">{top_hurter['Feature']}</b> contributed
                    <b style="color:#ff4757;">{top_hurter['Score Points']:+.1f} points</b> —
                    this borrower falls into a higher-risk bin for this feature.<br>
                    📊 <b style="color:#e0e6f0;">{len(positive_features)}</b> of {len(contrib_df)} features
                    pushed the score up, <b style="color:#e0e6f0;">{len(negative_features)}</b> pulled it down.<br>
                    🏁 Final score of <b style="color:#e0e6f0;">{score:.0f}</b> is {score_verdict}.
                    Model-implied probability of default:
                    <b style="color:{pd_color}">{pd_prob*100:.1f}%</b>
                    (population average: 18.5%).
                </div>
            </div>
            """
            st.markdown(summary_html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    st.subheader("Validation Metrics")
    col1, col2, col3 = st.columns(3)
    metrics = validation_metrics.set_index("Metric")

    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Gini (Test)</div>
                <div class="metric-value">{metrics.loc['Gini','Test']:.4f}</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">KS (Test)</div>
                <div class="metric-value">{metrics.loc['KS','Test']:.4f}</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">PSI</div>
                <div class="metric-value">{metrics.loc['PSI','Test']:.4f}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Score Distribution")
        st.caption("Train and test score distributions should overlap closely — large separation would indicate population shift or overfitting.")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')
        ax.hist(train_scores, bins=50, alpha=0.7, color='#3b82f6', label='Train (2010–2015)', edgecolor='none')
        ax.hist(test_scores, bins=50, alpha=0.7, color='#00c896', label='Test (2016–2018)', edgecolor='none')
        ax.set_xlabel("Score", color='#6b7280', fontsize=9)
        ax.set_ylabel("Count", color='#6b7280', fontsize=9)
        ax.tick_params(colors='#9ca3af')
        ax.spines[:].set_color('#1f2937')
        ax.legend(facecolor='#1f2937', edgecolor='#374151', labelcolor='#9ca3af')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Decile Table")
        st.caption("Borrowers ranked by score and split into 10 equal groups. Default rate should decrease monotonically from decile 0 (riskiest) to decile 9 (safest) — any inversion would be a red flag.")
        st.dataframe(
            decile_table.style.format({
                "mean_score": "{:.1f}",
                "default_rate": "{:.1%}",
                "count": "{:,.0f}"
            }),
            use_container_width=True,
            height=350
        )

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — GLOSSARY
# ═══════════════════════════════════════════════════════════════
elif page == "📖 Glossary":
    st.title("📖 Glossary")
    st.caption("Reference guide for all metrics, features, and concepts used in this scorecard.")

    glossary = [
        {
            "term": "Sub Grade",
            "definition": "LendingClub's internal risk classification, ranging from A1 (lowest risk) to G5 (highest risk). Each letter grade (A through G) is subdivided into 5 levels (1 through 5), giving 35 total categories. Sub grade is the strongest predictor in this model with an Information Value of 0.51. It is derived by LendingClub from the borrower's FICO score, requested loan amount, and loan purpose.",
            "formula": "A1 → A2 → ... → G4 → G5 (increasing risk)"
        },
        {
            "term": "FICO Score",
            "definition": "A standardized credit score developed by Fair Isaac Corporation, ranging from 300 to 850. It summarizes a borrower's credit history across five factors: payment history (35%), amounts owed (30%), length of credit history (15%), new credit (10%), and credit mix (10%). Higher scores indicate lower default risk. In this model we use the lower bound of the reported FICO range at loan origination.",
            "formula": "Score range: 300 (worst) → 850 (best)"
        },
        {
            "term": "DTI — Debt-to-Income Ratio",
            "definition": "The ratio of a borrower's total monthly debt obligations to their gross monthly income, expressed as a percentage. It measures how much of the borrower's income is already committed to servicing existing debt. A higher DTI indicates less financial flexibility and higher default risk.",
            "formula": "DTI = (Total Monthly Debt Payments / Gross Monthly Income) × 100"
        },
        {
            "term": "Revolving Utilization",
            "definition": "The percentage of revolving credit currently in use relative to the total available revolving credit limit. High utilization indicates the borrower is relying heavily on available credit, which can signal financial stress. It is one of the most important factors in FICO score calculation.",
            "formula": "Revol. Util = (Total Revolving Balance / Total Revolving Credit Limit) × 100"
        },
        {
            "term": "PD — Probability of Default",
            "definition": "The model's estimated probability that a borrower will default on their loan within the observation window. It is the direct output of the logistic regression model, converted from log-odds via the sigmoid function. PD is a key input to Expected Credit Loss (ECL) calculations under IFRS 9. In this model: green = below 10%, orange = 10–20%, red = above 20%.",
            "formula": "PD = 1 / (1 + e^(-log-odds))"
        },
        {
            "term": "WoE — Weight of Evidence",
            "definition": "A transformation applied to each feature that encodes the relative concentration of goods (non-defaults) vs bads (defaults) in each bin. Positive WoE indicates a bin has proportionally more goods than the overall population (lower risk). Negative WoE indicates more bads (higher risk). WoE linearizes the relationship between features and log-odds, making it ideal for logistic regression.",
            "formula": "WoE = ln(% of Goods in bin / % of Bads in bin)"
        },
        {
            "term": "IV — Information Value",
            "definition": "A single number summarizing the overall predictive power of a variable. It aggregates the WoE signal across all bins. Rule of thumb: IV < 0.02 = useless, 0.02–0.1 = weak, 0.1–0.3 = medium, 0.3–0.5 = strong, > 0.5 = suspicious (possible leakage). In this model, sub_grade (IV=0.51) and term (IV=0.24) are the two strongest predictors.",
            "formula": "IV = Σ (% Goods_i − % Bads_i) × WoE_i"
        },
        {
            "term": "PDO — Points to Double the Odds",
            "definition": "A scorecard scaling parameter that defines the sensitivity of the score scale. PDO = 20 means that every 20-point increase in score corresponds to a doubling of the good-to-bad odds ratio. In this model: anchor score = 600, anchor odds = 10:1. The score range is 635–805.",
            "formula": "B = PDO / ln(2) = 28.85,  Score = A − B × log-odds,  A = 666.44"
        },
        {
            "term": "Loan Term and Default Risk",
            "definition": "Loan term is the second strongest predictor in this model (IV = 0.24). 36-month loans have a default rate of 13.9% in the training data vs 31.9% for 60-month loans — a 2.3x difference. Switching from 36 to 60 months can move the score by approximately 15–20 points. This reflects that longer repayment periods introduce greater uncertainty.",
            "formula": "36 months: WoE = +0.339 | 60 months: WoE = −0.727"
        },
        {
            "term": "Default Imputation — Where Do Unseen Fields Come From?",
            "definition": "This scorecard uses 28 features in total. The input form collects 11 key fields from the user. The remaining 17 fields are set to the median value of borrowers with the same Sub Grade from the training population (819,364 LendingClub borrowers, 2010–2015). For example, if the user selects Sub Grade G5, the model uses median values observed among actual G5 borrowers — not the overall population average. This is more realistic because borrower characteristics are strongly correlated with risk grade. Limitation: this still assumes an average profile within the sub_grade — individual borrowers may deviate significantly.",
            "formula": "Missing field = median(training borrowers with same Sub Grade)"
        },
        {
            "term": "Multicollinearity and Sign Reversal",
            "definition": "In a multivariate logistic regression, individual feature coefficients represent the marginal effect of that feature controlling for all others — not its standalone effect. When features are correlated (e.g. revolving utilization, bc_util, percent_bc_gt_75 all measure credit usage), the model distributes the shared signal across all correlated features. This can cause individual coefficients to appear counterintuitive when examined in isolation. This is a known statistical artifact (Owen & Roediger 2014, Knaeble & Dutter 2015) and does not affect the model's overall predictive validity — the Gini of 0.41 confirms the model correctly ranks borrowers at the portfolio level.",
            "formula": "β_multivariate ≠ β_univariate when corr(Xᵢ, Xⱼ) ≠ 0"
        },
        {
            "term": "Gini Coefficient",
            "definition": "A measure of the model's ability to discriminate between goods and bads. Derived from the ROC curve: Gini = 2 × AUC − 1. Gini = 0 means no better than random. Gini = 1 means perfect discrimination. This model achieves Gini = 0.41 on the out-of-time test set (2016–2018). The train-test gap is 0.028, confirming the model generalizes without significant overfitting.",
            "formula": "Gini = 2 × AUC − 1 | This model: Train = 0.441, Test = 0.413"
        },
        {
            "term": "KS — Kolmogorov-Smirnov Statistic",
            "definition": "The maximum vertical distance between the cumulative distribution of goods and the cumulative distribution of bads, when borrowers are ranked by predicted score. It identifies the score threshold at which the model best separates good from bad borrowers. KS > 0.20 is generally considered acceptable. This model achieves KS = 0.30 on the test set.",
            "formula": "KS = max |Cumulative % Bads − Cumulative % Goods| | This model: Test = 0.2975"
        },
        {
            "term": "PSI — Population Stability Index",
            "definition": "A measure of how much the distribution of model scores has shifted between the training period and the scoring period. PSI < 0.10 = no significant shift. PSI 0.10–0.25 = moderate shift, monitor. PSI > 0.25 = major shift, model may need rebuilding. This model achieves PSI = 0.004 — essentially no population shift between 2010–2015 and 2016–2018.",
            "formula": "PSI = Σ (Actual% − Expected%) × ln(Actual% / Expected%) | This model: 0.004"
        },
        {
            "term": "Credit History Length",
            "definition": "The number of months between the borrower's earliest credit line and the loan issue date. Longer history generally indicates greater financial stability. This feature was engineered from raw date columns (earliest_cr_line and issue_d). The anchor is the loan issue date — not today's date — because the model must only see information available at the moment of the lending decision.",
            "formula": "Credit History Months = (Issue Date − Earliest Credit Line Date) in months"
        },
        {
            "term": "Out-of-Time Validation",
            "definition": "A validation methodology where the model is tested on data from a different time period than the training data. In this project, the model was trained on 2010–2015 LendingClub loans and validated on 2016–2018 loans. This simulates real deployment conditions and is the gold standard for credit model validation per SR 11-7 regulatory guidance. Note: the test period (2016–2018) had a higher observed default rate (22.4%) than the training period (18.5%), indicating a macroeconomic shift — the model's probability estimates are calibrated to the training period.",
            "formula": "Train: 2010–2015 (819,364 loans) | Test: 2016–2018 (516,405 loans)"
        },
        {
            "term": "Score Decile",
            "definition": "The decile (1–10) into which a borrower's score falls when the training population is divided into 10 equal groups by score. Decile 1 = bottom 10% of scores (riskiest, 45.2% default rate). Decile 10 = top 10% of scores (safest, 3.6% default rate). The 12x difference in default rate between decile 1 and decile 10 confirms strong model discrimination.",
            "formula": "Decile 1: avg score 672, default rate 45.2% | Decile 10: avg score 760, default rate 3.6%"
        }
    ]

    for item in glossary:
        st.markdown(f"""
            <div class="glossary-card">
                <div class="glossary-term">{item['term']}</div>
                <div class="glossary-def">{item['definition']}</div>
                <div class="glossary-formula">Formula: {item['formula']}</div>
            </div>
        """, unsafe_allow_html=True)