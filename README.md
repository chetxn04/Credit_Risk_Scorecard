# Credit Risk Scorecard
### End-to-End PD Modeling Pipeline on LendingClub Data

**Author:** Chetan Maheshwari  
**Stack:** Python · Logistic Regression · XGBoost · WoE/IV · PDO Scaling · Streamlit · Docker

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation and Goals](#2-motivation-and-goals)
3. [Dataset](#3-dataset)
4. [Repository Structure](#4-repository-structure)
5. [Methodology](#5-methodology)
   - 5.1 [Target Definition](#51-target-definition)
   - 5.2 [Data Cleaning](#52-data-cleaning)
   - 5.3 [Feature Engineering — WoE and IV](#53-feature-engineering--woe-and-iv)
   - 5.4 [Feature Selection](#54-feature-selection)
   - 5.5 [Logistic Regression Model](#55-logistic-regression-model)
   - 5.6 [PDO Scorecard Scaling](#56-pdo-scorecard-scaling)
   - 5.7 [XGBoost Challenger Model](#57-xgboost-challenger-model)
   - 5.8 [Validation Metrics](#58-validation-metrics)
6. [Results](#6-results)
7. [Known Limitations and Honest Caveats](#7-known-limitations-and-honest-caveats)
8. [Streamlit Dashboard](#8-streamlit-dashboard)
9. [Running the Project](#9-running-the-project)
10. [Docker](#10-docker)
11. [Testing](#11-testing)
12. [Regulatory Context](#12-regulatory-context)
13. [References](#13-references)

---

## 1. Project Overview

This project is a complete, end-to-end credit risk scorecard built on publicly available LendingClub loan data. It models the **Probability of Default (PD)** for individual borrowers and converts those probabilities into an interpretable points-based scorecard — the same format used by major retail banks for consumer lending decisions.

The pipeline covers every stage a production credit model goes through:

- Raw data ingestion and cleaning
- Leakage identification against the LendingClub data dictionary
- Weight of Evidence (WoE) and Information Value (IV) feature engineering
- Logistic regression as the primary model (statsmodels)
- XGBoost as a challenger model for performance benchmarking
- PDO scorecard scaling from log-odds to a 635–805 point scale
- Out-of-time validation using Gini, KS, and PSI — all implemented from scratch in NumPy
- Streamlit dashboard for interactive borrower scoring
- Docker containerization

The project is designed to be **fully defensible** — every decision has a documented rationale grounded in credit risk convention, statistical reasoning, or regulatory guidance.

---

## 2. Motivation and Goals

Credit scorecards are among the most widely deployed machine learning systems in the world — used by every retail bank, credit card issuer, and fintech lender to make lending decisions at scale. Despite their ubiquity, most data science portfolios treat credit modeling superficially.

This project was built to demonstrate:

- **Domain depth** — understanding of Basel II IRB, SR 11-7, IFRS 9 ECL, and credit scoring conventions
- **Statistical rigor** — WoE/IV from scratch before using optbinning; Gini/KS/PSI from scratch in NumPy
- **Engineering discipline** — modular `src/` structure, pytest coverage, Docker, GitHub Actions-ready CI
- **Intellectual honesty** — known limitations are documented, not hidden; every rule-of-thumb is explained as a convention rather than a law

---

## 3. Dataset

**Source:** LendingClub public loan dataset (2007–2018), available on Kaggle  
**Raw size:** 2.26M rows, 151 columns  
**Final modeling dataset:** 1,335,769 rows (after cleaning and splitting)

### Train/Test Split Strategy

A **chronological out-of-time split** was used rather than a random split:

| Split | Period | Rows |
|-------|--------|------|
| Excluded | 2007–2009 | ~300K |
| **Train** | 2010–2015 | **819,364** |
| **Test** | 2016–2018 | **516,405** |

**Why exclude 2007–2009?** The financial crisis introduced severe distributional distortion. A model trained on crisis-era data would not generalize to normal economic conditions.

**Why chronological?** Random splits allow future information to leak into training. In deployment, a model always scores borrowers from a future period it has never seen. Chronological splitting replicates this exactly.

---

## 4. Repository Structure

```
credit-risk-scorecard/
├── app.py                          ← Streamlit dashboard
├── Dockerfile                      ← Container definition
├── requirements.txt                ← Pinned dependencies
├── .dockerignore
├── notebooks/
│   └── 01_eda.ipynb               ← Full modeling pipeline
├── src/
│   ├── __init__.py
│   ├── data_prep.py               ← Data loading and cleaning functions
│   ├── features.py                ← WoE transformation functions
│   ├── model.py                   ← Logistic regression wrapper
│   ├── scorecard.py               ← PDO scaling functions
│   └── evaluate.py                ← Gini, KS, PSI from scratch
├── tests/
│   ├── test_evaluate.py           ← Pytest tests for metrics
│   └── test_scorecard.py          ← Pytest tests for scaling
└── data/
    ├── raw/                       ← Original LendingClub CSV (not tracked)
    └── processed/                 ← Cleaned datasets and model artifacts
        ├── train_selected.csv
        ├── test_selected.csv
        ├── train_woe.csv
        ├── test_woe.csv
        ├── logit_model.pkl
        ├── binning_models.pkl
        ├── train_scores.npy
        ├── test_scores.npy
        ├── decile_table.csv
        └── validation_metrics.csv
```

---

## 5. Methodology

### 5.1 Target Definition

The binary target variable was defined from `loan_status`:

| loan_status | Target |
|-------------|--------|
| Fully Paid | 0 (Good) |
| Charged Off | 1 (Bad) |
| Default | 1 (Bad) |
| Current, Late, Grace Period | Excluded |

**Class distribution:** 81.5% Good, 18.5% Bad. Due to this imbalance, accuracy was not used as an evaluation metric. Gini, KS, and AUC were used instead.

**Leakage columns removed:** `recoveries`, `collection_recovery_fee`, `last_pymnt_amnt`, `last_pymnt_d`, `last_credit_pull_d`, `last_fico_range_high`, `last_fico_range_low`, `loan_status`. These columns contain information only available after loan performance is known — using them would produce a model that appears perfect in validation but fails completely in deployment.

All leakage decisions were verified against the official LendingClub data dictionary rather than inferred from column names alone.

---

### 5.2 Data Cleaning

#### Missing Values
Columns with more than 40% missing values were dropped (58 columns). This threshold is an industry convention, not a hard rule — the rationale is that imputing more than 40% of a column's values introduces more noise than signal. Remaining missing values were imputed using **training set medians only**, applied to both train and test. Fitting imputation statistics on the test set would constitute leakage.

#### Outlier Treatment
`annual_inc` and `dti` were capped at the 99.9th percentile. Rows above this cap were dropped (~3,067 rows). This removes extreme values that would distort regression coefficients without removing the underlying signal.

#### Feature Engineering
`credit_history_months` was engineered from `earliest_cr_line` and `issue_d`:

```python
credit_history_months = (
    (issue_d.year - earliest_cr_line.year) * 12 +
    (issue_d.month - earliest_cr_line.month)
)
```

The anchor is `issue_d` (loan issue date) rather than today's date — because the model must only see information as it existed at the moment of the lending decision. Using today's date would give borrowers credit history length from a future vantage point they don't actually have at origination.

#### Post-Issue Leakage Drops
The following columns describe what happened after loan origination and were dropped:

- `out_prncp`, `out_prncp_inv` — outstanding principal remaining
- `total_pymnt`, `total_pymnt_inv` — total amount paid
- `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee` — amounts received
- `hardship_flag`, `debt_settlement_flag` — post-issue events
- `disbursement_method` — determined at loan issuance, not application
- `int_rate`, `installment`, `funded_amnt`, `funded_amnt_inv` — set after credit decision

---

### 5.3 Feature Engineering — WoE and IV

Weight of Evidence (WoE) and Information Value (IV) are the standard feature engineering framework for retail credit scorecards. WoE transforms each feature into a single number per bin that directly encodes the log-odds contribution of that bin relative to the overall population.

#### WoE Formula

For each bin `i` of a feature:

```
WoE_i = ln(% of Goods in bin_i / % of Bads in bin_i)
```

Where:
- **% of Goods in bin_i** = (number of non-defaults in bin_i) / (total non-defaults)
- **% of Bads in bin_i** = (number of defaults in bin_i) / (total defaults)

**Interpretation:**
- WoE > 0 → bin has proportionally more goods than the overall population → lower default risk
- WoE < 0 → bin has proportionally more bads → higher default risk
- WoE = 0 → bin has the same good/bad ratio as the overall population

#### Manual WoE Walkthrough — Grade A

To verify understanding before using any library, WoE was computed manually for the `grade` variable:

| Grade | Total | Goods | Bads | % Goods | % Bads | WoE |
|-------|-------|-------|------|---------|--------|-----|
| A | 144,705 | 136,656 | 8,049 | 20.46% | 5.32% | +1.347 |
| B | 239,374 | 210,566 | 28,808 | 31.52% | 19.04% | +0.504 |
| C | 223,369 | 177,789 | 45,580 | 26.61% | 30.12% | -0.124 |
| D | 125,006 | 89,760 | 35,246 | 13.44% | 23.29% | -0.550 |
| E | 61,494 | 38,835 | 22,659 | 5.81% | 14.97% | -0.946 |
| F | 20,587 | 11,857 | 8,730 | 1.77% | 5.77% | -1.179 |
| G | 4,829 | 2,568 | 2,261 | 0.38% | 1.49% | -1.358 |

WoE moves monotonically from +1.347 (Grade A, very safe) to -1.358 (Grade G, very risky) — confirming the grade variable has excellent discriminatory power. IV = 0.474 (Strong predictor).

#### IV Formula

```
IV = Σ_i (% Goods_i - % Bads_i) × WoE_i
```

IV summarizes the total predictive power of a variable across all bins.

| IV Range | Predictive Power |
|----------|-----------------|
| < 0.02 | Useless |
| 0.02 – 0.10 | Weak |
| 0.10 – 0.30 | Medium |
| 0.30 – 0.50 | Strong |
| > 0.50 | Suspicious — check for leakage |

#### Optimal Binning

For numeric variables, bin boundaries were determined using **optbinning**, which solves a constrained dynamic programming problem to find bin boundaries that:
1. Maximize IV
2. Enforce WoE monotonicity across bins
3. Enforce minimum bin size (statistical stability)

The dynamic programming approach is equivalent in structure to the classic rod-cutting problem — optimal substructure allows building the solution incrementally without exhaustive search over all 2^n possible boundary placements.

For categorical variables, optbinning finds the optimal grouping of categories rather than cut points, since categories have no inherent ordering (except `grade` and `sub_grade` which were specified as ordinal).

All binning models were **fit on training data only** and applied to test using the same learned boundaries — preventing leakage of test distribution into the transformation.

---

### 5.4 Feature Selection

The full IV table from optbinning:

| Feature | IV | Predictive Power |
|---------|----|-----------------|
| sub_grade | 0.5056 | Strong |
| term | 0.2411 | Medium |
| fico_range_low | 0.1213 | Medium |
| acc_open_past_24mths | 0.0814 | Weak |
| dti | 0.0758 | Weak |
| num_tl_op_past_12m | 0.0585 | Weak |
| bc_open_to_buy | 0.0555 | Weak |
| verification_status | 0.0514 | Weak |
| avg_cur_bal | 0.0406 | Weak |
| mo_sin_rcnt_tl | 0.0404 | Weak |
| total_bc_limit | 0.0390 | Weak |
| loan_amnt | 0.0370 | Weak |
| tot_hi_cred_lim | 0.0367 | Weak |
| tot_cur_bal | 0.0327 | Weak |
| annual_inc | 0.0324 | Weak |
| mths_since_recent_inq | 0.0304 | Weak |
| mo_sin_rcnt_rev_tl_op | 0.0302 | Weak |
| percent_bc_gt_75 | 0.0299 | Weak |
| num_actv_rev_tl | 0.0294 | Weak |
| num_rev_tl_bal_gt_0 | 0.0286 | Weak |
| inq_last_6mths | 0.0278 | Weak |
| bc_util | 0.0273 | Weak |
| mths_since_recent_bc | 0.0269 | Weak |
| mort_acc | 0.0255 | Weak |
| total_rev_hi_lim | 0.0249 | Weak |
| revol_util | 0.0224 | Weak |
| mo_sin_old_rev_tl_op | 0.0220 | Weak |
| home_ownership | 0.0212 | Weak |
| purpose | 0.0179 | Weak* |
| credit_history_months | 0.0099 | Weak* |

*Retained despite borderline IV due to domain convention (purpose is a standard scorecard feature; credit_history_months is a deliberately engineered feature with interpretive value).

**Features dropped from selection:**
- `grade` — redundant with `sub_grade` (hierarchically nested)
- `addr_state` — IV = 0.014, 50 categories, overfitting risk
- `initial_list_status` — IV = 0.0005, near zero
- `application_type` — IV = 0.000
- All zero-IV features

**Post-selection logistic regression drops:**
- `bc_util` — p-value = 0.444 (statistically insignificant)
- `total_rev_hi_lim` — p-value = 0.772 (statistically insignificant)

**Final model:** 28 features, all statistically significant (p < 0.05).

---

### 5.5 Logistic Regression Model

**Library:** statsmodels (for statistical inference — p-values, confidence intervals, standard errors)

**Why logistic regression for credit scoring?**

1. **Regulatory interpretability (SR 11-7):** Each feature has an explicit coefficient. The contribution of every variable to every score is transparent and auditable. Regulators require this.
2. **Scorecard conversion:** Logistic regression output (log-odds) maps directly to a points-based scorecard via PDO scaling. The math works because WoE + logistic regression produces log-odds that are linear and additive.
3. **Stability over complexity:** Credit models run for years on shifting populations. Simpler models degrade more predictably — which is measurable and manageable.

**Model equation:**

```
log-odds = β₀ + β₁×WoE(sub_grade) + β₂×WoE(term) + ... + β₂₈×WoE(credit_history_months)
```

**Key results from statsmodels summary:**

| Metric | Value |
|--------|-------|
| Observations | 819,364 |
| Pseudo R² (McFadden) | 0.1006 |
| Log-Likelihood | -352,590 |
| Null Log-Likelihood | -392,020 |
| Converged | Yes (6 iterations) |

**Intercept interpretation:** β₀ = -1.4871. Converting to probability:

```
p = 1 / (1 + e^1.4871) = 0.1847
```

This matches the training set default rate of 18.47% exactly — a mathematical property of MLE logistic regression where the mean predicted probability always equals the observed event rate on training data.

**A note on coefficient signs:**

Most coefficients are negative — meaning higher WoE (safer bin) reduces log-odds of default. Two features (`revol_util`, `mo_sin_rcnt_rev_tl_op`, `credit_history_months`) have positive coefficients despite their WoE correctly capturing the risk direction. This is a documented phenomenon in multivariate logistic regression when features are correlated.

When features share explanatory power (e.g. `revol_util`, `bc_util`, `percent_bc_gt_75` all measure credit utilization), the model distributes the signal across them during joint MLE estimation. Individual coefficients represent marginal effects controlling for all others — not standalone effects. This can produce apparent sign reversals for weak, correlated features. The model's overall predictive validity is confirmed by the Gini of 0.41 on out-of-time data. See references: Owen & Roediger (2014), Knaeble & Dutter (2015).

---

### 5.6 PDO Scorecard Scaling

The logistic regression produces log-odds. These are converted to a points-based scorecard using the PDO (Points to Double the Odds) framework — the industry standard for retail credit scorecards.

#### Scaling Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| PDO | 20 | Industry convention |
| Anchor Score | 600 | Industry convention |
| Anchor Odds | 10:1 | Chosen to reflect portfolio average PD (~18%) |

The anchor odds of 10:1 (good:bad) was chosen over the more common 50:1 because our portfolio has an 18.5% default rate. At anchor odds of 50:1, the score of 600 would correspond to PD ≈ 2% — far below our portfolio average, compressing all borrowers into the upper range. At 10:1 (PD ≈ 9%), the distribution is more spread and interpretable.

#### Derivation of Scaling Constants

From the PDO definition (doubling odds = +PDO points):

```
B = PDO / ln(2) = 20 / 0.6931 = 28.85
A = Anchor Score + B × ln(Anchor Odds) = 600 + 28.85 × ln(10) = 666.44
Score = A - B × log-odds = 666.44 - 28.85 × log-odds
```

**Why `A - B × log-odds` with a minus sign?**

Higher log-odds of default = riskier borrower. The minus sign ensures riskier borrowers receive lower scores — consistent with the convention that higher scores mean lower risk.

#### Score Distribution

| Statistic | Train | Test |
|-----------|-------|------|
| Min | 636.8 | 634.7 |
| Max | 805.4 | 801.3 |
| Mean | 715.7 | 716.4 |
| Std | 25.2 | 25.1 |

The near-normal distribution of scores is an expected property of WoE + logistic regression — WoE-encoded features fed into logistic regression produce log-odds that are approximately normally distributed.

#### Decile Table

| Decile | Mean Score | Default Rate | Count |
|--------|-----------|--------------|-------|
| 0 (Riskiest) | 672.1 | 45.2% | 81,937 |
| 1 | 688.7 | 31.2% | 81,936 |
| 2 | 698.1 | 24.9% | 81,936 |
| 3 | 705.5 | 20.7% | 81,937 |
| 4 | 712.2 | 17.2% | 81,936 |
| 5 | 718.6 | 14.3% | 81,936 |
| 6 | 725.3 | 11.7% | 81,937 |
| 7 | 732.8 | 9.1% | 81,936 |
| 8 | 742.7 | 6.8% | 81,936 |
| 9 (Safest) | 760.5 | 3.6% | 81,937 |

Monotonically decreasing default rate across all deciles with no inversions — confirming strong model discrimination. 12x difference in default rate between decile 0 (45.2%) and decile 9 (3.6%).

---

### 5.7 XGBoost Challenger Model

An XGBoost classifier was trained as a challenger model to benchmark logistic regression performance.

**Configuration:**
```python
XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    enable_categorical=True,
    tree_method="hist"
)
```

**Native categorical support** (XGBoost v1.5+) was used instead of label encoding or one-hot encoding. This allows XGBoost to find optimal partitions of categorical variables directly, without the false ordering introduced by label encoding or the dimensionality explosion of one-hot encoding.

**Raw features** (not WoE-transformed) were used for XGBoost. WoE transformation linearizes relationships for logistic regression — XGBoost's tree-splitting mechanism can capture non-linear relationships directly from raw values and benefits from having the full information rather than a pre-binned representation.

#### Model Comparison

| Metric | LR Train | LR Test | XGB Train | XGB Test |
|--------|----------|---------|-----------|----------|
| Gini | 0.4407 | 0.4128 | 0.4632 | 0.4279 |
| KS | 0.3199 | 0.2975 | 0.3380 | 0.3094 |
| PSI | 0.0044 | — | 0.0034 | — |

**Conclusion:** XGBoost outperforms logistic regression by ~1.5 Gini points on the test set. However:

1. The gain is modest — WoE + logistic regression captures most of the available signal
2. XGBoost cannot be directly converted to a scorecard without additional explainability work
3. XGBoost has a larger train/test Gini gap (0.035 vs 0.028), indicating marginally more overfitting
4. In a regulated environment, logistic regression remains the primary model per SR 11-7 requirements

XGBoost serves as validation that the logistic regression is not leaving significant performance on the table.

---

### 5.8 Validation Metrics

All three metrics were implemented from scratch in NumPy — no sklearn or scipy — to demonstrate mathematical understanding rather than library usage.

#### Gini Coefficient

```python
sorted_indices = np.argsort(-y_prob)
cum_bads = np.cumsum(y_sorted) / total_bads
cum_goods = np.cumsum(1 - y_sorted) / total_goods
auc = np.trapezoid(cum_bads, cum_goods)
gini = 2 * auc - 1
```

The trapezoidal rule approximates the area under the ROC curve by summing 819,364 tiny trapezoids. With this many points the approximation is effectively exact.

**Results:** Train Gini = 0.4407, Test Gini = 0.4128, Difference = 0.028

#### KS Statistic

```python
ks = np.max(np.abs(cum_bads - cum_goods))
```

**Results:** Train KS = 0.3199, Test KS = 0.2975, Difference = 0.022

#### PSI

```python
breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
psi_bins = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
psi = psi_bins.sum()
```

Bin boundaries are computed from the training distribution only — applying them to both train and test ensures an apples-to-apples comparison of the two distributions.

**Results:** PSI = 0.004 — no meaningful population shift between training and scoring periods.

---

## 6. Results

| Metric | Train | Test | Threshold |
|--------|-------|------|-----------|
| Gini | 0.441 | 0.413 | > 0.35 ✓ |
| KS | 0.320 | 0.298 | > 0.20 ✓ |
| PSI | 0.004 | — | < 0.10 ✓ |
| Pseudo R² | 0.101 | — | 0.10–0.30 ✓ |

**Calibration:** Mean predicted probability on test = 0.1805. Observed default rate on test = 0.2242. The model systematically underpredicts default probability on the test set. This is because the test period (2016–2018) had a higher default environment than the training period (2010–2015) — a macroeconomic shift rather than a model failure. The model's rank ordering (measured by Gini/KS) remains strong despite the calibration gap.

---

## 7. Known Limitations and Honest Caveats

**1. Probability underprediction on test set**
The model was calibrated to a training period with 18.5% default rate. The test period had 22.4%. Predicted probabilities should not be used directly for IFRS 9 ECL calculations without recalibration.

**2. Score range does not cover 300–850**
The score range (635–805) reflects the actual risk distribution of LendingClub borrowers — a filtered, non-subprime population. The model was not designed to score borrowers outside this population.

**3. Multicollinearity in weak features**
Several correlated credit utilization features (`revol_util`, `bc_util`, `percent_bc_gt_75`) share explanatory power. Individual coefficients for these features can appear to have counterintuitive signs when examined in isolation — a known consequence of multicollinearity in multivariate regression. This does not affect portfolio-level predictions. See: Owen & Roediger (2014).

**4. Imputation in the Streamlit dashboard**
The dashboard collects 11 of the 28 model features from the user. The remaining 17 are set to sub_grade-conditioned medians from the training population. A real deployment would collect all 28 fields from credit bureau data or the loan application.

**5. Financial crisis exclusion**
2007–2009 data was excluded due to distributional distortion from the financial crisis. The model may not perform well on data from severe economic downturns.

**6. LendingClub-specific population**
LendingClub borrowers are not representative of all consumer credit borrowers. The model should not be applied to populations with different underlying risk distributions without revalidation.

---

## 8. Streamlit Dashboard

The dashboard provides four pages:

### 🏦 Score Borrower
Input 11 key borrower attributes and receive:
- A credit score (635–805 scale)
- Probability of Default
- Risk band (Low / Medium / High)
- Score position on a visual gauge
- Score decile relative to training population

### 📊 Score Breakdown
Explains why the borrower received their specific score:
- Feature contribution table showing each feature's WoE value and point contribution
- Bar chart of top 5 positive and top 5 negative drivers
- Plain English summary identifying strongest positive and negative signals
- Source column distinguishing user-entered fields from imputed fields

### 📈 Model Performance
Model validation dashboard:
- Gini, KS, PSI metrics
- Score distribution comparison (train vs test)
- Decile table showing default rate by score band

### 📖 Glossary
Comprehensive reference covering all terms, formulas, and concepts used in the scorecard — including WoE, IV, PDO, Gini, KS, PSI, FICO, DTI, multicollinearity sign reversal, and imputation methodology.

---

## 9. Running the Project

### Prerequisites
- Python 3.9+
- Mac/Linux recommended
- ~4GB RAM for full dataset

### Setup

```bash
git clone <repo-url>
cd credit-risk-scorecard

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter lab notebooks/01_eda.ipynb
```

Run all cells in order. The notebook will:
1. Load and clean the raw LendingClub data
2. Engineer features including `credit_history_months`
3. Fit optbinning models and compute IV table
4. Train logistic regression and XGBoost
5. Apply PDO scaling
6. Compute and save all validation metrics
7. Save model artifacts to `data/processed/`

### Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

**Note:** The dashboard requires model artifacts in `data/processed/`. Run the notebook first to generate them, or download the pre-computed artifacts from the repository.

---

## 10. Docker

### Build

```bash
docker build -t credit-risk-scorecard .
```

### Run Tests

```bash
docker run credit-risk-scorecard
```

### Run Dashboard (once Streamlit is fully integrated)

```bash
docker run -p 8501:8501 credit-risk-scorecard streamlit run app.py
```

The dashboard will be accessible at `http://localhost:8501`.

---

## 11. Testing

pytest is used to verify core mathematical functions.

```bash
pytest tests/ -v
```

### Test Coverage

**`tests/test_evaluate.py`** — 6 tests:
- `test_gini_random_model` — random model should have Gini ≈ 0
- `test_gini_perfect_model` — perfect model should have Gini ≈ 1
- `test_ks_random_model` — random model should have KS ≈ 0
- `test_ks_perfect_model` — perfect model should have KS ≈ 1
- `test_psi_identical_distributions` — identical distributions should have PSI ≈ 0
- `test_psi_different_distributions` — very different distributions should have PSI > 0.25

**`tests/test_scorecard.py`** — 4 tests:
- `test_scaling_params` — B and A computed correctly from PDO formula
- `test_round_trip_conversion` — probability → score → probability recovers original
- `test_higher_pd_lower_score` — higher default probability produces lower score
- `test_pdo_property` — adding PDO points doubles the odds

---

## 12. Regulatory Context

This project was built with awareness of the regulatory environment credit models operate in:

**SR 11-7 (Federal Reserve, 2011)** — Model Risk Management guidance. Requires models to be conceptually sound, validated independently, and their limitations documented. The logistic regression's interpretability directly supports SR 11-7 compliance. XGBoost's opacity makes it unsuitable as a primary model without additional explainability work (SHAP values, LIME).

**Basel II IRB Approach** — Internal Ratings-Based approach allows banks to use internal PD models for regulatory capital calculation, subject to supervisory approval. The structure of this project (PD model, out-of-time validation, challenger model) mirrors the IRB model development process.

**IFRS 9 ECL** — International Financial Reporting Standard 9 requires Expected Credit Loss provisioning using PD × LGD × EAD. This model produces PD estimates. The underprediction on the test set (18.5% calibrated vs 22.4% observed) highlights why recalibration is required before using model outputs for ECL calculations.

---

## 13. References

**Methodology:**
- Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring.* Wiley.
- Thomas, L.C., Edelman, D.B., & Crook, J.N. (2002). *Credit Scoring and Its Applications.* SIAM.
- Anderson, R. (2007). *The Credit Scoring Toolkit.* Oxford University Press.

**Multicollinearity and Coefficient Sign Reversal:**
- Owen, A.B., & Roediger, P.A. (2014). *The sign of the logistic regression coefficient.* arXiv:1402.0845.
- Knaeble, B., & Dutter, S. (2015). *Reversals of Least-Squares Estimates and Model-Independent Estimation for Directions of Unique Effects.* arXiv:1503.02722.
- Vatcheva, K.P., et al. (2016). *Multicollinearity in Regression Analyses Conducted in Epidemiologic Studies.* Epidemiology (Sunnyvale), 6(2). PMC4318006.

**Regulatory:**
- Federal Reserve / OCC (2011). *SR 11-7: Supervisory Guidance on Model Risk Management.*
- Bank for International Settlements (2006). *Basel II: International Convergence of Capital Measurement and Capital Standards.*
- IASB (2014). *IFRS 9: Financial Instruments.*

**Optimal Binning:**
- Navas-Palencia, G. (2020). *Optimal binning for scoring modeling.* optbinning library documentation.

---

*Built by Chetan Maheshwari*
