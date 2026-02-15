# Double Machine Learning: A Beginner's Guide to Causal Inference

**Author:** Jahnavi Patel  
**Course:** INFO 7390 — Crash Course in Causality, Northeastern University  
**Date:** February 2026  
**Topic:** Double Machine Learning (DML) for Causal Effect Estimation from Observational Data

---

## Project Overview

This project demonstrates how Double Machine Learning (DML) estimates causal effects from observational data where randomized experiments are infeasible. The notebook walks through the complete pipeline — from causal theory and DAG construction through data preparation, DML estimation, heterogeneity analysis, and robustness checks — using two real-world datasets and one synthetic validation exercise.

**Core method:** DML (Chernozhukov et al., 2018) uses flexible machine learning models to estimate nuisance functions (outcome and treatment propensity) via cross-fitting, then recovers the causal effect from residualized variables. This eliminates regularization bias while providing valid statistical inference.

**Key results:**

| Analysis | Naive Estimate | DML Causal Estimate | 95% CI | Confounding Bias |
|---|---|---|---|---|
| Bank Incentive → Churn | +7.7% | +2.3% | [−1.5%, +6.0%] | 4.4 pp (57%) |
| Having Kids → Wine Spending | −$348 | −$95 | [−$152, −$7] | $253 (73%) |
| Training → Productivity (synthetic, true=6.0) | +7.7 | +6.65 | [4.5, 8.8] | Recovered ✅ |

---

## Repository Structure

```
Individual_JainamPatel_DoubleMachineLearning/
├── Causality_Notebook.ipynb          # Main Jupyter notebook (fully executable)
├── Example1_Churn/
│   ├── Customer-Churn-Records.csv    # Bank customer dataset (10,000 customers)
│   └── graphs/                       # Auto-saved visualizations
├── Example2_Marketing/
│   ├── marketing_campaign.csv        # Consumer spending dataset (2,237 customers)
│   └── graphs/                       # Auto-saved visualizations
├── QuizQuestions.md                   # 15 multiple-choice questions with explanations
├── Video_Link.txt                    # Link to video presentation
├── README.md                         # This file
└── LICENSE                           # MIT License
```

---

## Datasets

### Dataset 1: Customer Churn Records
- **Source:** [Publicly available bank customer dataset](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)
- **Size:** 10,000 customers × 18 features
- **Treatment:** `HighBalanceIncentive` — probabilistic financial incentive targeting high-balance customers
- **Outcome:** `Exited` — whether the customer left the bank (binary)
- **Confounders:** Age, CreditScore, Tenure, Geography, Gender
- **Causal question:** Does offering financial incentives to high-balance customers reduce churn?

### Dataset 2: Marketing Campaign
- **Source:** [Publicly available bank customer dataset](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign)
- **Size:** 2,237 customers × 29 features (after cleaning)
- **Treatment:** `Has_Kids` — whether the customer has young children at home (binary)
- **Outcome:** `MntWines` — annual wine spending in dollars (continuous)
- **Confounders:** Income, Age, Education, Marital Status
- **Causal question:** How much does having children affect wine spending, after controlling for income and demographics?

---

## Methodology

### Double Machine Learning (DML)

DML is a two-stage procedure:

1. **Stage 1 (Nuisance estimation):** Use ML models to predict the outcome from confounders (E[Y|X]) and predict treatment from confounders (E[T|X]). Compute residuals for both.
2. **Stage 2 (Causal estimation):** Regress outcome residuals on treatment residuals. The coefficient is the Average Treatment Effect (ATE).

**Why DML over traditional regression:** DML handles high-dimensional confounders, captures non-linear relationships, and eliminates regularization bias through Neyman-orthogonal cross-fitting.

### Implementation Details

- **Library:** EconML (Microsoft Research) — `LinearDML` with `discrete_treatment=True`
- **Stage 1 models:** Gradient Boosting (Example 1), Random Forest (Example 2)
- **Cross-fitting:** 5-fold sample splitting for debiased inference
- **Inference:** Bootstrap confidence intervals (100 resamples)
- **Diagnostics:** Propensity score overlap, covariate balance (SMD), heterogeneous treatment effects (CATE)

### Robustness Checks

Both examples include sensitivity analyses:
- **Alternative ML models:** Random Forest vs. Gradient Boosting
- **Alternative confounder sets:** Minimal (2 vars) vs. full specification
- **Placebo test (Example 1):** Using Satisfaction Score as a falsification outcome — null result confirms no spurious effects

---

## Key Findings

### Example 1: Bank Customer Churn
- **Naive correlation:** High-balance customers churn 7.7% more (confounded by age/demographics)
- **DML causal estimate:** +2.3% (CI: [−1.5%, +6.0%]) — not statistically significant at 5%
- **Confounding correction:** DML removes 57% of the naive estimate as spurious
- **Heterogeneity:** Treatment effect increases with age and credit score
- **Sensitivity:** Estimates stable across model specifications (GB: 2.3%, RF: 2.5%)

### Example 2: Consumer Wine Spending
- **Naive correlation:** Parents spend $348 less on wine (confounded by income)
- **DML causal estimate:** −$95 (CI: [−$152, −$7]) — statistically significant
- **Confounding correction:** 73% of the naive estimate was due to income/age differences
- **Heterogeneity:** Effect varies by income — stronger for higher-income families
- **Sensitivity:** Estimates stable (RF: −$95, GB: −$86, Minimal: −$95)

### Synthetic Validation
- **True ATE:** 6.0 (built into the data-generating process)
- **Naive estimate:** 7.7 (upward biased by confounding)
- **DML estimate:** 6.65 (CI: [4.5, 8.8]) — correctly recovers the truth
- **Heterogeneity:** DML correctly identifies age-varying effects

---

## Notebook Structure

| Section | Content |
|---|---|
| Part 1 | Why correlation ≠ causation: confounding, backdoor paths |
| Part 2 | DML theory: two-stage residualization, Neyman orthogonality, cross-fitting |
| Part 2.5 | Data preparation: variable selection (confounders vs. mediators vs. colliders), missing data, encoding |
| Part 3 | Setup: imports, helper functions |
| Example 1 | Churn analysis: DAG → EDA → DML → naive vs. causal → CATE → balance → sensitivity |
| Example 2 | Wine spending: DAG → EDA → DML → naive vs. causal → CATE → balance → sensitivity |
| Cross-Dataset | Comparing heterogeneity patterns across datasets |
| Synthetic | Ground-truth validation with known causal structure |
| Conclusion | Key findings, assumptions, limitations, future directions |

---

## How to Run

### Prerequisites

```bash
pip install econml scikit-learn pandas numpy matplotlib seaborn graphviz scipy
```

### Execution

1. Clone this repository
2. Open `Causality_Notebook.ipynb` in Jupyter Notebook or JupyterLab
3. Run **Kernel → Restart & Run All**
4. All outputs (figures, tables, estimates) will regenerate

**Expected runtime:** ~10–15 minutes (DML fitting + bootstrap inference)

**Python version:** 3.8+  
**Key dependencies:** EconML ≥ 0.14, scikit-learn ≥ 1.3, pandas ≥ 1.5

---

## Visualizations

The notebook produces 14 publication-quality visualizations including:

- **DAGs** for each causal model (3 total)
- **Confounding diagnostics:** KDE density plots, propensity score overlap
- **Naive vs. DML comparison** bar charts with confidence intervals
- **Heterogeneous treatment effects:** Scatter plots with smoothed trends + binned bar charts
- **Covariate balance:** Standardized mean difference plots
- **Cross-dataset comparison:** Standardized CATE patterns by age
- **Synthetic validation:** DML recovery of known ground truth

All figures auto-save to `Example1_Churn/graphs/` and `Example2_Marketing/graphs/`.

---

## Limitations

1. **Unconfoundedness is untestable:** We assume all important confounders are measured. Unmeasured confounders would bias estimates.
2. **Example 1 treatment is synthetic:** The probabilistic incentive is constructed from observed data, not a real randomized intervention. External validity is limited.
3. **Example 2 treatment is non-manipulable:** "Having children" cannot be assigned — results are descriptive conditional effects, not policy-actionable.
4. **LinearDML assumes linear treatment effect model:** Heterogeneity is captured through linear functions of covariates. CausalForestDML would allow non-linear effect surfaces.
5. **No formal sensitivity analysis for unmeasured confounding:** E-values or Cinelli & Hazlett (2020) bounds would strengthen the analysis.

---

## References

1. Chernozhukov, V., et al. (2018). "Double/debiased machine learning for treatment and structural parameters." *The Econometrics Journal*, 21(1), C1–C68.
2. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
3. Rubin, D. B. (1974). "Estimating causal effects of treatments in randomized and nonrandomized studies." *Journal of Educational Psychology*, 66(5), 688–701.
4. Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
5. Battocchi, K., et al. (2019–2024). *EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation*. Microsoft Research.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Video Presentation

Link: https://youtu.be/1vb76ej6o_g

---

## Contact

**Jahnavi Patel**  
Northeastern University  
patel.jahnavi@northeastern.edu
