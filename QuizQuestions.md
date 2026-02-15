# Quiz Questions: Double Machine Learning for Causal Inference

**Course:** INFO 7390 — Crash Course in Causality  
**Author:** Jahnavi Patel  
**Topic:** Double Machine Learning (DML) and Data Preparation for Causal Analysis

---

## Question 1: Confounding Variables

A bank observes that customers enrolled in its loyalty program churn less than non-enrolled customers. The bank concludes the program *causes* lower churn. What is the most likely threat to this causal claim?

A) The sample size is too small for statistical significance  
B) Wealthier, more satisfied customers are more likely to both enroll and stay, creating confounding  
C) The loyalty program was poorly designed  
D) Churn is measured incorrectly  

**Correct Answer:** B

**Explanation:** A confounder is a variable (like wealth or satisfaction) that influences both treatment assignment (who enrolls) and the outcome (who churns). When such a variable exists, the naive comparison conflates the program's effect with pre-existing differences between groups. Option A is about power, not bias; C and D are about measurement, not confounding. Confounding bias is the fundamental reason correlation does not imply causation in observational studies.

---

## Question 2: Mediators vs. Confounders

In a study of whether a marketing email (Treatment) increases purchases (Outcome), a data scientist includes "number of website visits after receiving the email" as a control variable. What mistake are they making?

A) They are including a collider  
B) They are including a mediator, which blocks the causal pathway  
C) They are including an instrumental variable  
D) They are correctly adjusting for a confounder  

**Correct Answer:** B

**Explanation:** Website visits *after* the email are on the causal pathway: Email → Website Visits → Purchases. Controlling for a mediator blocks the mechanism through which the treatment operates, attenuating the estimated causal effect toward zero. A confounder must affect both treatment and outcome but not lie on the causal path between them. Option A (collider) requires both treatment and outcome to cause the variable. Option D is wrong because post-treatment variables should never be controlled for in causal analysis.

---

## Question 3: DML Two-Stage Process

In Double Machine Learning, what happens in Stage 1?

A) The causal effect is estimated by comparing treated and control group means  
B) Machine learning models predict the outcome and treatment from confounders, and residuals are computed  
C) A randomized controlled trial is simulated using bootstrapping  
D) Propensity scores are used to match treated and control units one-to-one  

**Correct Answer:** B

**Explanation:** DML's Stage 1 uses flexible ML models to estimate two nuisance functions: E[Y|X] (expected outcome given confounders) and E[T|X] (expected treatment given confounders). Residuals from both predictions represent the "surprise" variation not explained by confounders. Stage 2 then regresses outcome residuals on treatment residuals to isolate the causal effect. Option A describes a naive comparison. Option C is not what DML does. Option D describes propensity score matching, a different causal method entirely.

---

## Question 4: Why "Double"?

Why is Double Machine Learning called "double"?

A) It uses two different datasets for training and testing  
B) It estimates the causal effect twice and averages the results  
C) It uses machine learning twice — once to model the outcome and once to model the treatment  
D) It applies two layers of neural networks  

**Correct Answer:** C

**Explanation:** The "double" refers to the two ML models used in Stage 1: one predicting Y from X (outcome model) and one predicting T from X (treatment model). Both must be estimated to fully remove confounding — if only one is estimated, confounding remains in the other variable's residual. This double residualization is what gives DML its debiasing property. Options A, B, and D do not describe the DML methodology.

---

## Question 5: Cross-Fitting (Neyman Orthogonality)

DML uses cross-fitting (sample splitting) during estimation. What problem does this solve?

A) It increases the sample size by generating synthetic data  
B) It eliminates regularization bias that arises when the same data is used to estimate nuisance functions and the causal effect  
C) It ensures the treatment and control groups are balanced  
D) It corrects for multiple hypothesis testing  

**Correct Answer:** B

**Explanation:** When ML models are trained and used for prediction on the same data, they overfit — producing residuals that are artificially small and correlated with the estimation error. Cross-fitting (e.g., 5-fold) trains models on folds {1,2,3,4} and predicts on fold 5, ensuring predictions are out-of-sample. This eliminates regularization bias and achieves Neyman orthogonality, making the causal estimator robust to imperfect nuisance function estimation. This is a key theoretical contribution of Chernozhukov et al. (2018).

---

## Question 6: Positivity Assumption

What does the positivity (overlap) assumption require in causal inference?

A) The treatment must have a positive effect on the outcome  
B) All confounders must be positively correlated with the treatment  
C) For every combination of confounder values, there must be a non-zero probability of receiving either treatment or control  
D) The sample must contain more treated than control units  

**Correct Answer:** C

**Explanation:** Positivity requires 0 < P(T=1|X) < 1 for all X — meaning that at every point in the covariate space, both treated and untreated units exist. Without this, DML must extrapolate (guess) rather than interpolate (compare similar units), producing unreliable estimates. This is checked via propensity score overlap diagnostics. Option A confuses positivity with effect direction. Options B and D are unrelated to the assumption.

---

## Question 7: Collider Bias

A researcher studies whether exercise (Treatment) affects heart disease (Outcome). They control for "hospitalization status," which is caused by both exercise injuries and heart disease. What bias does this introduce?

A) Confounding bias  
B) Selection bias (collider bias)  
C) Measurement error  
D) Attrition bias  

**Correct Answer:** B

**Explanation:** Hospitalization is a collider: Exercise → Hospitalization ← Heart Disease. Conditioning on a collider opens a non-causal path between treatment and outcome, creating a spurious association. Among hospitalized patients, those without heart disease are disproportionately there due to exercise injuries, making exercise appear to *cause* heart disease. This is collider bias (a form of selection bias). The correct approach is to exclude colliders from the conditioning set entirely.

---

## Question 8: Interpreting the ATE

A DML model estimates an Average Treatment Effect (ATE) of −$85 for the effect of having children on wine spending, with a 95% confidence interval of [−$152, −$7]. Which interpretation is correct?

A) Every parent spends exactly $85 less on wine than non-parents  
B) On average, having children causally reduces wine spending by approximately $85, and we can be confident the true effect is negative  
C) The effect is not statistically significant because the confidence interval is wide  
D) 85% of parents reduced their wine spending  

**Correct Answer:** B

**Explanation:** The ATE is an average across the population — individual effects may vary (heterogeneity). The CI of [−$152, −$7] excludes zero, meaning the effect is statistically significant at the 5% level. Option A incorrectly treats the ATE as a constant. Option C is wrong because the CI does not include zero. Option D confuses the dollar magnitude with a percentage of the population.

---

## Question 9: Naive vs. Causal Estimate

A naive comparison shows parents spend $348 less on wine than non-parents. After DML adjustment for income, age, and education, the causal estimate is −$95. What does the $253 difference represent?

A) Measurement error in the wine spending variable  
B) The amount of confounding bias — the portion of the naive estimate attributable to differences in income, age, and education rather than having children  
C) The standard error of the estimate  
D) The effect of children on income  

**Correct Answer:** B

**Explanation:** The gap between the naive estimate (−$348) and the DML estimate (−$95) represents confounding bias: $253 of the naive difference was driven by the fact that parents systematically differ from non-parents in income, age, and education — not by having children per se. DML removes this bias by residualizing both treatment and outcome on confounders. This demonstrates exactly why causal methods are necessary for valid business decisions.

---

## Question 10: Post-Treatment Variables

When estimating the causal effect of having children on wine spending, which variable should be EXCLUDED from the confounder set?

A) Income (measured before having children)  
B) Age (demographic characteristic)  
C) Number of web visits per month (affected by having children)  
D) Education level (completed before having children)  

**Correct Answer:** C

**Explanation:** Web visits per month is a post-treatment variable — having children reduces available browsing time, so this variable is on the causal pathway (a mediator): Kids → Less Free Time → Fewer Web Visits → Different Purchases. Controlling for it blocks the causal mechanism and biases the estimate. Income, Age, and Education are pre-treatment variables that existed before the treatment (having children) and are appropriate confounders. The key rule: only condition on variables that temporally precede the treatment.

---

## Question 11: DAGs and Backdoor Criterion

In a DAG where Age → Treatment, Age → Outcome, and Treatment → Outcome, what is the minimum set of variables needed to identify the causal effect of Treatment on Outcome?

A) No adjustment needed — the effect is already identified  
B) Adjust for Age only  
C) Adjust for Age and Outcome  
D) Adjust for Treatment and Age  

**Correct Answer:** B

**Explanation:** Age creates a backdoor path: Treatment ← Age → Outcome. To block this path and identify the causal effect, we must condition on Age. This satisfies the backdoor criterion (Pearl, 2009). Option A is wrong because the backdoor path remains open. Option C is wrong because you never condition on the outcome variable. Option D is wrong because conditioning on the treatment itself is not part of confounder adjustment — the treatment is what we're estimating the effect of.

---

## Question 12: Heterogeneous Treatment Effects (CATE)

DML estimates that the average effect of a financial incentive on churn is +2.3%, but CATE analysis shows the effect ranges from −3% to +8% across customers. What does this heterogeneity imply?

A) The DML model is incorrectly specified and should be discarded  
B) The treatment effect varies across subgroups, and targeted interventions may be more effective than a one-size-fits-all approach  
C) The average effect is meaningless because it cancels out  
D) Confounding was not adequately controlled  

**Correct Answer:** B

**Explanation:** Heterogeneous treatment effects (CATEs) reveal that the incentive works differently for different customer segments. Some customers may benefit (negative effect on churn) while others experience no change or even increased churn. This information is valuable for targeting: focus the incentive on customers who respond positively. Option A is wrong — heterogeneity is expected, not a model failure. Option C is wrong because the ATE is still a valid population summary. Option D conflates heterogeneity with confounding.

---

## Question 13: Missing Data in Causal Contexts

A dataset has 15% missing values in the Income variable. Investigation shows that younger customers are more likely to skip the income question, regardless of their actual income. What type of missingness is this?

A) MCAR (Missing Completely at Random)  
B) MAR (Missing at Random)  
C) MNAR (Missing Not at Random)  
D) Selection bias  

**Correct Answer:** B

**Explanation:** Missingness depends on an observed variable (Age) but not on the missing value itself (Income). Among young customers, rich and poor skip equally — the missingness pattern is fully explained by Age. This is MAR. The correct approach is to impute Income using other variables (e.g., median imputation) AND create a missingness indicator (`Income_Missing`) as an additional confounder. MCAR (A) would mean missingness is completely random. MNAR (C) would mean low-income people specifically avoid reporting income because it is low.

---

## Question 14: Sensitivity Analysis

A researcher estimates ATE = 0.023 using Gradient Boosting and ATE = 0.025 using Random Forest in the DML first stage. They also find that a placebo test (using an unrelated outcome) returns a null result. What can they conclude?

A) The causal estimate is proven to be correct  
B) The estimate is robust to model specification, and the placebo test rules out spurious model artifacts, increasing confidence in the finding  
C) The two models disagree, so the results are unreliable  
D) The placebo test failure means all results are invalid  

**Correct Answer:** B

**Explanation:** Stability across ML model specifications (0.023 vs 0.025, <10% difference) suggests the causal estimate does not depend on a particular modeling choice. The placebo test — estimating the "effect" on a variable that treatment cannot plausibly affect — returning null confirms the model is not generating spurious effects. Together, these checks increase confidence without proving causation (which requires untestable assumptions). Option A overstates the conclusion. Option C misinterprets small differences as disagreement. Option D reverses the placebo logic — a null placebo *supports* validity.

---

## Question 15: Synthetic Data Validation

A researcher generates synthetic data where the true ATE is 6.0, the naive estimate is 7.7, and the DML estimate is 6.65 with 95% CI [4.5, 8.8]. What does this exercise demonstrate?

A) DML always overestimates treatment effects  
B) The naive estimate is more accurate than DML because 7.7 is closer to some other value  
C) DML successfully recovers the true causal effect within the confidence interval, while the naive estimate is upward-biased due to confounding  
D) Synthetic data is unreliable for validating methods  

**Correct Answer:** C

**Explanation:** The true ATE (6.0) falls within the DML confidence interval [4.5, 8.8], confirming that DML provides valid inference. The DML point estimate (6.65) is much closer to the truth than the naive estimate (7.7), which is biased upward because high-performing employees were more likely to receive training (confounding). This synthetic validation — where we know the ground truth — is the gold standard for method evaluation. It builds confidence that DML works correctly before applying it to real data where the truth is unknown. Option A is incorrect as the overestimation here is small and within normal sampling variation. Option D dismisses a widely accepted methodological practice.

---

