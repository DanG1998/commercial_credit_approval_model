# Commercial Credit Approval Model

A credit approval model for commercial trade credit. An L2-regularized (Ridge) logistic regression estimates, for each credit request, the probability that the credit line is granted. The estimate draws on firm-level characteristics and the macroeconomic context, and a second decision layer applies business rules that the lender can configure.

The project was developed as a university case study (Data Science for Economics and Business, Università Politecnica delle Marche) to support the credit decision process of a company operating in the leasing sector.

## Problem

Granting trade credit means balancing two kinds of error:

- **False positives**: credit granted to risky firms, which can lead to losses.
- **False negatives**: reliable firms rejected, which means lost business.

Larger credit lines amplify both. In the dataset the approval rate is roughly constant across amount bands (about 80%), but the amount of money at stake in the €10,000–20,000 band is more than four times that of the base band. A wrong approval there can wipe out the margin earned on many smaller lines.

## Data

About 5,800 historical credit requests. Each request includes:

| Group | Variables |
|---|---|
| Target | `Esito_finale`: credit granted (1) / not granted (0) |
| Firm | requested amount, employees, number of properties, firm age, ITP (payment timeliness index, average days of delay) |
| Relationship | `REVISIONE` (existing client under review), `AFFIDATA` (credit lines already active) |
| Macroeconomic | expected inflation in Italy at 12 months (Bank of Italy), expected GDP growth (ISTAT) |
| Rules layer | activity status, legal form |

> **Note:** the original dataset is confidential and is not included in this repository.

## Methodology

### 1. Data preparation
- Removal of duplicates and of data-entry errors (`ITP = -999`).
- Firm age computed in days, as the difference between the score calculation date and the activity start date. Inactive or newly registered firms with no start date get age 0.
- Log transformation (`log1p`) of skewed variables: firm age, ITP, employees, number of properties.
- Exclusion of redundant or leaking columns, such as the external scores and the amount band. Revenue is also excluded, because about 40% of its values are missing and it correlates strongly with employees.
- Multicollinearity check with VIF.

### 2. Model
- Holdout set (20%, stratified) kept aside for final validation.
- Train/test split (70/30, stratified) on the remaining data.
- Continuous variables standardized with `StandardScaler`, fitted on the training set only. Binary and macroeconomic variables are left on their original scale.
- `LogisticRegressionCV` with L2 penalty, 5-fold cross-validation over 10 values of `C`. Best `C ≈ 0.046`.
- Learning curves used to check for over- or underfitting.

### 3. Decision threshold
Because the target is imbalanced (about 80% approvals), a request is classified as approvable only when the predicted probability is **≥ 0.80**. This is a more conservative criterion than the default 0.50. The threshold is a parameter and should be calibrated with the lender according to its risk appetite.

### 4. Business rules layer
The model output then goes through a configurable decision engine:

1. **Blocking filters (KO):** requests from firms that are inactive, withdrawn, suspended, ceased or under insolvency proceedings are rejected automatically.
2. **Dynamic thresholds by legal form:** the approval threshold depends on the legal protection and assets behind the applicant.

| Legal form | Threshold |
|---|---|
| Limited companies (S.r.l., S.p.A.) | 0.68 |
| Partnerships | 0.75 |
| Other (cooperatives, entities, foundations) | 0.80 |
| Sole proprietorships / self-employed | 0.88 |

These values are illustrative and are meant to be set by the lender.

## Results

**Main drivers** (Ridge coefficients):

- **Positive:** previous credit relationship (`REVISIONE`, by far the strongest factor), number of employees, firm age, number of properties.
- **Negative:** ITP (late payments are the earliest warning sign), requested amount, credit lines already active.
- **Macroeconomic variables:** close to zero impact. They do not change the risk assessment on their own.

**Confusion matrices on the holdout set:**

| | Model only | Model + business rules |
|---|---|---|
| True negatives | 76 | 52 |
| False positives | 70 | 94 |
| False negatives | 147 | 106 |
| True positives | 580 | 621 |

Adding the rules makes the system more permissive. It recovers 41 reliable firms that the model alone would have rejected, at the cost of admitting 24 more firms that should not have been financed. That is roughly 1.7 good clients gained for every bad one accepted.

The script also compares the model's probabilities with the lender's internal alphanumeric rating, decoded to a 0–1 scale, to find the cases where the two disagree most.

## Repository structure

```
├── regressione_binaria.py   # full pipeline: preparation, model, rules, charts
├── report.pdf               # final report (in Italian)
└── README.md
```

## Authors

- Daniel Galizia
- Martina Leporati
- Chiara Paoloni
- Matheus Soares Campos

Università Politecnica delle Marche, March 2026
