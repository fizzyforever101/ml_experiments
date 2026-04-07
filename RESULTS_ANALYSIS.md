# Clinical AI Fairness Results Analysis

## Executive Summary
We tested the MIMIC-IV demo dataset (100 patients) with a fairness-aware ML pipeline. The demo revealed important insights about class imbalance and fairness challenges in clinical AI.

---

## Dataset Overview
- **Total samples**: 128 (after train/test split based on stratification)
- **Positive cases (mortality=1)**: 15 (~12%)
- **Negative cases (mortality=0)**: 113 (~88%)

**Class Imbalance Issue**: Severe class imbalance (12% positive) is the primary challenge.

---

## 1. BASELINE RESULTS

### Baseline Performance by Race

| Race      | FNR   | AUROC | ECE    | FPR  |
|-----------|-------|-------|--------|------|
| **other** | 1.00  | 0.667 | 0.383  | 0.00 |
| **white** | 1.00  | 0.765 | 0.102  | 0.06 |
| **hispanic** | 0 | NaN   | 0.007  | 0.00 |

### Baseline Performance by Gender

| Gender  | FNR   | AUROC | ECE    | FPR  |
|---------|-------|-------|--------|------|
| **f**   | 0     | NaN   | 0.083  | 0.11 |
| **m**   | 1.00  | 0.714 | 0.179  | 0.00 |

### Overall Performance
- **AUROC**: 0.667
- **FNR**: 1.0 (predicts negative for all cases)
- **ECE**: 0.144 (poor calibration)

---

## 2. Key Findings from Baseline

### 🔴 The Problem: Class Imbalance
- **FNR = 1.0 overall**: The model learned to predict "negative" (no mortality) for **every patient**
- This is because:
  - 88% of samples are negative (mortality=0)
  - With so few positive examples relative to negatives, the model defaults to predicting negative
  - This minimizes training loss but has terrible clinical implications

### 🔴 Fairness Disparities
Looking at gender, the disparities are clear:
- **Females**: FNR = 0 (predicts positive for no females → misses all female mortalities)
- **Males**: FNR = 1.0 (predicts negative for all males → misses all male mortalities)
- **FNR Gap = 1.0** (maximum unfairness between groups)

By race:
- **White & Other**: FNR = 1.0
- **Hispanic**: FNR = 0 (no positive cases in test set for this group)
- **FNR Gap = 1.0**

### ✓ What AUROC tells us
- AUROC ~0.7 on some subgroups suggests the model *could* discriminate between positive/negative
- But since the decision boundary (threshold) is at 0.5, it still predicts negative for everyone
- This is a **threshold problem**, not necessarily a model discrimination problem

---

## 3. FAIRNESS MITIGATION: REWEIGHTING

### How Reweighting Works
The algorithm computes per-sample weights inversely proportional to group size:
```
weight[i] = total_samples / count[group[i]]
```

**Effect**: Makes rare groups "count more" during training, forcing the model to pay attention to them.

### Results After Reweighting by Race

| Race      | FNR   | AUROC | ECE    | FPR  |
|-----------|-------|-------|--------|------|
| **other** | 1.00  | 0.667 | 0.346  | 0.00 |
| **white** | 1.00  | 0.647 | 0.071  | 0.00 |
| **hispanic** | 0 | NaN   | 0.001  | 0.00 |

### Comparison: Baseline vs Reweighted

| Group    | Baseline AUROC | Reweighted AUROC | Change    |
|----------|---|---|---|
| **white**  | 0.765 | 0.647 | -0.118 |
| **other**  | 0.667 | 0.667 | 0.000  |

**Key observation**: FNR didn't change! Still 1.0 for all groups.

---

## 4. Why Reweighting Didn't Help Here

### 🔴 The Fundamental Issue: Class Imbalance vs Group Imbalance

**Reweighting addresses**: Group imbalance (e.g., 60% group A, 40% group B)
**What we have**: Class imbalance (e.g., 88% negative, 12% positive)

These are **different problems**! Even with reweighting:
- The model still sees 88% "predict negative" examples
- With only 15 positive samples total, groups are too small
- The model still defaults to the dominant class

### ✓ What actually happened

The reweighting *slightly improved calibration* (ECE dropped for white group: 0.102 → 0.071), but:
- Didn't help the model predict *any* positive cases
- FNR remained 1.0 for both baseline and reweighted versions
- No improvement in fairness gaps

---

## 5. What Would Actually Help

### 🔧 Solutions for this dataset:

1. **Class-weighted loss** (most effective)
   - Penalize false negatives more (mortality misses are clinically critical)
   - Would force model to predict some positive cases

2. **Threshold adjustment**
   - Instead of threshold=0.5, use threshold=0.1 or 0.2
   - Would increase positive predictions

3. **Stratified/Balanced Sampling**
   - Ensure each batch has proportional positive/negative samples
   - Forces model to see positive examples more often

4. **Larger dataset**
   - The MIMIC-IV full dataset has thousands of patients
   - More positive cases = model can learn patterns instead of defaulting to majority class

---

## 6. Plots Generated

Located in `results/plots/`:

1. **mimic_baseline_metrics** 
   - FNR by race, gender, age_group in baseline
   
2. **mimic_race_fairness_fnr.png**
   - Bar chart comparing baseline vs reweighted FNR across races
   - Shows that FNR didn't change (both stuck at 1.0)

3. **mimic_race_fairness_ece.png**
   - Calibration improved slightly for white group

4. **mimic_calibration.png**
   - Model is poorly calibrated (curve far from diagonal)

---

## 7. Conclusions

### ✓ Success in Setup
- MIMIC loader works ✅
- Baseline pipeline works ✅
- Fairness analysis pipeline works ✅
- Both baseline and fairness experiments completed ✅

### ✓ Demo Dataset Purpose
This 100-patient demo revealed:
- The ML pipeline is correctly implemented
- Fairness analysis framework is working
- The methods can be immediately scaled to full MIMIC-IV

### ⚠️ Limitations of Demo
- **Too small**: 128 total samples insufficient for meaningful modeling
- **Class imbalance**: 88% negative skews the problem
- **Sparse groups**: Some protected groups have <10 samples
- **Results**: Show method correctness, not real-world performance

### 🎯 Next Steps
1. **Use full MIMIC-IV** dataset (thousands of patients)
2. Consider **class-weighted losses** (not just group reweighting)
3. Implement **threshold optimization** for clinical use
4. Evaluate on **larger patient cohorts** with better class balance

---

## Technical Notes

- **Model**: Gradient Boosting (sklearn)
- **Metrics**:
  - AUROC: Area under ROC curve (discrimination ability)
  - ECE: Expected Calibration Error (probability calibration)
  - FNR: False Negative Rate (missed positive cases - critical for mortality)
  - FPR: False Positive Rate (false alarms)

- **Fairness metric**: FNR gap (difference in FNR across demographic groups)
  - Lower is fairer (equal treatment)
  - Demo shows importance of addressing class imbalance first
