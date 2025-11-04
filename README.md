# md_plus_datathon_2025
Flare Prediction from Chronic Illness Dataset

## Analysis Categories

### Category 1: Preoperative Determinants of Intensive Care
**Focus:** Cohort construction, feature engineering, univariate/multivariable analysis, predictive modeling for ICU admission

- **verify_true_spine_cohort.py** - Validates spine surgery cohort definition and ensures exclusion of non-spine procedures (hip/knee), critical for proper cohort construction
- **comprehensive_icu_analysis.py** - Implements the complete predictive modeling pipeline: feature engineering, train/test split, class weights, regularization, cross-validation, and model evaluation (Random Forest, Gradient Boosting, Logistic Regression)
- **procedure_stratified_analysis.py** - Implements the procedure-type filtering approach described in your exclusion criteria, ensuring ≥10 patients per procedure category in both ICU/non-ICU groups to control for confounding

### Category 2: Long-Term Outcomes Analysis
**Focus:** Longitudinal follow-up, mortality, readmissions, new diagnoses, discharge disposition, IPTW adjustment

- **icu_outcomes_analysis.py** - Assesses mortality, readmissions, and adverse outcomes using propensity score matching and IPTW adjustment methods described in your methods
- **true_long_term_outcomes_analysis.py** - Analyzes unambiguous temporal outcomes at 30, 90, 180, and 365 days post-discharge including mortality and readmissions with proper follow-up time calculations
- **extended_long_term_factors_analysis.py** - Identifies new diagnoses post-discharge, calculates multi-system organ involvement (Si), and implements composite measures (MAE, PO) as described in your methods

### Category 3: Post-Surgical Complications
**Focus:** 27 predefined complications across 7 clinical domains, ICD-based identification, complication rates and odds ratios

- **specific_complications_analysis.py** - Implements systematic evaluation of complications across cardiopulmonary, CNS, renal, electrolyte, GI, infectious, and surgical domains with ICD code and keyword matching as described

### Category 4: Prognostic Factors in ICU Patients
**Focus:** Analysis restricted to ICU patients, predictors of good vs poor outcomes within ICU cohort

- **icu_prognostic_factors_analysis.py** - Restricts analysis to the 104 ICU patients, implements composite poor outcome definition, performs univariate and multivariable logistic regression with standardized predictors

### Support/Infrastructure Files
*(Not part of core methods but essential for analysis)*

- **verify_setup.py** - Environment and data validation
- **paper_analysis_figures.py** - Publication-quality figure generation
- **create_patient_data_visual.py** - Data structure visualization
- **examine_patient_examples.py** - Individual patient data verification

---

## Section 1: Preoperative Determinants of Intensive Care

**Navigation Flow:**  
Begin with `verify_true_spine_cohort.py` to validate your cohort definition and ensure exclusion of non-spine procedures (hip/knee replacements). This script compares an "OLD" filter (lines 39-48) that incorrectly includes joint replacements against a "NEW" filter (lines 70-130) that uses both ICD body part characters and procedure title keywords to exclude non-spine surgeries.

Next, critically, run `procedure_stratified_analysis.py` BEFORE `comprehensive_icu_analysis.py` to implement your methods' exclusion criteria. This file (lines 92-150) categorizes procedures into clinically meaningful groups, calculates representation scores (minimum of ICU and non-ICU counts), and filters to retain only procedures with ≥10 patients in both groups.

Finally, execute `comprehensive_icu_analysis.py`, which implements the complete predictive modeling pipeline:
- Feature engineering (Section 2, lines 120-250)
- Train/test split with stratified sampling (Section 3, lines 250-350)
- Class weight calculation and VIF multicollinearity checks (lines 400-450)
- Logistic regression with L2 regularization and 5-fold cross-validation (Section 4, lines 450-600)
- Model evaluation with AUC-ROC and calibration curves (Section 5, lines 600-750)


---

## Section 2: Long-Term Outcomes Analysis

**Navigation Flow:**  
This section represents an iterative refinement process with five complementary analyses.

1. **Start with `icu_outcomes_analysis.py`** (583 lines), which establishes the foundation by building the full cohort (ICU and non-ICU), identifying basic outcomes (mortality, readmissions, cardiac/pulmonary complications), and implementing propensity score matching with IPTW adjustment (Section 4, lines 300-450).

2. **Then execute `true_long_term_outcomes_analysis.py`**, which shifts strategy entirely to avoid diagnosis timing ambiguity by focusing exclusively on unambiguous temporal outcomes with clear timestamps: hospital LOS, mortality at 30/90/180/365 days, and readmissions at multiple intervals (Section 2, lines 100-250).

3. **Follow with `extended_long_term_factors_analysis.py`** to identify additional long-term factors:
   - New chronic conditions developed post-discharge (comparing index vs. subsequent encounters, Section 4, lines 200-350)
   - Multi-system organ involvement calculations (Si = count of affected organ systems, Section 5, lines 350-450)
   - Composite adverse outcomes (MAE and PO definitions, Section 6, lines 450-550)


---

## Section 3: Post-Surgical Complications

**Navigation Flow:**  
Execute `specific_complications_analysis.py` as a single comprehensive analysis. This 612-line script systematically evaluates 27 predefined complications organized into 7 clinical domains (Section 2, lines 60-150 defines the complication mappings).

The code:
- Loads diagnoses with ICD descriptions (lines 50-80)
- Defines each complication using both ICD code prefixes (e.g., ICD-9 518.81/518.82/518.84 and ICD-10 J96 for respiratory failure) and keyword searches in the long_title field (Section 3, lines 150-250)
- For each complication c, calculates rates per group (pc,g = Σ Cicg / Ng), odds ratios using the Woolf method with 95% CIs, and performs chi-square or Fisher's exact tests depending on expected cell counts (Section 4, lines 250-400)
- Results are stratified by statistical significance (p < 0.001, p < 0.01, p < 0.05) and organized by clinical domain (cardiopulmonary, CNS, renal, etc.) in Section 5-6 (lines 400-550)
- Generates comprehensive visualizations showing complication rates, odds ratios, risk differences, and domain-specific patterns

---

## Section 4: Prognostic Factors in ICU Patients

**Navigation Flow:**  
Run `icu_prognostic_factors_analysis.py` as a standalone analysis restricted to the 104 ICU patients.

The script:
- Begins by loading the ICU cohort from `cohort_elective_spine_icu.csv` (lines 20-40) and merging with patient demographics, admissions data for discharge information, and diagnoses for comorbidities (Section 1, lines 40-100)
- Section 2 (lines 100-200) defines the composite poor outcome (POi = 1 if any of: 90-day mortality, ≥2 readmissions in 1 year, facility discharge, ≥5 new diagnoses, or LOS > 75th percentile)
- Baseline variables are extracted from the index hospitalization including age, sex, total diagnosis count, specific comorbidities, and ICU length of stay (Section 3, lines 200-280)
- Continuous variables are standardized to z-scores (Xj* = (Xj - X̄j)/sj) in Section 4 (lines 280-320) to enable coefficient comparability
- Section 5 (lines 320-450) performs univariate testing (t-tests for continuous, chi-square/Fisher's for categorical variables), followed by multivariable logistic regression with 9 baseline predictors:
  
  **logit(P(GoodOutcome)) = β0 + β1(Age) + β2(Sex) + β3(Diagnoses) + β4(Cardiac) + β5(Hypertension) + β6(Diabetes) + β7(Pulmonary) + β8(Renal) + β9(ICU_LOS)**

- Adjusted odds ratios, 95% CIs, and AUC are calculated in Section 6 (lines 450-520), with visualizations showing outcome distributions, predictor effects, and model performance
