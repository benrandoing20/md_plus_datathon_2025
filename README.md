# ICU Outcomes Prediction in Orthopedic Surgery Patients

A machine learning framework for predicting composite long-term outcomes in ICU patients following orthopedic procedures (spine, hip, knee, shoulder/elbow, and hand surgeries).

## Overview

This repository contains predictive models and statistical analyses to identify patient and procedural factors associated with poor outcomes in orthopedic surgery patients requiring intensive care unit (ICU) admission. The analysis uses clinical data including ICD procedure codes, CPT codes, comorbidities, and temporal outcomes to develop risk stratification tools.

### Research Objectives

- **Primary Goal**: Predict composite long-term outcomes for ICU patients after orthopedic surgery
- **Secondary Goals**: 
  - Identify preoperative risk factors for ICU admission
  - Characterize post-surgical complications across clinical domains
  - Assess long-term mortality, readmission, and functional outcomes

### Key Features

- Multi-procedure orthopedic analysis (spine, hip, knee, shoulder/elbow, hand)
- Composite outcome modeling (90-day mortality, 7-day readmission, extended LOS)
- Machine learning models: Random Forest, Gradient Boosting, Logistic Regression
- Feature importance analysis using SHAP values
- Temporal outcome tracking (30, 90, 180, 365 days post-discharge)
- Propensity score matching and inverse probability of treatment weighting (IPTW)

## Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install pandas numpy matplotlib seaborn scikit-learn scipy
pip install shap  # Optional, for feature importance visualization
```

### Installation

```bash
git clone https://github.com/yourusername/md_plus_datathon_2025.git
cd md_plus_datathon_2025
```

### Basic Usage

Run the main analysis script:

```bash
python 04_icu_ortho_all_procedures_analysis.py
```

This will:
1. Load and preprocess patient data
2. Engineer features from clinical variables
3. Train predictive models on ICU patients
4. Generate evaluation metrics and visualizations
5. Output results to `analysis_output/` directory

### Expected Outputs

- **Trained models**: Random Forest, Gradient Boosting, Logistic Regression classifiers
- **Performance metrics**: ROC curves, AUC scores, precision-recall curves, calibration plots
- **Feature importance**: SHAP summary plots and individual feature contributions
- **Statistical reports**: Classification metrics, confusion matrices, cross-validation scores

## Project Structure

```
md_plus_datathon_2025/
│
├── 04_icu_ortho_all_procedures_analysis.py  # Main analysis pipeline
├── stats/                                    # Legacy statistical analyses (archived)
├── README.md                                 # This file
└── analysis_output/                          # Generated results (created at runtime)
    ├── figures/                              # Visualizations
    ├── models/                               # Saved model objects
    └── reports/                              # Statistical summaries
```

## Methodology

### Cohort Definition

**Inclusion Criteria:**
- Patients ≥18 years who underwent orthopedic surgery (ICD procedure codes + CPT codes)
- Procedures: spine, hip, knee, shoulder/elbow, hand
- Required ICU admission during index hospitalization

**Exclusion Criteria:**
- Non-surgical orthopedic encounters (diagnostic/observation only)
- Missing critical outcome data (discharge status, mortality, readmission)

### Composite Outcome Definition

**Poor Outcome** (any of the following):
- Mortality within 90 days of discharge
- Hospital readmission within 7 days of discharge
- Index hospitalization length of stay >7 days

**Good Outcome**: Survived ≥90 days AND no 7-day readmission AND LOS ≤7 days

### Feature Engineering

**Demographic Features:**
- Age at surgery, sex, race, insurance type

**Clinical Features:**
- Comorbidity burden (total diagnosis count)
- Specific comorbidities: cardiac disease, hypertension, diabetes, pulmonary disease, renal disease, obesity
- Procedure type and anatomical location
- Emergency vs. elective admission

**Temporal Features:**
- Length of stay (hospital and ICU)
- Time to readmission
- Follow-up duration

### Statistical Methods

- **Univariate Analysis**: Chi-square tests, t-tests, Mann-Whitney U tests
- **Multivariable Models**: Logistic regression with L2 regularization
- **Machine Learning**: Random Forest, Gradient Boosting with hyperparameter tuning
- **Model Evaluation**: AUC-ROC, precision-recall, calibration curves, cross-validation
- **Feature Selection**: Variance inflation factor (VIF) for multicollinearity, SHAP for importance

### Model Training

```python
# Stratified train/test split (80/20)
# Class weight adjustment for imbalanced outcomes
# 5-fold cross-validation for hyperparameter tuning
# Sample weighting for robust predictions
```

## Results Interpretation

### Model Performance Metrics

- **AUC-ROC**: Discrimination ability (0.5 = random, 1.0 = perfect)
- **Precision-Recall**: Performance on imbalanced datasets
- **Calibration**: Agreement between predicted probabilities and observed outcomes
- **Feature Importance**: Clinical variables driving predictions (via SHAP)

### Clinical Significance

High-risk patients identified by the model may benefit from:
- Enhanced preoperative optimization
- Extended perioperative monitoring
- Early intervention protocols
- Targeted discharge planning

## Data Requirements

The analysis expects CSV files with the following structure:

**Patients** (`patients.csv`):
- `patient_id`, `age`, `sex`, `race`

**Admissions** (`admissions.csv`):
- `admission_id`, `patient_id`, `admission_date`, `discharge_date`, `discharge_disposition`

**Procedures** (`procedures.csv`):
- `admission_id`, `icd_code`, `cpt_code`, `procedure_date`, `procedure_description`

**Diagnoses** (`diagnoses.csv`):
- `admission_id`, `icd_code`, `diagnosis_description`

**ICU Stays** (`icu_stays.csv`):
- `admission_id`, `icu_admit_date`, `icu_discharge_date`, `icu_los_days`

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add your feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@software{icu_ortho_prediction_2025,
  author = {Your Name/Team},
  title = {ICU Outcomes Prediction in Orthopedic Surgery Patients},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/md_plus_datathon_2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions, issues, or collaboration inquiries:

- **Primary Contact**: [Your Name] - your.email@institution.edu
- **Institution**: [Your Institution]
- **Research Group**: [Lab/Department Name]

## Acknowledgments

- Data source: [Hospital/Database Name]
- Funding: [Grant Information, if applicable]
- Computational resources: [Cluster/Cloud Platform, if applicable]

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'shap'`  
**Solution**: SHAP is optional. Install with `pip install shap` or the analysis will continue without SHAP visualizations.

**Issue**: `FileNotFoundError: data/patients.csv not found`  
**Solution**: Ensure data files are in the expected directory structure or update file paths in the script.

**Issue**: Low model performance (AUC < 0.6)  
**Solution**: Check for data quality issues, class imbalance, or insufficient sample size in the ICU cohort.

## Version History

- **v1.0.0** (2025-11-14): Initial release with composite outcome modeling
- **v0.1.0** (2025-10-01): Development version with exploratory analyses

---

**Status**: Active Development  
**Last Updated**: November 14, 2025
