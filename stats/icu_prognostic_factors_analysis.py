"""
Analysis 2: Prognostic Factors in ICU Patients
Identifies factors associated with better vs worse long-term outcomes in ICU patients only
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PROGNOSTIC FACTORS IN ICU PATIENTS")
print("What Predicts Better Outcomes Among ICU Patients?")
print("="*80)

# Load data
print("\n[1/6] Loading data...")
icu_cohort = pd.read_csv('cohort_elective_spine_icu.csv')
print(f"✓ ICU cohort: {len(icu_cohort)} patients")

# Load patients table for demographics and mortality
print("Loading patients table...")
patients = pd.read_csv('physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz', compression='gzip')

# Merge demographics if not present
if 'gender' not in icu_cohort.columns or 'anchor_age' not in icu_cohort.columns:
    icu_cohort = icu_cohort.merge(patients[['subject_id', 'gender', 'anchor_age', 'dod']], 
                                   on='subject_id', how='left')
else:
    icu_cohort = icu_cohort.merge(patients[['subject_id', 'dod']], on='subject_id', how='left')

print("Loading admissions for readmission data...")
admissions = pd.read_csv('physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz', compression='gzip')
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])

# Merge discharge info from admissions
icu_cohort = icu_cohort.merge(
    admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'discharge_location']],
    on=['subject_id', 'hadm_id'], how='left', suffixes=('', '_adm')
)

# Use merged columns or keep existing ones
if 'dischtime_adm' in icu_cohort.columns:
    icu_cohort['dischtime'] = icu_cohort['dischtime_adm']
if 'admittime_adm' in icu_cohort.columns:
    icu_cohort['admittime'] = icu_cohort['admittime_adm']
    
icu_cohort['dischtime'] = pd.to_datetime(icu_cohort['dischtime'])
icu_cohort['admittime'] = pd.to_datetime(icu_cohort.get('admittime', icu_cohort['intime']))
icu_cohort['dod'] = pd.to_datetime(icu_cohort['dod'])

print("Loading diagnoses...")
diag_chunks = []
for chunk in pd.read_csv('physionet.org/files/mimiciv/3.1/hosp/diagnoses_icd.csv.gz',
                          chunksize=100000, compression='gzip'):
    diag_chunks.append(chunk[chunk['subject_id'].isin(icu_cohort['subject_id'])])
diagnoses = pd.concat(diag_chunks, ignore_index=True)

# Load ICD descriptions
d_icd_diag = pd.read_csv('physionet.org/files/mimiciv/3.1/hosp/d_icd_diagnoses.csv.gz',
                         compression='gzip')
diagnoses = diagnoses.merge(d_icd_diag[['icd_code', 'icd_version', 'long_title']],
                            on=['icd_code', 'icd_version'], how='left')

print(f"✓ Loaded {len(diagnoses)} diagnoses for ICU patients")

print("\n" + "="*80)
print("[2/6] DEFINING OUTCOME MEASURES")
print("="*80)

# Define good vs poor outcomes
print("\nCalculating outcome measures...")

# 1. Mortality outcomes
icu_cohort['died_in_hospital'] = icu_cohort['dod'].notna() & (icu_cohort['dod'] <= icu_cohort['dischtime'])
icu_cohort['days_to_death'] = (icu_cohort['dod'] - icu_cohort['dischtime']).dt.days

icu_cohort['died_90day'] = (icu_cohort['days_to_death'] <= 90) & (icu_cohort['days_to_death'] >= 0)
icu_cohort['died_1year'] = (icu_cohort['days_to_death'] <= 365) & (icu_cohort['days_to_death'] >= 0)

# 2. Readmission outcomes
readmissions = []
for idx, row in icu_cohort.iterrows():
    subj_readmits = admissions[
        (admissions['subject_id'] == row['subject_id']) &
        (admissions['hadm_id'] != row['hadm_id']) &
        (admissions['admittime'] > row['dischtime'])
    ]
    
    if len(subj_readmits) > 0:
        first_readmit = subj_readmits.sort_values('admittime').iloc[0]
        days_to_readmit = (first_readmit['admittime'] - row['dischtime']).days
        readmissions.append({
            'subject_id': row['subject_id'],
            'hadm_id': row['hadm_id'],
            'readmitted_30day': days_to_readmit <= 30,
            'readmitted_90day': days_to_readmit <= 90,
            'readmitted_1year': days_to_readmit <= 365,
            'num_readmissions_1year': len(subj_readmits[
                (subj_readmits['admittime'] - row['dischtime']).dt.days <= 365
            ])
        })
    else:
        readmissions.append({
            'subject_id': row['subject_id'],
            'hadm_id': row['hadm_id'],
            'readmitted_30day': False,
            'readmitted_90day': False,
            'readmitted_1year': False,
            'num_readmissions_1year': 0
        })

readmit_df = pd.DataFrame(readmissions)
icu_cohort = icu_cohort.merge(readmit_df, on=['subject_id', 'hadm_id'], how='left')

# 3. Discharge disposition
icu_cohort['discharged_home'] = icu_cohort['discharge_location'].str.contains('HOME', case=False, na=False)

# 4. Hospital LOS
icu_cohort['admittime'] = pd.to_datetime(icu_cohort['admittime'])
icu_cohort['los_days'] = (icu_cohort['dischtime'] - icu_cohort['admittime']).dt.days
icu_cohort['prolonged_los'] = icu_cohort['los_days'] > icu_cohort['los_days'].quantile(0.75)

# 5. New diagnoses in follow-up
future_diag = []
for idx, row in icu_cohort.iterrows():
    # Get index admission diagnoses
    index_diag = diagnoses[
        (diagnoses['subject_id'] == row['subject_id']) &
        (diagnoses['hadm_id'] == row['hadm_id'])
    ]['icd_code'].unique()
    
    # Get future admission diagnoses
    future_admissions = admissions[
        (admissions['subject_id'] == row['subject_id']) &
        (admissions['hadm_id'] != row['hadm_id']) &
        (admissions['admittime'] > row['dischtime']) &
        ((admissions['admittime'] - row['dischtime']).dt.days <= 365)
    ]
    
    if len(future_admissions) > 0:
        future_hadm_ids = future_admissions['hadm_id'].unique()
        future_diag_codes = diagnoses[
            (diagnoses['subject_id'] == row['subject_id']) &
            (diagnoses['hadm_id'].isin(future_hadm_ids))
        ]['icd_code'].unique()
        
        # Count new diagnoses
        new_diag = set(future_diag_codes) - set(index_diag)
        num_new_diag = len(new_diag)
    else:
        num_new_diag = 0
    
    future_diag.append({
        'subject_id': row['subject_id'],
        'hadm_id': row['hadm_id'],
        'num_new_diagnoses_1year': num_new_diag
    })

future_diag_df = pd.DataFrame(future_diag)
icu_cohort = icu_cohort.merge(future_diag_df, on=['subject_id', 'hadm_id'], how='left')

# Define composite "good outcome" vs "poor outcome"
print("\nDefining composite outcomes...")

# POOR OUTCOME: Any of the following
icu_cohort['poor_outcome'] = (
    icu_cohort['died_90day'] |
    (icu_cohort['num_readmissions_1year'] >= 2) |
    (~icu_cohort['discharged_home']) |
    (icu_cohort['num_new_diagnoses_1year'] >= 5) |
    icu_cohort['prolonged_los']
)

# GOOD OUTCOME: None of the poor outcome criteria
icu_cohort['good_outcome'] = ~icu_cohort['poor_outcome']

print(f"\n✓ Good outcomes: {icu_cohort['good_outcome'].sum()} ({icu_cohort['good_outcome'].mean()*100:.1f}%)")
print(f"✓ Poor outcomes: {icu_cohort['poor_outcome'].sum()} ({icu_cohort['poor_outcome'].mean()*100:.1f}%)")

print("\nOutcome breakdown:")
print(f"  • Died within 90 days: {icu_cohort['died_90day'].sum()} ({icu_cohort['died_90day'].mean()*100:.1f}%)")
print(f"  • 2+ readmissions in 1 year: {(icu_cohort['num_readmissions_1year'] >= 2).sum()} ({(icu_cohort['num_readmissions_1year'] >= 2).mean()*100:.1f}%)")
print(f"  • Not discharged home: {(~icu_cohort['discharged_home']).sum()} ({(~icu_cohort['discharged_home']).mean()*100:.1f}%)")
print(f"  • 5+ new diagnoses: {(icu_cohort['num_new_diagnoses_1year'] >= 5).sum()} ({(icu_cohort['num_new_diagnoses_1year'] >= 5).mean()*100:.1f}%)")
print(f"  • Prolonged LOS: {icu_cohort['prolonged_los'].sum()} ({icu_cohort['prolonged_los'].mean()*100:.1f}%)")

print("\n" + "="*80)
print("[3/6] EXTRACTING BASELINE CHARACTERISTICS")
print("="*80)

# Get baseline characteristics for each ICU patient
print("\nExtracting patient features...")

# Demographics
features_df = icu_cohort[['subject_id', 'hadm_id', 'anchor_age', 'gender']].copy()
features_df['is_male'] = (features_df['gender'] == 'M').astype(int)

# Baseline diagnoses (seq_num = 1)
baseline_diag = diagnoses[
    (diagnoses['subject_id'].isin(icu_cohort['subject_id'])) &
    (diagnoses['hadm_id'].isin(icu_cohort['hadm_id'])) &
    (diagnoses['seq_num'] == 1)
]

# Count total diagnoses per patient
diag_counts = diagnoses[
    (diagnoses['subject_id'].isin(icu_cohort['subject_id'])) &
    (diagnoses['hadm_id'].isin(icu_cohort['hadm_id']))
].groupby(['subject_id', 'hadm_id']).size().reset_index(name='num_total_diagnoses')

features_df = features_df.merge(diag_counts, on=['subject_id', 'hadm_id'], how='left')
features_df['num_total_diagnoses'] = features_df['num_total_diagnoses'].fillna(0)

# Specific comorbidities
def has_condition(subj_id, hadm_id, icd_patterns, keywords):
    """Check if patient has a specific condition"""
    pat_diag = diagnoses[
        (diagnoses['subject_id'] == subj_id) &
        (diagnoses['hadm_id'] == hadm_id)
    ]
    
    for pattern in icd_patterns:
        if pat_diag['icd_code'].str.startswith(pattern).any():
            return True
    
    titles = pat_diag['long_title'].str.lower().fillna('')
    for keyword in keywords:
        if titles.str.contains(keyword).any():
            return True
    
    return False

print("  Identifying comorbidities...")
comorbidities = {
    'has_cardiac': (['I', '414', '410', '428'], ['heart', 'cardiac', 'myocardial']),
    'has_hypertension': (['I10', '401'], ['hypertension', 'high blood pressure']),
    'has_diabetes': (['E11', 'E10', '250'], ['diabetes']),
    'has_pulmonary': (['J44', '496', '493'], ['copd', 'chronic obstructive', 'asthma']),
    'has_renal': (['N18', '585'], ['chronic kidney', 'renal insufficiency']),
    'has_liver': (['K70', '571'], ['cirrhosis', 'liver disease']),
    'has_cancer': (['C', '140', '150', '160', '170'], ['malignant', 'cancer', 'neoplasm'])
}

for comorb_name, (patterns, keywords) in comorbidities.items():
    features_df[comorb_name] = features_df.apply(
        lambda row: has_condition(row['subject_id'], row['hadm_id'], patterns, keywords),
        axis=1
    ).astype(int)
    print(f"    ✓ {comorb_name}: {features_df[comorb_name].sum()} patients")

# ICU-specific factors (if available in ICU stays table)
try:
    print("\n  Loading ICU stay data...")
    icu_stays = pd.read_csv('physionet.org/files/mimiciv/3.1/icu/icustays.csv.gz', compression='gzip')
    icu_stays['intime'] = pd.to_datetime(icu_stays['intime'])
    icu_stays['outtime'] = pd.to_datetime(icu_stays['outtime'])
    icu_stays['icu_los_days'] = (icu_stays['outtime'] - icu_stays['intime']).dt.total_seconds() / 86400
    
    # Get ICU LOS for each admission
    icu_los = icu_stays.groupby(['subject_id', 'hadm_id']).agg({
        'icu_los_days': 'sum',
        'stay_id': 'count'
    }).reset_index()
    icu_los.columns = ['subject_id', 'hadm_id', 'total_icu_los_days', 'num_icu_stays']
    
    features_df = features_df.merge(icu_los, on=['subject_id', 'hadm_id'], how='left')
    features_df['total_icu_los_days'] = features_df['total_icu_los_days'].fillna(0)
    features_df['num_icu_stays'] = features_df['num_icu_stays'].fillna(0)
    features_df['prolonged_icu'] = features_df['total_icu_los_days'] > features_df['total_icu_los_days'].quantile(0.75)
    
    print(f"    ✓ Mean ICU LOS: {features_df['total_icu_los_days'].mean():.1f} days")
    print(f"    ✓ Prolonged ICU stay (>Q3): {features_df['prolonged_icu'].sum()} patients")
except Exception as e:
    print(f"    ⚠️  Could not load ICU stays data: {e}")
    features_df['total_icu_los_days'] = 0
    features_df['num_icu_stays'] = 0
    features_df['prolonged_icu'] = 0

# Merge outcomes with features
analysis_df = features_df.merge(
    icu_cohort[['subject_id', 'hadm_id', 'good_outcome', 'poor_outcome', 
                'died_90day', 'died_1year', 'readmitted_90day', 'discharged_home',
                'los_days', 'num_readmissions_1year', 'num_new_diagnoses_1year']],
    on=['subject_id', 'hadm_id'], how='left'
)

print(f"\n✓ Analysis dataset: {len(analysis_df)} ICU patients with {len(features_df.columns)} features")

print("\n" + "="*80)
print("[4/6] UNIVARIATE ASSOCIATIONS WITH GOOD OUTCOME")
print("="*80)

univariate_results = []

# Continuous variables
continuous_vars = ['anchor_age', 'num_total_diagnoses', 'total_icu_los_days', 'los_days']
print("\n" + "-"*80)
print("CONTINUOUS VARIABLES")
print("-"*80)
print(f"{'Variable':<25} {'Good Outcome':<15} {'Poor Outcome':<15} {'p-value':<10}")
print("-"*80)

for var in continuous_vars:
    if var in analysis_df.columns:
        good_vals = analysis_df[analysis_df['good_outcome'] == True][var].dropna()
        poor_vals = analysis_df[analysis_df['poor_outcome'] == True][var].dropna()
        
        if len(good_vals) > 0 and len(poor_vals) > 0:
            t_stat, p_val = stats.ttest_ind(good_vals, poor_vals)
            
            print(f"{var:<25} {good_vals.mean():>10.2f}     {poor_vals.mean():>10.2f}     {p_val:>8.4f} {'***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))}")
            
            univariate_results.append({
                'Variable': var,
                'Type': 'Continuous',
                'Good_Outcome_Mean': good_vals.mean(),
                'Poor_Outcome_Mean': poor_vals.mean(),
                'p_value': p_val,
                'Significant': p_val < 0.05
            })

# Binary variables
binary_vars = ['is_male', 'has_cardiac', 'has_hypertension', 'has_diabetes', 
               'has_pulmonary', 'has_renal', 'has_liver', 'has_cancer', 'prolonged_icu']

print("\n" + "-"*80)
print("BINARY VARIABLES")
print("-"*80)
print(f"{'Variable':<25} {'Good Outcome %':<15} {'Poor Outcome %':<15} {'p-value':<10}")
print("-"*80)

for var in binary_vars:
    if var in analysis_df.columns:
        good_pct = analysis_df[analysis_df['good_outcome'] == True][var].mean() * 100
        poor_pct = analysis_df[analysis_df['poor_outcome'] == True][var].mean() * 100
        
        # Chi-square test
        contingency = pd.crosstab(analysis_df[var], analysis_df['good_outcome'])
        if contingency.shape == (2, 2):
            _, p_val = stats.fisher_exact(contingency) if contingency.min().min() < 5 else stats.chi2_contingency(contingency)[:2]
            
            print(f"{var:<25} {good_pct:>10.1f}%     {poor_pct:>10.1f}%     {p_val:>8.4f} {'***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))}")
            
            univariate_results.append({
                'Variable': var,
                'Type': 'Binary',
                'Good_Outcome_Mean': good_pct,
                'Poor_Outcome_Mean': poor_pct,
                'p_value': p_val,
                'Significant': p_val < 0.05
            })

univar_df = pd.DataFrame(univariate_results)
univar_df.to_csv('icu_prognostic_factors_univariate.csv', index=False)
print("\n✓ Saved: icu_prognostic_factors_univariate.csv")

print("\n" + "="*80)
print("[5/6] MULTIVARIABLE MODEL")
print("="*80)

# Prepare data for logistic regression
print("\nBuilding multivariable logistic regression model...")

# Select predictors (exclude outcome variables)
predictor_vars = ['anchor_age', 'is_male', 'num_total_diagnoses', 
                  'has_cardiac', 'has_hypertension', 'has_diabetes',
                  'has_pulmonary', 'has_renal', 'total_icu_los_days']

# Only use variables that exist
predictor_vars = [v for v in predictor_vars if v in analysis_df.columns]

X = analysis_df[predictor_vars].fillna(0)
y = analysis_df['good_outcome'].astype(int)

# Remove any rows with missing outcome
mask = y.notna()
X = X[mask]
y = y[mask]

print(f"  • Predictors: {len(predictor_vars)}")
print(f"  • Sample size: {len(X)}")
print(f"  • Good outcomes: {y.sum()} ({y.mean()*100:.1f}%)")

# Standardize continuous variables
scaler = StandardScaler()
cont_vars_in_model = [v for v in ['anchor_age', 'num_total_diagnoses', 'total_icu_los_days'] if v in predictor_vars]
if len(cont_vars_in_model) > 0:
    X[cont_vars_in_model] = scaler.fit_transform(X[cont_vars_in_model])

# Fit model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X, y)

# Get coefficients and odds ratios
coef_df = pd.DataFrame({
    'Variable': predictor_vars,
    'Coefficient': model.coef_[0],
    'Odds_Ratio': np.exp(model.coef_[0])
})
coef_df = coef_df.sort_values('Odds_Ratio', ascending=False)

print("\n" + "-"*80)
print("MULTIVARIABLE MODEL RESULTS")
print("-"*80)
print(f"{'Variable':<25} {'Coefficient':<15} {'Odds Ratio':<15} {'Interpretation':<20}")
print("-"*80)

for _, row in coef_df.iterrows():
    interp = "Better outcome" if row['Odds_Ratio'] > 1 else "Worse outcome"
    print(f"{row['Variable']:<25} {row['Coefficient']:>10.3f}     {row['Odds_Ratio']:>10.3f}     {interp:<20}")

coef_df.to_csv('icu_prognostic_factors_multivariable.csv', index=False)
print("\n✓ Saved: icu_prognostic_factors_multivariable.csv")

# Model performance
from sklearn.metrics import roc_auc_score, classification_report
y_pred_proba = model.predict_proba(X)[:, 1]
auc = roc_auc_score(y, y_pred_proba)
print(f"\nModel AUC: {auc:.3f}")

print("\n" + "="*80)
print("[6/6] CREATING VISUALIZATION")
print("="*80)

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.35,
                      left=0.08, right=0.96, top=0.94, bottom=0.06)

# Panel A: Outcome distribution
ax_out = fig.add_subplot(gs[0, 0])
outcome_counts = [icu_cohort['good_outcome'].sum(), icu_cohort['poor_outcome'].sum()]
colors_out = ['#2ecc71', '#e74c3c']
bars = ax_out.bar(['Good Outcome', 'Poor Outcome'], outcome_counts, color=colors_out, alpha=0.8, edgecolor='black', linewidth=2)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax_out.text(bar.get_x() + bar.get_width()/2, height + 1.5,
               f'n={int(height)}\n({height/len(icu_cohort)*100:.1f}%)',
               ha='center', va='bottom', fontweight='bold', fontsize=10)
ax_out.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
ax_out.set_title('A. Outcome Distribution in ICU Patients', fontsize=13, fontweight='bold', loc='left', pad=12)
ax_out.spines['top'].set_visible(False)
ax_out.spines['right'].set_visible(False)
ax_out.grid(axis='y', alpha=0.3, linestyle='--')

# Panel B: Poor outcome components
ax_comp = fig.add_subplot(gs[0, 1])
components = {
    'Died ≤90d': icu_cohort['died_90day'].sum(),
    '≥2 Readmissions': (icu_cohort['num_readmissions_1year'] >= 2).sum(),
    'Not Home Discharge': (~icu_cohort['discharged_home']).sum(),
    '≥5 New Diagnoses': (icu_cohort['num_new_diagnoses_1year'] >= 5).sum(),
    'Prolonged LOS': icu_cohort['prolonged_los'].sum()
}
y_pos = np.arange(len(components))
counts = list(components.values())
bars = ax_comp.barh(y_pos, counts, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
for i, bar in enumerate(bars):
    width = bar.get_width()
    pct = width / len(icu_cohort) * 100
    ax_comp.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{int(width)} ({pct:.1f}%)', va='center', fontweight='bold', fontsize=9)
ax_comp.set_yticks(y_pos)
ax_comp.set_yticklabels(components.keys(), fontsize=10)
ax_comp.set_xlabel('Number of Patients', fontsize=12, fontweight='bold')
ax_comp.set_title('B. Poor Outcome Components', fontsize=13, fontweight='bold', loc='left', pad=12)
ax_comp.spines['top'].set_visible(False)
ax_comp.spines['right'].set_visible(False)
ax_comp.grid(axis='x', alpha=0.3, linestyle='--')

# Panel C: Significant univariate predictors
ax_univ = fig.add_subplot(gs[1, :])
sig_univar = univar_df[univar_df['Significant'] == True].sort_values('p_value')
if len(sig_univar) > 0:
    y_pos = np.arange(len(sig_univar))
    # Calculate effect direction
    effects = []
    for _, row in sig_univar.iterrows():
        if row['Type'] == 'Continuous':
            effect = row['Good_Outcome_Mean'] - row['Poor_Outcome_Mean']
        else:  # Binary
            effect = row['Good_Outcome_Mean'] - row['Poor_Outcome_Mean']
        effects.append(effect)
    
    colors_univ = ['#2ecc71' if e > 0 else '#e74c3c' for e in effects]
    bars = ax_univ.barh(y_pos, effects, color=colors_univ, alpha=0.8, edgecolor='black')
    ax_univ.set_yticks(y_pos)
    ax_univ.set_yticklabels(sig_univar['Variable'], fontsize=10)
    ax_univ.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax_univ.set_xlabel('Effect Size (Good - Poor Outcome)', fontsize=12, fontweight='bold')
    ax_univ.set_title('C. Significant Univariate Predictors of Good Outcome (p<0.05)', 
                     fontsize=13, fontweight='bold', loc='left', pad=12)
    ax_univ.spines['top'].set_visible(False)
    ax_univ.spines['right'].set_visible(False)
    ax_univ.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add p-value annotations
    for i, (_, row) in enumerate(sig_univar.iterrows()):
        sig_marker = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01 else '*')
        x_pos = effects[i] + (0.5 if effects[i] > 0 else -0.5)
        ax_univ.text(x_pos, i, sig_marker, va='center', ha='center', fontweight='bold', fontsize=12)
else:
    ax_univ.text(0.5, 0.5, 'No significant univariate predictors found',
                ha='center', va='center', transform=ax_univ.transAxes, fontsize=12)
    ax_univ.set_title('C. Significant Univariate Predictors', fontsize=13, fontweight='bold', loc='left', pad=12)

# Panel D: Multivariable odds ratios
ax_multi = fig.add_subplot(gs[2, :])
y_pos = np.arange(len(coef_df))
colors_multi = ['#2ecc71' if or_val > 1 else '#e74c3c' for or_val in coef_df['Odds_Ratio']]
bars = ax_multi.barh(y_pos, coef_df['Odds_Ratio'], color=colors_multi, alpha=0.8, edgecolor='black')
ax_multi.set_yticks(y_pos)
ax_multi.set_yticklabels(coef_df['Variable'], fontsize=10)
ax_multi.axvline(x=1, color='black', linestyle='--', linewidth=2, label='OR=1 (No effect)')
ax_multi.set_xlabel('Odds Ratio for Good Outcome', fontsize=12, fontweight='bold')
ax_multi.set_title('D. Multivariable Model: Adjusted Odds Ratios', fontsize=13, fontweight='bold', loc='left', pad=12)
ax_multi.legend(frameon=True, fontsize=10)
ax_multi.spines['top'].set_visible(False)
ax_multi.spines['right'].set_visible(False)
ax_multi.grid(axis='x', alpha=0.3, linestyle='--')

plt.savefig('icu_prognostic_factors_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: icu_prognostic_factors_analysis.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nKey Findings:")
print(f"  • {icu_cohort['good_outcome'].sum()} of {len(icu_cohort)} ICU patients had good outcomes ({icu_cohort['good_outcome'].mean()*100:.1f}%)")
print(f"  • {len(sig_univar)} significant univariate predictors identified")
print(f"  • Multivariable model AUC: {auc:.3f}")

# Get top protective and risk factors from coefficients
top_protective = coef_df[coef_df['Odds_Ratio'] > 1].head(3)
top_risk = coef_df[coef_df['Odds_Ratio'] < 1].tail(3)

if len(top_protective) > 0:
    best_protective = top_protective.iloc[0]
    print(f"  • Strongest protective factor: {best_protective['Variable']} (OR={best_protective['Odds_Ratio']:.2f})")

if len(top_risk) > 0:
    worst_risk = top_risk.iloc[-1]
    print(f"  • Strongest risk factor: {worst_risk['Variable']} (OR={worst_risk['Odds_Ratio']:.2f})")

print("\nOutput files:")
print("  ✓ icu_prognostic_factors_univariate.csv")
print("  ✓ icu_prognostic_factors_multivariable.csv")
print("  ✓ icu_prognostic_factors_analysis.png")
print("="*80)

