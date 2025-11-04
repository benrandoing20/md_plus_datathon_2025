"""
TRUE LONG-TERM OUTCOMES ANALYSIS
=================================

GOAL: Avoid diagnosis ambiguity by using UNAMBIGUOUS temporal outcomes

Key Improvements:
1. Use outcomes with CLEAR timestamps (not diagnoses)
2. Focus on LONG-TERM outcomes (30-day, 90-day, 6-month, 1-year)
3. Use healthcare utilization metrics
4. Compare ICU vs non-ICU adjusting for baseline risk

Outcomes (all with clear temporal ordering):
1. Hospital length of stay (during index admission)
2. ICU length of stay (for ICU patients)
3. In-hospital mortality (during index admission)
4. 30-day mortality (post-discharge)
5. 90-day mortality (post-discharge)
6. 30-day readmission
7. 90-day readmission
8. 6-month readmission
9. 1-year readmission
10. Emergency department visits (30-day, 90-day, 6-month)
11. Total hospital days in 1 year (including readmissions)
12. Number of readmissions in 1 year
13. Cost implications (based on LOS)

Method: Propensity-score matched analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (20, 16)

print("="*80)
print("TRUE LONG-TERM OUTCOMES ANALYSIS")
print("="*80)
print("Using unambiguous temporal outcomes (not diagnosis codes)")
print("="*80)

# ============================================================================
# SECTION 1: DATA LOADING WITH TEMPORAL INFORMATION
# ============================================================================

print("\n[1/7] Loading data with temporal markers...")

DATA_ROOT = Path('physionet.org/files/mimiciv/3.1')
HOSP_PATH = DATA_ROOT / 'hosp'
ICU_PATH = DATA_ROOT / 'icu'

cohort_icu = pd.read_csv('cohort_elective_spine_icu.csv')
admissions = pd.read_csv(HOSP_PATH / 'admissions.csv.gz', compression='gzip')
patients = pd.read_csv(HOSP_PATH / 'patients.csv.gz', compression='gzip')
diagnoses_icd = pd.read_csv(HOSP_PATH / 'diagnoses_icd.csv.gz', compression='gzip')
d_icd_diagnoses = pd.read_csv(HOSP_PATH / 'd_icd_diagnoses.csv.gz', compression='gzip')
procedures_icd = pd.read_csv(HOSP_PATH / 'procedures_icd.csv.gz', compression='gzip')
icustays = pd.read_csv(ICU_PATH / 'icustays.csv.gz', compression='gzip')

print(f"✓ Loaded cohort: {len(cohort_icu)} ICU patients")

# Build full cohort
def is_spine_surgery(row):
    code = str(row['icd_code'])
    version = row['icd_version']
    if version == 9:
        return code.startswith(('810', '813', '816', '03'))
    elif version == 10 and len(code) >= 4:
        bs, op, bp = code[1], code[2], code[3]
        if bs == 'R' and op in 'BGNTQSHJPRUW' and bp in '0123467689AB':
            return True
        if bs == 'S' and op in 'BGNTQSHJPRUW' and bp in '012345678':
            return True
    return False

procedures_icd['is_spine'] = procedures_icd.apply(is_spine_surgery, axis=1)
spine_procs = procedures_icd[procedures_icd['is_spine']].copy()

elective = admissions[admissions['admission_type'].str.upper() == 'ELECTIVE']
adults = patients[patients['anchor_age'] >= 18][['subject_id', 'gender', 'anchor_age', 'dod']]
spine_hadm = spine_procs[['hadm_id', 'subject_id']].drop_duplicates()

cohort = elective.merge(spine_hadm, on=['hadm_id', 'subject_id'], how='inner')
cohort = cohort.merge(adults, on='subject_id', how='inner')

icu_ids = set(cohort_icu['hadm_id'])
cohort['icu_admission'] = cohort['hadm_id'].isin(icu_ids).astype(int)

# Convert dates
for col in ['admittime', 'dischtime', 'deathtime', 'dod']:
    if col in cohort.columns:
        cohort[col] = pd.to_datetime(cohort[col])
    if col in admissions.columns:
        admissions[col] = pd.to_datetime(admissions[col])

print(f"✓ Full cohort: {len(cohort)} patients")
print(f"  - ICU: {cohort['icu_admission'].sum()}")
print(f"  - Non-ICU: {(1-cohort['icu_admission']).sum()}")

# ============================================================================
# SECTION 2: BASELINE CHARACTERISTICS (PRE-ADMISSION ONLY)
# ============================================================================

print("\n[2/7] Engineering BASELINE characteristics (pre-admission)...")

cohort['age'] = cohort['anchor_age']
cohort['is_male'] = (cohort['gender'] == 'M').astype(int)

# For baseline comorbidities, use seq_num=1 (admission diagnosis) only
cohort_hadm_ids = set(cohort['hadm_id'])
admission_diagnoses = diagnoses_icd[
    (diagnoses_icd['hadm_id'].isin(cohort_hadm_ids)) &
    (diagnoses_icd['seq_num'] == 1)  # Only admission diagnosis
].copy()

admission_diagnoses = admission_diagnoses.merge(
    d_icd_diagnoses[['icd_code', 'icd_version', 'long_title']],
    on=['icd_code', 'icd_version'], how='left'
)

print(f"✓ Using seq_num=1 only for baseline: {len(admission_diagnoses)} admission diagnoses")

baseline_counts = admission_diagnoses.groupby('hadm_id').size().reset_index(name='num_baseline_diagnoses')
cohort = cohort.merge(baseline_counts, on='hadm_id', how='left')
cohort['num_baseline_diagnoses'] = cohort['num_baseline_diagnoses'].fillna(0)

# Baseline comorbidities (admission diagnosis only)
def has_admission_comorbidity(hadm_id, keywords):
    dx = admission_diagnoses[admission_diagnoses['hadm_id'] == hadm_id]['long_title'].fillna('')
    return int(any(any(kw.lower() in title.lower() for kw in keywords) 
                   for title in dx if isinstance(title, str)))

comorbidity_keywords = {
    'has_cardiac': ['cardiac', 'coronary', 'myocardial', 'heart failure', 'atrial fibrillation'],
    'has_hypertension': ['hypertension'],
    'has_diabetes': ['diabetes'],
    'has_pulmonary': ['copd', 'asthma', 'pulmonary', 'sleep apnea'],
    'has_renal': ['renal', 'kidney', 'chronic kidney']
}

for comorb, keywords in comorbidity_keywords.items():
    cohort[comorb] = cohort['hadm_id'].apply(lambda x: has_admission_comorbidity(x, keywords))

print(f"✓ Baseline comorbidities identified (admission dx only)")
for comorb in comorbidity_keywords.keys():
    print(f"  - {comorb}: {cohort[comorb].sum()} ({cohort[comorb].mean()*100:.1f}%)")

# ============================================================================
# SECTION 3: UNAMBIGUOUS TEMPORAL OUTCOMES
# ============================================================================

print("\n[3/7] Calculating outcomes with CLEAR temporal ordering...")

outcomes_df = cohort[['hadm_id', 'subject_id', 'icu_admission', 
                       'admittime', 'dischtime', 'deathtime', 'dod']].copy()

# 1. Index admission outcomes (during hospitalization)
outcomes_df['los_days'] = (outcomes_df['dischtime'] - outcomes_df['admittime']).dt.total_seconds() / 86400
outcomes_df['in_hospital_death'] = outcomes_df['deathtime'].notna().astype(int)

# ICU LOS
icu_los = icustays.groupby('hadm_id').agg({
    'los': 'sum',
    'stay_id': 'count'
}).reset_index()
icu_los.columns = ['hadm_id', 'icu_los_days', 'num_icu_stays']
outcomes_df = outcomes_df.merge(icu_los, on='hadm_id', how='left')
outcomes_df['icu_los_days'] = outcomes_df['icu_los_days'].fillna(0)
outcomes_df['num_icu_stays'] = outcomes_df['num_icu_stays'].fillna(0)

print(f"✓ Index admission outcomes:")
print(f"  - Mean LOS: {outcomes_df['los_days'].mean():.1f} days")
print(f"  - In-hospital deaths: {outcomes_df['in_hospital_death'].sum()}")

# 2. Post-discharge mortality (clear temporal ordering!)
outcomes_df['days_alive_post_discharge'] = (outcomes_df['dod'] - outcomes_df['dischtime']).dt.days

for window in [30, 90, 180, 365]:
    col_name = f'mortality_{window}d'
    outcomes_df[col_name] = (
        (outcomes_df['days_alive_post_discharge'] >= 0) & 
        (outcomes_df['days_alive_post_discharge'] <= window)
    ).astype(int)
    
    # Also include in-hospital deaths
    outcomes_df[col_name] = (outcomes_df[col_name] | outcomes_df['in_hospital_death']).astype(int)
    
    print(f"  - {window}-day mortality: {outcomes_df[col_name].sum()} ({outcomes_df[col_name].mean()*100:.1f}%)")

# 3. Readmissions (unambiguous - occurs AFTER discharge)
print("\n  Calculating readmissions...")

# Sort all admissions by patient and time
all_admissions = admissions.sort_values(['subject_id', 'admittime']).copy()
all_admissions['admittime'] = pd.to_datetime(all_admissions['admittime'])

# For each patient, find subsequent admissions
outcomes_df['first_readmit_days'] = np.nan
outcomes_df['num_readmits_1yr'] = 0
outcomes_df['total_hospital_days_1yr'] = outcomes_df['los_days']

for idx, row in outcomes_df.iterrows():
    subject_id = row['subject_id']
    index_discharge = row['dischtime']
    
    # Find all subsequent admissions within 1 year
    future_admits = all_admissions[
        (all_admissions['subject_id'] == subject_id) &
        (all_admissions['admittime'] > index_discharge) &
        (all_admissions['admittime'] <= index_discharge + timedelta(days=365))
    ]
    
    if len(future_admits) > 0:
        # Time to first readmission
        first_readmit = (future_admits.iloc[0]['admittime'] - index_discharge).days
        outcomes_df.at[idx, 'first_readmit_days'] = first_readmit
        
        # Total number of readmissions in 1 year
        outcomes_df.at[idx, 'num_readmits_1yr'] = len(future_admits)
        
        # Total hospital days in 1 year (including readmissions)
        readmit_los = (pd.to_datetime(future_admits['dischtime']) - 
                       pd.to_datetime(future_admits['admittime'])).dt.total_seconds() / 86400
        outcomes_df.at[idx, 'total_hospital_days_1yr'] = row['los_days'] + readmit_los.sum()

# Create binary readmission variables
for window in [30, 90, 180, 365]:
    col_name = f'readmit_{window}d'
    outcomes_df[col_name] = (
        (outcomes_df['first_readmit_days'].notna()) &
        (outcomes_df['first_readmit_days'] <= window)
    ).astype(int)
    print(f"  - {window}-day readmission: {outcomes_df[col_name].sum()} ({outcomes_df[col_name].mean()*100:.1f}%)")

print(f"  - Mean readmissions in 1 year: {outcomes_df['num_readmits_1yr'].mean():.2f}")
print(f"  - Mean total hospital days in 1 year: {outcomes_df['total_hospital_days_1yr'].mean():.1f}")

# 4. Healthcare utilization composite outcome
outcomes_df['high_utilization'] = (
    (outcomes_df['num_readmits_1yr'] >= 2) |
    (outcomes_df['total_hospital_days_1yr'] > 14)
).astype(int)

print(f"  - High healthcare utilization (≥2 readmits or >14 days): {outcomes_df['high_utilization'].sum()} ({outcomes_df['high_utilization'].mean()*100:.1f}%)")

# ============================================================================
# SECTION 4: PROPENSITY SCORE MATCHING
# ============================================================================

print("\n[4/7] Propensity score matching on BASELINE characteristics...")

# Merge baseline features with outcomes
ps_features = ['age', 'is_male', 'num_baseline_diagnoses', 'has_cardiac',
               'has_hypertension', 'has_diabetes', 'has_pulmonary', 'has_renal']

analysis_df = cohort[['hadm_id'] + ps_features + ['icu_admission']].merge(
    outcomes_df, on=['hadm_id', 'icu_admission'], how='inner'
)

# Remove rows with missing data
analysis_df = analysis_df.dropna(subset=ps_features)

print(f"✓ Analysis cohort: {len(analysis_df)} patients")
print(f"  - ICU: {analysis_df['icu_admission'].sum()}")
print(f"  - Non-ICU: {(1-analysis_df['icu_admission']).sum()}")

# Estimate propensity scores
X_ps = analysis_df[ps_features].copy()
y_ps = analysis_df['icu_admission'].copy()

scaler = StandardScaler()
X_ps_scaled = scaler.fit_transform(X_ps)

ps_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
ps_model.fit(X_ps_scaled, y_ps)
analysis_df['propensity_score'] = ps_model.predict_proba(X_ps_scaled)[:, 1]

print(f"✓ Propensity scores:")
print(f"  - ICU mean: {analysis_df[analysis_df['icu_admission']==1]['propensity_score'].mean():.3f}")
print(f"  - Non-ICU mean: {analysis_df[analysis_df['icu_admission']==0]['propensity_score'].mean():.3f}")

# IPTW weights
mean_treatment = analysis_df['icu_admission'].mean()
analysis_df['iptw_weight'] = np.where(
    analysis_df['icu_admission'] == 1,
    mean_treatment / analysis_df['propensity_score'],
    (1 - mean_treatment) / (1 - analysis_df['propensity_score'])
)
# Trim extreme weights
analysis_df['iptw_weight'] = analysis_df['iptw_weight'].clip(
    upper=analysis_df['iptw_weight'].quantile(0.99)
)

print(f"  - IPTW weights created (trimmed at 99th percentile)")

# ============================================================================
# SECTION 5: COMPARATIVE OUTCOMES ANALYSIS
# ============================================================================

print("\n[5/7] Comparing outcomes between ICU and non-ICU...")

# Outcomes to analyze
continuous_outcomes = [
    ('los_days', 'Hospital Length of Stay (days)'),
    ('icu_los_days', 'ICU Length of Stay (days)'),
    ('total_hospital_days_1yr', 'Total Hospital Days in 1 Year'),
    ('num_readmits_1yr', 'Number of Readmissions in 1 Year')
]

binary_outcomes = [
    ('in_hospital_death', 'In-Hospital Mortality'),
    ('mortality_30d', '30-Day Mortality'),
    ('mortality_90d', '90-Day Mortality'),
    ('mortality_180d', '6-Month Mortality'),
    ('mortality_365d', '1-Year Mortality'),
    ('readmit_30d', '30-Day Readmission'),
    ('readmit_90d', '90-Day Readmission'),
    ('readmit_180d', '6-Month Readmission'),
    ('readmit_365d', '1-Year Readmission'),
    ('high_utilization', 'High Healthcare Utilization')
]

print("\n" + "="*80)
print("UNADJUSTED COMPARISONS")
print("="*80)

results_unadjusted = []

print("\nCONTINUOUS OUTCOMES:")
print("-" * 80)
for outcome, label in continuous_outcomes:
    icu_mean = analysis_df[analysis_df['icu_admission']==1][outcome].mean()
    non_icu_mean = analysis_df[analysis_df['icu_admission']==0][outcome].mean()
    
    icu_vals = analysis_df[analysis_df['icu_admission']==1][outcome].dropna()
    non_icu_vals = analysis_df[analysis_df['icu_admission']==0][outcome].dropna()
    
    if len(icu_vals) > 0 and len(non_icu_vals) > 0:
        t_stat, p_val = stats.ttest_ind(icu_vals, non_icu_vals)
    else:
        p_val = np.nan
    
    diff = icu_mean - non_icu_mean
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    
    print(f"{label:40s}: ICU={icu_mean:6.2f} | Non-ICU={non_icu_mean:6.2f} | Diff={diff:+6.2f} | p={p_val:.4f} {sig}")
    
    results_unadjusted.append({
        'outcome': label,
        'type': 'continuous',
        'icu_value': icu_mean,
        'non_icu_value': non_icu_mean,
        'difference': diff,
        'p_value': p_val
    })

print("\nBINARY OUTCOMES:")
print("-" * 80)
for outcome, label in binary_outcomes:
    icu_rate = analysis_df[analysis_df['icu_admission']==1][outcome].mean() * 100
    non_icu_rate = analysis_df[analysis_df['icu_admission']==0][outcome].mean() * 100
    
    contingency = pd.crosstab(analysis_df[outcome], analysis_df['icu_admission'])
    if contingency.shape[0] > 1:
        chi2, p_val, _, _ = stats.chi2_contingency(contingency)
        
        a = contingency.iloc[1, 1]
        b = contingency.iloc[1, 0]
        c = contingency.iloc[0, 1]
        d = contingency.iloc[0, 0]
        
        if b > 0 and c > 0:
            or_val = (a * d) / (b * c)
        else:
            or_val = np.nan
    else:
        p_val = np.nan
        or_val = np.nan
    
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    
    print(f"{label:40s}: ICU={icu_rate:5.1f}% | Non-ICU={non_icu_rate:5.1f}% | OR={or_val:.2f} | p={p_val:.4f} {sig}")
    
    results_unadjusted.append({
        'outcome': label,
        'type': 'binary',
        'icu_value': icu_rate,
        'non_icu_value': non_icu_rate,
        'or': or_val,
        'p_value': p_val
    })

# IPTW-adjusted
print("\n" + "="*80)
print("IPTW-ADJUSTED COMPARISONS (Controlling for Baseline Risk)")
print("="*80)

results_adjusted = []

print("\nCONTINUOUS OUTCOMES:")
print("-" * 80)
for outcome, label in continuous_outcomes:
    icu_adj = np.average(
        analysis_df[analysis_df['icu_admission']==1][outcome],
        weights=analysis_df[analysis_df['icu_admission']==1]['iptw_weight']
    )
    non_icu_adj = np.average(
        analysis_df[analysis_df['icu_admission']==0][outcome],
        weights=analysis_df[analysis_df['icu_admission']==0]['iptw_weight']
    )
    diff_adj = icu_adj - non_icu_adj
    
    print(f"{label:40s}: ICU={icu_adj:6.2f} | Non-ICU={non_icu_adj:6.2f} | Diff={diff_adj:+6.2f}")
    
    results_adjusted.append({
        'outcome': label,
        'type': 'continuous',
        'icu_adj': icu_adj,
        'non_icu_adj': non_icu_adj,
        'diff_adj': diff_adj
    })

print("\nBINARY OUTCOMES:")
print("-" * 80)
for outcome, label in binary_outcomes:
    icu_adj = np.average(
        analysis_df[analysis_df['icu_admission']==1][outcome],
        weights=analysis_df[analysis_df['icu_admission']==1]['iptw_weight']
    ) * 100
    non_icu_adj = np.average(
        analysis_df[analysis_df['icu_admission']==0][outcome],
        weights=analysis_df[analysis_df['icu_admission']==0]['iptw_weight']
    ) * 100
    risk_diff = icu_adj - non_icu_adj
    
    print(f"{label:40s}: ICU={icu_adj:5.1f}% | Non-ICU={non_icu_adj:5.1f}% | RD={risk_diff:+5.1f}%")
    
    results_adjusted.append({
        'outcome': label,
        'type': 'binary',
        'icu_adj': icu_adj,
        'non_icu_adj': non_icu_adj,
        'risk_diff': risk_diff
    })

# ============================================================================
# SECTION 6: CLEAN VISUALIZATION
# ============================================================================

print("\n[6/7] Creating publication-quality figure...")

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.35,
                      left=0.06, right=0.97, top=0.95, bottom=0.05)

# Panel A: Propensity score distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(analysis_df[analysis_df['icu_admission']==1]['propensity_score'],
         bins=20, alpha=0.6, label='ICU', color='#ef5350', edgecolor='black', linewidth=1)
ax1.hist(analysis_df[analysis_df['icu_admission']==0]['propensity_score'],
         bins=20, alpha=0.6, label='Non-ICU', color='#66bb6a', edgecolor='black', linewidth=1)
ax1.set_xlabel('Propensity Score', fontweight='bold', fontsize=11)
ax1.set_ylabel('Frequency', fontweight='bold', fontsize=11)
ax1.set_title('A. Baseline Risk Distribution', fontweight='bold', fontsize=12, pad=12)
ax1.legend(frameon=True, fontsize=10)
ax1.grid(alpha=0.3)

# Panel B: Length of stay
ax2 = fig.add_subplot(gs[0, 1:])
los_data = [
    analysis_df[analysis_df['icu_admission']==0]['los_days'],
    analysis_df[analysis_df['icu_admission']==1]['los_days']
]
bp = ax2.boxplot(los_data, labels=['Non-ICU', 'ICU'],
                 patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('#66bb6a')
bp['boxes'][1].set_facecolor('#ef5350')
for box in bp['boxes']:
    box.set_alpha(0.7)
ax2.set_ylabel('Hospital Length of Stay (days)', fontweight='bold', fontsize=11)
ax2.set_title('B. Hospital Length of Stay', fontweight='bold', fontsize=12, pad=12)
ax2.grid(axis='y', alpha=0.3)

# Panel C: Mortality over time
ax3 = fig.add_subplot(gs[1, :])
mort_outcomes = ['In-Hospital', '30-Day', '90-Day', '6-Month', '1-Year']
mort_keys = ['in_hospital_death', 'mortality_30d', 'mortality_90d', 'mortality_180d', 'mortality_365d']
icu_mort = [analysis_df[analysis_df['icu_admission']==1][k].mean()*100 for k in mort_keys]
non_icu_mort = [analysis_df[analysis_df['icu_admission']==0][k].mean()*100 for k in mort_keys]

x = np.arange(len(mort_outcomes))
width = 0.35
bars1 = ax3.bar(x - width/2, icu_mort, width, label='ICU',
                color='#ef5350', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax3.bar(x + width/2, non_icu_mort, width, label='Non-ICU',
                color='#66bb6a', alpha=0.8, edgecolor='black', linewidth=1.5)

for bar in bars1 + bars2:
    height = bar.get_height()
    if height > 0:
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

ax3.set_ylabel('Mortality Rate (%)', fontweight='bold', fontsize=11)
ax3.set_title('C. Mortality Outcomes Over Time', fontweight='bold', fontsize=12, pad=12)
ax3.set_xticks(x)
ax3.set_xticklabels(mort_outcomes, fontsize=10)
ax3.legend(loc='upper left', frameon=True, fontsize=10)
ax3.grid(axis='y', alpha=0.3)

# Panel D: Readmission over time
ax4 = fig.add_subplot(gs[2, :])
readmit_outcomes = ['30-Day', '90-Day', '6-Month', '1-Year']
readmit_keys = ['readmit_30d', 'readmit_90d', 'readmit_180d', 'readmit_365d']
icu_readmit = [analysis_df[analysis_df['icu_admission']==1][k].mean()*100 for k in readmit_keys]
non_icu_readmit = [analysis_df[analysis_df['icu_admission']==0][k].mean()*100 for k in readmit_keys]

x = np.arange(len(readmit_outcomes))
bars1 = ax4.bar(x - width/2, icu_readmit, width, label='ICU',
                color='#ef5350', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax4.bar(x + width/2, non_icu_readmit, width, label='Non-ICU',
                color='#66bb6a', alpha=0.8, edgecolor='black', linewidth=1.5)

for bar in bars1 + bars2:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

ax4.set_ylabel('Readmission Rate (%)', fontweight='bold', fontsize=11)
ax4.set_title('D. Readmission Outcomes Over Time', fontweight='bold', fontsize=12, pad=12)
ax4.set_xticks(x)
ax4.set_xticklabels(readmit_outcomes, fontsize=10)
ax4.legend(loc='upper left', frameon=True, fontsize=10)
ax4.grid(axis='y', alpha=0.3)

# Panel E: Healthcare utilization
ax5 = fig.add_subplot(gs[3, 0])
util_metrics = ['Total Hospital\nDays (1 yr)', 'Number of\nReadmissions (1 yr)']
util_keys = ['total_hospital_days_1yr', 'num_readmits_1yr']
icu_util = [analysis_df[analysis_df['icu_admission']==1][k].mean() for k in util_keys]
non_icu_util = [analysis_df[analysis_df['icu_admission']==0][k].mean() for k in util_keys]

x = np.arange(len(util_metrics))
bars1 = ax5.bar(x - width/2, icu_util, width, label='ICU',
                color='#ef5350', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax5.bar(x + width/2, non_icu_util, width, label='Non-ICU',
                color='#66bb6a', alpha=0.8, edgecolor='black', linewidth=1.5)

for bar in bars1 + bars2:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom', fontsize=9)

ax5.set_ylabel('Value', fontweight='bold', fontsize=11)
ax5.set_title('E. Healthcare Utilization (1 Year)', fontweight='bold', fontsize=12, pad=12)
ax5.set_xticks(x)
ax5.set_xticklabels(util_metrics, fontsize=9)
ax5.legend(loc='upper left', frameon=True, fontsize=10)
ax5.grid(axis='y', alpha=0.3)

# Panel F: IPTW-adjusted differences (key outcomes)
ax6 = fig.add_subplot(gs[3, 1:])
key_outcomes = ['90-Day\nMortality', '6-Month\nReadmission', '1-Year\nReadmission',
                'High Healthcare\nUtilization']
# Get adjusted results for binary outcomes (indices 4-13 in results_adjusted)
# Offset by 4 since first 4 are continuous outcomes
key_indices_adj = [2+4, 7+4, 8+4, 9+4]  # 90d mortality, 6m readmit, 1y readmit, high util
risk_diffs = [results_adjusted[i]['risk_diff'] for i in key_indices_adj]

colors_rd = ['#c62828' if rd > 2 else '#2e7d32' if rd < -2 else '#757575' for rd in risk_diffs]

y_pos = np.arange(len(key_outcomes))
bars = ax6.barh(y_pos, risk_diffs, color=colors_rd, alpha=0.8, 
                edgecolor='black', linewidth=1.5)
ax6.axvline(0, color='black', linestyle='--', linewidth=2.5)
ax6.set_yticks(y_pos)
ax6.set_yticklabels(key_outcomes, fontsize=10)
ax6.set_xlabel('IPTW-Adjusted Risk Difference (% points, ICU - Non-ICU)', fontweight='bold', fontsize=11)
ax6.set_title('F. Adjusted Long-Term Outcome Differences', fontweight='bold', fontsize=12, pad=12)
ax6.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, risk_diffs)):
    if abs(val) > 1:
        ax6.text(val, i, f'{val:+.1f}%', ha='left' if val > 0 else 'right',
                va='center', fontsize=9, fontweight='bold')

plt.savefig('true_long_term_outcomes.png', dpi=300, bbox_inches='tight')
print("✓ Saved: true_long_term_outcomes.png")
plt.close()

# ============================================================================
# SECTION 7: SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY: UNAMBIGUOUS TEMPORAL OUTCOMES")
print("="*80)

print("\n1. METHODOLOGY:")
print("   ✓ Used ONLY admission diagnosis (seq_num=1) for baseline")
print("   ✓ Outcomes have clear temporal ordering (LOS, mortality dates, readmissions)")
print("   ✓ No ambiguity about diagnosis timing")

print("\n2. KEY SIGNIFICANT FINDINGS (Unadjusted, p<0.05):")
sig_continuous = [r for r in results_unadjusted if r['type']=='continuous' and r['p_value'] < 0.05]
sig_binary = [r for r in results_unadjusted if r['type']=='binary' and r['p_value'] < 0.05]

if sig_continuous:
    print("\n   CONTINUOUS OUTCOMES:")
    for r in sig_continuous:
        print(f"   • {r['outcome']}: ICU {r['icu_value']:.1f} vs Non-ICU {r['non_icu_value']:.1f} (p={r['p_value']:.4f})")

if sig_binary:
    print("\n   BINARY OUTCOMES:")
    for r in sig_binary:
        print(f"   • {r['outcome']}: ICU {r['icu_value']:.1f}% vs Non-ICU {r['non_icu_value']:.1f}% (OR={r['or']:.2f}, p={r['p_value']:.4f})")

print("\n3. IPTW-ADJUSTED FINDINGS:")
print("   After controlling for baseline risk:")
for r in results_adjusted:
    if r['type'] == 'binary' and r['outcome'] in ['90-Day Mortality', '6-Month Readmission', '1-Year Readmission']:
        print(f"   • {r['outcome']}: RD = {r['risk_diff']:+.1f}%")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Save results
pd.DataFrame(results_unadjusted).to_csv('true_long_term_outcomes_unadjusted.csv', index=False)
pd.DataFrame(results_adjusted).to_csv('true_long_term_outcomes_adjusted.csv', index=False)
print("\nFiles saved:")
print("  • true_long_term_outcomes.png")
print("  • true_long_term_outcomes_unadjusted.csv")
print("  • true_long_term_outcomes_adjusted.csv")

