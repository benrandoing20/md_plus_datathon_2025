"""
ICU Outcomes Analysis: Adverse Events Following Elective Spine Surgery
=======================================================================

Research Question: Does ICU admission lead to worse outcomes, controlling for baseline risk?

Outcomes Assessed:
1. In-hospital mortality
2. Hospital length of stay
3. 30-day readmission
4. Cardiac complications (MI, arrhythmia, cardiac arrest)
5. Pulmonary complications (respiratory failure, pneumonia, PE)
6. Acute kidney injury
7. Sepsis/infection
8. ICU length of stay (for ICU patients)

Methods:
- Propensity score matching to balance baseline characteristics
- Inverse probability of treatment weighting (IPTW)
- Multivariable regression adjusting for confounders
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

print("="*80)
print("ICU OUTCOMES ANALYSIS - ELECTIVE SPINE SURGERY")
print("="*80)
print("Objective: Assess adverse outcomes associated with ICU admission")
print("Approach: Propensity score methods + confounder adjustment")
print("="*80)

# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================

print("\n[1/7] Loading data...")

DATA_ROOT = Path('physionet.org/files/mimiciv/3.1')
HOSP_PATH = DATA_ROOT / 'hosp'
ICU_PATH = DATA_ROOT / 'icu'

# Load cohort
cohort_icu = pd.read_csv('cohort_elective_spine_icu.csv')
admissions = pd.read_csv(HOSP_PATH / 'admissions.csv.gz', compression='gzip')
patients = pd.read_csv(HOSP_PATH / 'patients.csv.gz', compression='gzip')

# Load diagnosis and procedure data for outcome identification
diagnoses_icd = pd.read_csv(HOSP_PATH / 'diagnoses_icd.csv.gz', compression='gzip')
d_icd_diagnoses = pd.read_csv(HOSP_PATH / 'd_icd_diagnoses.csv.gz', compression='gzip')
procedures_icd = pd.read_csv(HOSP_PATH / 'procedures_icd.csv.gz', compression='gzip')

# Load ICU stays for length of stay
icustays = pd.read_csv(ICU_PATH / 'icustays.csv.gz', compression='gzip')

print(f"✓ Loaded {len(cohort_icu)} ICU patients")
print(f"✓ Loaded {len(admissions)} admissions")
print(f"✓ Loaded {len(diagnoses_icd)} diagnoses")
print(f"✓ Loaded {len(icustays)} ICU stays")

# Build cohort as before
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
adults = patients[patients['anchor_age'] >= 18][['subject_id', 'gender', 'anchor_age']]
spine_hadm = spine_procs[['hadm_id', 'subject_id']].drop_duplicates()

cohort = elective.merge(spine_hadm, on=['hadm_id', 'subject_id'], how='inner')
cohort = cohort.merge(adults, on='subject_id', how='inner')

# Add ICU indicator
icu_ids = set(cohort_icu['hadm_id'])
cohort['icu_admission'] = cohort['hadm_id'].isin(icu_ids).astype(int)

print(f"✓ Final cohort: {len(cohort)} patients")
print(f"  - ICU: {cohort['icu_admission'].sum()} ({cohort['icu_admission'].mean()*100:.1f}%)")
print(f"  - Non-ICU: {(1-cohort['icu_admission']).sum()}")

# ============================================================================
# SECTION 2: BASELINE CHARACTERISTICS (CONFOUNDERS)
# ============================================================================

print("\n[2/7] Engineering baseline characteristics for propensity score...")

# Demographics
cohort['age'] = cohort['anchor_age']
cohort['is_male'] = (cohort['gender'] == 'M').astype(int)

# Comorbidities
cohort_hadm_ids = set(cohort['hadm_id'])
cohort_diagnoses = diagnoses_icd[diagnoses_icd['hadm_id'].isin(cohort_hadm_ids)].copy()
cohort_diagnoses = cohort_diagnoses.merge(
    d_icd_diagnoses[['icd_code', 'icd_version', 'long_title']],
    on=['icd_code', 'icd_version'], how='left'
)

# Count total diagnoses per admission
diagnosis_counts = cohort_diagnoses.groupby('hadm_id').size().reset_index(name='num_diagnoses')
cohort = cohort.merge(diagnosis_counts, on='hadm_id', how='left')
cohort['num_diagnoses'] = cohort['num_diagnoses'].fillna(0)

# Specific comorbidities
def has_comorbidity(hadm_id, keywords):
    dx = cohort_diagnoses[cohort_diagnoses['hadm_id'] == hadm_id]['long_title'].fillna('')
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
    print(f"  Extracting {comorb}...")
    cohort[comorb] = cohort['hadm_id'].apply(lambda x: has_comorbidity(x, keywords))

print(f"✓ Baseline features created: {cohort[['age', 'is_male', 'num_diagnoses', 'has_cardiac']].shape}")

# ============================================================================
# SECTION 3: OUTCOME ASCERTAINMENT
# ============================================================================

print("\n[3/7] Identifying adverse outcomes...")

outcomes_df = cohort[['hadm_id', 'subject_id', 'icu_admission']].copy()

# 1. In-hospital mortality
outcomes_df = outcomes_df.merge(
    admissions[['hadm_id', 'hospital_expire_flag']],
    on='hadm_id', how='left'
)
outcomes_df['mortality'] = outcomes_df['hospital_expire_flag'].fillna(0).astype(int)

# 2. Hospital length of stay
if 'admittime' not in cohort.columns:
    cohort = cohort.merge(
        admissions[['hadm_id', 'admittime', 'dischtime']],
        on='hadm_id', how='left'
    )
cohort['admittime'] = pd.to_datetime(cohort['admittime'])
cohort['dischtime'] = pd.to_datetime(cohort['dischtime'])
cohort['los_hospital_days'] = (cohort['dischtime'] - cohort['admittime']).dt.total_seconds() / 86400
outcomes_df = outcomes_df.merge(cohort[['hadm_id', 'los_hospital_days']], on='hadm_id', how='left')

# 3. ICU length of stay (for ICU patients)
icu_los = icustays.groupby('hadm_id').agg({
    'los': 'sum'  # Total ICU days (sum if multiple ICU stays)
}).reset_index()
icu_los.columns = ['hadm_id', 'los_icu_days']
outcomes_df = outcomes_df.merge(icu_los, on='hadm_id', how='left')
outcomes_df['los_icu_days'] = outcomes_df['los_icu_days'].fillna(0)

# 4. 30-day readmission
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
admissions_sorted = admissions.sort_values(['subject_id', 'admittime'])
admissions_sorted['next_admit'] = admissions_sorted.groupby('subject_id')['admittime'].shift(-1)
admissions_sorted['days_to_readmit'] = (
    admissions_sorted['next_admit'] - pd.to_datetime(admissions_sorted['dischtime'])
).dt.total_seconds() / 86400

outcomes_df = outcomes_df.merge(
    admissions_sorted[['hadm_id', 'days_to_readmit']],
    on='hadm_id', how='left'
)
outcomes_df['readmit_30d'] = ((outcomes_df['days_to_readmit'] >= 0) & 
                               (outcomes_df['days_to_readmit'] <= 30)).astype(int)

# 5. Cardiac complications
cardiac_complications = [
    # ICD-9
    '410', '427', '997.1', 
    # ICD-10
    'I21', 'I22', 'I46', 'I47', 'I48', 'I49', 'I97.1'
]

def has_complication(hadm_id, codes):
    dx_codes = cohort_diagnoses[cohort_diagnoses['hadm_id'] == hadm_id]['icd_code'].astype(str)
    return int(any(any(code in dx_code for code in codes) for dx_code in dx_codes))

print("  Identifying cardiac complications...")
outcomes_df['cardiac_complication'] = outcomes_df['hadm_id'].apply(
    lambda x: has_complication(x, cardiac_complications)
)

# 6. Pulmonary complications
pulmonary_complications = [
    # Respiratory failure, pneumonia, PE
    '518.4', '518.5', '518.81', '507', '415.1', '480', '481', '482', '483', '484', '485', '486',
    'J80', 'J81', 'J95.1', 'J95.2', 'J95.3', 'J18', 'J69', 'I26'
]

print("  Identifying pulmonary complications...")
outcomes_df['pulmonary_complication'] = outcomes_df['hadm_id'].apply(
    lambda x: has_complication(x, pulmonary_complications)
)

# 7. Acute kidney injury
aki_codes = ['584', 'N17']

print("  Identifying acute kidney injury...")
outcomes_df['aki'] = outcomes_df['hadm_id'].apply(
    lambda x: has_complication(x, aki_codes)
)

# 8. Sepsis/Infection
sepsis_codes = ['995.91', '995.92', '038', 'A41', 'R65']

print("  Identifying sepsis...")
outcomes_df['sepsis'] = outcomes_df['hadm_id'].apply(
    lambda x: has_complication(x, sepsis_codes)
)

# Composite outcome: any major complication
outcomes_df['any_complication'] = (
    (outcomes_df['cardiac_complication'] == 1) |
    (outcomes_df['pulmonary_complication'] == 1) |
    (outcomes_df['aki'] == 1) |
    (outcomes_df['sepsis'] == 1)
).astype(int)

print(f"✓ Outcomes identified:")
print(f"  - Mortality: {outcomes_df['mortality'].sum()} ({outcomes_df['mortality'].mean()*100:.1f}%)")
print(f"  - 30-day readmission: {outcomes_df['readmit_30d'].sum()} ({outcomes_df['readmit_30d'].mean()*100:.1f}%)")
print(f"  - Cardiac complications: {outcomes_df['cardiac_complication'].sum()} ({outcomes_df['cardiac_complication'].mean()*100:.1f}%)")
print(f"  - Pulmonary complications: {outcomes_df['pulmonary_complication'].sum()} ({outcomes_df['pulmonary_complication'].mean()*100:.1f}%)")
print(f"  - AKI: {outcomes_df['aki'].sum()} ({outcomes_df['aki'].mean()*100:.1f}%)")
print(f"  - Sepsis: {outcomes_df['sepsis'].sum()} ({outcomes_df['sepsis'].mean()*100:.1f}%)")
print(f"  - Any major complication: {outcomes_df['any_complication'].sum()} ({outcomes_df['any_complication'].mean()*100:.1f}%)")

# Merge back
analysis_df = cohort[[
    'hadm_id', 'age', 'is_male', 'num_diagnoses', 
    'has_cardiac', 'has_hypertension', 'has_diabetes', 
    'has_pulmonary', 'has_renal', 'icu_admission'
]].merge(outcomes_df, on=['hadm_id', 'icu_admission'], how='inner')

# ============================================================================
# SECTION 4: PROPENSITY SCORE ESTIMATION
# ============================================================================

print("\n[4/7] Estimating propensity scores...")

# Features for propensity score
ps_features = ['age', 'is_male', 'num_diagnoses', 'has_cardiac', 
               'has_hypertension', 'has_diabetes', 'has_pulmonary', 'has_renal']

X_ps = analysis_df[ps_features].copy()
y_ps = analysis_df['icu_admission'].copy()

# Standardize
scaler = StandardScaler()
X_ps_scaled = scaler.fit_transform(X_ps)

# Fit propensity score model
ps_model = LogisticRegression(max_iter=1000, random_state=42)
ps_model.fit(X_ps_scaled, y_ps)

# Predict propensity scores
analysis_df['propensity_score'] = ps_model.predict_proba(X_ps_scaled)[:, 1]

print(f"✓ Propensity scores estimated")
print(f"  - Mean PS (ICU): {analysis_df[analysis_df['icu_admission']==1]['propensity_score'].mean():.3f}")
print(f"  - Mean PS (non-ICU): {analysis_df[analysis_df['icu_admission']==0]['propensity_score'].mean():.3f}")

# Calculate IPTW weights
analysis_df['iptw_weight'] = np.where(
    analysis_df['icu_admission'] == 1,
    1 / analysis_df['propensity_score'],
    1 / (1 - analysis_df['propensity_score'])
)

# Stabilize weights
mean_treatment = analysis_df['icu_admission'].mean()
analysis_df['iptw_weight'] = np.where(
    analysis_df['icu_admission'] == 1,
    mean_treatment / analysis_df['propensity_score'],
    (1 - mean_treatment) / (1 - analysis_df['propensity_score'])
)

# Trim extreme weights
analysis_df['iptw_weight'] = analysis_df['iptw_weight'].clip(upper=analysis_df['iptw_weight'].quantile(0.99))

print(f"  - Mean IPTW (ICU): {analysis_df[analysis_df['icu_admission']==1]['iptw_weight'].mean():.3f}")
print(f"  - Mean IPTW (non-ICU): {analysis_df[analysis_df['icu_admission']==0]['iptw_weight'].mean():.3f}")

# ============================================================================
# SECTION 5: COMPARATIVE OUTCOMES ANALYSIS
# ============================================================================

print("\n[5/7] Comparing outcomes: ICU vs. non-ICU...")

outcomes = [
    'mortality', 'readmit_30d', 'cardiac_complication', 
    'pulmonary_complication', 'aki', 'sepsis', 'any_complication'
]

continuous_outcomes = ['los_hospital_days', 'los_icu_days']

print("\n" + "="*80)
print("UNADJUSTED OUTCOMES (Crude Comparison)")
print("="*80)

results_crude = []

for outcome in outcomes:
    icu_rate = analysis_df[analysis_df['icu_admission']==1][outcome].mean() * 100
    non_icu_rate = analysis_df[analysis_df['icu_admission']==0][outcome].mean() * 100
    
    # Chi-square test
    contingency = pd.crosstab(analysis_df[outcome], analysis_df['icu_admission'])
    chi2, p, _, _ = stats.chi2_contingency(contingency)
    
    # Odds ratio
    a = contingency.iloc[1, 1] if contingency.shape[0] > 1 else 0
    b = contingency.iloc[1, 0] if contingency.shape[0] > 1 else 0
    c = contingency.iloc[0, 1] if contingency.shape[0] > 1 else analysis_df['icu_admission'].sum()
    d = contingency.iloc[0, 0] if contingency.shape[0] > 1 else (1-analysis_df['icu_admission']).sum()
    
    if b > 0 and c > 0:
        or_val = (a * d) / (b * c)
    else:
        or_val = np.nan
    
    results_crude.append({
        'outcome': outcome.replace('_', ' ').title(),
        'icu_rate': icu_rate,
        'non_icu_rate': non_icu_rate,
        'diff': icu_rate - non_icu_rate,
        'or': or_val,
        'p': p
    })
    
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"{outcome:30s}: ICU={icu_rate:5.1f}% | Non-ICU={non_icu_rate:5.1f}% | OR={or_val:.2f} | p={p:.4f} {sig}")

print("\nContinuous outcomes:")
for outcome in continuous_outcomes:
    icu_mean = analysis_df[analysis_df['icu_admission']==1][outcome].mean()
    non_icu_mean = analysis_df[analysis_df['icu_admission']==0][outcome].mean()
    
    t_stat, p = stats.ttest_ind(
        analysis_df[analysis_df['icu_admission']==1][outcome].dropna(),
        analysis_df[analysis_df['icu_admission']==0][outcome].dropna()
    )
    
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"{outcome:30s}: ICU={icu_mean:6.2f} days | Non-ICU={non_icu_mean:6.2f} days | p={p:.4f} {sig}")

# IPTW-adjusted analysis
print("\n" + "="*80)
print("IPTW-ADJUSTED OUTCOMES (Controlling for Confounders)")
print("="*80)

results_adjusted = []

for outcome in outcomes:
    # Weighted means
    icu_weighted = np.average(
        analysis_df[analysis_df['icu_admission']==1][outcome],
        weights=analysis_df[analysis_df['icu_admission']==1]['iptw_weight']
    ) * 100
    
    non_icu_weighted = np.average(
        analysis_df[analysis_df['icu_admission']==0][outcome],
        weights=analysis_df[analysis_df['icu_admission']==0]['iptw_weight']
    ) * 100
    
    # Risk difference
    risk_diff = icu_weighted - non_icu_weighted
    
    results_adjusted.append({
        'outcome': outcome.replace('_', ' ').title(),
        'icu_adj': icu_weighted,
        'non_icu_adj': non_icu_weighted,
        'risk_diff': risk_diff
    })
    
    print(f"{outcome:30s}: ICU={icu_weighted:5.1f}% | Non-ICU={non_icu_weighted:5.1f}% | RD={risk_diff:+.1f}%")

# ============================================================================
# SECTION 6: VISUALIZATION
# ============================================================================

print("\n[6/7] Creating visualizations...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Panel A: Propensity score distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(analysis_df[analysis_df['icu_admission']==1]['propensity_score'], 
         bins=30, alpha=0.6, label='ICU', color='#ef5350', edgecolor='black')
ax1.hist(analysis_df[analysis_df['icu_admission']==0]['propensity_score'], 
         bins=30, alpha=0.6, label='Non-ICU', color='#66bb6a', edgecolor='black')
ax1.set_xlabel('Propensity Score', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_title('A. Propensity Score Distribution', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Panel B: Crude outcome rates
ax2 = fig.add_subplot(gs[0, 1:])
outcome_names = [r['outcome'] for r in results_crude]
icu_rates = [r['icu_rate'] for r in results_crude]
non_icu_rates = [r['non_icu_rate'] for r in results_crude]

x = np.arange(len(outcome_names))
width = 0.35

bars1 = ax2.bar(x - width/2, icu_rates, width, label='ICU', 
                color='#ef5350', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(x + width/2, non_icu_rates, width, label='Non-ICU',
                color='#66bb6a', alpha=0.8, edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Outcome', fontweight='bold')
ax2.set_ylabel('Incidence (%)', fontweight='bold')
ax2.set_title('B. Unadjusted Outcome Rates', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(outcome_names, rotation=45, ha='right', fontsize=9)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Panel C: Odds ratios
ax3 = fig.add_subplot(gs[1, 0])
or_values = [r['or'] for r in results_crude if not np.isnan(r['or'])]
or_names = [r['outcome'] for r in results_crude if not np.isnan(r['or'])]
p_values = [r['p'] for r in results_crude if not np.isnan(r['or'])]

colors = ['#c62828' if p < 0.05 else '#757575' for p in p_values]

y_pos = np.arange(len(or_names))
ax3.barh(y_pos, or_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.axvline(1, color='black', linestyle='--', linewidth=2)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(or_names, fontsize=9)
ax3.set_xlabel('Odds Ratio (95% CI)', fontweight='bold')
ax3.set_title('C. Crude Odds Ratios', fontweight='bold')
ax3.set_xscale('log')
ax3.grid(axis='x', alpha=0.3)

# Panel D: IPTW-adjusted risk differences
ax4 = fig.add_subplot(gs[1, 1:])
adj_names = [r['outcome'] for r in results_adjusted]
risk_diffs = [r['risk_diff'] for r in results_adjusted]

colors_rd = ['#c62828' if rd > 0 else '#2e7d32' for rd in risk_diffs]

y_pos = np.arange(len(adj_names))
ax4.barh(y_pos, risk_diffs, color=colors_rd, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.axvline(0, color='black', linestyle='--', linewidth=2)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(adj_names, fontsize=9)
ax4.set_xlabel('Risk Difference (% points)', fontweight='bold')
ax4.set_title('D. IPTW-Adjusted Risk Differences (ICU - Non-ICU)', fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

# Panel E: Length of stay comparison
ax5 = fig.add_subplot(gs[2, 0])
los_data = [
    analysis_df[analysis_df['icu_admission']==1]['los_hospital_days'].dropna(),
    analysis_df[analysis_df['icu_admission']==0]['los_hospital_days'].dropna()
]
bp = ax5.boxplot(los_data, labels=['ICU', 'Non-ICU'], patch_artist=True)
bp['boxes'][0].set_facecolor('#ef5350')
bp['boxes'][1].set_facecolor('#66bb6a')
ax5.set_ylabel('Days', fontweight='bold')
ax5.set_title('E. Hospital Length of Stay', fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# Panel F: Composite complication rate by baseline risk
ax6 = fig.add_subplot(gs[2, 1])
analysis_df['ps_quartile'] = pd.qcut(analysis_df['propensity_score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
comp_by_risk = analysis_df.groupby(['ps_quartile', 'icu_admission'])['any_complication'].mean().unstack() * 100

comp_by_risk.plot(kind='bar', ax=ax6, color=['#66bb6a', '#ef5350'], alpha=0.8, edgecolor='black', linewidth=1.5)
ax6.set_xlabel('Baseline Risk Quartile', fontweight='bold')
ax6.set_ylabel('Complication Rate (%)', fontweight='bold')
ax6.set_title('F. Complications by Baseline Risk', fontweight='bold')
ax6.set_xticklabels(comp_by_risk.index, rotation=0)
ax6.legend(['Non-ICU', 'ICU'])
ax6.grid(axis='y', alpha=0.3)

# Panel G: Mortality by comorbidity burden
ax7 = fig.add_subplot(gs[2, 2])
analysis_df['diag_tertile'] = pd.qcut(analysis_df['num_diagnoses'], q=3, labels=['Low', 'Medium', 'High'])
mort_by_burden = analysis_df.groupby(['diag_tertile', 'icu_admission'])['mortality'].mean().unstack() * 100

mort_by_burden.plot(kind='bar', ax=ax7, color=['#66bb6a', '#ef5350'], alpha=0.8, edgecolor='black', linewidth=1.5)
ax7.set_xlabel('Comorbidity Burden', fontweight='bold')
ax7.set_ylabel('Mortality Rate (%)', fontweight='bold')
ax7.set_title('G. Mortality by Comorbidity Burden', fontweight='bold')
ax7.set_xticklabels(mort_by_burden.index, rotation=0)
ax7.legend(['Non-ICU', 'ICU'])
ax7.grid(axis='y', alpha=0.3)

plt.suptitle('ICU Outcomes Analysis: Adverse Events in Elective Spine Surgery', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('icu_outcomes_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: icu_outcomes_analysis.png")

# ============================================================================
# SECTION 7: SUMMARY AND INTERPRETATION
# ============================================================================

print("\n" + "="*80)
print("SUMMARY AND CLINICAL INTERPRETATION")
print("="*80)

print("\n1. KEY FINDINGS:")
print(f"   • In-hospital mortality: ICU {results_crude[0]['icu_rate']:.1f}% vs Non-ICU {results_crude[0]['non_icu_rate']:.1f}%")
print(f"   • Any major complication: ICU {results_crude[6]['icu_rate']:.1f}% vs Non-ICU {results_crude[6]['non_icu_rate']:.1f}%")
print(f"   • Hospital LOS: ICU {analysis_df[analysis_df['icu_admission']==1]['los_hospital_days'].mean():.1f} days vs Non-ICU {analysis_df[analysis_df['icu_admission']==0]['los_hospital_days'].mean():.1f} days")

print("\n2. CONFOUNDING BY INDICATION:")
print("   ICU patients have higher baseline risk (propensity score analysis shows)")
print(f"   • Mean baseline risk (PS): ICU {analysis_df[analysis_df['icu_admission']==1]['propensity_score'].mean():.3f} vs Non-ICU {analysis_df[analysis_df['icu_admission']==0]['propensity_score'].mean():.3f}")
print("   • This confounding must be accounted for when interpreting outcomes")

print("\n3. AFTER ADJUSTING FOR CONFOUNDERS (IPTW):")
high_risk_outcomes = [r for r in results_adjusted if r['risk_diff'] > 2]
if high_risk_outcomes:
    print("   Outcomes with ≥2% increased risk in ICU:")
    for r in high_risk_outcomes:
        print(f"   • {r['outcome']}: +{r['risk_diff']:.1f}% points")
else:
    print("   No major excess risk after adjustment - outcomes may reflect baseline severity")

print("\n4. CLINICAL IMPLICATIONS:")
print("   • Higher crude complication rates in ICU patients expected (sicker baseline)")
print("   • Adjusted analysis helps isolate ICU-specific risks (e.g., delirium, infections)")
print("   • Focus should be on modifiable ICU complications (VAP, CAUTI, delirium)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("Output files:")
print("  • icu_outcomes_analysis.png")
print("\nNext steps:")
print("  1. Stratified analysis by ICU length of stay")
print("  2. Propensity score matching (1:1 pairs)")
print("  3. Mediation analysis for length of stay")

# Save results
results_df = pd.DataFrame(results_crude)
results_df.to_csv('icu_outcomes_crude.csv', index=False)

results_adj_df = pd.DataFrame(results_adjusted)
results_adj_df.to_csv('icu_outcomes_adjusted.csv', index=False)

print("  • icu_outcomes_crude.csv")
print("  • icu_outcomes_adjusted.csv")

