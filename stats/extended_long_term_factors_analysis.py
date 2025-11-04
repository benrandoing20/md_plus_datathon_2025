"""
EXTENDED LONG-TERM FACTORS ANALYSIS
====================================

Goal: Identify ADDITIONAL long-term factors beyond basic mortality/readmission

Focus on:
1. New chronic conditions developed (cardiovascular, renal, pulmonary)
2. Time-stratified complications (30d, 90d, 6m, 1yr)
3. Repeated hospitalizations by cause
4. Progressive organ dysfunction
5. Healthcare utilization intensity
6. Multiple system involvement
7. Composite adverse outcomes

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (22, 20)

print("="*80)
print("EXTENDED LONG-TERM FACTORS ANALYSIS")
print("="*80)
print("Finding ADDITIONAL factors beyond basic mortality/readmission")
print("="*80)

# ============================================================================
# DATA LOADING
# ============================================================================

print("\n[1/10] Loading data...")

DATA_ROOT = Path('physionet.org/files/mimiciv/3.1')
HOSP_PATH = DATA_ROOT / 'hosp'
ICU_PATH = DATA_ROOT / 'icu'

cohort_icu = pd.read_csv('cohort_elective_spine_icu.csv')
admissions = pd.read_csv(HOSP_PATH / 'admissions.csv.gz', compression='gzip')
patients = pd.read_csv(HOSP_PATH / 'patients.csv.gz', compression='gzip')
diagnoses_icd = pd.read_csv(HOSP_PATH / 'diagnoses_icd.csv.gz', compression='gzip')
d_icd_diagnoses = pd.read_csv(HOSP_PATH / 'd_icd_diagnoses.csv.gz', compression='gzip')
procedures_icd = pd.read_csv(HOSP_PATH / 'procedures_icd.csv.gz', compression='gzip')

# Build cohort
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

# Dates
for col in ['admittime', 'dischtime', 'deathtime', 'dod']:
    if col in cohort.columns:
        cohort[col] = pd.to_datetime(cohort[col])
    if col in admissions.columns:
        admissions[col] = pd.to_datetime(admissions[col])

print(f"✓ Cohort: {len(cohort)} patients (ICU: {cohort['icu_admission'].sum()}, Non-ICU: {(1-cohort['icu_admission']).sum()})")

# ============================================================================
# SECTION 2: NEW CHRONIC CONDITIONS DEVELOPED
# ============================================================================

print("\n[2/10] Analyzing new chronic conditions developed post-discharge...")

# Get index admission diagnoses (baseline)
index_diagnoses = diagnoses_icd[diagnoses_icd['hadm_id'].isin(set(cohort['hadm_id']))].copy()
index_diagnoses = index_diagnoses.merge(
    d_icd_diagnoses[['icd_code', 'icd_version', 'long_title']],
    on=['icd_code', 'icd_version'], how='left'
)

# Get all subsequent admissions
cohort_with_dates = cohort[['hadm_id', 'subject_id', 'icu_admission', 'dischtime']].copy()
all_admissions_sorted = admissions.sort_values(['subject_id', 'admittime']).copy()

# Define chronic conditions to track
chronic_conditions = {
    'new_heart_failure': ['heart failure', 'congestive heart', 'chf', 'cardiac failure'],
    'new_ckd': ['chronic kidney disease', 'ckd', 'chronic renal', 'esrd', 'end stage renal'],
    'new_copd': ['chronic obstructive', 'copd', 'chronic bronchitis', 'emphysema'],
    'new_diabetes_complications': ['diabetic neuropathy', 'diabetic retinopathy', 'diabetic nephropathy'],
    'new_stroke_tia': ['stroke', 'cerebrovascular accident', 'cva', 'transient ischemic', 'tia'],
    'new_afib': ['atrial fibrillation', 'atrial flutter'],
    'new_depression': ['major depressive', 'depression'],
    'new_dementia': ['dementia', 'cognitive decline', 'alzheimer'],
    'new_osteoporosis': ['osteoporosis', 'osteopenia'],
    'new_chronic_pain': ['chronic pain', 'fibromyalgia']
}

def had_baseline_condition(hadm_id, keywords):
    """Check if condition existed at baseline"""
    dx = index_diagnoses[index_diagnoses['hadm_id'] == hadm_id]['long_title'].fillna('')
    return any(any(kw.lower() in str(title).lower() for kw in keywords) 
               for title in dx if isinstance(title, str))

def developed_condition_postdischarge(subject_id, index_discharge, keywords, timeframe_days=365):
    """Check if condition appeared in subsequent admissions"""
    future_admits = all_admissions_sorted[
        (all_admissions_sorted['subject_id'] == subject_id) &
        (all_admissions_sorted['admittime'] > index_discharge) &
        (all_admissions_sorted['admittime'] <= index_discharge + timedelta(days=timeframe_days))
    ]
    
    if len(future_admits) == 0:
        return False
    
    future_hadm_ids = set(future_admits['hadm_id'])
    future_dx = diagnoses_icd[diagnoses_icd['hadm_id'].isin(future_hadm_ids)].merge(
        d_icd_diagnoses[['icd_code', 'icd_version', 'long_title']],
        on=['icd_code', 'icd_version'], how='left'
    )
    
    return any(any(kw.lower() in str(title).lower() for kw in keywords)
               for title in future_dx['long_title'].fillna('') if isinstance(title, str))

print("\nIdentifying new chronic conditions (not present at baseline)...")
chronic_df = cohort_with_dates.copy()

for condition, keywords in chronic_conditions.items():
    print(f"  Analyzing {condition}...")
    
    # Check baseline
    chronic_df[f'{condition}_baseline'] = chronic_df['hadm_id'].apply(
        lambda x: had_baseline_condition(x, keywords)
    )
    
    # Check development within 1 year
    chronic_df[f'{condition}_developed'] = chronic_df.apply(
        lambda row: developed_condition_postdischarge(
            row['subject_id'], row['dischtime'], keywords
        ) if not chronic_df.loc[chronic_df['hadm_id']==row['hadm_id'], f'{condition}_baseline'].iloc[0] else False,
        axis=1
    )

# Calculate rates
print("\nNEW CHRONIC CONDITIONS DEVELOPED (among those WITHOUT condition at baseline):")
new_conditions_summary = []

for condition in chronic_conditions.keys():
    # Only among those who didn't have it at baseline
    eligible_icu = chronic_df[
        (chronic_df['icu_admission']==1) & 
        (chronic_df[f'{condition}_baseline']==False)
    ]
    eligible_non_icu = chronic_df[
        (chronic_df['icu_admission']==0) & 
        (chronic_df[f'{condition}_baseline']==False)
    ]
    
    if len(eligible_icu) > 0 and len(eligible_non_icu) > 0:
        icu_rate = eligible_icu[f'{condition}_developed'].mean() * 100
        non_icu_rate = eligible_non_icu[f'{condition}_developed'].mean() * 100
        
        # Statistical test
        contingency = pd.crosstab(
            chronic_df[chronic_df[f'{condition}_baseline']==False][f'{condition}_developed'],
            chronic_df[chronic_df[f'{condition}_baseline']==False]['icu_admission']
        )
        
        if contingency.shape[0] > 1:
            chi2, p, _, _ = stats.chi2_contingency(contingency)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            
            print(f"  {condition:35s}: ICU {icu_rate:5.1f}% | Non-ICU {non_icu_rate:5.1f}% | p={p:.4f} {sig}")
            
            new_conditions_summary.append({
                'condition': condition,
                'icu_rate': icu_rate,
                'non_icu_rate': non_icu_rate,
                'p_value': p,
                'n_eligible_icu': len(eligible_icu),
                'n_eligible_non_icu': len(eligible_non_icu)
            })

# ============================================================================
# SECTION 3: TIME-STRATIFIED READMISSION REASONS
# ============================================================================

print("\n[3/10] Analyzing readmission reasons by timeframe...")

readmission_categories = {
    'cardiac': ['heart', 'cardiac', 'myocardial', 'angina', 'coronary'],
    'infection': ['infection', 'sepsis', 'pneumonia', 'uti', 'cellulitis'],
    'gi': ['gastrointestinal', 'bowel', 'ileus', 'obstruction', 'gi bleed'],
    'respiratory': ['respiratory', 'pulmonary', 'dyspnea', 'copd exacerbation'],
    'neurological': ['stroke', 'seizure', 'altered mental', 'confusion'],
    'pain': ['pain', 'chronic pain', 'acute pain'],
    'surgical_complication': ['wound', 'surgical site', 'dehiscence', 'hematoma']
}

def categorize_readmission(hadm_id, categories_dict):
    """Categorize readmission by primary diagnosis"""
    dx = diagnoses_icd[
        (diagnoses_icd['hadm_id'] == hadm_id) &
        (diagnoses_icd['seq_num'] == 1)
    ].merge(
        d_icd_diagnoses[['icd_code', 'icd_version', 'long_title']],
        on=['icd_code', 'icd_version'], how='left'
    )
    
    if len(dx) == 0:
        return 'other'
    
    primary_dx = str(dx['long_title'].iloc[0]).lower()
    
    for category, keywords in categories_dict.items():
        if any(kw in primary_dx for kw in keywords):
            return category
    
    return 'other'

# Analyze readmission reasons
print("\nCategorizing readmissions by timeframe...")
readmit_reasons = []

for idx, row in cohort_with_dates.iterrows():
    subject_id = row['subject_id']
    index_discharge = row['dischtime']
    icu = row['icu_admission']
    
    # Find readmissions at different timepoints
    future_admits = all_admissions_sorted[
        (all_admissions_sorted['subject_id'] == subject_id) &
        (all_admissions_sorted['admittime'] > index_discharge)
    ].copy()
    
    if len(future_admits) > 0:
        future_admits['days_since_discharge'] = (
            future_admits['admittime'] - index_discharge
        ).dt.days
        
        # 30-day readmissions
        readmit_30d = future_admits[future_admits['days_since_discharge'] <= 30]
        if len(readmit_30d) > 0:
            reason = categorize_readmission(readmit_30d.iloc[0]['hadm_id'], readmission_categories)
            readmit_reasons.append({
                'icu': icu,
                'timeframe': '30-day',
                'reason': reason
            })
        
        # 31-90 day readmissions
        readmit_90d = future_admits[
            (future_admits['days_since_discharge'] > 30) &
            (future_admits['days_since_discharge'] <= 90)
        ]
        if len(readmit_90d) > 0:
            reason = categorize_readmission(readmit_90d.iloc[0]['hadm_id'], readmission_categories)
            readmit_reasons.append({
                'icu': icu,
                'timeframe': '31-90-day',
                'reason': reason
            })

readmit_df = pd.DataFrame(readmit_reasons)

if len(readmit_df) > 0:
    print("\nREADMISSION REASONS BY TIMEFRAME:")
    for timeframe in ['30-day', '31-90-day']:
        print(f"\n{timeframe} Readmissions:")
        tf_data = readmit_df[readmit_df['timeframe'] == timeframe]
        if len(tf_data) > 0:
            reason_summary = tf_data.groupby(['icu', 'reason']).size().unstack(fill_value=0)
            reason_pct = reason_summary.div(reason_summary.sum(axis=1), axis=0) * 100
            
            if 1 in reason_pct.index:
                print("  ICU patients:")
                for reason in reason_pct.columns:
                    print(f"    {reason}: {reason_pct.loc[1, reason]:.1f}%")
            
            if 0 in reason_pct.index:
                print("  Non-ICU patients:")
                for reason in reason_pct.columns:
                    print(f"    {reason}: {reason_pct.loc[0, reason]:.1f}%")

# ============================================================================
# SECTION 4: MULTIPLE SYSTEM INVOLVEMENT
# ============================================================================

print("\n[4/10] Analyzing multiple organ system involvement...")

organ_systems = {
    'cardiovascular': ['heart', 'cardiac', 'vascular', 'coronary'],
    'respiratory': ['lung', 'pulmonary', 'respiratory', 'pneumonia'],
    'renal': ['kidney', 'renal', 'urinary'],
    'hepatic': ['liver', 'hepatic', 'cirrhosis'],
    'neurological': ['brain', 'neurological', 'stroke', 'seizure'],
    'endocrine': ['diabetes', 'thyroid', 'endocrine'],
    'hematologic': ['anemia', 'coagulation', 'thrombosis']
}

def count_systems_involved(hadm_id, systems_dict):
    """Count number of organ systems with diagnoses"""
    dx = index_diagnoses[index_diagnoses['hadm_id'] == hadm_id]['long_title'].fillna('')
    systems_affected = set()
    
    for system, keywords in systems_dict.items():
        if any(any(kw.lower() in str(title).lower() for kw in keywords)
               for title in dx if isinstance(title, str)):
            systems_affected.add(system)
    
    return len(systems_affected)

cohort_with_dates['num_systems_involved'] = cohort_with_dates['hadm_id'].apply(
    lambda x: count_systems_involved(x, organ_systems)
)

cohort_with_dates['multiple_systems'] = (cohort_with_dates['num_systems_involved'] >= 3).astype(int)

icu_systems = cohort_with_dates[cohort_with_dates['icu_admission']==1]['num_systems_involved'].mean()
non_icu_systems = cohort_with_dates[cohort_with_dates['icu_admission']==0]['num_systems_involved'].mean()

icu_multiple = cohort_with_dates[cohort_with_dates['icu_admission']==1]['multiple_systems'].mean() * 100
non_icu_multiple = cohort_with_dates[cohort_with_dates['icu_admission']==0]['multiple_systems'].mean() * 100

t_stat, p_systems = stats.ttest_ind(
    cohort_with_dates[cohort_with_dates['icu_admission']==1]['num_systems_involved'],
    cohort_with_dates[cohort_with_dates['icu_admission']==0]['num_systems_involved']
)

print(f"\nOrgan Systems Involved:")
print(f"  ICU: {icu_systems:.2f} systems | Non-ICU: {non_icu_systems:.2f} systems (p={p_systems:.4f})")
print(f"  Multiple systems (≥3): ICU {icu_multiple:.1f}% | Non-ICU {non_icu_multiple:.1f}%")

# ============================================================================
# SECTION 5: HEALTHCARE UTILIZATION INTENSITY
# ============================================================================

print("\n[5/10] Analyzing healthcare utilization intensity...")

# Calculate metrics per patient
util_df = cohort_with_dates.copy()

# From previous analysis
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])

utilization_metrics = []

for idx, row in util_df.iterrows():
    subject_id = row['subject_id']
    index_discharge = row['dischtime']
    
    future_admits = admissions[
        (admissions['subject_id'] == subject_id) &
        (admissions['admittime'] > index_discharge) &
        (admissions['admittime'] <= index_discharge + timedelta(days=365))
    ]
    
    num_admits = len(future_admits)
    
    if num_admits > 0:
        total_days = (pd.to_datetime(future_admits['dischtime']) - 
                     pd.to_datetime(future_admits['admittime'])).dt.total_seconds().sum() / 86400
        num_emergency = (future_admits['admission_type'].str.upper() == 'EMERGENCY').sum()
    else:
        total_days = 0
        num_emergency = 0
    
    utilization_metrics.append({
        'hadm_id': row['hadm_id'],
        'icu_admission': row['icu_admission'],
        'readmissions_1yr': num_admits,
        'total_hospital_days_1yr': total_days,
        'emergency_admits_1yr': num_emergency,
        'frequent_utilizer': 1 if num_admits >= 3 else 0
    })

util_metrics_df = pd.DataFrame(utilization_metrics)

print("\nHealthcare Utilization Metrics (1 year post-discharge):")
for metric in ['readmissions_1yr', 'total_hospital_days_1yr', 'emergency_admits_1yr']:
    icu_mean = util_metrics_df[util_metrics_df['icu_admission']==1][metric].mean()
    non_icu_mean = util_metrics_df[util_metrics_df['icu_admission']==0][metric].mean()
    
    t_stat, p_val = stats.ttest_ind(
        util_metrics_df[util_metrics_df['icu_admission']==1][metric],
        util_metrics_df[util_metrics_df['icu_admission']==0][metric]
    )
    
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
    print(f"  {metric:30s}: ICU {icu_mean:6.2f} | Non-ICU {non_icu_mean:6.2f} | p={p_val:.4f} {sig}")

# Frequent utilizers
icu_freq = util_metrics_df[util_metrics_df['icu_admission']==1]['frequent_utilizer'].mean() * 100
non_icu_freq = util_metrics_df[util_metrics_df['icu_admission']==0]['frequent_utilizer'].mean() * 100
print(f"  Frequent utilizers (≥3 admits): ICU {icu_freq:.1f}% | Non-ICU {non_icu_freq:.1f}%")

# ============================================================================
# SECTION 6: COMPOSITE ADVERSE OUTCOMES
# ============================================================================

print("\n[6/10] Creating composite outcome measures...")

# Merge all data
composite_df = util_metrics_df.merge(cohort_with_dates, on=['hadm_id', 'icu_admission'])
composite_df = composite_df.merge(chronic_df[[c for c in chronic_df.columns if 'developed' in c] + ['hadm_id']], 
                                   on='hadm_id', how='left')

# Define composite outcomes
composite_df['major_adverse_event'] = (
    (composite_df['readmissions_1yr'] >= 2) |
    (composite_df['emergency_admits_1yr'] >= 1) |
    (composite_df['multiple_systems'] == 1)
).astype(int)

composite_df['poor_outcome'] = (
    (composite_df['total_hospital_days_1yr'] > 14) |
    (composite_df['frequent_utilizer'] == 1) |
    (composite_df['multiple_systems'] == 1)
).astype(int)

print("\nComposite Outcomes:")
for outcome in ['major_adverse_event', 'poor_outcome']:
    icu_rate = composite_df[composite_df['icu_admission']==1][outcome].mean() * 100
    non_icu_rate = composite_df[composite_df['icu_admission']==0][outcome].mean() * 100
    
    contingency = pd.crosstab(composite_df[outcome], composite_df['icu_admission'])
    chi2, p, _, _ = stats.chi2_contingency(contingency)
    
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"  {outcome:30s}: ICU {icu_rate:5.1f}% | Non-ICU {non_icu_rate:5.1f}% | p={p:.4f} {sig}")

# ============================================================================
# SECTION 7: VISUALIZATION
# ============================================================================

print("\n[7/10] Creating comprehensive visualization...")

fig = plt.figure(figsize=(22, 16))
gs = fig.add_gridspec(4, 3, hspace=0.50, wspace=0.35,
                      left=0.06, right=0.97, top=0.98, bottom=0.05)

# Panel A: New Chronic Conditions
ax1 = fig.add_subplot(gs[0, :])
if len(new_conditions_summary) > 0:
    new_cond_df = pd.DataFrame(new_conditions_summary)
    new_cond_sig = new_cond_df[new_cond_df['p_value'] < 0.1].sort_values('p_value')
    
    if len(new_cond_sig) > 0:
        conditions = [c.replace('new_', '').replace('_', ' ').title() for c in new_cond_sig['condition']]
        x = np.arange(len(conditions))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, new_cond_sig['icu_rate'], width, label='ICU',
                       color='#ef5350', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax1.bar(x + width/2, new_cond_sig['non_icu_rate'], width, label='Non-ICU',
                       color='#66bb6a', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax1.set_ylabel('Incidence (%)', fontweight='bold', fontsize=12)
        ax1.set_title('A. New Chronic Conditions Developed Within 1 Year (p<0.1)', 
                     fontweight='bold', fontsize=13, pad=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(conditions, rotation=45, ha='right', fontsize=10)
        ax1.legend(loc='upper left', frameon=True, fontsize=10)
        ax1.grid(axis='y', alpha=0.3)

# Panel B: Readmission Reasons (30-day)
ax2 = fig.add_subplot(gs[1, 0:2])
if len(readmit_df) > 0:
    readmit_30d = readmit_df[readmit_df['timeframe'] == '30-day']
    if len(readmit_30d) > 0:
        reason_comparison = readmit_30d.groupby(['icu', 'reason']).size().unstack(fill_value=0)
        reason_pct = reason_comparison.div(reason_comparison.sum(axis=1), axis=0) * 100
        
        reasons = reason_pct.columns
        x = np.arange(len(reasons))
        width = 0.35
        
        icu_vals = [reason_pct.loc[1, r] if 1 in reason_pct.index else 0 for r in reasons]
        non_icu_vals = [reason_pct.loc[0, r] if 0 in reason_pct.index else 0 for r in reasons]
        
        bars1 = ax2.bar(x - width/2, icu_vals, width, label='ICU',
                       color='#ef5350', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax2.bar(x + width/2, non_icu_vals, width, label='Non-ICU',
                       color='#66bb6a', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax2.set_ylabel('Percentage of Readmissions (%)', fontweight='bold', fontsize=11)
        ax2.set_title('B. Primary Reasons for 30-Day Readmission', fontweight='bold', fontsize=12, pad=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels([r.title() for r in reasons], rotation=20, ha='right', fontsize=10)
        ax2.legend(loc='upper right', frameon=True, fontsize=10)
        ax2.grid(axis='y', alpha=0.3)

# Panel C: Multiple System Involvement
ax3 = fig.add_subplot(gs[1, 2])
systems_data = cohort_with_dates.groupby(['icu_admission', 'num_systems_involved']).size().unstack(fill_value=0)
systems_pct = systems_data.div(systems_data.sum(axis=1), axis=0) * 100

if 1 in systems_pct.index and 0 in systems_pct.index:
    systems = systems_pct.columns
    x = np.arange(len(systems))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, systems_pct.loc[1], width, label='ICU',
                   color='#ef5350', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x + width/2, systems_pct.loc[0], width, label='Non-ICU',
                   color='#66bb6a', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax3.set_xlabel('Number of Organ Systems', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Percentage (%)', fontweight='bold', fontsize=11)
    ax3.set_title('C. Multi-System Involvement', fontweight='bold', fontsize=12, pad=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(systems, fontsize=10)
    ax3.legend(loc='upper right', frameon=True, fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

# Panel D: Healthcare Utilization Intensity
ax4 = fig.add_subplot(gs[2, :])
util_metrics = ['readmissions_1yr', 'total_hospital_days_1yr', 'emergency_admits_1yr']
util_labels = ['Total Readmissions\n(1 year)', 'Total Hospital Days\n(1 year)', 'Emergency Admissions\n(1 year)']

icu_means = [util_metrics_df[util_metrics_df['icu_admission']==1][m].mean() for m in util_metrics]
non_icu_means = [util_metrics_df[util_metrics_df['icu_admission']==0][m].mean() for m in util_metrics]

x = np.arange(len(util_labels))
width = 0.35

bars1 = ax4.bar(x - width/2, icu_means, width, label='ICU',
               color='#ef5350', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax4.bar(x + width/2, non_icu_means, width, label='Non-ICU',
               color='#66bb6a', alpha=0.8, edgecolor='black', linewidth=1.5)

ax4.set_ylabel('Count / Days', fontweight='bold', fontsize=12)
ax4.set_title('D. Healthcare Utilization (1 Year Follow-up)', fontweight='bold', fontsize=13, pad=12)
ax4.set_xticks(x)
ax4.set_xticklabels(util_labels, fontsize=10)
ax4.legend(loc='upper left', frameon=True, fontsize=10)
ax4.grid(axis='y', alpha=0.3)

# Add values on bars
for bar in bars1 + bars2:
    height = bar.get_height()
    if height > 0:
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

# Panel E: Composite Outcomes
ax5 = fig.add_subplot(gs[3, 0:2])
composite_outcomes = ['major_adverse_event', 'poor_outcome', 'frequent_utilizer']
composite_labels = ['Major Adverse\nEvent', 'Poor Overall\nOutcome', 'Frequent\nUtilizer (≥3 admits)']

icu_comp = [composite_df[composite_df['icu_admission']==1][c].mean()*100 for c in composite_outcomes]
non_icu_comp = [composite_df[composite_df['icu_admission']==0][c].mean()*100 for c in composite_outcomes]

x = np.arange(len(composite_labels))
width = 0.35

bars1 = ax5.bar(x - width/2, icu_comp, width, label='ICU',
               color='#ef5350', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax5.bar(x + width/2, non_icu_comp, width, label='Non-ICU',
               color='#66bb6a', alpha=0.8, edgecolor='black', linewidth=1.5)

ax5.set_ylabel('Incidence (%)', fontweight='bold', fontsize=12)
ax5.set_title('E. Composite Adverse Outcomes', fontweight='bold', fontsize=13, pad=12)
ax5.set_xticks(x)
ax5.set_xticklabels(composite_labels, fontsize=10)
ax5.legend(loc='upper left', frameon=True, fontsize=10)
ax5.grid(axis='y', alpha=0.3)

for bar in bars1 + bars2:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Panel F: Summary Table
ax6 = fig.add_subplot(gs[3, 2])
ax6.axis('off')

summary_data = [
    ['Metric', 'ICU', 'Non-ICU', 'P-value'],
    ['', '', '', ''],
    ['New chronic conditions*', f"{len([c for c in new_conditions_summary if c['p_value']<0.1])}", '-', '-'],
    ['Organ systems involved', f"{icu_systems:.1f}", f"{non_icu_systems:.1f}", f"{p_systems:.3f}"],
    ['1-yr readmissions', f"{util_metrics_df[util_metrics_df['icu_admission']==1]['readmissions_1yr'].mean():.1f}",
     f"{util_metrics_df[util_metrics_df['icu_admission']==0]['readmissions_1yr'].mean():.1f}", '-'],
    ['Major adverse event', f"{icu_comp[0]:.1f}%", f"{non_icu_comp[0]:.1f}%", '-'],
]

table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
                  colWidths=[0.4, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)

for i in range(len(summary_data)):
    for j in range(4):
        cell = table[(i, j)]
        if i == 0:
            cell.set_facecolor('#34495e')
            cell.set_text_props(weight='bold', color='white')
        elif i % 2 == 0:
            cell.set_facecolor('#ecf0f1')

ax6.text(0.5, 0.98, 'F. Summary Statistics', transform=ax6.transAxes,
         fontsize=13, fontweight='bold', ha='center')

plt.savefig('extended_long_term_factors.png', dpi=300, bbox_inches='tight')
print("✓ Saved: extended_long_term_factors.png")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n[8/10] Saving results...")

if len(new_conditions_summary) > 0:
    pd.DataFrame(new_conditions_summary).to_csv('new_chronic_conditions.csv', index=False)

if len(readmit_df) > 0:
    readmit_df.to_csv('readmission_reasons.csv', index=False)

util_metrics_df.to_csv('utilization_metrics.csv', index=False)
composite_df.to_csv('composite_outcomes.csv', index=False)

print("✓ Saved all CSV files")

print("\n" + "="*80)
print("EXTENDED ANALYSIS COMPLETE")
print("="*80)
print("\nFiles created:")
print("  • extended_long_term_factors.png")
print("  • new_chronic_conditions.csv")
print("  • readmission_reasons.csv")
print("  • utilization_metrics.csv")
print("  • composite_outcomes.csv")

