"""
Analysis 1: Specific Post-Surgical Complications (ICU vs Non-ICU)
Examines specific complication types from surgical literature
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SPECIFIC POST-SURGICAL COMPLICATIONS ANALYSIS")
print("ICU vs Non-ICU Comparison")
print("="*80)

# Setup paths
DATA_ROOT = Path('physionet.org/files/mimiciv/3.1')
HOSP_PATH = DATA_ROOT / 'hosp'

# Load data
print("\n[1/5] Loading data...")
cohort_icu = pd.read_csv('cohort_elective_spine_icu.csv')
admissions = pd.read_csv(HOSP_PATH / 'admissions.csv.gz', compression='gzip')
patients = pd.read_csv(HOSP_PATH / 'patients.csv.gz', compression='gzip')
procedures_icd = pd.read_csv(HOSP_PATH / 'procedures_icd.csv.gz', compression='gzip')

print(f"✓ Loaded {len(cohort_icu)} ICU patients")
print(f"✓ Loaded {len(admissions):,} total admissions")

# Define spine surgery filter (same as comprehensive_icu_analysis.py)
def is_spine_surgery(row):
    """Identify spine surgery procedures from ICD codes"""
    code = str(row['icd_code'])
    version = row['icd_version']
    
    if version == 9:
        return code.startswith(('810', '813', '816', '03'))
    elif version == 10 and len(code) >= 4:
        body_system = code[1]
        operation = code[2]
        body_part = code[3]
        
        if body_system == 'R' and operation in 'BGNTQSHJPRUW' and body_part in '0123467689AB':
            return True
        if body_system == 'S' and operation in 'BGNTQSHJPRUW' and body_part in '012345678':
            return True
    return False

# Build full cohort
print("Identifying spine surgeries...")
procedures_icd['is_spine'] = procedures_icd.apply(is_spine_surgery, axis=1)
spine_procs = procedures_icd[procedures_icd['is_spine']].copy()

elective = admissions[admissions['admission_type'].str.upper() == 'ELECTIVE']
adults = patients[patients['anchor_age'] >= 18][['subject_id', 'gender', 'anchor_age']]

spine_hadm = spine_procs[['hadm_id', 'subject_id']].drop_duplicates()
cohort = elective.merge(spine_hadm, on=['hadm_id', 'subject_id'], how='inner')
cohort = cohort.merge(adults, on='subject_id', how='inner')

# Label ICU admissions
icu_ids = set(cohort_icu['hadm_id'])
cohort['icu_stay'] = cohort['hadm_id'].isin(icu_ids)

print(f"✓ Final cohort: {len(cohort)} patients")
print(f"  - ICU: {cohort['icu_stay'].sum()} ({cohort['icu_stay'].mean()*100:.1f}%)")
print(f"  - Non-ICU: {(~cohort['icu_stay']).sum()}")

# Load diagnoses
print("Loading diagnoses...")
diag_chunks = []
for chunk in pd.read_csv(HOSP_PATH / 'diagnoses_icd.csv.gz', 
                          chunksize=100000, compression='gzip'):
    diag_chunks.append(chunk[chunk['subject_id'].isin(cohort['subject_id'])])
diagnoses = pd.concat(diag_chunks, ignore_index=True)
print(f"✓ Loaded {len(diagnoses)} diagnoses")

# Load ICD descriptions
d_icd_diag = pd.read_csv(HOSP_PATH / 'd_icd_diagnoses.csv.gz', compression='gzip')
diagnoses = diagnoses.merge(d_icd_diag[['icd_code', 'icd_version', 'long_title']], 
                            on=['icd_code', 'icd_version'], how='left')

# Merge with cohort
diagnoses = diagnoses.merge(cohort[['subject_id', 'hadm_id', 'icu_stay']], 
                            on=['subject_id', 'hadm_id'], how='inner')

print(f"✓ Diagnoses merged with cohort: {len(diagnoses)} records")

print("\n" + "="*80)
print("[2/5] DEFINING SPECIFIC COMPLICATION CATEGORIES")
print("="*80)

# Define ICD code patterns for specific complications
complications = {
    # 1. CARDIOPULMONARY COMPLICATIONS
    'Acute MI': {
        'icd9': ['410'],
        'icd10': ['I21', 'I22'],
        'keywords': ['acute myocardial infarction', 'acute mi', 'stemi', 'nstemi']
    },
    'Heart Failure': {
        'icd9': ['428'],
        'icd10': ['I50'],
        'keywords': ['heart failure', 'cardiac failure', 'congestive heart']
    },
    'Arrhythmia': {
        'icd9': ['427'],
        'icd10': ['I47', 'I48', 'I49'],
        'keywords': ['arrhythmia', 'atrial fibrillation', 'ventricular tachycardia', 'ventricular fibrillation']
    },
    'Pulmonary Embolism': {
        'icd9': ['415.1'],
        'icd10': ['I26'],
        'keywords': ['pulmonary embolism', 'pulmonary thromboembolism']
    },
    'Respiratory Failure': {
        'icd9': ['518.81', '518.82', '518.84'],
        'icd10': ['J96'],
        'keywords': ['respiratory failure', 'acute respiratory failure']
    },
    'Pneumonia': {
        'icd9': ['481', '482', '483', '484', '485', '486', '997.31'],
        'icd10': ['J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J95.851'],
        'keywords': ['pneumonia', 'postoperative pneumonia']
    },
    
    # 2. CENTRAL NERVOUS SYSTEM COMPLICATIONS
    'Stroke': {
        'icd9': ['434', '997.02'],
        'icd10': ['I63', 'I97.81'],
        'keywords': ['stroke', 'cerebral infarction', 'postoperative stroke']
    },
    'Delirium': {
        'icd9': ['293.0', '293.1'],
        'icd10': ['F05'],
        'keywords': ['delirium', 'acute confusional state']
    },
    'Seizure': {
        'icd9': ['780.39', '345'],
        'icd10': ['R56', 'G40'],
        'keywords': ['seizure', 'convulsion']
    },
    
    # 3. RENAL COMPLICATIONS
    'Acute Kidney Injury': {
        'icd9': ['584'],
        'icd10': ['N17'],
        'keywords': ['acute kidney injury', 'acute renal failure', 'aki']
    },
    'Urinary Retention': {
        'icd9': ['788.2'],
        'icd10': ['R33'],
        'keywords': ['urinary retention', 'retention of urine']
    },
    
    # 4. ELECTROLYTE COMPLICATIONS
    'Hyponatremia': {
        'icd9': ['276.1'],
        'icd10': ['E87.1'],
        'keywords': ['hyponatremia', 'low sodium']
    },
    'Hypernatremia': {
        'icd9': ['276.0'],
        'icd10': ['E87.0'],
        'keywords': ['hypernatremia', 'high sodium']
    },
    'Hypokalemia': {
        'icd9': ['276.8'],
        'icd10': ['E87.6'],
        'keywords': ['hypokalemia', 'low potassium']
    },
    'SIADH': {
        'icd9': ['276.1'],
        'icd10': ['E22.2'],
        'keywords': ['siadh', 'syndrome of inappropriate antidiuretic']
    },
    
    # 5. GASTROINTESTINAL COMPLICATIONS
    'GI Bleeding': {
        'icd9': ['578', '998.11'],
        'icd10': ['K92.2', 'K92.1', 'T81.0'],
        'keywords': ['gastrointestinal hemorrhage', 'gi bleed', 'gastrointestinal bleeding']
    },
    'Ileus': {
        'icd9': ['560.1', '997.4'],
        'icd10': ['K56.0', 'K91.89'],
        'keywords': ['ileus', 'paralytic ileus', 'postoperative ileus']
    },
    'Liver Failure': {
        'icd9': ['570', '572.2'],
        'icd10': ['K72'],
        'keywords': ['liver failure', 'hepatic failure', 'acute liver failure']
    },
    
    # 6. INFECTION COMPLICATIONS
    'Surgical Site Infection': {
        'icd9': ['998.5', '998.51', '998.59'],
        'icd10': ['T81.4', 'T84.6'],
        'keywords': ['surgical site infection', 'wound infection', 'postoperative infection']
    },
    'Sepsis': {
        'icd9': ['995.91', '995.92', '038'],
        'icd10': ['A41', 'R65.2'],
        'keywords': ['sepsis', 'severe sepsis', 'septicemia']
    },
    'UTI': {
        'icd9': ['599.0'],
        'icd10': ['N39.0'],
        'keywords': ['urinary tract infection', 'uti', 'cystitis']
    },
    
    # 7. ANESTHESIA-RELATED COMPLICATIONS
    'Malignant Hyperthermia': {
        'icd9': ['995.86'],
        'icd10': ['T88.3'],
        'keywords': ['malignant hyperthermia']
    },
    'Anesthesia Complications': {
        'icd9': ['995.4', '668'],
        'icd10': ['T88', 'O74', 'O89'],
        'keywords': ['anesthesia complication', 'adverse effect of anesthesia']
    },
    
    # 8. HEMATOLOGIC COMPLICATIONS
    'DVT': {
        'icd9': ['453.4'],
        'icd10': ['I82.4'],
        'keywords': ['deep vein thrombosis', 'deep venous thrombosis', 'dvt']
    },
    'Bleeding/Hemorrhage': {
        'icd9': ['998.1', '998.11', '998.12'],
        'icd10': ['T81.0'],
        'keywords': ['postoperative hemorrhage', 'postoperative bleeding']
    },
    
    # 9. SURGICAL COMPLICATIONS
    'CSF Leak': {
        'icd9': ['349.81', '997.09'],
        'icd10': ['G96.0', 'G97.0'],
        'keywords': ['cerebrospinal fluid leak', 'csf leak', 'dural tear']
    },
    'Hardware Complication': {
        'icd9': ['996.4'],
        'icd10': ['T84'],
        'keywords': ['device complication', 'hardware complication', 'implant complication']
    }
}

def check_complication(row, comp_def):
    """Check if a diagnosis matches the complication definition"""
    icd_code = str(row['icd_code'])
    icd_version = row['icd_version']
    title = str(row['long_title']).lower() if pd.notna(row['long_title']) else ''
    
    # Check ICD code patterns
    if icd_version == 9:
        for pattern in comp_def['icd9']:
            if icd_code.startswith(pattern):
                return True
    elif icd_version == 10:
        for pattern in comp_def['icd10']:
            if icd_code.startswith(pattern):
                return True
    
    # Check keywords in description
    for keyword in comp_def['keywords']:
        if keyword in title:
            return True
    
    return False

print("\nIdentifying complications in dataset...")
results = []

for comp_name, comp_def in complications.items():
    # Find matching diagnoses
    mask = diagnoses.apply(lambda row: check_complication(row, comp_def), axis=1)
    comp_diag = diagnoses[mask].copy()
    
    if len(comp_diag) == 0:
        print(f"  ⚠️  {comp_name}: No cases found")
        results.append({
            'Complication': comp_name,
            'Available': 'No',
            'Total_Cases': 0,
            'ICU_Cases': 0,
            'NonICU_Cases': 0,
            'ICU_Rate': 0,
            'NonICU_Rate': 0,
            'OR': np.nan,
            'p_value': np.nan
        })
        continue
    
    # Calculate rates
    total_cases = comp_diag['subject_id'].nunique()
    icu_cases = comp_diag[comp_diag['icu_stay'] == True]['subject_id'].nunique()
    noicu_cases = comp_diag[comp_diag['icu_stay'] == False]['subject_id'].nunique()
    
    n_icu = cohort['icu_stay'].sum()
    n_noicu = len(cohort) - n_icu
    
    icu_rate = (icu_cases / n_icu) * 100
    noicu_rate = (noicu_cases / n_noicu) * 100
    
    # Statistical test
    contingency = [[icu_cases, n_icu - icu_cases],
                   [noicu_cases, n_noicu - noicu_cases]]
    
    try:
        _, p_value = stats.fisher_exact(contingency) if min(contingency[0] + contingency[1]) < 10 else stats.chi2_contingency(contingency)[:2]
        
        # Odds ratio
        if noicu_cases > 0 and icu_cases < n_icu:
            or_val = (icu_cases / (n_icu - icu_cases)) / (noicu_cases / (n_noicu - noicu_cases))
        else:
            or_val = np.nan
    except:
        p_value = np.nan
        or_val = np.nan
    
    print(f"  ✓ {comp_name}: {total_cases} cases ({icu_cases} ICU, {noicu_cases} non-ICU)")
    
    results.append({
        'Complication': comp_name,
        'Available': 'Yes',
        'Total_Cases': total_cases,
        'ICU_Cases': icu_cases,
        'NonICU_Cases': noicu_cases,
        'ICU_Rate': icu_rate,
        'NonICU_Rate': noicu_rate,
        'OR': or_val,
        'p_value': p_value
    })

results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("[3/5] DATA AVAILABILITY SUMMARY")
print("="*80)

available = results_df[results_df['Available'] == 'Yes']
unavailable = results_df[results_df['Available'] == 'No']

print(f"\n✓ Available in dataset: {len(available)} complications")
print(f"✗ Not found in dataset: {len(unavailable)} complications")

if len(unavailable) > 0:
    print("\nComplications NOT found:")
    for comp in unavailable['Complication'].values:
        print(f"  • {comp}")

print("\n" + "="*80)
print("[4/5] STATISTICAL RESULTS")
print("="*80)

available_sorted = available.sort_values('p_value')

print("\n" + "-"*80)
print("SIGNIFICANT COMPLICATIONS (p < 0.05)")
print("-"*80)
print(f"{'Complication':<30} {'ICU %':<10} {'Non-ICU %':<12} {'OR':<10} {'p-value':<10}")
print("-"*80)

sig_comps = []
for _, row in available_sorted.iterrows():
    if pd.notna(row['p_value']) and row['p_value'] < 0.05:
        sig = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01 else '*')
        print(f"{row['Complication']:<30} {row['ICU_Rate']:>6.1f}%   {row['NonICU_Rate']:>6.1f}%      {row['OR']:>6.2f}    {row['p_value']:.4f} {sig}")
        sig_comps.append(row['Complication'])

if len(sig_comps) == 0:
    print("  No significant complications found at p<0.05")

print("\n" + "-"*80)
print("NON-SIGNIFICANT COMPLICATIONS (p >= 0.05)")
print("-"*80)
print(f"{'Complication':<30} {'ICU %':<10} {'Non-ICU %':<12} {'OR':<10} {'p-value':<10}")
print("-"*80)

for _, row in available_sorted.iterrows():
    if pd.notna(row['p_value']) and row['p_value'] >= 0.05:
        print(f"{row['Complication']:<30} {row['ICU_Rate']:>6.1f}%   {row['NonICU_Rate']:>6.1f}%      {row['OR']:>6.2f}    {row['p_value']:.4f}")

# Save results
results_df.to_csv('specific_complications_results.csv', index=False)
print("\n✓ Saved: specific_complications_results.csv")

print("\n" + "="*80)
print("[5/5] CREATING VISUALIZATION")
print("="*80)

# Create comprehensive figure
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.35)

# Panel A: Data Availability
ax_avail = fig.add_subplot(gs[0, :])
avail_counts = results_df['Available'].value_counts()
colors_avail = ['#2ecc71', '#e74c3c']
bars = ax_avail.barh(['Available', 'Not Found'], 
                     [len(available), len(unavailable)],
                     color=colors_avail, alpha=0.7, edgecolor='black', linewidth=1.5)
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax_avail.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                 f'n={int(width)}', va='center', fontweight='bold', fontsize=12)
ax_avail.set_xlabel('Number of Complication Types', fontsize=11, fontweight='bold')
ax_avail.set_title('A. Data Availability for Specific Complications', 
                   fontsize=13, fontweight='bold', loc='left')
ax_avail.spines['top'].set_visible(False)
ax_avail.spines['right'].set_visible(False)
ax_avail.grid(axis='x', alpha=0.3, linestyle='--')

# Panel B: Complication Rates Comparison
ax_rates = fig.add_subplot(gs[1, :])
available_plot = available.sort_values('ICU_Rate', ascending=False).head(15)
x = np.arange(len(available_plot))
width = 0.35

bars1 = ax_rates.bar(x - width/2, available_plot['ICU_Rate'], width, 
                     label='ICU', color='#e74c3c', alpha=0.8, edgecolor='black')
bars2 = ax_rates.bar(x + width/2, available_plot['NonICU_Rate'], width,
                     label='Non-ICU', color='#3498db', alpha=0.8, edgecolor='black')

ax_rates.set_xlabel('Complication Type', fontsize=11, fontweight='bold', labelpad=8)
ax_rates.set_ylabel('Rate (%)', fontsize=11, fontweight='bold')
ax_rates.set_title('B. Complication Rates: ICU vs Non-ICU (Top 15 by ICU Rate)', 
                   fontsize=13, fontweight='bold', loc='left', pad=15)
ax_rates.set_xticks(x)
ax_rates.set_xticklabels(available_plot['Complication'], rotation=15, ha='right', fontsize=9)
ax_rates.legend(frameon=True, loc='upper right')
ax_rates.spines['top'].set_visible(False)
ax_rates.spines['right'].set_visible(False)
ax_rates.grid(axis='y', alpha=0.3, linestyle='--')

# Panel C: Odds Ratios (Significant only)
ax_or = fig.add_subplot(gs[2, :])
sig_data = available[available['p_value'] < 0.05].copy()
# Remove complications with undefined OR (NaN - occurs when one group has 0 cases)
sig_data = sig_data[sig_data['OR'].notna()].sort_values('OR', ascending=True)
if len(sig_data) > 0:
    y_pos = np.arange(len(sig_data))
    colors_or = ['#e74c3c' if or_val > 1 else '#3498db' for or_val in sig_data['OR']]
    
    # Calculate appropriate x-axis limits to make all bars visible
    or_max = sig_data['OR'].max()
    or_min = sig_data['OR'].min()
    or_range = or_max - or_min
    
    # Ensure all bars are visible with appropriate scaling
    x_min = 0
    x_max = or_max * 1.15
    
    # Draw reference line FIRST (behind bars) with lower zorder
    ax_or.axvline(x=1, color='gray', linestyle='--', linewidth=2, alpha=0.7, 
                  label='OR = 1 (No effect)', zorder=1)
    
    # Draw bars from 0 to OR value to ensure all are visible
    # Ensure minimum visual width for bars close to 1
    for i, (idx, row) in enumerate(sig_data.iterrows()):
        or_val = row['OR']
        color = colors_or[i]
        
        # Draw bar with high zorder to appear on top of reference line
        bar = ax_or.barh(i, or_val, left=0, color=color, alpha=0.7, 
                        edgecolor='black', linewidth=2, height=0.65, zorder=3)
    
    ax_or.set_yticks(y_pos)
    ax_or.set_yticklabels(sig_data['Complication'], fontsize=10)
    ax_or.set_xlabel('Odds Ratio (OR)', fontsize=11, fontweight='bold', labelpad=8)
    ax_or.set_title('C. Odds Ratios for Significant Complications (p < 0.05)', 
                   fontsize=13, fontweight='bold', loc='left', pad=15)
    ax_or.set_xlim(x_min, x_max)
    ax_or.legend(frameon=True, loc='best')
    ax_or.spines['top'].set_visible(False)
    ax_or.spines['right'].set_visible(False)
    ax_or.grid(axis='x', alpha=0.3, linestyle='--', zorder=0)
    
    # Add OR value labels and p-value annotations
    for i, (_, row) in enumerate(sig_data.iterrows()):
        sig_marker = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01 else '*')
        # Position label at the end of the bar
        label_x = row['OR'] + (x_max - x_min) * 0.02
        ax_or.text(label_x, i, f"{row['OR']:.2f} {sig_marker}", va='center', 
                  fontweight='bold', fontsize=10, ha='left', zorder=4)
    
    # Check if any significant complications were excluded due to undefined OR
    sig_excluded = available[(available['p_value'] < 0.05) & (available['OR'].isna())]
    if len(sig_excluded) > 0:
        excluded_names = ', '.join(sig_excluded['Complication'].values)
        ax_or.text(0.98, 0.02, f'Note: {excluded_names} excluded (OR undefined: 0 cases in one group)', 
                  transform=ax_or.transAxes, fontsize=8, ha='right', va='bottom',
                  style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
else:
    ax_or.text(0.5, 0.5, 'No significant complications found', 
               ha='center', va='center', transform=ax_or.transAxes, fontsize=12)
    ax_or.set_title('C. Odds Ratios for Significant Complications', 
                   fontsize=13, fontweight='bold', loc='left')

# Panel D: Category Summary
ax_cat = fig.add_subplot(gs[3, 0])
categories = {
    'Cardiopulmonary': ['Acute MI', 'Heart Failure', 'Arrhythmia', 'Pulmonary Embolism', 
                       'Respiratory Failure', 'Pneumonia'],
    'CNS': ['Stroke', 'Delirium', 'Seizure'],
    'Renal': ['Acute Kidney Injury', 'Urinary Retention'],
    'Electrolyte': ['Hyponatremia', 'Hypernatremia', 'Hypokalemia', 'SIADH'],
    'GI': ['GI Bleeding', 'Ileus', 'Liver Failure'],
    'Infection': ['Surgical Site Infection', 'Sepsis', 'UTI'],
    'Other': ['DVT', 'Bleeding/Hemorrhage', 'CSF Leak', 'Hardware Complication']
}

cat_sig_counts = []
for cat, comps in categories.items():
    sig_count = sum([1 for c in comps if c in sig_comps])
    cat_sig_counts.append({'Category': cat, 'Significant': sig_count, 'Total': len(comps)})

cat_df = pd.DataFrame(cat_sig_counts)
x_cat = np.arange(len(cat_df))
width_cat = 0.35

bars1_cat = ax_cat.bar(x_cat - width_cat/2, cat_df['Significant'], width_cat,
                       label='Significant (p<0.05)', color='#e74c3c', alpha=0.8, edgecolor='black')
bars2_cat = ax_cat.bar(x_cat + width_cat/2, cat_df['Total'], width_cat,
                       label='Total', color='#95a5a6', alpha=0.8, edgecolor='black')

ax_cat.set_ylabel('Number of Complications', fontsize=10, fontweight='bold')
ax_cat.set_title('D. Significant Complications\nby Category', fontsize=11, fontweight='bold', loc='left')
ax_cat.set_xticks(x_cat)
ax_cat.set_xticklabels(cat_df['Category'], rotation=15, ha='right', fontsize=9)
ax_cat.legend(frameon=True, fontsize=8)
ax_cat.spines['top'].set_visible(False)
ax_cat.spines['right'].set_visible(False)
ax_cat.grid(axis='y', alpha=0.3, linestyle='--')

# Panel E: Top complications by absolute difference
ax_diff = fig.add_subplot(gs[3, 1])
available_plot2 = available.copy()
available_plot2['Abs_Diff'] = available_plot2['ICU_Rate'] - available_plot2['NonICU_Rate']
available_plot2 = available_plot2.sort_values('Abs_Diff', ascending=False).head(10)

colors_diff = ['#e74c3c' if d > 0 else '#3498db' for d in available_plot2['Abs_Diff']]
bars_diff = ax_diff.barh(np.arange(len(available_plot2)), available_plot2['Abs_Diff'],
                         color=colors_diff, alpha=0.8, edgecolor='black')
ax_diff.set_yticks(np.arange(len(available_plot2)))
ax_diff.set_yticklabels(available_plot2['Complication'], fontsize=9)
ax_diff.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
ax_diff.set_xlabel('Rate Difference (%)', fontsize=10, fontweight='bold', labelpad=5)
ax_diff.set_title('E. Largest Rate Differences (ICU - Non-ICU)\nTop 10 Complications', fontsize=11, fontweight='bold', loc='left', pad=12)
ax_diff.spines['top'].set_visible(False)
ax_diff.spines['right'].set_visible(False)
ax_diff.grid(axis='x', alpha=0.3, linestyle='--')

# Panel F: Summary table (instead of text)
ax_power = fig.add_subplot(gs[3, 2])

summary_data = [
    ['Metric', 'Value'],
    ['', ''],
    ['Total Complications', str(len(results_df))],
    ['Available in Dataset', f"{len(available)} ({len(available)/len(results_df)*100:.0f}%)"],
    ['Significant (p<0.05)', str(len(sig_data))],
    ['', ''],
    ['ICU Patients', str(int(cohort['icu_stay'].sum()))],
    ['Non-ICU Patients', str(int(len(cohort) - cohort['icu_stay'].sum()))]
]

table = ax_power.table(cellText=summary_data, cellLoc='left', loc='center',
                       colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

for i in range(len(summary_data)):
    for j in range(2):
        cell = table[(i, j)]
        if i == 0:
            cell.set_facecolor('#34495e')
            cell.set_text_props(weight='bold', color='white')
        elif i % 2 == 0 and i > 1:
            cell.set_facecolor('#ecf0f1')

ax_power.axis('off')
ax_power.text(0.1, 0.95, 'F. Summary', transform=ax_power.transAxes,
             fontsize=12, fontweight='bold', ha='left', va='top')

plt.savefig('specific_complications_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: specific_complications_analysis.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nKey Findings:")
print(f"  • {len(available)} of {len(results_df)} complications found in dataset")
print(f"  • {len(sig_data)} complications significantly different between ICU/non-ICU")
if len(available) > 0:
    print(f"  • Highest ICU rate: {available.loc[available['ICU_Rate'].idxmax(), 'Complication']} ({available['ICU_Rate'].max():.1f}%)")

if len(sig_data) > 0:
    highest_or = sig_data.loc[sig_data['OR'].idxmax()]
    print(f"  • Highest odds ratio: {highest_or['Complication']} (OR={highest_or['OR']:.2f}, p={highest_or['p_value']:.4f})")

print("\nOutput files:")
print("  ✓ specific_complications_results.csv")
print("  ✓ specific_complications_analysis.png")
print("="*80)
