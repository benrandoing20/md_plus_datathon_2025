"""
Create Visual Representation of Patient Data Structure
Shows exactly what data we extract from MIMIC-IV for ICU vs non-ICU patients
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

print("Creating patient data structure visualization...")

mimic_path = 'physionet.org/files/mimiciv/3.1/hosp'

# Load data
admissions = pd.read_csv(os.path.join(mimic_path, 'admissions.csv.gz'), compression='gzip')
patients = pd.read_csv(os.path.join(mimic_path, 'patients.csv.gz'), compression='gzip')
diagnoses = pd.read_csv(os.path.join(mimic_path, 'diagnoses_icd.csv.gz'), compression='gzip')
d_icd_diagnoses = pd.read_csv(os.path.join(mimic_path, 'd_icd_diagnoses.csv.gz'), compression='gzip')
procedures = pd.read_csv(os.path.join(mimic_path, 'procedures_icd.csv.gz'), compression='gzip')
d_icd_procedures = pd.read_csv(os.path.join(mimic_path, 'd_icd_procedures.csv.gz'), compression='gzip')
icustays = pd.read_csv(os.path.join(mimic_path, '../icu/icustays.csv.gz'), compression='gzip')

# Load cohort
cohort = pd.read_csv('cohort_elective_spine_icu.csv')

# Get one ICU and one non-ICU example
icu_example = cohort.iloc[0]
# Find a non-ICU elective spine patient
spine_procs = procedures.merge(d_icd_procedures[['icd_code', 'long_title']], on='icd_code', how='left')
spine_procs = spine_procs[spine_procs['long_title'].str.contains('vertebr|spine|fusion|disc', case=False, na=False)]
non_icu_candidates = admissions[
    (admissions['admission_type'] == 'ELECTIVE') &
    (admissions['hadm_id'].isin(spine_procs['hadm_id'])) &
    (~admissions['hadm_id'].isin(cohort['hadm_id']))
]
if len(non_icu_candidates) > 0:
    non_icu_example = non_icu_candidates.iloc[0]
else:
    # Just use a different patient from cohort if needed
    non_icu_example = None

def extract_patient_data(subject_id, hadm_id, is_icu):
    """Extract all relevant data for a patient"""
    pt = patients[patients['subject_id'] == subject_id].iloc[0]
    adm = admissions[admissions['hadm_id'] == hadm_id].iloc[0]
    
    # Diagnoses
    adm_diags = diagnoses[diagnoses['hadm_id'] == hadm_id].merge(
        d_icd_diagnoses[['icd_code', 'long_title']], on='icd_code', how='left'
    ).sort_values('seq_num')
    
    # Procedures
    adm_procs = procedures[procedures['hadm_id'] == hadm_id].merge(
        d_icd_procedures[['icd_code', 'long_title']], on='icd_code', how='left'
    ).sort_values('seq_num')
    
    # ICU stays
    icu_stay = icustays[icustays['hadm_id'] == hadm_id]
    
    # Future admissions
    discharge_dt = pd.to_datetime(adm['dischtime'])
    future_admits = admissions[
        (admissions['subject_id'] == subject_id) &
        (pd.to_datetime(admissions['admittime']) > discharge_dt)
    ].sort_values('admittime')
    
    # Calculate features
    has_cardiac = any('cardiac' in str(d).lower() or 'heart' in str(d).lower() 
                     for d in adm_diags['long_title'])
    has_diabetes = any('diabetes' in str(d).lower() for d in adm_diags['long_title'])
    has_htn = any('hypertension' in str(d).lower() for d in adm_diags['long_title'])
    
    # Get all diagnoses formatted
    all_diagnoses = []
    for _, diag in adm_diags.iterrows():
        all_diagnoses.append(f"[{diag['seq_num']}] {diag['long_title'][:60]}")
    
    data = {
        'subject_id': subject_id,
        'hadm_id': hadm_id,
        'age': pt['anchor_age'],
        'gender': pt['gender'],
        'admission_type': adm['admission_type'],
        'admittime': adm['admittime'],
        'dischtime': adm['dischtime'],
        'discharge_location': adm['discharge_location'],
        'num_diagnoses': len(adm_diags),
        'all_diagnoses': all_diagnoses,
        'primary_diagnosis': adm_diags.iloc[0]['long_title'][:50] if len(adm_diags) > 0 else 'N/A',
        'has_cardiac': has_cardiac,
        'has_diabetes': has_diabetes,
        'has_hypertension': has_htn,
        'num_procedures': len(adm_procs),
        'spine_procedure': adm_procs.iloc[0]['long_title'][:50] if len(adm_procs) > 0 else 'N/A',
        'had_icu': is_icu,
        'icu_los': (pd.to_datetime(icu_stay.iloc[0]['outtime']) - pd.to_datetime(icu_stay.iloc[0]['intime'])).days if is_icu and len(icu_stay) > 0 else 0,
        'hospital_los': (discharge_dt - pd.to_datetime(adm['admittime'])).days,
        'death_date': pt['dod'] if pd.notna(pt['dod']) else 'None',
        'num_readmissions_1yr': len(future_admits[(pd.to_datetime(future_admits['admittime']) - discharge_dt).dt.days <= 365]),
        'days_to_first_readmit': (pd.to_datetime(future_admits.iloc[0]['admittime']) - discharge_dt).days if len(future_admits) > 0 else 'N/A'
    }
    
    return data

# Extract data for both patients
icu_data = extract_patient_data(icu_example['subject_id'], icu_example['hadm_id'], True)

if non_icu_example is not None:
    non_icu_data = extract_patient_data(non_icu_example['subject_id'], non_icu_example['hadm_id'], False)
else:
    non_icu_data = None

# Create figure - Only show flow diagram and table
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.5], hspace=0.40)

# ============================================================================
# Panel A: Data Flow Diagram
# ============================================================================
ax_flow = fig.add_subplot(gs[0])
ax_flow.set_xlim(0, 10)
ax_flow.set_ylim(0, 3)
ax_flow.axis('off')

# Boxes
boxes = [
    {'x': 0.5, 'y': 1.5, 'w': 1.5, 'h': 1, 'text': 'MIMIC-IV\nDatabase', 'color': '#e3f2fd'},
    {'x': 2.5, 'y': 2, 'w': 1.5, 'h': 0.6, 'text': 'Demographics\n(age, gender)', 'color': '#fff3e0'},
    {'x': 2.5, 'y': 1.2, 'w': 1.5, 'h': 0.6, 'text': 'Diagnoses\n(ICD codes)', 'color': '#fff3e0'},
    {'x': 2.5, 'y': 0.4, 'w': 1.5, 'h': 0.6, 'text': 'Procedures\n(ICD codes)', 'color': '#fff3e0'},
    {'x': 4.5, 'y': 1.5, 'w': 2, 'h': 1, 'text': 'PRE-SURGERY\nFEATURES', 'color': '#c8e6c9'},
    {'x': 7, 'y': 2, 'w': 1.5, 'h': 0.6, 'text': 'ICU Stay\n(Yes/No)', 'color': '#ffccbc'},
    {'x': 7, 'y': 1.2, 'w': 1.5, 'h': 0.6, 'text': 'Hospital LOS', 'color': '#ffccbc'},
    {'x': 7, 'y': 0.4, 'w': 1.5, 'h': 0.6, 'text': 'Discharge\nLocation', 'color': '#ffccbc'},
    {'x': 8.5, 'y': 0.8, 'w': 1.3, 'h': 1.4, 'text': 'LONG-TERM\nOUTCOMES\n(mortality,\nreadmission)', 'color': '#f8bbd0'},
]

for box in boxes:
    fancy_box = FancyBboxPatch(
        (box['x'], box['y']), box['w'], box['h'],
        boxstyle="round,pad=0.05", 
        edgecolor='black', 
        facecolor=box['color'],
        linewidth=2
    )
    ax_flow.add_patch(fancy_box)
    ax_flow.text(box['x'] + box['w']/2, box['y'] + box['h']/2, 
                box['text'], ha='center', va='center', 
                fontsize=10, fontweight='bold')

# Arrows
arrows = [
    ((2, 1.5), (2.5, 2.3)),
    ((2, 1.5), (2.5, 1.5)),
    ((2, 1.5), (2.5, 0.7)),
    ((4, 1.5), (4.5, 1.5)),
    ((6.5, 1.5), (7, 2.3)),
    ((6.5, 1.5), (7, 1.5)),
    ((6.5, 1.5), (7, 0.7)),
    ((8.5, 1.5), (8.5, 1.5)),
]

for start, end in arrows:
    ax_flow.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

ax_flow.text(5, 2.8, 'A. Data Extraction and Aggregation Flow', 
            fontsize=14, fontweight='bold', ha='left')

# ============================================================================
# Panel B: Patient Data Tables
# ============================================================================
ax_table = fig.add_subplot(gs[1])
ax_table.axis('off')

# Create comparison table
table_data = []
fields = [
    ('Demographics', '', ''),
    ('  Subject ID', f"{icu_data['subject_id']}", f"{non_icu_data['subject_id']}" if non_icu_data else 'N/A'),
    ('  Admission ID', f"{icu_data['hadm_id']}", f"{non_icu_data['hadm_id']}" if non_icu_data else 'N/A'),
    ('  Age', f"{icu_data['age']} years", f"{non_icu_data['age']} years" if non_icu_data else 'N/A'),
    ('  Gender', icu_data['gender'], non_icu_data['gender'] if non_icu_data else 'N/A'),
    ('', '', ''),
    ('Pre-Surgery Features', '', ''),
    ('  Total Diagnoses', str(icu_data['num_diagnoses']), str(non_icu_data['num_diagnoses']) if non_icu_data else 'N/A'),
]

# Add all diagnoses
fields.append(('  All Diagnoses (seq_num order):', '', ''))
max_diags = max(len(icu_data['all_diagnoses']), len(non_icu_data['all_diagnoses']) if non_icu_data else 0)
for i in range(max_diags):
    icu_diag = icu_data['all_diagnoses'][i] if i < len(icu_data['all_diagnoses']) else ''
    non_icu_diag = non_icu_data['all_diagnoses'][i] if non_icu_data and i < len(non_icu_data['all_diagnoses']) else ''
    fields.append(('    ', icu_diag, non_icu_diag))

fields.extend([
    ('', '', ''),
    ('  Cardiac Disease', '✓ Yes' if icu_data['has_cardiac'] else '✗ No', 
     '✓ Yes' if non_icu_data and non_icu_data['has_cardiac'] else '✗ No' if non_icu_data else 'N/A'),
    ('  Diabetes', '✓ Yes' if icu_data['has_diabetes'] else '✗ No', 
     '✓ Yes' if non_icu_data and non_icu_data['has_diabetes'] else '✗ No' if non_icu_data else 'N/A'),
    ('  Hypertension', '✓ Yes' if icu_data['has_hypertension'] else '✗ No', 
     '✓ Yes' if non_icu_data and non_icu_data['has_hypertension'] else '✗ No' if non_icu_data else 'N/A'),
    ('  Procedures Performed', str(icu_data['num_procedures']), str(non_icu_data['num_procedures']) if non_icu_data else 'N/A'),
    ('', '', ''),
    ('Acute Outcomes (THIS ADMISSION)', '', ''),
    ('  ICU Admission', 'YES', 'NO' if non_icu_data else 'N/A'),
    ('  ICU Length of Stay', f"{icu_data['icu_los']} days" if icu_data['icu_los'] > 0 else 'N/A', 
     '0 days' if non_icu_data else 'N/A'),
    ('  Hospital Length of Stay', f"{icu_data['hospital_los']} days", 
     f"{non_icu_data['hospital_los']} days" if non_icu_data else 'N/A'),
    ('  Discharge Location', icu_data['discharge_location'], 
     non_icu_data['discharge_location'] if non_icu_data else 'N/A'),
    ('', '', ''),
    ('Long-Term Outcomes (AFTER DISCHARGE)', '', ''),
    ('  Death Date', icu_data['death_date'], non_icu_data['death_date'] if non_icu_data else 'N/A'),
    ('  Days to First Readmit', str(icu_data['days_to_first_readmit']), 
     str(non_icu_data['days_to_first_readmit']) if non_icu_data else 'N/A'),
    ('  Readmissions (1 year)', str(icu_data['num_readmissions_1yr']), 
     str(non_icu_data['num_readmissions_1yr']) if non_icu_data else 'N/A'),
])

# Create table
table = ax_table.table(cellText=[[row[0], row[1], row[2] if len(row) > 2 else ''] for row in fields],
                       colLabels=['Data Field', '🔴 ICU Patient Example', '🟢 Non-ICU Patient Example'],
                       cellLoc='left',
                       loc='center',
                       colWidths=[0.35, 0.325, 0.325])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(3):
    table[(0, i)].set_facecolor('#1976d2')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style section headers - will be dynamically identified
for idx, field in enumerate(fields):
    if field[0] in ['Demographics', 'Pre-Surgery Features', 'Acute Outcomes (THIS ADMISSION)', 'Long-Term Outcomes (AFTER DISCHARGE)']:
        for col in range(3):
            try:
                table[(idx+1, col)].set_facecolor('#e3f2fd')
                table[(idx+1, col)].set_text_props(weight='bold')
            except:
                pass

# Find and highlight ICU admission row
for idx, field in enumerate(fields):
    if 'ICU Admission' in field[0]:
        try:
            table[(idx+1, 1)].set_facecolor('#ffcdd2')  # ICU YES
            table[(idx+1, 2)].set_facecolor('#c8e6c9')  # ICU NO
        except:
            pass

plt.tight_layout()
plt.savefig('patient_data_structure_visual.png', dpi=300, bbox_inches='tight')
print("✓ Saved: patient_data_structure_visual.png")
print()
print("This figure shows:")
print("  A. How data flows from MIMIC-IV → Features → Outcomes")
print("  B. Actual example patient data side-by-side (ICU vs non-ICU)")
print("  C. How we address causality vs correlation concerns")

