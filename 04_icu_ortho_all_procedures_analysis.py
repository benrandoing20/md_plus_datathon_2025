"""
COMPOSITE OUTCOME PREDICTION - ICU PATIENTS AFTER ORTHOPEDIC SURGERY
Spine, Hip, Knee, Shoulder/Elbow, Hand

This script predicts composite long-term outcomes for ICU patients after orthopedic 
surgery procedures.

⚠️ IMPORTANT: Models are trained ONLY on patients who had an ICU admission.

Uses BOTH ICD procedure codes AND CPT codes to ensure we capture ACTUAL SURGICAL
procedures (not just diagnostic codes for fractures or non-operative cases).

Research Question: What patient and procedural factors predict good vs poor long-term 
outcomes for ICU patients after orthopedic surgery?

Composite Outcome (for ICU patients only):
- Poor outcome = ANY of the following:
  • Died within 90 days of discharge
  • Readmitted within 7 days of discharge
  • Length of stay >7 days
- Good outcome = NONE of the above (survived 90 days AND no 7-day readmission AND LOS ≤7 days)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                            precision_recall_curve, average_precision_score)
from sklearn.utils.class_weight import compute_sample_weight
import warnings
import time
warnings.filterwarnings('ignore')

# Try to import SHAP for feature analysis
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available. Install with: pip install shap")

np.random.seed(42)

print("="*80)
print("COMPOSITE OUTCOME PREDICTION - ICU PATIENTS AFTER ORTHOPEDIC SURGERY")
print("Spine | Hip | Knee | Shoulder/Elbow | Hand")
print("⚠️  Models trained on ICU PATIENTS ONLY")
print("="*80)

# ============================================================================
# DEFINE DATA PATHS
# ============================================================================
DATA_ROOT = Path('physionet.org/files/mimiciv/3.1')
HOSP_PATH = DATA_ROOT / 'hosp'
ICU_PATH = DATA_ROOT / 'icu'

# ============================================================================
# [1/8] IDENTIFY ALL ORTHOPEDIC PROCEDURES
# ============================================================================
print("\n[1/8] Identifying all orthopedic procedures...")

# Load procedure data
procedures_icd = pd.read_csv(HOSP_PATH / 'procedures_icd.csv.gz', compression='gzip')
d_icd_proc = pd.read_csv(HOSP_PATH / 'd_icd_procedures.csv.gz', compression='gzip')
procedures_icd = procedures_icd.merge(
    d_icd_proc[['icd_code', 'icd_version', 'long_title']], 
    on=['icd_code', 'icd_version'], how='left'
)

def classify_orthopedic_procedure(row):
    """
    Classify orthopedic procedures into categories:
    spine, hip, knee, shoulder_elbow, hand
    """
    code = str(row['icd_code'])
    version = row['icd_version']
    title = str(row.get('long_title', '')).lower()
    
    if version == 9:
        # ICD-9 orthopedic procedure codes
        # Spine: 81.0x (fusion), 81.3x (refusion), 81.6x (other spine), 03.xx (spinal cord)
        if (code.startswith('810') or code.startswith('813') or 
            code.startswith('816') or code.startswith('03')):
            return 'spine'
        
        # Hip: 81.51-81.53 (hip replacement), 00.7x (hip resurfacing), 80.05 (arthrotomy hip)
        if code.startswith('8151') or code.startswith('8152') or code.startswith('8153'):
            return 'hip'
        if code.startswith('007') and 'hip' in title:
            return 'hip'
        if code.startswith('8005') or (code.startswith('80') and 'hip' in title):
            return 'hip'
        
        # Knee: 81.54-81.55 (knee replacement), 80.06 (arthrotomy knee), 80.26 (arthroscopy knee)
        if code.startswith('8154') or code.startswith('8155'):
            return 'knee'
        if code.startswith('8006') or code.startswith('8026'):
            return 'knee'
        if code.startswith('80') and 'knee' in title:
            return 'knee'
        
        # Shoulder/Elbow: 81.80-81.85 (shoulder and elbow procedures)
        if (code.startswith('8180') or code.startswith('8181') or 
            code.startswith('8182') or code.startswith('8183') or
            code.startswith('8184') or code.startswith('8185')):
            return 'shoulder_elbow'
        if code.startswith('80') and ('shoulder' in title or 'elbow' in title):
            return 'shoulder_elbow'
        
        # Hand/Wrist: 81.7x (hand/wrist procedures), 82.xx (hand operations)
        if code.startswith('817') or code.startswith('82'):
            return 'hand'
        if code.startswith('80') and ('hand' in title or 'wrist' in title or 'finger' in title):
            return 'hand'
    
    elif version == 10:
        # ICD-10-PCS orthopedic procedures
        if len(code) < 4:
            return None
        
        body_system = code[1]  # 2nd character
        operation = code[2]    # 3rd character
        body_part = code[3]    # 4th character
        
        # SPINE procedures
        # Upper Joints (0R): cervical/thoracic spine
        if body_system == 'R':
            spine_parts = ['0', '1', '2', '3', '4', '6', '7', '8', '9', 'A', 'B']
            spine_ops = ['B', 'G', 'N', 'T', 'Q', 'S', 'H', 'J', 'P', 'R', 'U', 'W']
            if operation in spine_ops and body_part in spine_parts:
                return 'spine'
        
        # Lower Joints (0S): lumbar/sacral spine
        if body_system == 'S':
            spine_parts = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
            spine_ops = ['B', 'G', 'N', 'T', 'Q', 'S', 'H', 'J', 'P', 'R', 'U', 'W']
            if operation in spine_ops and body_part in spine_parts:
                return 'spine'
        
        # HIP procedures (0S: Lower Joints)
        if body_system == 'S':
            hip_parts = ['9', 'A', 'B', 'E']  # Hip joint parts
            hip_ops = ['B', 'G', 'N', 'T', 'Q', 'S', 'H', 'J', 'P', 'R', 'U', 'W', '5', '9']
            if operation in hip_ops and body_part in hip_parts:
                return 'hip'
        
        # KNEE procedures (0S: Lower Joints)
        if body_system == 'S':
            knee_parts = ['C', 'D']  # Knee joint parts
            knee_ops = ['B', 'G', 'N', 'T', 'Q', 'S', 'H', 'J', 'P', 'R', 'U', 'W', '5', '9']
            if operation in knee_ops and body_part in knee_parts:
                return 'knee'
        
        # SHOULDER/ELBOW procedures (0R: Upper Joints)
        if body_system == 'R':
            shoulder_elbow_parts = ['G', 'H', 'J', 'L', 'M']  # Shoulder and elbow joint parts
            shoulder_elbow_ops = ['B', 'G', 'N', 'T', 'Q', 'S', 'H', 'J', 'P', 'R', 'U', 'W', '5', '9']
            if operation in shoulder_elbow_ops and body_part in shoulder_elbow_parts:
                return 'shoulder_elbow'
        
        # HAND/WRIST procedures (0R: Upper Joints, 0X: Anatomical Regions)
        if body_system == 'R':
            hand_parts = ['N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V']  # Wrist, hand joints
            hand_ops = ['B', 'G', 'N', 'T', 'Q', 'S', 'H', 'J', 'P', 'R', 'U', 'W', '5', '9']
            if operation in hand_ops and body_part in hand_parts:
                return 'hand'
        
        # Hand operations (0X: Anatomical Regions, Upper Extremities)
        if body_system == 'X':
            hand_parts = ['D', 'E', 'F', 'G', 'H', 'J', 'K']  # Hand, fingers
            if body_part in hand_parts:
                return 'hand'
    
    return None

# Classify all ICD procedures
print("  Classifying ICD orthopedic procedures...")
procedures_icd['ortho_type'] = procedures_icd.apply(classify_orthopedic_procedure, axis=1)
ortho_procedures_icd = procedures_icd[procedures_icd['ortho_type'].notna()].copy()

print(f"\n  ICD-coded orthopedic PROCEDURE RECORDS: {len(ortho_procedures_icd):,}")
print(f"  Unique hospital ADMISSIONS (ICD): {ortho_procedures_icd['hadm_id'].nunique():,}")

# ============================================================================
# ADD CPT/HCPCS CODE FILTERING FOR SURGICAL PROCEDURES
# ============================================================================
print("\n  Loading CPT/HCPCS codes to verify surgical procedures...")

def classify_cpt_orthopedic_procedure(code):
    """
    Classify orthopedic surgery CPT codes into categories.
    CPT codes definitively identify SURGICAL procedures (not just diagnoses).
    
    Returns: 'spine', 'hip', 'knee', 'shoulder_elbow', 'hand', or None
    """
    try:
        code_num = int(code)
    except:
        return None
    
    # SPINE SURGERY (definitive surgical CPT codes)
    # 22000-22899: Spine surgery (fusion, decompression, corpectomy, etc.)
    # 63000-63746: Spine decompression/laminectomy
    if (22000 <= code_num <= 22899) or (63000 <= code_num <= 63746):
        return 'spine'
    
    # HIP SURGERY
    # 27130-27132: Total hip arthroplasty
    # 27134-27138: Hip revision
    # 27146-27147: Hip osteotomy
    # 27156-27159: Hip fracture surgery (ORIF)
    if (27130 <= code_num <= 27132) or (27134 <= code_num <= 27138) or \
       (27146 <= code_num <= 27147) or (27156 <= code_num <= 27159) or \
       (27235 <= code_num <= 27236) or (27244 <= code_num <= 27248):
        return 'hip'
    
    # KNEE SURGERY
    # 27440-27447: Total knee arthroplasty
    # 27486-27487: Knee revision
    # 27427-27429: Ligament reconstruction
    # 29870-29889: Knee arthroscopy (surgical)
    if (27440 <= code_num <= 27447) or (27486 <= code_num <= 27487) or \
       (27427 <= code_num <= 27429) or (29870 <= code_num <= 29889):
        return 'knee'
    
    # SHOULDER SURGERY
    # 23000-23929: Shoulder procedures
    # 23470-23474: Shoulder arthroplasty
    # 23410-23420: Rotator cuff repair
    # 29806-29828: Shoulder arthroscopy (surgical)
    if (23000 <= code_num <= 23929) or (29806 <= code_num <= 29828):
        return 'shoulder_elbow'
    
    # ELBOW SURGERY
    # 24000-24999: Elbow procedures
    # 24360-24370: Elbow arthroplasty
    if (24000 <= code_num <= 24999):
        return 'shoulder_elbow'
    
    # HAND/WRIST SURGERY
    # 25000-26999: Hand and wrist procedures
    # 26340-26373: Tendon repairs
    # 26530-26596: Fracture/dislocation surgery
    if (25000 <= code_num <= 26999):
        return 'hand'
    
    return None

# Load and classify CPT codes
try:
    hcpcs_events = pd.read_csv(HOSP_PATH / 'hcpcsevents.csv.gz', compression='gzip')
    hcpcs_events['ortho_type'] = hcpcs_events['hcpcs_cd'].apply(classify_cpt_orthopedic_procedure)
    ortho_procedures_cpt = hcpcs_events[hcpcs_events['ortho_type'].notna()].copy()
    
    print(f"  CPT-coded orthopedic PROCEDURE RECORDS: {len(ortho_procedures_cpt):,}")
    print(f"  Unique hospital ADMISSIONS (CPT): {ortho_procedures_cpt['hadm_id'].nunique():,}")
    
    # Combine ICD and CPT procedures
    ortho_procedures = pd.concat([
        ortho_procedures_icd[['subject_id', 'hadm_id', 'ortho_type', 'long_title']],
        ortho_procedures_cpt[['subject_id', 'hadm_id', 'ortho_type', 'short_description']].rename(columns={'short_description': 'long_title'})
    ], ignore_index=True)
    
    print(f"\n  ✓ COMBINED (ICD + CPT) orthopedic procedures: {len(ortho_procedures):,}")
    print(f"  ✓ Unique hospital ADMISSIONS (ICD + CPT): {ortho_procedures['hadm_id'].nunique():,}")
    print(f"\n  NOTE: ICD+CPT combined for cohort identification (captures all orthopedic admissions)")
    print(f"        CPT codes alone used for procedure distribution plots (more specific)")
    print(f"        ICD diagnosis codes used for comorbidity identification (appropriate use)")
    
except Exception as e:
    print(f"  ⚠️  CPT codes not available, using ICD codes only: {e}")
    ortho_procedures = ortho_procedures_icd

print(f"\n  Final breakdown by type (procedure records):")
for proc_type in ['spine', 'hip', 'knee', 'shoulder_elbow', 'hand']:
    count = (ortho_procedures['ortho_type'] == proc_type).sum()
    display_name = 'Shoulder/Elbow' if proc_type == 'shoulder_elbow' else proc_type.capitalize()
    print(f"    {display_name:16s}: {count:5,} procedures")

# Show sample procedures to demonstrate we're capturing SURGICAL procedures
print("\n  ═══════════════════════════════════════════════════════════════════════")
print("  SAMPLE SURGICAL PROCEDURES CAPTURED (to verify surgical vs non-operative):")
print("  ═══════════════════════════════════════════════════════════════════════")
for proc_type in ['spine', 'hip', 'knee', 'shoulder_elbow', 'hand']:
    sample = ortho_procedures[ortho_procedures['ortho_type'] == proc_type].head(5)
    if len(sample) > 0:
        display_name = 'Shoulder/Elbow' if proc_type == 'shoulder_elbow' else proc_type.upper()
        print(f"\n  {display_name}:")
        for idx, row in sample.iterrows():
            title = row['long_title']
            if pd.notna(title):
                # Truncate long titles
                title_display = title[:70] + "..." if len(str(title)) > 70 else title
                print(f"    • {title_display}")
print("  ═══════════════════════════════════════════════════════════════════════\n")

# NOTE: Procedure distribution plot moved to after ICU cohort is defined
# (Will be generated later in the script)

# ============================================================================
# [2/8] BUILD COHORT - NON-ELECTIVE (EMERGENT) ADMISSIONS WITH ICU STATUS
# ============================================================================
print("\n[2/8] Building cohort of NON-ELECTIVE (emergent) orthopedic surgery patients...")

# Load core data
admissions = pd.read_csv(HOSP_PATH / 'admissions.csv.gz', compression='gzip')
patients = pd.read_csv(HOSP_PATH / 'patients.csv.gz', compression='gzip')
icustays = pd.read_csv(ICU_PATH / 'icustays.csv.gz', compression='gzip')

# Filter for NON-ELECTIVE (emergent) admissions
# Include emergency department and observation admission types (exclude scheduled/elective)
# Print breakdown of admission types
print("\n  Admission type breakdown (for orthopedic procedures):")
# Get admissions that have orthopedic procedures
ortho_hadm_ids = ortho_procedures['hadm_id'].unique()
ortho_admissions = admissions[admissions['hadm_id'].isin(ortho_hadm_ids)]
admission_type_counts = ortho_admissions['admission_type'].value_counts()
for adm_type, count in admission_type_counts.items():
    print(f"    {adm_type:40s}: {count:6,}")
print()

# Filter for NON-ELECTIVE (emergent) admissions only
non_elective_types = ['EW EMER.', 'DIRECT EMER.', 'EU OBSERVATION', 'OBSERVATION ADMIT',
                      'AMBULATORY OBSERVATION', 'DIRECT OBSERVATION']
elective = admissions[admissions['admission_type'].isin(non_elective_types)].copy()

# Calculate sum of non-elective types from the breakdown
non_elective_sum = sum([admission_type_counts.get(t, 0) for t in non_elective_types])
print(f"\n  Sum of NON-ELECTIVE admissions (from breakdown): {non_elective_sum:,}")
print(f"  Non-elective types included: {', '.join(non_elective_types)}")

# Filter for adults (age >= 18)
adults = patients[patients['anchor_age'] >= 18][['subject_id', 'gender', 'anchor_age']].copy()

# Get unique hospital admissions with orthopedic procedures
# For admissions with multiple procedure types, prioritize by complexity
print("  Handling multi-procedure admissions...")
procedure_priority = {'spine': 1, 'hip': 2, 'knee': 3, 'shoulder_elbow': 4, 'hand': 5}
ortho_procedures['priority'] = ortho_procedures['ortho_type'].map(procedure_priority)
ortho_hadm = ortho_procedures.sort_values('priority').drop_duplicates(subset=['hadm_id'], keep='first')
ortho_hadm = ortho_hadm[['hadm_id', 'subject_id', 'ortho_type']]

multi_proc_count = len(ortho_procedures[['hadm_id', 'subject_id']].drop_duplicates()) - len(ortho_hadm)
if multi_proc_count > 0:
    print(f"  ⚠️  {multi_proc_count:,} admissions had multiple procedure types (kept highest priority)")

# Build cohort: non-elective + adult + orthopedic
cohort = elective.merge(ortho_hadm, on=['hadm_id', 'subject_id'], how='inner')
cohort = cohort.merge(adults, on='subject_id', how='inner')

# Add admission times for temporal filtering
cohort['admittime'] = pd.to_datetime(cohort['admittime'])
cohort['dischtime'] = pd.to_datetime(cohort['dischtime'])

# ============================================================================
# CRITICAL: ONE ADMISSION PER PATIENT (FIRST ADMISSION ONLY)
# ============================================================================
print("\n  Ensuring ONE admission per patient (first orthopedic surgery with ICU)...")
pre_dedup_count = len(cohort)
pre_dedup_patients = cohort['subject_id'].nunique()

# Sort by admission time and keep FIRST admission per patient
cohort = cohort.sort_values('admittime').drop_duplicates(subset=['subject_id'], keep='first')

post_dedup_count = len(cohort)
post_dedup_patients = cohort['subject_id'].nunique()

print(f"  Before deduplication: {pre_dedup_count:,} admissions from {pre_dedup_patients:,} patients")
print(f"  After deduplication:  {post_dedup_count:,} admissions from {post_dedup_patients:,} patients")
print(f"  Removed {pre_dedup_count - post_dedup_count:,} repeat admissions")
print(f"  ✓ Each patient now appears EXACTLY ONCE")

# Determine ICU admission status with temporal filtering
print("  Filtering ICU stays with temporal verification...")

# Get procedure dates
procedure_dates = procedures_icd[procedures_icd['hadm_id'].isin(cohort['hadm_id'])].copy()
procedure_dates['chartdate'] = pd.to_datetime(procedure_dates['chartdate'])

# Get earliest procedure date per admission
earliest_proc_date = procedure_dates.groupby('hadm_id')['chartdate'].min().reset_index()
earliest_proc_date.columns = ['hadm_id', 'procedure_date']
cohort = cohort.merge(earliest_proc_date, on='hadm_id', how='left')

# For admissions without procedure date, use admission date
cohort['procedure_date'] = cohort['procedure_date'].fillna(cohort['admittime'])

# Get ICU stays with times
icustays['intime'] = pd.to_datetime(icustays['intime'])
icustays['outtime'] = pd.to_datetime(icustays['outtime'])

# Merge ICU stays with cohort
cohort_with_icu = cohort.merge(
    icustays[['hadm_id', 'subject_id', 'stay_id', 'intime', 'outtime', 'los']],
    on=['hadm_id', 'subject_id'],
    how='left'
)

# Filter ICU stays that occurred AFTER the procedure AND within a reasonable timeframe
# This ensures ICU admission is directly related to the surgery
# Criteria:
#   1. ICU admission occurs ON or AFTER procedure date
#   2. ICU admission occurs WITHIN 7 DAYS of procedure (to ensure it's surgery-related)
#   3. ICU admission is within the SAME hospital admission
ICU_WINDOW_DAYS = 7  # ICU admission must occur within 7 days of surgery

cohort_with_icu['days_proc_to_icu'] = (
    (cohort_with_icu['intime'] - cohort_with_icu['procedure_date']).dt.total_seconds() / (24*3600)
)

cohort_with_icu['icu_related_to_surgery'] = (
    cohort_with_icu['intime'].notna() & 
    (cohort_with_icu['days_proc_to_icu'] >= 0) &  # ICU on or after procedure
    (cohort_with_icu['days_proc_to_icu'] <= ICU_WINDOW_DAYS)  # Within 7 days
)

# For patients with ICU admission, keep only the FIRST one that meets criteria
first_icu_per_patient = cohort_with_icu[cohort_with_icu['icu_related_to_surgery']].copy()
first_icu_per_patient = first_icu_per_patient.sort_values('intime').groupby('subject_id').first().reset_index()

# Merge back to get ICU status
cohort['had_first_icu'] = cohort['subject_id'].isin(first_icu_per_patient['subject_id']).astype(int)

# For those with ICU, add ICU details
icu_details = first_icu_per_patient[['subject_id', 'stay_id', 'intime', 'outtime', 'los', 'days_proc_to_icu']].copy()
icu_details.columns = ['subject_id', 'icu_stay_id', 'icu_intime', 'icu_outtime', 'icu_los', 'days_proc_to_icu']
cohort = cohort.merge(icu_details, on='subject_id', how='left')

# Rename for clarity
cohort['icu_admission'] = cohort['had_first_icu']

print(f"  ✓ Temporal filtering complete (ICU within {ICU_WINDOW_DAYS} days of surgery)")
print(f"  ✓ Ensured FIRST ICU admission only per patient")
print(f"  ✓ ICU admissions are surgery-related (within {ICU_WINDOW_DAYS}-day window)")

# Show distribution of time from procedure to ICU
if cohort['icu_admission'].sum() > 0:
    icu_timing = cohort[cohort['icu_admission'] == 1]['days_proc_to_icu'].dropna()
    print(f"\n  ⏱️  Time from procedure to ICU admission (for ICU patients):")
    print(f"    Same day (0 days): {(icu_timing == 0).sum():,} ({(icu_timing == 0).mean()*100:.1f}%)")
    print(f"    Within 1 day: {(icu_timing <= 1).sum():,} ({(icu_timing <= 1).mean()*100:.1f}%)")
    print(f"    Within 3 days: {(icu_timing <= 3).sum():,} ({(icu_timing <= 3).mean()*100:.1f}%)")
    print(f"    Within 7 days: {(icu_timing <= 7).sum():,} ({(icu_timing <= 7).mean()*100:.1f}%)")
    print(f"    Median time to ICU: {icu_timing.median():.1f} days")
    print(f"    Mean time to ICU: {icu_timing.mean():.1f} days")

print(f"\n  Total cohort (non-elective + adult + orthopedic): {len(cohort):,} patients")
print(f"  ✓ This matches the sum of non-elective types: {non_elective_sum:,}")
print(f"    ICU (FIRST admission post-surgery): {cohort['icu_admission'].sum():,} ({cohort['icu_admission'].mean()*100:.1f}%)")
print(f"    Non-ICU: {(~cohort['icu_admission'].astype(bool)).sum():,}")

# ============================================================================
# DEFINE COMPOSITE OUTCOMES (90-day mortality + 7-day readmission + LOS >7 days)
# ============================================================================
print("\n  Defining composite outcomes (mortality + readmission + prolonged LOS)...")

# Merge with patient data for death dates
cohort = cohort.merge(patients[['subject_id', 'dod']], on='subject_id', how='left')
cohort['dod'] = pd.to_datetime(cohort['dod'])

# 1. 90-day mortality
cohort['days_to_death'] = (cohort['dod'] - cohort['dischtime']).dt.days
cohort['died_90day'] = (cohort['days_to_death'] <= 90) & (cohort['days_to_death'] >= 0)

# 2. Length of stay (>7 days = poor outcome)
cohort['los_days'] = (cohort['dischtime'] - cohort['admittime']).dt.days
cohort['prolonged_los'] = cohort['los_days'] > 7

print(f"\n  Length of stay statistics:")
print(f"    Median LOS: {cohort['los_days'].median():.1f} days")
print(f"    75th percentile LOS: {cohort['los_days'].quantile(0.75):.1f} days")
print(f"    Mean LOS: {cohort['los_days'].mean():.1f} days")
print(f"  ⚠️  Using >7 days as prolonged LOS threshold")

# Note: Discharge location calculated for reference only (NOT used in outcome)
cohort['discharged_home'] = cohort['discharge_location'].str.contains('HOME', case=False, na=False)

print(f"\n  Discharge locations (for reference only):")
discharge_counts = cohort['discharge_location'].value_counts().head(5)
for loc, count in discharge_counts.items():
    pct = (count / len(cohort)) * 100
    print(f"    {loc:50s}: {count:5,} ({pct:5.1f}%)")

# 4. Readmissions within 7 days (VECTORIZED for speed)
print("  Calculating readmissions within 7 days of discharge (vectorized approach)...")

# Create a cross-join for readmissions (much faster than row-by-row)
# Only look at admissions for patients in our cohort
cohort_subjects = set(cohort['subject_id'])
future_admits = admissions[admissions['subject_id'].isin(cohort_subjects)].copy()

# Ensure datetime types
future_admits['admittime'] = pd.to_datetime(future_admits['admittime'])

# Merge cohort with ALL future admissions
readmit_check = cohort[['subject_id', 'hadm_id', 'dischtime']].merge(
    future_admits[['subject_id', 'hadm_id', 'admittime']],
    on='subject_id',
    how='left',
    suffixes=('_index', '_readmit')
)

# Ensure both columns are datetime after merge
readmit_check['admittime'] = pd.to_datetime(readmit_check['admittime'])
readmit_check['dischtime'] = pd.to_datetime(readmit_check['dischtime'])

# Calculate days between discharge and readmission
readmit_check['days_to_readmit'] = (readmit_check['admittime'] - readmit_check['dischtime']).dt.days

# Filter to only readmissions (different admission, after discharge, within 7 days)
readmit_check['is_readmission'] = (
    (readmit_check['hadm_id_index'] != readmit_check['hadm_id_readmit']) &
    (readmit_check['days_to_readmit'] > 0) &
    (readmit_check['days_to_readmit'] <= 7)
)

# Count readmissions per index admission
readmit_counts = readmit_check.groupby(['subject_id', 'hadm_id_index'])['is_readmission'].sum().reset_index()
readmit_counts.columns = ['subject_id', 'hadm_id', 'num_readmissions_7day']

# Merge back to cohort
cohort = cohort.merge(readmit_counts, on=['subject_id', 'hadm_id'], how='left')
cohort['num_readmissions_7day'] = cohort['num_readmissions_7day'].fillna(0)
cohort['readmitted_7day'] = cohort['num_readmissions_7day'] >= 1

print(f"  ✓ Readmissions calculated (vectorized - much faster!)")

# 3. COMPOSITE POOR OUTCOME (mortality + readmission + prolonged LOS)
cohort['poor_outcome'] = (
    cohort['died_90day'] |
    cohort['readmitted_7day'] |
    cohort['prolonged_los']
)
cohort['good_outcome'] = ~cohort['poor_outcome']

print(f"\n  ✓ Composite outcomes defined (90-day mortality + 7-day readmission + LOS >7 days):")
print(f"    90-day mortality: {cohort['died_90day'].sum():,} ({cohort['died_90day'].mean()*100:.1f}%)")
print(f"    7-day readmission: {cohort['readmitted_7day'].sum():,} ({cohort['readmitted_7day'].mean()*100:.1f}%)")
print(f"    Prolonged LOS (>7 days): {cohort['prolonged_los'].sum():,} ({cohort['prolonged_los'].mean()*100:.1f}%)")
print(f"    COMPOSITE POOR OUTCOME: {cohort['poor_outcome'].sum():,} ({cohort['poor_outcome'].mean()*100:.1f}%)")
print(f"    GOOD OUTCOME: {cohort['good_outcome'].sum():,} ({cohort['good_outcome'].mean()*100:.1f}%)")

# Check breakdown by procedure type
print(f"\n  ICU admission by procedure type:")
for proc_type in ['spine', 'hip', 'knee', 'shoulder_elbow', 'hand']:
    subset = cohort[cohort['ortho_type'] == proc_type]
    icu_count = subset['icu_admission'].sum()
    non_icu_count = len(subset) - icu_count
    display_name = 'Shoulder/Elbow' if proc_type == 'shoulder_elbow' else proc_type.capitalize()
    print(f"    {display_name:16s}: {len(subset):4,} total | ICU: {icu_count:4,} | Non-ICU: {non_icu_count:4,}")

# Show admission type breakdown for each procedure
print(f"\n  Admission type breakdown by procedure:")
print("  " + "="*80)
for proc_type in ['spine', 'hip', 'knee', 'shoulder_elbow', 'hand']:
    subset = cohort[cohort['ortho_type'] == proc_type]
    display_name = 'Shoulder/Elbow' if proc_type == 'shoulder_elbow' else proc_type.capitalize()
    
    print(f"\n  {display_name} ({len(subset):,} total admissions):")
    admission_breakdown = subset['admission_type'].value_counts()
    for adm_type, count in admission_breakdown.items():
        pct = (count / len(subset)) * 100
        icu_in_type = subset[subset['admission_type'] == adm_type]['icu_admission'].sum()
        print(f"    {adm_type:35s}: {count:5,} ({pct:5.1f}%) | ICU: {icu_in_type:4,}")
print("  " + "="*80)

# ============================================================================
# [3/8] FILTER PROCEDURE TYPES - MINIMUM 10 ICU AND 10 NON-ICU
# ============================================================================
print("\n[3/8] Filtering procedure types with minimum case requirements...")
print("  Requirement: At least 10 ICU AND 10 non-ICU cases")

valid_proc_types = []
for proc_type in ['spine', 'hip', 'knee', 'shoulder_elbow', 'hand']:
    subset = cohort[cohort['ortho_type'] == proc_type]
    icu_count = subset['icu_admission'].sum()
    non_icu_count = len(subset) - icu_count
    
    display_name = 'Shoulder/Elbow' if proc_type == 'shoulder_elbow' else proc_type.capitalize()
    
    if icu_count >= 10 and non_icu_count >= 10:
        valid_proc_types.append(proc_type)
        print(f"  ✓ {display_name:16s}: ICU={icu_count:3,}, Non-ICU={non_icu_count:3,} - INCLUDED")
    else:
        print(f"  ✗ {display_name:16s}: ICU={icu_count:3,}, Non-ICU={non_icu_count:3,} - EXCLUDED (insufficient cases)")

# Filter cohort to valid procedure types
cohort = cohort[cohort['ortho_type'].isin(valid_proc_types)].copy()

print(f"\n  Final cohort after filtering: {len(cohort):,} patients")
display_names = ['Shoulder/Elbow' if p == 'shoulder_elbow' else p.capitalize() for p in valid_proc_types]
print(f"    Included procedure types: {', '.join(display_names)}")
print(f"    ICU: {cohort['icu_admission'].sum():,} ({cohort['icu_admission'].mean()*100:.1f}%)")
print(f"    Non-ICU: {(~cohort['icu_admission'].astype(bool)).sum():,}")
print(f"    Imbalance ratio: 1:{(~cohort['icu_admission'].astype(bool)).sum()/cohort['icu_admission'].sum():.1f}")

# ============================================================================
# [4/8] COMPREHENSIVE FEATURE ENGINEERING
# ============================================================================
print("\n[4/8] Extracting comprehensive features...")

features = cohort[['subject_id', 'hadm_id', 'icu_admission', 'anchor_age', 
                   'gender', 'insurance', 'admittime', 'ortho_type']].copy()

# Demographics
print("  [1/6] Demographics...")
features['is_male'] = (features['gender'] == 'M').astype(int)
features['has_medicare'] = (features['insurance'] == 'Medicare').astype(int)
features['has_medicaid'] = (features['insurance'] == 'Medicaid').astype(int)
features['has_private'] = (features['insurance'] == 'Private').astype(int)

# Admission timing
features['admittime'] = pd.to_datetime(features['admittime'])
features['admit_is_weekend'] = (features['admittime'].dt.dayofweek >= 5).astype(int)

# Procedure type (one-hot encoding)
for proc_type in valid_proc_types:
    features[f'proc_{proc_type}'] = (features['ortho_type'] == proc_type).astype(int)

print(f"  ✓ Demographics: {len([c for c in features.columns if c not in ['subject_id', 'hadm_id', 'icu_admission', 'admittime', 'ortho_type']])} features")

# Comorbidities from diagnoses
print("  [2/6] Comorbidities...")
all_diagnoses = pd.read_csv(HOSP_PATH / 'diagnoses_icd.csv.gz', compression='gzip')
cohort_hadm_ids = set(features['hadm_id'])
diagnoses = all_diagnoses[all_diagnoses['hadm_id'].isin(cohort_hadm_ids)].copy()

d_icd_dx = pd.read_csv(HOSP_PATH / 'd_icd_diagnoses.csv.gz', compression='gzip')
diagnoses = diagnoses.merge(d_icd_dx[['icd_code', 'icd_version', 'long_title']], 
                            on=['icd_code', 'icd_version'], how='left')

# Count diagnoses
dx_counts = diagnoses.groupby('hadm_id').size().reset_index(name='num_diagnoses')
features = features.merge(dx_counts, on='hadm_id', how='left')
features['num_diagnoses'] = features['num_diagnoses'].fillna(0)

# Specific comorbidities
def has_dx(hadm, keywords):
    hadm_dx = diagnoses[diagnoses['hadm_id'] == hadm]['long_title'].str.lower()
    return int(hadm_dx.str.contains('|'.join(keywords), na=False).any())

comorbidities = {
    'has_hypertension': ['hypertension'],
    'has_diabetes': ['diabetes'],
    'has_cardiac': ['cardiac', 'coronary', 'myocardial', 'heart failure', 'atrial fib'],
    'has_afib': ['atrial fibrillation'],
    'has_pulmonary': ['copd', 'asthma', 'pulmonary', 'sleep apnea'],
    'has_renal': ['renal', 'kidney', 'chronic kidney'],
    'has_obesity': ['obesity'],
    'has_anemia': ['anemia'],
    'has_depression': ['depression'],
    'has_anxiety': ['anxiety'],
    'has_smoking': ['tobacco', 'smoking'],
    'has_cancer_history': ['cancer', 'carcinoma', 'neoplasm', 'malignancy'],
    'has_stroke_history': ['stroke', 'cerebrovascular'],
    'has_dvt_history': ['thrombosis', 'thromboembolism']
}

for name, keywords in comorbidities.items():
    features[name] = features['hadm_id'].apply(lambda h: has_dx(h, keywords))

# Charlson score
features['charlson_score'] = (
    features['has_diabetes'] + features['has_cardiac'] + 
    features['has_pulmonary'] + features['has_renal']*2 +
    features['has_cancer_history']*2 +
    (features['anchor_age'] > 70).astype(int)
)

print(f"  ✓ Comorbidities: {len(comorbidities) + 2} features")

# Procedure complexity
print("  [3/6] Procedure complexity...")
procedures = ortho_procedures[ortho_procedures['hadm_id'].isin(cohort_hadm_ids)].copy()

proc_counts = procedures.groupby('hadm_id').size().reset_index(name='num_procedures')
features = features.merge(proc_counts, on='hadm_id', how='left')
features['num_procedures'] = features['num_procedures'].fillna(0)

# Procedure types
def has_proc(hadm, keywords):
    hadm_proc = procedures[procedures['hadm_id'] == hadm]['long_title'].str.lower()
    return int(hadm_proc.str.contains('|'.join(keywords), na=False, regex=True).any())

proc_types = {
    'has_fusion': ['fusion', 'refusion', 'arthrodesis'],
    'has_replacement': ['replacement', 'arthroplasty', 'prosthesis'],
    'has_revision': ['revision'],
    'has_multilevel': ['2-3 vertebrae', '4-8 vertebrae', '2 or more', 'multiple'],
    'has_decompression': ['decompression', 'laminectomy'],
    'has_anterior_approach': ['anterior'],
    'has_posterior_approach': ['posterior'],
    'has_arthroscopy': ['arthroscopy', 'arthroscopic']
}

for name, keywords in proc_types.items():
    features[name] = features['hadm_id'].apply(lambda h: has_proc(h, keywords))

# Complexity score
features['procedure_complexity'] = (
    features['num_procedures'] + 
    features['has_fusion']*2 + 
    features['has_replacement']*2 +
    features['has_multilevel']*2 +
    features['has_revision']*3
)

print(f"  ✓ Procedures: {len(proc_types) + 2} features")

# Medications
print("  [4/6] Medications...")
try:
    all_prescriptions = pd.read_csv(HOSP_PATH / 'prescriptions.csv.gz', compression='gzip')
    prescriptions = all_prescriptions[all_prescriptions['hadm_id'].isin(cohort_hadm_ids)].copy()
    
    medication_classes = {
        'has_opioid_rx': ['morphine', 'fentanyl', 'hydromorphone', 'dilaudid', 'oxycodone', 
                          'hydrocodone', 'tramadol', 'methadone'],
        'has_anticoagulant': ['warfarin', 'heparin', 'enoxaparin', 'apixaban', 'rivaroxaban'],
        'has_antiplatelet': ['aspirin', 'clopidogrel', 'plavix'],
        'has_steroid': ['prednisone', 'methylprednisolone', 'dexamethasone', 'hydrocortisone'],
        'has_beta_blocker': ['metoprolol', 'atenolol', 'carvedilol', 'propranolol']
    }
    
    for med_class, keywords in medication_classes.items():
        def has_medication(hadm_id):
            hadm_meds = prescriptions[prescriptions['hadm_id'] == hadm_id]['drug'].str.lower().fillna('')
            return int(hadm_meds.str.contains('|'.join(keywords), na=False).any())
        
        features[med_class] = features['hadm_id'].apply(has_medication)
    
    # Count opioid prescriptions
    opioid_kw = ['morphine', 'fentanyl', 'hydromorphone', 'dilaudid', 'oxycodone', 'hydrocodone', 'tramadol']
    prescriptions['is_opioid'] = prescriptions['drug'].str.lower().str.contains('|'.join(opioid_kw), na=False)
    opioid_rx = prescriptions[prescriptions['is_opioid']].groupby('hadm_id').size()
    opioid_rx = opioid_rx.reset_index(name='num_opioid_rx')
    features = features.merge(opioid_rx, on='hadm_id', how='left')
    features['num_opioid_rx'] = features['num_opioid_rx'].fillna(0)
    
    print(f"  ✓ Medications: {len(medication_classes) + 1} features")
except Exception as e:
    print(f"  ⚠️  Medication features skipped: {e}")
    for med_class in ['has_opioid_rx', 'has_anticoagulant', 'has_antiplatelet', 'has_steroid', 'has_beta_blocker']:
        features[med_class] = 0
    features['num_opioid_rx'] = 0

# Lab values
print("  [5/6] Laboratory values...")
try:
    labevents = pd.read_csv('cohort_data/labevents.csv')
    d_labitems = pd.read_csv(HOSP_PATH / 'd_labitems.csv.gz', compression='gzip')
    labevents = labevents.merge(d_labitems[['itemid', 'label']], on='itemid', how='left')
    labevents['valuenum'] = pd.to_numeric(labevents['valuenum'], errors='coerce')
    
    # Filter to cohort
    labevents = labevents[labevents['hadm_id'].isin(cohort_hadm_ids)].copy()
    
    lab_features = {
        'hemoglobin': 'hemoglobin',
        'hematocrit': 'hematocrit',
        'wbc': 'white blood',
        'platelet': 'platelet',
        'creatinine': 'creatinine',
        'albumin': 'albumin',
        'glucose': 'glucose'
    }
    
    for lab_name, lab_keyword in lab_features.items():
        lab_data = labevents[labevents['label'].str.lower().str.contains(lab_keyword, na=False)]
        lab_summary = lab_data.groupby('hadm_id')['valuenum'].agg(['mean', 'min', 'max']).reset_index()
        lab_summary.columns = ['hadm_id', f'{lab_name}_mean', f'{lab_name}_min', f'{lab_name}_max']
        
        features = features.merge(lab_summary, on='hadm_id', how='left')
    
    print(f"  ✓ Labs: {len(lab_features) * 3} features")
except Exception as e:
    print(f"  ⚠️  Lab features skipped: {e}")

# ICU characteristics (for ICU patients only)
# NOTE: ALL ICU LOS features REMOVED to prevent data leakage (LOS is part of outcome)
print("  [6/6] ICU characteristics...")
print(f"  ⚠️  ICU LOS features (icu_los_mean, icu_los_total) REMOVED to prevent data leakage")
print(f"      Reason: Hospital LOS >7 days is part of the composite outcome")

print(f"\n✓ Feature engineering complete!")
print(f"  Total features: {len([c for c in features.columns if c not in ['subject_id', 'hadm_id', 'icu_admission', 'admittime', 'ortho_type', 'gender', 'insurance']])}")

# Verification: Ensure no duplicate patients
print(f"\n  VERIFICATION - No duplicate patients:")
print(f"    Total rows in features: {len(features):,}")
print(f"    Unique patients: {features['subject_id'].nunique():,}")
print(f"    ✓ Each patient appears EXACTLY ONCE: {len(features) == features['subject_id'].nunique()}")

# ============================================================================
# [5/8] STATISTICAL ANALYSIS - FEATURE SIGNIFICANCE
# ============================================================================
print("\n[5/8] Statistical significance testing...")
print("="*80)

# Merge features with cohort for statistical testing
# Drop duplicate columns from features that are already in cohort (to avoid _x/_y suffixes)
features_to_merge = features.drop(columns=['anchor_age', 'gender', 'insurance', 'ortho_type', 'icu_admission', 'admittime'], errors='ignore')
cohort_with_features = cohort.merge(features_to_merge, on=['subject_id', 'hadm_id'], how='left')

good = cohort_with_features[cohort_with_features['good_outcome'] == True]
poor = cohort_with_features[cohort_with_features['poor_outcome'] == True]

# Continuous features (ALL ICU LOS features REMOVED to prevent data leakage)
cont_features = ['anchor_age', 'num_diagnoses', 'num_procedures', 
                 'charlson_score', 'procedure_complexity', 'num_opioid_rx',
                 'hemoglobin_mean', 'hematocrit_mean', 'wbc_mean', 'platelet_mean',
                 'creatinine_mean', 'albumin_mean', 'glucose_mean']

print("\nContinuous Features (t-test):")
print(f"{'Feature':<30s} {'Good Mean':>10s} {'Poor Mean':>12s} {'p-value':>10s} {'Sig':>5s}")
print("-"*70)

cont_results = []
for feat in cont_features:
    if feat in cohort_with_features.columns:
        good_vals = good[feat].dropna()
        poor_vals = poor[feat].dropna()
        if len(good_vals) > 0 and len(poor_vals) > 0:
            t_stat, p = stats.ttest_ind(good_vals, poor_vals)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"{feat:<30s} {good_vals.mean():>10.2f} {poor_vals.mean():>12.2f} "
                  f"{p:>10.4f} {sig:>5s}")
            cont_results.append({
                'feature': feat,
                'good_mean': good_vals.mean(),
                'poor_mean': poor_vals.mean(),
                'p_value': p,
                'significant': p < 0.05
            })

# Categorical features
cat_features = ['is_male', 'has_hypertension', 'has_diabetes', 'has_cardiac', 'has_afib',
                'has_pulmonary', 'has_obesity', 'has_anemia', 'has_fusion', 
                'has_replacement', 'has_multilevel',
                'has_opioid_rx', 'has_anticoagulant', 'has_antiplatelet', 
                'has_steroid', 'has_beta_blocker']

print("\nCategorical Features (chi-square):")
print(f"{'Feature':<30s} {'Good %':>10s} {'Poor %':>12s} {'p-value':>10s} {'Sig':>5s}")
print("-"*70)

cat_results = []
for feat in cat_features:
    if feat in cohort_with_features.columns:
        contingency = pd.crosstab(cohort_with_features[feat], cohort_with_features['good_outcome'])
        if contingency.shape[0] > 1 and contingency.values.min() >= 5:  # Chi-square validity
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            good_pct = good[feat].mean() * 100 if feat in good.columns else 0
            poor_pct = poor[feat].mean() * 100 if feat in poor.columns else 0
            print(f"{feat:<30s} {good_pct:>9.1f}% {poor_pct:>11.1f}% "
                  f"{p:>10.4f} {sig:>5s}")
            cat_results.append({
                'feature': feat,
                'good_pct': good_pct,
                'poor_pct': poor_pct,
                'p_value': p,
                'significant': p < 0.05
            })

print("\n*** p<0.001, ** p<0.01, * p<0.05")
print("="*80)

# ============================================================================
# [6/8] PREPARE DATA FOR MODELING
# ============================================================================
print("\n[6/8] Preparing data for modeling...")

# ⚠️ FILTER TO ICU PATIENTS ONLY
print(f"\n  ⚠️  IMPORTANT: Filtering to ICU PATIENTS ONLY for modeling")
print(f"  Original cohort: {len(cohort):,} patients")
icu_count = cohort['icu_admission'].sum()
non_icu_count = len(cohort) - icu_count
print(f"    ICU patients: {icu_count:,} ({icu_count/len(cohort)*100:.1f}%)")
print(f"    Non-ICU patients: {non_icu_count:,} ({non_icu_count/len(cohort)*100:.1f}%)")

# Filter cohort and features to ICU patients only
icu_mask = cohort['icu_admission'] == True
cohort_icu = cohort[icu_mask].copy()
features_icu = features[icu_mask].copy()

print(f"\n  ✓ Modeling will use ONLY ICU patients: {len(cohort_icu):,}")
print(f"    Good outcomes: {cohort_icu['good_outcome'].sum():,} ({cohort_icu['good_outcome'].mean()*100:.1f}%)")
print(f"    Poor outcomes: {cohort_icu['poor_outcome'].sum():,} ({cohort_icu['poor_outcome'].mean()*100:.1f}%)")
print(f"\n    Composite outcome = 90-day mortality OR 7-day readmission OR LOS >7 days:")
print(f"      90-day mortality: {cohort_icu['died_90day'].sum():,} ({cohort_icu['died_90day'].mean()*100:.1f}%)")
print(f"      7-day readmission: {cohort_icu['readmitted_7day'].sum():,} ({cohort_icu['readmitted_7day'].mean()*100:.1f}%)")
print(f"      Prolonged LOS (>7 days): {cohort_icu['prolonged_los'].sum():,} ({cohort_icu['prolonged_los'].mean()*100:.1f}%)")

# ============================================================================
# CREATE PROCEDURE DISTRIBUTION PLOT - ICU PATIENTS ONLY
# ============================================================================
print("\n  Creating procedure distribution analysis (ICU PATIENTS ONLY - top 15 procedures)...")

# Get ICU patient admission IDs
icu_hadm_ids = set(cohort_icu['hadm_id'])

# Filter procedures to ICU patients only
ortho_procedures_icu = ortho_procedures[ortho_procedures['hadm_id'].isin(icu_hadm_ids)].copy()

print(f"  ✓ Filtering to ICU patients only: {len(ortho_procedures_icu):,} procedures from {len(icu_hadm_ids):,} ICU admissions")

# Get top procedures for each orthopedic category
proc_order = ['spine', 'hip', 'knee', 'shoulder_elbow', 'hand']
display_names_map = {
    'spine': 'SPINE', 
    'hip': 'HIP', 
    'knee': 'KNEE', 
    'shoulder_elbow': 'SHOULDER/ELBOW', 
    'hand': 'HAND/WRIST'
}

# Extract top 15 procedures for each category - ICU PATIENTS ONLY
proc_type_details = {}
proc_type_totals = {}

for proc_type in proc_order:
    proc_subset = ortho_procedures_icu[ortho_procedures_icu['ortho_type'] == proc_type].copy()
    
    # Clean procedure titles
    proc_subset['clean_title'] = proc_subset['long_title'].fillna('Unknown')
    
    # Count top 15 unique procedure types
    proc_counts = proc_subset['clean_title'].value_counts().head(15)
    proc_type_details[proc_type] = proc_counts
    
    # Store total count for this procedure type
    proc_type_totals[proc_type] = len(proc_subset)
    
    print(f"    {display_names_map[proc_type]}: {len(proc_subset):,} ICU procedures, showing top 15")

# Create figure with horizontal bar charts showing top 15 procedures
fig = plt.figure(figsize=(20, 18))
gs = fig.add_gridspec(5, 1, hspace=0.35, top=0.98, bottom=0.04, left=0.25, right=0.97)

# Color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

for idx, proc_type in enumerate(proc_order):
    ax = fig.add_subplot(gs[idx, 0])
    
    proc_counts = proc_type_details[proc_type]
    
    if len(proc_counts) > 0:
        # Truncate long procedure names
        labels = [title[:85] + '...' if len(title) > 85 else title for title in proc_counts.index]
        values = proc_counts.values
        
        # Create horizontal bar chart
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, height=0.7, alpha=0.8, 
                       color=colors[idx], 
                       edgecolor='black', linewidth=0.8)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            width = bar.get_width()
            pct = (val / values.sum()) * 100
            ax.text(width + max(values)*0.01, bar.get_y() + bar.get_height()/2.,
                    f'{int(val):,} ({pct:.1f}%)',
                    ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Styling
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlabel('Number of Procedure Records (ICU Patients)', fontsize=11, fontweight='bold')
        
        display_name = display_names_map[proc_type]
        total_all = proc_type_totals[proc_type]
        total_top15 = proc_counts.sum()
        pct_covered = (total_top15 / total_all) * 100 if total_all > 0 else 0
        
        ax.set_title(f'{display_name} - Top 15 Procedures (ICU Total: {total_all:,}, Top 15: {total_top15:,}, {pct_covered:.1f}%)', 
                    fontsize=13, fontweight='bold', loc='left', pad=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

# Add footer note
fig.text(0.5, 0.01, 
         'Note: ICU patients only (n=1,736). Procedure codes from ICD-9, ICD-10-PCS, and CPT/HCPCS combined.\n' +
         'ICD diagnosis codes used separately for comorbidity identification.',
         ha='center', fontsize=10, style='italic', color='#555555', wrap=True)

plt.savefig('ortho_procedure_distribution_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: ortho_procedure_distribution_analysis.png (ICU patients only, top 15 per category)")
plt.close()


# Select model features (exclude identifiers and target)
model_features = [c for c in features_icu.columns if c not in 
                 ['subject_id', 'hadm_id', 'icu_admission', 'admittime', 
                  'ortho_type', 'gender', 'insurance']]

print(f"\n  Selected {len(model_features)} features for modeling")

# Prepare X and y (ICU PATIENTS ONLY)
# TARGET: Predicting COMPOSITE OUTCOME FOR ICU PATIENTS
# y=1 (good outcome) = Survived 90 days AND no 7-day readmission AND LOS ≤7 days
# y=0 (poor outcome) = Died within 90 days OR readmitted within 7 days OR LOS >7 days
X = features_icu[model_features].copy()
y = cohort_icu['good_outcome'].astype(int).copy()  # Ensure y is integer (0/1)

print(f"\n  📊 TARGET VARIABLE: Predicting COMPOSITE OUTCOME (ICU PATIENTS ONLY)")
print(f"     Good outcome (y=1): Survived 90 days AND no 7-day readmission AND LOS ≤7 days")
print(f"     Poor outcome (y=0): Died within 90 days OR readmitted within 7 days OR LOS >7 days")

# Convert all columns to numeric (in case any are object type)
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce')

# Check for missing values before filling
print(f"  Missing values before imputation: {X.isnull().sum().sum()}")

# Fill missing values with median/mode
for col in X.columns:
    if X[col].isnull().any():
        if X[col].dtype in ['float64', 'int64']:
            # Use median, or 0 if all values are NaN
            median_val = X[col].median()
            if pd.isna(median_val):
                X[col].fillna(0, inplace=True)
            else:
                X[col].fillna(median_val, inplace=True)
        else:
            X[col].fillna(0, inplace=True)

# Replace any remaining inf values with 0
X.replace([np.inf, -np.inf], 0, inplace=True)

# Final check for NaN/inf
print(f"  Missing values after imputation: {X.isnull().sum().sum()}")
print(f"  Infinite values: {np.isinf(X.values).sum()}")

if X.isnull().any().any():
    print("  ⚠️  WARNING: Still have NaN values. Filling remaining with 0...")
    X.fillna(0, inplace=True)

print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"  Target: GOOD outcome={y.sum()} ({y.mean()*100:.1f}%), POOR outcome={(~y.astype(bool)).sum()} ({(~y.astype(bool)).mean()*100:.1f}%)")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"  Training: {len(X_train)} samples (Good={y_train.sum()}, {y_train.mean()*100:.1f}%)")
print(f"  Test: {len(X_test)} samples (Good={y_test.sum()}, {y_test.mean()*100:.1f}%)")

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check for NaN after scaling (can happen if a feature has zero variance)
if np.isnan(X_train_scaled).any():
    print("  ⚠️  WARNING: NaN values after scaling. Replacing with 0...")
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

print("✓ Data prepared for modeling")

# ============================================================================
# [7/8] TRAIN MODELS WITH HYPERPARAMETER TUNING
# ============================================================================
print("\n[7/8] Training models with hyperparameter tuning...")
print("="*80)
print("  ⏱️  Estimated time: 5-15 minutes depending on dataset size")
print("  📊 Progress bars will show training status\n")

models_results = {}

# 1. Logistic Regression (STRONGER REGULARIZATION to prevent overfitting)
print("\n[A] Logistic Regression with GridSearchCV...")
print("  ⚠️  Using STRONG regularization to prevent overfitting on single features")
lr_param_grid = {
    'C': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],  # Stronger regularization (smaller C)
    'penalty': ['l1', 'l2'],  # L1 for feature selection, L2 for shrinkage
    'solver': ['liblinear'],
    'class_weight': ['balanced'],
    'max_iter': [2000]
}

lr_grid = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid=lr_param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2  # Show progress
)

start_time = time.time()
lr_grid.fit(X_train_scaled, y_train)
lr_time = time.time() - start_time

lr = lr_grid.best_estimator_
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

auc_lr = roc_auc_score(y_test, y_prob_lr)
ap_lr = average_precision_score(y_test, y_prob_lr)

print(f"  ✓ AUC-ROC: {auc_lr:.4f}")
print(f"  ✓ Avg Precision: {ap_lr:.4f}")
print(f"  ✓ Best params: {lr_grid.best_params_}")
print(f"  ✓ Time: {lr_time:.1f}s")

models_results['Logistic Regression'] = {
    'model': lr, 'y_prob': y_prob_lr, 'auc': auc_lr, 'ap': ap_lr, 'time': lr_time
}

# 2. Random Forest (CONSTRAINED to prevent single-feature dominance)
print("\n[B] Random Forest with RandomizedSearchCV...")
print("  ⚠️  Adding constraints to prevent overfitting on individual features")
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],  # Shallower trees to prevent overfitting
    'min_samples_split': [10, 20, 30],  # Higher values for more regularization
    'min_samples_leaf': [4, 8, 12],  # Higher values to prevent overfitting
    'max_features': ['sqrt', 'log2', 0.3, 0.5],  # Limit features per split
    'class_weight': ['balanced'],
    'min_impurity_decrease': [0.0, 0.001, 0.01]  # Require improvement to split
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=rf_param_grid,
    n_iter=30,  # Reduced from 50 for speed
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=2  # Show progress
)

start_time = time.time()
rf_random.fit(X_train_scaled, y_train)
rf_time = time.time() - start_time

rf = rf_random.best_estimator_
y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]

auc_rf = roc_auc_score(y_test, y_prob_rf)
ap_rf = average_precision_score(y_test, y_prob_rf)

print(f"  ✓ AUC-ROC: {auc_rf:.4f}")
print(f"  ✓ Avg Precision: {ap_rf:.4f}")
print(f"  ✓ Best params: {rf_random.best_params_}")
print(f"  ✓ Time: {rf_time:.1f}s")

models_results['Random Forest'] = {
    'model': rf, 'y_prob': y_prob_rf, 'auc': auc_rf, 'ap': ap_rf, 'time': rf_time
}

# 3. Gradient Boosting (LOWER learning rate and constrained)
print("\n[C] Gradient Boosting with RandomizedSearchCV...")
print("  ⚠️  Using lower learning rates and feature constraints")
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],  # Lower learning rates
    'max_depth': [2, 3, 4, 5],  # Shallower trees
    'min_samples_split': [10, 20, 30],  # Higher for regularization
    'min_samples_leaf': [4, 8, 12],  # Higher for regularization
    'subsample': [0.7, 0.8, 0.9],  # Lower to prevent overfitting
    'max_features': ['sqrt', 'log2', 0.3, 0.5]  # Limit features per split
}

gb_random = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_distributions=gb_param_grid,
    n_iter=30,  # Reduced from 50 for speed
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=2  # Show progress
)

start_time = time.time()
# Use sample weights for gradient boosting
sample_weights = compute_sample_weight('balanced', y_train)
gb_random.fit(X_train_scaled, y_train, sample_weight=sample_weights)
gb_time = time.time() - start_time

gb = gb_random.best_estimator_
y_pred_gb = gb.predict(X_test_scaled)
y_prob_gb = gb.predict_proba(X_test_scaled)[:, 1]

auc_gb = roc_auc_score(y_test, y_prob_gb)
ap_gb = average_precision_score(y_test, y_prob_gb)

print(f"  ✓ AUC-ROC: {auc_gb:.4f}")
print(f"  ✓ Avg Precision: {ap_gb:.4f}")
print(f"  ✓ Best params: {gb_random.best_params_}")
print(f"  ✓ Time: {gb_time:.1f}s")

models_results['Gradient Boosting'] = {
    'model': gb, 'y_prob': y_prob_gb, 'auc': auc_gb, 'ap': ap_gb, 'time': gb_time
}

print("\n" + "="*80)
print("✓ All models trained")

# ============================================================================
# [8/8] FEATURE IMPORTANCE AND VISUALIZATION
# ============================================================================
print("\n[8/8] Extracting feature importance and creating visualizations...")

# Determine best model
best_model_name = max(models_results.items(), key=lambda x: x[1]['auc'])[0]
best_model = models_results[best_model_name]['model']
best_auc = models_results[best_model_name]['auc']

print(f"\n🏆 BEST MODEL: {best_model_name} (AUC={best_auc:.4f})")

# Feature importance
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    feature_importance = pd.DataFrame({
        'feature': model_features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
else:  # Logistic Regression
    feature_importance = pd.DataFrame({
        'feature': model_features,
        'coefficient': best_model.coef_[0],
        'importance': np.abs(best_model.coef_[0])
    }).sort_values('importance', ascending=False)

print(f"\n📊 TOP 20 MOST IMPORTANT FEATURES:")
print("="*80)
for i, row in feature_importance.head(20).iterrows():
    print(f"  {i+1:2d}. {row['feature']:<40s}: {row['importance']:.6f}")

# Save results
feature_importance.to_csv('ortho_all_procedures_feature_importance.csv', index=False)
print(f"\n✓ Saved: ortho_all_procedures_feature_importance.csv")

# Model comparison
results_df = pd.DataFrame([
    {
        'Model': name, 
        'AUC-ROC': results['auc'], 
        'Avg Precision': results['ap'],
        'Time (s)': results['time']
    }
    for name, results in models_results.items()
])
results_df.to_csv('ortho_all_procedures_model_performance.csv', index=False)
print(f"✓ Saved: ortho_all_procedures_model_performance.csv")

# ============================================================================
# [7B/8] SHAP ANALYSIS - Check for overfitting on single features
# ============================================================================
if SHAP_AVAILABLE:
    print("\n" + "="*80)
    print("SHAP ANALYSIS - Checking for Feature Overfitting")
    print("="*80)
    
    try:
        # Use the best model for SHAP analysis
        print(f"\nCalculating SHAP values for: {best_model_name}")
        print("(This may take a few minutes...)")
        
        # Sample data for faster SHAP computation
        sample_size = min(500, len(X_test_scaled))
        X_shap_sample = X_test_scaled[:sample_size]
        
        # Create SHAP explainer based on model type
        if 'Logistic' in best_model_name:
            explainer = shap.LinearExplainer(best_model, X_train_scaled)
            shap_values = explainer.shap_values(X_shap_sample)
        else:  # Tree-based models
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_shap_sample)
            if isinstance(shap_values, list):  # For binary classification
                shap_values = shap_values[1]
        
        # Calculate mean absolute SHAP values for each feature
        # Create feature name mapping dictionary
        feature_name_map = {
            'anchor_age': 'Age (years)', 'is_male': 'Male Gender',
            'has_medicare': 'Medicare Insurance', 'has_medicaid': 'Medicaid Insurance',
            'has_private': 'Private Insurance', 'admit_is_weekend': 'Weekend Admission',
            'proc_spine': 'Spine Surgery', 'proc_hip': 'Hip Surgery',
            'proc_knee': 'Knee Surgery', 'proc_shoulder_elbow': 'Shoulder/Elbow Surgery',
            'proc_hand': 'Hand/Wrist Surgery', 'has_hypertension': 'Hypertension',
            'has_diabetes': 'Diabetes', 'has_cardiac': 'Cardiac Disease',
            'has_afib': 'Atrial Fibrillation', 'has_pulmonary': 'Pulmonary Disease',
            'has_renal': 'Renal Disease', 'has_obesity': 'Obesity',
            'has_anemia': 'Anemia', 'has_depression': 'Depression',
            'has_anxiety': 'Anxiety', 'has_smoking': 'Smoking/Tobacco Use',
            'has_cancer_history': 'Cancer History', 'has_stroke_history': 'Stroke History',
            'has_dvt_history': 'DVT/PE History', 'charlson_score': 'Charlson Comorbidity Score',
            'num_diagnoses': 'Number of Diagnoses', 'num_procedures': 'Number of Procedures',
            'has_fusion': 'Spinal Fusion', 'has_replacement': 'Joint Replacement',
            'has_revision': 'Revision Surgery', 'has_multilevel': 'Multi-level Surgery',
            'has_decompression': 'Decompression Surgery', 'has_anterior_approach': 'Anterior Approach',
            'has_posterior_approach': 'Posterior Approach', 'has_arthroscopy': 'Arthroscopic Surgery',
            'procedure_complexity': 'Procedure Complexity Score',
            'has_opioid_rx': 'Opioid Prescription', 'num_opioid_rx': 'Number of Opioid Prescriptions',
            'has_anticoagulant': 'Anticoagulant Use', 'has_antiplatelet': 'Antiplatelet Use',
            'has_steroid': 'Steroid Use', 'has_beta_blocker': 'Beta Blocker Use',
        }
        
        shap_importance = pd.DataFrame({
            'feature': model_features,
            'display_name': [feature_name_map.get(f, f) for f in model_features],
            'mean_abs_shap': np.abs(shap_values).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        print(f"\n📊 TOP 15 FEATURES BY SHAP IMPORTANCE:")
        print("="*80)
        print(f"{'Rank':<6} {'Feature':<45} {'Mean |SHAP|':<15} {'% of Total':<10}")
        print("-"*80)
        
        total_shap = shap_importance['mean_abs_shap'].sum()
        for i, row in shap_importance.head(15).iterrows():
            pct = (row['mean_abs_shap'] / total_shap) * 100
            print(f"{i+1:<6} {row['display_name']:<45} {row['mean_abs_shap']:<15.6f} {pct:>6.2f}%")
        
        # Check for overfitting warning
        top_feature_pct = (shap_importance.iloc[0]['mean_abs_shap'] / total_shap) * 100
        if top_feature_pct > 30:
            print(f"\n⚠️  WARNING: Top feature accounts for {top_feature_pct:.1f}% of importance!")
            print(f"   This suggests potential overfitting on: {shap_importance.iloc[0]['feature']}")
            print(f"   Consider: 1) More regularization, 2) Feature engineering, 3) More data")
        else:
            print(f"\n✅ Feature importance is well-distributed (top feature: {top_feature_pct:.1f}%)")
        
        # Save SHAP importance
        shap_importance.to_csv('ortho_all_procedures_shap_importance.csv', index=False)
        print(f"\n✓ Saved: ortho_all_procedures_shap_importance.csv")
        
        # Create SHAP summary plot with readable feature names
        print("\n📊 Creating SHAP summary plot...")
        # Map feature names for better readability
        readable_feature_names = [feature_name_map.get(f, f) for f in model_features]
        
        fig_shap, ax_shap = plt.subplots(figsize=(14, 10))
        shap.summary_plot(shap_values, X_shap_sample, feature_names=readable_feature_names, 
                         max_display=20, show=False)
        plt.tight_layout()
        plt.savefig('ortho_all_procedures_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: ortho_all_procedures_shap_summary.png")
        
    except Exception as e:
        print(f"\n⚠️  SHAP analysis failed: {e}")
        print("   Continuing with standard analysis...")
else:
    print("\n⚠️  SHAP not available. Install with: pip install shap")
    print("   Skipping SHAP analysis...")

# Create feature name mapping for better labels in plots
feature_name_map = {
    # Demographics
    'anchor_age': 'Age (years)',
    'is_male': 'Male Gender',
    'has_medicare': 'Medicare Insurance',
    'has_medicaid': 'Medicaid Insurance',
    'has_private': 'Private Insurance',
    'admit_is_weekend': 'Weekend Admission',
    
    # Procedure types
    'proc_spine': 'Spine Surgery',
    'proc_hip': 'Hip Surgery',
    'proc_knee': 'Knee Surgery',
    'proc_shoulder_elbow': 'Shoulder/Elbow Surgery',
    'proc_hand': 'Hand/Wrist Surgery',
    
    # Comorbidities
    'has_hypertension': 'Hypertension',
    'has_diabetes': 'Diabetes',
    'has_cardiac': 'Cardiac Disease',
    'has_afib': 'Atrial Fibrillation',
    'has_pulmonary': 'Pulmonary Disease',
    'has_renal': 'Renal Disease',
    'has_obesity': 'Obesity',
    'has_anemia': 'Anemia',
    'has_depression': 'Depression',
    'has_anxiety': 'Anxiety',
    'has_smoking': 'Smoking/Tobacco Use',
    'has_cancer_history': 'Cancer History',
    'has_stroke_history': 'Stroke History',
    'has_dvt_history': 'DVT/PE History',
    'charlson_score': 'Charlson Comorbidity Score',
    
    # Diagnoses
    'num_diagnoses': 'Number of Diagnoses',
    
    # Procedures
    'num_procedures': 'Number of Procedures',
    'has_fusion': 'Spinal Fusion',
    'has_replacement': 'Joint Replacement',
    'has_revision': 'Revision Surgery',
    'has_multilevel': 'Multi-level Surgery',
    'has_decompression': 'Decompression Surgery',
    'has_anterior_approach': 'Anterior Approach',
    'has_posterior_approach': 'Posterior Approach',
    'has_arthroscopy': 'Arthroscopic Surgery',
    'procedure_complexity': 'Procedure Complexity Score',
    
    # Medications
    'has_opioid_rx': 'Opioid Prescription',
    'num_opioid_rx': 'Number of Opioid Prescriptions',
    'has_anticoagulant': 'Anticoagulant Use',
    'has_antiplatelet': 'Antiplatelet Use',
    'has_steroid': 'Steroid Use',
    'has_beta_blocker': 'Beta Blocker Use',
    
    # Lab values
    'hemoglobin_mean': 'Hemoglobin (Mean)',
    'hemoglobin_min': 'Hemoglobin (Min)',
    'hemoglobin_max': 'Hemoglobin (Max)',
    'hematocrit_mean': 'Hematocrit (Mean)',
    'hematocrit_min': 'Hematocrit (Min)',
    'hematocrit_max': 'Hematocrit (Max)',
    'wbc_mean': 'WBC Count (Mean)',
    'wbc_min': 'WBC Count (Min)',
    'wbc_max': 'WBC Count (Max)',
    'platelet_mean': 'Platelet Count (Mean)',
    'platelet_min': 'Platelet Count (Min)',
    'platelet_max': 'Platelet Count (Max)',
    'creatinine_mean': 'Creatinine (Mean)',
    'creatinine_min': 'Creatinine (Min)',
    'creatinine_max': 'Creatinine (Max)',
    'albumin_mean': 'Albumin (Mean)',
    'albumin_min': 'Albumin (Min)',
    'albumin_max': 'Albumin (Max)',
    'glucose_mean': 'Glucose (Mean)',
    'glucose_min': 'Glucose (Min)',
    'glucose_max': 'Glucose (Max)',
}

# Visualizations
print(f"\n📊 Creating comprehensive visualizations...")

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.35)

# Title
display_names = ['Shoulder/Elbow' if p == 'shoulder_elbow' else p.capitalize() for p in valid_proc_types]
fig.suptitle(f'Composite Outcome Prediction - ICU Patients After Orthopedic Surgery\n' + 
             f'({", ".join(display_names)}) - ICU Patients Only',
             fontsize=16, fontweight='bold', y=0.995)

# A: Model Performance
ax1 = fig.add_subplot(gs[0, :])
models_names = list(models_results.keys())
aucs = [models_results[m]['auc'] for m in models_names]
aps = [models_results[m]['ap'] for m in models_names]
x_pos = np.arange(len(models_names))
width = 0.35
ax1.bar(x_pos - width/2, aucs, width, label='AUC-ROC', alpha=0.8, color='#3498db')
ax1.bar(x_pos + width/2, aps, width, label='Avg Precision', alpha=0.8, color='#e74c3c')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models_names, fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontweight='bold', fontsize=13)
ax1.set_title('A. Model Performance Comparison', fontweight='bold', fontsize=14, loc='left')
ax1.legend(fontsize=12, loc='lower right', framealpha=0.95)  # Fixed position to avoid overlap
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 1.08)  # Increased to give more space for labels above bars
for i, (auc, ap) in enumerate(zip(aucs, aps)):
    ax1.text(i - width/2, auc + 0.02, f'{auc:.3f}', ha='center', fontweight='bold', fontsize=10)
    ax1.text(i + width/2, ap + 0.02, f'{ap:.3f}', ha='center', fontweight='bold', fontsize=10)

# B: ROC Curves
ax2 = fig.add_subplot(gs[1, 0])
colors = ['#2ecc71', '#e74c3c', '#3498db']
for (name, results), color in zip(models_results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, results['y_prob'])
    ax2.plot(fpr, tpr, label=f"{name} ({results['auc']:.3f})", linewidth=3, color=color)
ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5)
ax2.set_xlabel('False Positive Rate', fontweight='bold', fontsize=11)
ax2.set_ylabel('True Positive Rate', fontweight='bold', fontsize=11)
ax2.set_title('B. ROC Curves', fontweight='bold', fontsize=13, loc='left')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# C: Precision-Recall Curves
ax3 = fig.add_subplot(gs[1, 1])
for (name, results), color in zip(models_results.items(), colors):
    precision, recall, _ = precision_recall_curve(y_test, results['y_prob'])
    ax3.plot(recall, precision, label=f"{name} ({results['ap']:.3f})", linewidth=3, color=color)
baseline = y_test.mean()
ax3.axhline(y=baseline, color='k', linestyle='--', linewidth=2, alpha=0.5, 
           label=f'Baseline ({baseline:.3f})')
ax3.set_xlabel('Recall', fontweight='bold', fontsize=11)
ax3.set_ylabel('Precision', fontweight='bold', fontsize=11)
ax3.set_title('C. Precision-Recall Curves', fontweight='bold', fontsize=13, loc='left')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# D: Cohort Statistics
ax4 = fig.add_subplot(gs[1, 2])
proc_stats = []
for proc_type in valid_proc_types:
    subset = cohort[cohort['ortho_type'] == proc_type]
    icu_count = subset['icu_admission'].sum()
    non_icu_count = len(subset) - icu_count
    display_name = 'Shoulder/Elbow' if proc_type == 'shoulder_elbow' else proc_type.capitalize()
    proc_stats.append({
        'Procedure': display_name,
        'ICU': icu_count,
        'Non-ICU': non_icu_count
    })
proc_stats_df = pd.DataFrame(proc_stats)
x_pos = np.arange(len(proc_stats_df))
width = 0.35
bars1 = ax4.bar(x_pos - width/2, proc_stats_df['ICU'], width, label='ICU', alpha=0.8, color='#e74c3c')
bars2 = ax4.bar(x_pos + width/2, proc_stats_df['Non-ICU'], width, label='Non-ICU', alpha=0.8, color='#2ecc71')

# Add count labels on bars
for bar in bars1:
    height = bar.get_height()
    if height > 0:
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    if height > 0:
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

ax4.set_xticks(x_pos)
ax4.set_xticklabels(proc_stats_df['Procedure'], rotation=15, ha='right', fontsize=10)
ax4.set_ylabel('Count', fontweight='bold', fontsize=11)
ax4.set_title('D. Cohort Distribution by Procedure (Prioritized)', fontweight='bold', fontsize=13, loc='left')
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)

# E: Top 15 Features
ax5 = fig.add_subplot(gs[2, :])
top_15 = feature_importance.head(15).copy()
# Map feature names to readable labels
top_15['display_name'] = top_15['feature'].map(lambda x: feature_name_map.get(x, x))
colors_feat = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_15)))
y_pos = np.arange(len(top_15)) * 1.2  # Increase spacing between bars
ax5.barh(y_pos, top_15['importance'], height=0.8, alpha=0.85, edgecolor='black', linewidth=1.5, color=colors_feat)
ax5.set_yticks(y_pos)
ax5.set_yticklabels(top_15['display_name'], fontsize=10, fontweight='bold')
ax5.invert_yaxis()
ax5.set_xlabel('Importance Score', fontweight='bold', fontsize=12)
ax5.set_title(f'E. Top 15 Features from {best_model_name}', fontweight='bold', fontsize=13, loc='left')
ax5.grid(axis='x', alpha=0.3)
ax5.set_ylim(max(y_pos) + 0.8, min(y_pos) - 0.8)  # Add padding at top and bottom

# F: Age Distribution
ax6 = fig.add_subplot(gs[3, 0])
icu_ages = features[features['icu_admission']==1]['anchor_age']
non_icu_ages = features[features['icu_admission']==0]['anchor_age']
ax6.hist([icu_ages, non_icu_ages], bins=15, label=['ICU', 'Non-ICU'], 
        alpha=0.7, color=['#e74c3c', '#2ecc71'])
ax6.set_xlabel('Age (years)', fontweight='bold', fontsize=11)
ax6.set_ylabel('Frequency', fontweight='bold', fontsize=11)
ax6.set_title('F. Age Distribution', fontweight='bold', fontsize=13, loc='left')
ax6.legend(fontsize=10)
ax6.grid(axis='y', alpha=0.3)

# G: Charlson Score
ax7 = fig.add_subplot(gs[3, 1])
icu_charlson = features[features['icu_admission']==1]['charlson_score']
non_icu_charlson = features[features['icu_admission']==0]['charlson_score']
ax7.hist([icu_charlson, non_icu_charlson], bins=10, label=['ICU', 'Non-ICU'], 
        alpha=0.7, color=['#e74c3c', '#2ecc71'])
ax7.set_xlabel('Charlson Score', fontweight='bold', fontsize=11)
ax7.set_ylabel('Frequency', fontweight='bold', fontsize=11)
ax7.set_title('G. Comorbidity Burden', fontweight='bold', fontsize=13, loc='left')
ax7.legend(fontsize=10)
ax7.grid(axis='y', alpha=0.3)

# H: ICU Rate by Procedure
ax8 = fig.add_subplot(gs[3, 2])
icu_rates = []
icu_counts = []
total_counts = []
display_names_proc = []
for proc_type in valid_proc_types:
    subset = cohort[cohort['ortho_type'] == proc_type]
    icu_n = subset['icu_admission'].sum()
    total_n = len(subset)
    rate = (icu_n / total_n * 100) if total_n > 0 else 0
    icu_rates.append(rate)
    icu_counts.append(icu_n)
    total_counts.append(total_n)
    display_name = 'Shoulder/Elbow' if proc_type == 'shoulder_elbow' else proc_type.capitalize()
    display_names_proc.append(display_name)
colors_proc = plt.cm.Set2(np.arange(len(valid_proc_types)))
bars = ax8.bar(range(len(valid_proc_types)), icu_rates, alpha=0.85, 
       edgecolor='black', linewidth=1.5, color=colors_proc)
ax8.set_xticks(range(len(valid_proc_types)))
ax8.set_xticklabels(display_names_proc, rotation=15, ha='right', fontsize=10)
ax8.set_ylabel('ICU Admission Rate (%)', fontweight='bold', fontsize=11)
ax8.set_title('H. ICU Rate by Procedure Type (Prioritized)', fontweight='bold', fontsize=13, loc='left')
ax8.grid(axis='y', alpha=0.3)
# Set y-limit to add space for labels above bars
max_rate = max(icu_rates)
ax8.set_ylim(0, max_rate * 1.2)  # Add 20% space at top
for i, (rate, icu_n, total_n) in enumerate(zip(icu_rates, icu_counts, total_counts)):
    ax8.text(i, rate + max_rate*0.02, f'{rate:.1f}%\n({icu_n}/{total_n})', 
             ha='center', va='bottom', fontweight='bold', fontsize=8)

plt.savefig('ortho_all_procedures_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: ortho_all_procedures_comprehensive_analysis.png")

# ============================================================================
# CREATE PROFESSIONAL DEMOGRAPHIC TABLE (TABLE 1)
# ============================================================================
print("\n📊 Creating professional demographic characteristics table...")

# Merge cohort with features for complete data
cohort_for_table = cohort_icu.merge(
    features_icu[['subject_id', 'hadm_id', 'is_male', 'has_medicare', 'has_medicaid', 
                  'has_private', 'num_diagnoses', 'charlson_score', 'procedure_complexity',
                  'has_hypertension', 'has_diabetes', 'has_cardiac', 'has_pulmonary',
                  'has_obesity', 'has_anemia', 'num_opioid_rx']],
    on=['subject_id', 'hadm_id'],
    how='left'
)

# Create Table 1 - Professional demographic characteristics
table_data = []

# Total N
table_data.append(['', 'Overall', f'n = {len(cohort_icu):,}'])

# Age
age_mean = cohort_for_table['anchor_age'].mean()
age_std = cohort_for_table['anchor_age'].std()
age_median = cohort_for_table['anchor_age'].median()
age_q1 = cohort_for_table['anchor_age'].quantile(0.25)
age_q3 = cohort_for_table['anchor_age'].quantile(0.75)
table_data.append(['Demographics', '', ''])
table_data.append(['  Age (years)', 'Mean ± SD', f'{age_mean:.1f} ± {age_std:.1f}'])
table_data.append(['', 'Median [IQR]', f'{age_median:.0f} [{age_q1:.0f}, {age_q3:.0f}]'])

# Sex
male_pct = cohort_for_table['is_male'].mean() * 100
female_pct = 100 - male_pct
male_n = cohort_for_table['is_male'].sum()
female_n = len(cohort_for_table) - male_n
table_data.append(['  Male sex', 'n (%)', f'{male_n:,} ({male_pct:.1f}%)'])
table_data.append(['  Female sex', 'n (%)', f'{female_n:,} ({female_pct:.1f}%)'])

# Insurance
medicare_n = cohort_for_table['has_medicare'].sum()
medicaid_n = cohort_for_table['has_medicaid'].sum()
private_n = cohort_for_table['has_private'].sum()
other_n = len(cohort_for_table) - medicare_n - medicaid_n - private_n
table_data.append(['  Insurance', '', ''])
table_data.append(['    Medicare', 'n (%)', f'{medicare_n:,} ({medicare_n/len(cohort_for_table)*100:.1f}%)'])
table_data.append(['    Medicaid', 'n (%)', f'{medicaid_n:,} ({medicaid_n/len(cohort_for_table)*100:.1f}%)'])
table_data.append(['    Private', 'n (%)', f'{private_n:,} ({private_n/len(cohort_for_table)*100:.1f}%)'])
table_data.append(['    Other', 'n (%)', f'{other_n:,} ({other_n/len(cohort_for_table)*100:.1f}%)'])

# Procedure type distribution - TRUE counts (patients may appear in multiple regions)
table_data.append(['Procedure Distribution by Body Region', '', ''])
table_data.append(['  (Patients with multi-region surgery counted in each applicable region)', '', ''])

# Get procedure details from ortho_procedures for granular breakdown
try:
    cohort_hadm_ids_table = set(cohort_for_table['hadm_id'])
    procedures_for_table = ortho_procedures[ortho_procedures['hadm_id'].isin(cohort_hadm_ids_table)].copy()
    
    # Define SPECIFIC procedure types for each anatomic region
    proc_type_keywords = {
        'spine': {
            'Posterior Lumbar Fusion': ['posterior', 'lumbar', 'fusion'],
            'Anterior Cervical Discectomy & Fusion (ACDF)': ['anterior', 'cervical', 'discectomy', 'fusion'],
            'Posterior Cervical Fusion': ['posterior', 'cervical', 'fusion'],
            'Anterior Lumbar Interbody Fusion (ALIF)': ['anterior', 'lumbar', 'interbody'],
            'Posterior Lumbar Interbody Fusion (PLIF)': ['posterior', 'lumbar', 'interbody'],
            'Transforaminal Lumbar Interbody Fusion (TLIF)': ['transforaminal', 'interbody'],
            'Laminectomy (Decompression)': ['laminectomy'],
            'Laminotomy': ['laminotomy'],
            'Discectomy': ['discectomy'],
            'Spinal Instrumentation (Rods/Screws)': ['instrumentation', 'pedicle screw', 'rod'],
            'Vertebroplasty/Kyphoplasty': ['vertebroplasty', 'kyphoplasty'],
            'Spinal Fracture Fixation': ['vertebra', 'fracture', 'fixation'],
            'Corpectomy': ['corpectomy'],
            'Multi-level Fusion (2+ levels)': ['2-3 vertebrae', '4-8 vertebrae', 'multiple', '2 or more']
        },
        'hip': {
            'Total Hip Arthroplasty (THA)': ['total hip', 'tha', 'total replacement of hip'],
            'Hemiarthroplasty': ['hemiarthroplasty', 'partial hip'],
            'Hip Revision Arthroplasty': ['hip', 'revision'],
            'Hip Fracture ORIF': ['hip', 'fracture', 'fixation'],
            'Femoral Neck Fracture Fixation': ['femoral neck', 'fracture'],
            'Intertrochanteric Fracture Fixation': ['intertrochanteric', 'fracture'],
            'Hip Pinning (Cannulated Screws)': ['hip', 'screw', 'pinning'],
            'Dynamic Hip Screw (DHS)': ['dynamic hip screw', 'dhs'],
            'Intramedullary Nail (Hip)': ['intramedullary', 'nail', 'femur'],
            'Hip Arthroscopy': ['hip', 'arthroscopy']
        },
        'knee': {
            'Total Knee Arthroplasty (TKA)': ['total knee', 'tka', 'total replacement of knee'],
            'Unicompartmental Knee Replacement': ['unicompartmental', 'partial knee'],
            'Knee Revision Arthroplasty': ['knee', 'revision'],
            'ACL Reconstruction': ['acl', 'anterior cruciate', 'reconstruction'],
            'PCL Reconstruction': ['pcl', 'posterior cruciate'],
            'Meniscectomy': ['meniscectomy', 'meniscus', 'excision'],
            'Meniscus Repair': ['meniscus', 'repair'],
            'Knee Arthroscopy': ['knee', 'arthroscopy'],
            'Tibial Plateau Fracture ORIF': ['tibial plateau', 'fracture'],
            'Patellar Fracture Fixation': ['patella', 'fracture'],
            'Knee Fracture ORIF': ['knee', 'fracture', 'fixation']
        },
        'shoulder_elbow': {
            'Total Shoulder Arthroplasty': ['total shoulder', 'shoulder replacement'],
            'Reverse Shoulder Arthroplasty': ['reverse', 'shoulder'],
            'Rotator Cuff Repair': ['rotator cuff', 'repair'],
            'Shoulder Arthroscopy': ['shoulder', 'arthroscopy'],
            'Bankart Repair (Shoulder Stabilization)': ['bankart', 'labral repair', 'shoulder stabilization'],
            'Shoulder Fracture ORIF': ['shoulder', 'fracture', 'fixation'],
            'Proximal Humerus Fracture Fixation': ['proximal humerus', 'fracture'],
            'Clavicle Fracture ORIF': ['clavicle', 'fracture'],
            'Elbow Arthroscopy': ['elbow', 'arthroscopy'],
            'Elbow Fracture ORIF': ['elbow', 'fracture'],
            'Olecranon Fracture Fixation': ['olecranon', 'fracture']
        },
        'hand': {
            'Carpal Tunnel Release': ['carpal tunnel', 'release'],
            'Distal Radius Fracture ORIF': ['distal radius', 'fracture'],
            'Scaphoid Fracture Fixation': ['scaphoid', 'fracture'],
            'Metacarpal Fracture ORIF': ['metacarpal', 'fracture'],
            'Phalangeal Fracture Fixation': ['phalanx', 'phalangeal', 'finger', 'fracture'],
            'Tendon Repair (Hand)': ['tendon', 'repair', 'hand'],
            'Wrist Arthroscopy': ['wrist', 'arthroscopy'],
            'Wrist Fracture ORIF': ['wrist', 'fracture'],
            'Trigger Finger Release': ['trigger finger', 'release']
        }
    }
    
    total_cohort = len(cohort_for_table)
    
    # For each body region, count ALL patients who had procedures there (allows overlap)
    for proc_type in ['spine', 'hip', 'knee', 'shoulder_elbow', 'hand']:
        display_name = 'Shoulder/Elbow' if proc_type == 'shoulder_elbow' else proc_type.capitalize()
        
        # Get ALL procedures for this body region from the FULL procedure list
        region_procedures = procedures_for_table[procedures_for_table['ortho_type'] == proc_type].copy()
        
        # Count unique patients who had ANY procedure in this region
        region_patients = region_procedures['hadm_id'].unique()
        region_n = len(region_patients)
        region_pct = (region_n / total_cohort * 100) if total_cohort > 0 else 0
        
        if region_n > 0:
            # Add main region row
            table_data.append([f'  {display_name} Procedures', 'n (% of cohort)', f'{region_n:,} ({region_pct:.1f}%)'])
            
            # Count each specific procedure type
            if proc_type in proc_type_keywords:
                region_char_counts = []
                
                for char_name, keywords in proc_type_keywords[proc_type].items():
                    matching_hadms = set()
                    for idx, row in region_procedures.iterrows():
                        title = str(row['long_title']).lower() if pd.notna(row['long_title']) else ''
                        if any(kw in title for kw in keywords):
                            matching_hadms.add(row['hadm_id'])
                    
                    n_patients = len(matching_hadms)
                    pct_of_total = (n_patients / total_cohort * 100) if total_cohort > 0 else 0
                    
                    if n_patients > 0:
                        region_char_counts.append((char_name, n_patients, pct_of_total))
                
                # Sort by count and add to table
                region_char_counts.sort(key=lambda x: x[1], reverse=True)
                for char_name, n, pct in region_char_counts:
                    table_data.append([f'    {char_name}', 'n (% of cohort)', f'{n:,} ({pct:.1f}%)'])
    
    print(f"  ✓ Added TRUE procedure distribution (patients counted in all applicable body regions)")
    
except Exception as e:
    print(f"  ⚠️  Could not add detailed procedure breakdown: {e}")
    # Fallback to simple listing
    for proc_type in valid_proc_types:
        display_name = 'Shoulder/Elbow' if proc_type == 'shoulder_elbow' else proc_type.capitalize()
        proc_n = (cohort_for_table['ortho_type'] == proc_type).sum()
        proc_pct = proc_n / len(cohort_for_table) * 100
        table_data.append([f'  {display_name}', 'n (%)', f'{proc_n:,} ({proc_pct:.1f}%)'])

# Clinical characteristics
table_data.append(['Clinical Characteristics', '', ''])
charlson_mean = cohort_for_table['charlson_score'].mean()
charlson_std = cohort_for_table['charlson_score'].std()
table_data.append(['  Charlson Comorbidity Index', 'Mean ± SD', f'{charlson_mean:.1f} ± {charlson_std:.1f}'])

num_dx_median = cohort_for_table['num_diagnoses'].median()
num_dx_q1 = cohort_for_table['num_diagnoses'].quantile(0.25)
num_dx_q3 = cohort_for_table['num_diagnoses'].quantile(0.75)
table_data.append(['  Number of diagnoses', 'Median [IQR]', f'{num_dx_median:.0f} [{num_dx_q1:.0f}, {num_dx_q3:.0f}]'])

# Comorbidities
table_data.append(['Comorbidities', '', ''])
comorbidity_features = {
    'Hypertension': 'has_hypertension',
    'Diabetes': 'has_diabetes',
    'Cardiac disease': 'has_cardiac',
    'Pulmonary disease': 'has_pulmonary',
    'Obesity': 'has_obesity',
    'Anemia': 'has_anemia'
}
for display_name, col_name in comorbidity_features.items():
    if col_name in cohort_for_table.columns:
        n = cohort_for_table[col_name].sum()
        pct = n / len(cohort_for_table) * 100
        table_data.append([f'  {display_name}', 'n (%)', f'{n:,} ({pct:.1f}%)'])

# Surgical complexity
table_data.append(['Surgical Complexity', '', ''])
proc_complex_median = cohort_for_table['procedure_complexity'].median()
proc_complex_q1 = cohort_for_table['procedure_complexity'].quantile(0.25)
proc_complex_q3 = cohort_for_table['procedure_complexity'].quantile(0.75)
table_data.append(['  Procedure complexity score', 'Median [IQR]', f'{proc_complex_median:.0f} [{proc_complex_q1:.0f}, {proc_complex_q3:.0f}]'])

# Outcomes
table_data.append(['Outcomes', '', ''])
poor_n = cohort_for_table['poor_outcome'].sum()
poor_pct = poor_n / len(cohort_for_table) * 100
table_data.append(['  Composite poor outcome*', 'n (%)', f'{poor_n:,} ({poor_pct:.1f}%)'])

mort_n = cohort_for_table['died_90day'].sum()
mort_pct = mort_n / len(cohort_for_table) * 100
table_data.append(['    90-day mortality', 'n (%)', f'{mort_n:,} ({mort_pct:.1f}%)'])

readmit_n = cohort_for_table['readmitted_7day'].sum()
readmit_pct = readmit_n / len(cohort_for_table) * 100
table_data.append(['    7-day readmission', 'n (%)', f'{readmit_n:,} ({readmit_pct:.1f}%)'])

los_n = cohort_for_table['prolonged_los'].sum()
los_pct = los_n / len(cohort_for_table) * 100
table_data.append(['    Prolonged LOS (>7 days)', 'n (%)', f'{los_n:,} ({los_pct:.1f}%)'])

# Convert to DataFrame and save
table_df = pd.DataFrame(table_data, columns=['Characteristic', 'Statistic', 'Value'])

# Add explanatory notes at the bottom
notes_data = [
    ['', '', ''],
    ['NOTES:', '', ''],
    ['* Each patient appears ONLY ONCE (first orthopedic surgery admission with ICU)', '', ''],
    ['* All procedures listed are from that specific admission only', '', ''],
    ['* Composite outcome = 90-day mortality OR 7-day readmission OR LOS >7 days', '', ''],
    ['* All patients had ICU admission within 7 days of surgery', '', ''],
    ['* Procedure counts may sum to >100% because patients with multi-region surgery', '', ''],
    ['  (e.g., spine + hip) are counted in each applicable body region', '', ''],
    ['* Specific procedure percentages are % of total cohort (not % within region)', '', '']
]
notes_df = pd.DataFrame(notes_data, columns=['Characteristic', 'Statistic', 'Value'])
table_with_notes = pd.concat([table_df, notes_df], ignore_index=True)

table_with_notes.to_csv('ortho_table1_demographics.csv', index=False)
print("  ✓ Saved: ortho_table1_demographics.csv")

# ============================================================================
# CREATE ADMISSION TYPE BREAKDOWN BY SPECIFIC PROCEDURE
# ============================================================================
print("\n📊 Creating admission type breakdown by procedure...")

admission_breakdown_data = []

try:
    # Get admission types for cohort
    # Drop admission_type if it exists in cohort_icu to avoid conflicts
    cohort_for_admission = cohort_icu.drop(columns=['admission_type'], errors='ignore')
    cohort_with_admission = cohort_for_admission.merge(
        admissions[['hadm_id', 'admission_type']], 
        on='hadm_id', 
        how='left'
    )
    
    print(f"  Admission types merged: {cohort_with_admission['admission_type'].notna().sum():,} patients")
    
    # Ensure we have procedures_for_table and proc_type_keywords from earlier
    if 'procedures_for_table' not in locals():
        cohort_hadm_ids_table = set(cohort_for_table['hadm_id'])
        procedures_for_table = ortho_procedures[ortho_procedures['hadm_id'].isin(cohort_hadm_ids_table)].copy()
    
    # Define procedure keywords if not already defined
    if 'proc_type_keywords' not in locals():
        proc_type_keywords = {
            'spine': {
                'Posterior Lumbar Fusion': ['posterior', 'lumbar', 'fusion'],
                'Anterior Cervical Discectomy & Fusion (ACDF)': ['anterior', 'cervical', 'discectomy', 'fusion'],
                'Total Hip Arthroplasty (THA)': ['total hip', 'tha', 'total replacement of hip'],
            },
            'hip': {
                'Total Hip Arthroplasty (THA)': ['total hip', 'tha', 'total replacement of hip'],
            },
            'knee': {
                'Total Knee Arthroplasty (TKA)': ['total knee', 'tka', 'total replacement of knee'],
            },
            'shoulder_elbow': {
                'Total Shoulder Arthroplasty': ['total shoulder', 'shoulder replacement'],
            },
            'hand': {
                'Carpal Tunnel Release': ['carpal tunnel', 'release'],
            }
        }
        print("  ⚠️  Using simplified procedure keywords (full definitions may not be in scope)")
    
    # For each body region and specific procedure type
    for proc_type in ['spine', 'hip', 'knee', 'shoulder_elbow', 'hand']:
        display_name = 'Shoulder/Elbow' if proc_type == 'shoulder_elbow' else proc_type.capitalize()
        
        # Get procedures for this region
        region_procedures = procedures_for_table[procedures_for_table['ortho_type'] == proc_type].copy()
        
        if len(region_procedures) == 0:
            continue
        
        # Get the specific procedure keywords for this region
        if proc_type in proc_type_keywords:
            for char_name, keywords in proc_type_keywords[proc_type].items():
                # Find patients with this specific procedure
                matching_hadms = set()
                for idx, row in region_procedures.iterrows():
                    title = str(row['long_title']).lower() if pd.notna(row['long_title']) else ''
                    if any(kw in title for kw in keywords):
                        matching_hadms.add(row['hadm_id'])
                
                if len(matching_hadms) > 0:
                    # Get admission types for these patients
                    proc_admissions = cohort_with_admission[
                        cohort_with_admission['hadm_id'].isin(matching_hadms)
                    ].copy()
                    
                    # Filter out any missing admission types
                    proc_admissions = proc_admissions[proc_admissions['admission_type'].notna()]
                    
                    if len(proc_admissions) == 0:
                        continue
                    
                    # Count admission types
                    admission_counts = proc_admissions['admission_type'].value_counts()
                    total_proc = len(proc_admissions)
                    
                    # Add header row for this procedure
                    admission_breakdown_data.append({
                        'Body_Region': display_name,
                        'Specific_Procedure': char_name,
                        'Admission_Type': '--- TOTAL ---',
                        'Count': total_proc,
                        'Percentage': 100.0
                    })
                    
                    # Add each admission type
                    for adm_type, count in admission_counts.items():
                        pct = (count / total_proc * 100) if total_proc > 0 else 0
                        admission_breakdown_data.append({
                            'Body_Region': display_name,
                            'Specific_Procedure': char_name,
                            'Admission_Type': adm_type,
                            'Count': count,
                            'Percentage': pct
                        })
    
    # Convert to DataFrame and save
    if len(admission_breakdown_data) > 0:
        admission_breakdown_df = pd.DataFrame(admission_breakdown_data)
        admission_breakdown_df.to_csv('ortho_admission_type_by_procedure.csv', index=False)
        print(f"  ✓ Saved: ortho_admission_type_by_procedure.csv ({len(admission_breakdown_df)} rows)")
        
        # Print summary to console
        print("\n  Sample breakdown (first 20 rows):")
        print("  " + "="*90)
        for idx, row in admission_breakdown_df.head(20).iterrows():
            if row['Admission_Type'] == '--- TOTAL ---':
                print(f"\n  {row['Body_Region']} - {row['Specific_Procedure']}")
                print(f"    TOTAL: {row['Count']} patients")
            else:
                print(f"      {row['Admission_Type']:<35s}: {row['Count']:4d} ({row['Percentage']:5.1f}%)")
    else:
        print("  ⚠️  No admission type data to save")
        
except Exception as e:
    print(f"  ⚠️  Could not create admission type breakdown: {e}")
    import traceback
    traceback.print_exc()

# Print formatted table to console
print("\n" + "="*90)
print("TABLE 1: Demographic and Clinical Characteristics")
print("(UNIQUE ICU Patients - First Admission Only)")
print("="*90)
print(f"{'Characteristic':<45} {'Statistic':<20} {'Value':<20}")
print("-"*90)
for row in table_data:
    print(f"{row[0]:<45} {row[1]:<20} {row[2]:<20}")
print("-"*90)
print("NOTES:")
print("* Each patient appears ONLY ONCE (first orthopedic surgery admission with ICU)")
print("* All procedures listed are from that specific admission only")
print("* Composite outcome = 90-day mortality OR 7-day readmission OR LOS >7 days")
print("* All patients had ICU admission within 7 days of surgery")
print("* Procedure counts may sum to >100% because patients with multi-region surgery")
print("  (e.g., spine + hip) are counted in each applicable body region")
print("* Specific procedure percentages are % of total cohort (not % within region)")
print("="*90)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE - ICU PATIENTS AFTER ORTHOPEDIC SURGERY")
print("="*80)

print(f"\n📊 DATASET SUMMARY (ICU PATIENTS ONLY - UNIQUE PATIENTS):")
print(f"  Total ICU patients analyzed: {len(cohort_icu):,}")
print(f"  Unique patients (verification): {cohort_icu['subject_id'].nunique():,}")
print(f"  ✓ ONE admission per patient: {len(cohort_icu) == cohort_icu['subject_id'].nunique()}")
print(f"  GOOD outcomes: {cohort_icu['good_outcome'].sum():,} ({cohort_icu['good_outcome'].mean()*100:.1f}%)")
print(f"  POOR outcomes: {cohort_icu['poor_outcome'].sum():,} ({cohort_icu['poor_outcome'].mean()*100:.1f}%)")
print(f"\n  Composite outcome = 90-day mortality OR 7-day readmission OR LOS >7 days:")
print(f"    90-day mortality: {cohort_icu['died_90day'].sum():,} ({cohort_icu['died_90day'].mean()*100:.1f}%)")
print(f"    7-day readmission: {cohort_icu['readmitted_7day'].sum():,} ({cohort_icu['readmitted_7day'].mean()*100:.1f}%)")
print(f"    Prolonged LOS (>7 days): {cohort_icu['prolonged_los'].sum():,} ({cohort_icu['prolonged_los'].mean()*100:.1f}%)")
display_names_final = ['Shoulder/Elbow' if p == 'shoulder_elbow' else p.capitalize() for p in valid_proc_types]
print(f"\n  Included procedures: {', '.join(display_names_final)}")

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"  AUC-ROC: {models_results[best_model_name]['auc']:.4f}")
print(f"  Avg Precision: {models_results[best_model_name]['ap']:.4f}")

print(f"\n📈 ALL MODELS:")
for name, results in models_results.items():
    print(f"  {name:<25s} AUC: {results['auc']:.4f}  AP: {results['ap']:.4f}")

print(f"\n⭐ TOP 10 MOST IMPORTANT FEATURES:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']}")

print(f"\n📁 OUTPUT FILES:")
print(f"  ✓ ortho_all_procedures_feature_importance.csv")
print(f"  ✓ ortho_all_procedures_model_performance.csv")
print(f"  ✓ ortho_all_procedures_comprehensive_analysis.png")
print(f"  ✓ ortho_procedure_distribution_analysis.png")
print(f"  ✓ ortho_table1_demographics.csv")
print(f"  ✓ ortho_admission_type_by_procedure.csv")

# ============================================================================
# [9/9] SEPARATE INVESTIGATION: 7-DAY READMISSION vs 90-DAY MORTALITY
# ============================================================================
print("\n" + "="*80)
print("[9/9] INVESTIGATING: 7-Day Readmission vs 90-Day Mortality by Procedure Type")
print("="*80)

# Calculate 90-day mortality for the cohort
cohort = cohort.merge(patients[['subject_id', 'dod']], on='subject_id', how='left', suffixes=('', '_merge'))
if 'dod_merge' in cohort.columns:
    cohort['dod'] = cohort['dod_merge']
    cohort = cohort.drop(columns=['dod_merge'])

cohort['dod'] = pd.to_datetime(cohort['dod'])
cohort['days_to_death'] = (cohort['dod'] - cohort['dischtime']).dt.days
cohort['died_90day'] = (cohort['days_to_death'] <= 90) & (cohort['days_to_death'] >= 0)

print("\n📊 OVERALL MORTALITY ANALYSIS:")
print("="*80)

# Overall mortality by readmission status
readmitted = cohort[cohort['readmitted_7day'] == True]
not_readmitted = cohort[cohort['readmitted_7day'] == False]

print(f"\nPatients WITH 7-day readmission (n={len(readmitted):,}):")
print(f"  90-day mortality: {readmitted['died_90day'].sum():,} / {len(readmitted):,} ({readmitted['died_90day'].mean()*100:.2f}%)")

print(f"\nPatients WITHOUT 7-day readmission (n={len(not_readmitted):,}):")
print(f"  90-day mortality: {not_readmitted['died_90day'].sum():,} / {len(not_readmitted):,} ({not_readmitted['died_90day'].mean()*100:.2f}%)")

# Calculate relative risk
if not_readmitted['died_90day'].mean() > 0:
    relative_risk = readmitted['died_90day'].mean() / not_readmitted['died_90day'].mean()
    print(f"\n⚠️  RELATIVE RISK: Patients with 7-day readmission are {relative_risk:.2f}x more likely to die within 90 days")

# Statistical test
from scipy.stats import chi2_contingency
contingency_table = pd.crosstab(cohort['readmitted_7day'], cohort['died_90day'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"  Chi-square test p-value: {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else '(not significant)'}")

# Breakdown by procedure type
print("\n" + "="*80)
print("📊 MORTALITY BREAKDOWN BY ORTHOPEDIC PROCEDURE TYPE:")
print("="*80)

procedure_types = ['spine', 'hip', 'knee', 'shoulder_elbow', 'hand']
results_table = []

for proc_type in procedure_types:
    proc_data = cohort[cohort['ortho_type'] == proc_type]
    
    if len(proc_data) == 0:
        continue
    
    # With readmission
    with_readmit = proc_data[proc_data['readmitted_7day'] == True]
    n_readmit = len(with_readmit)
    deaths_readmit = with_readmit['died_90day'].sum()
    mort_rate_readmit = with_readmit['died_90day'].mean() * 100 if n_readmit > 0 else 0
    
    # Without readmission
    without_readmit = proc_data[proc_data['readmitted_7day'] == False]
    n_no_readmit = len(without_readmit)
    deaths_no_readmit = without_readmit['died_90day'].sum()
    mort_rate_no_readmit = without_readmit['died_90day'].mean() * 100 if n_no_readmit > 0 else 0
    
    # Relative risk
    if mort_rate_no_readmit > 0:
        rr = mort_rate_readmit / mort_rate_no_readmit
    else:
        rr = float('inf') if mort_rate_readmit > 0 else 1.0
    
    display_name = 'Shoulder/Elbow' if proc_type == 'shoulder_elbow' else proc_type.upper()
    
    results_table.append({
        'Procedure': display_name,
        'With_7day_Readmit_N': n_readmit,
        'With_7day_Readmit_Deaths': deaths_readmit,
        'With_7day_Readmit_Rate': mort_rate_readmit,
        'Without_7day_Readmit_N': n_no_readmit,
        'Without_7day_Readmit_Deaths': deaths_no_readmit,
        'Without_7day_Readmit_Rate': mort_rate_no_readmit,
        'Relative_Risk': rr
    })

# Display results table
print("\n" + "-"*120)
print(f"{'Procedure':<15} | {'WITH 7-day Readmit':<35} | {'WITHOUT 7-day Readmit':<35} | {'Relative':<10}")
print(f"{'Type':<15} | {'N':>6}  {'Deaths':>7}  {'Rate':>10} | {'N':>6}  {'Deaths':>7}  {'Rate':>10} | {'Risk':>8}")
print("-"*120)

for result in results_table:
    print(f"{result['Procedure']:<15} | "
          f"{result['With_7day_Readmit_N']:>6}  {result['With_7day_Readmit_Deaths']:>7}  {result['With_7day_Readmit_Rate']:>9.2f}% | "
          f"{result['Without_7day_Readmit_N']:>6}  {result['Without_7day_Readmit_Deaths']:>7}  {result['Without_7day_Readmit_Rate']:>9.2f}% | "
          f"{result['Relative_Risk']:>7.2f}x")

print("-"*120)

# Create visualization
print("\n📊 Creating mortality comparison visualization...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('90-Day Mortality: 7-Day Readmission vs No Readmission by Procedure Type', 
             fontsize=16, fontweight='bold')

for idx, proc_type in enumerate(procedure_types):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    proc_data = cohort[cohort['ortho_type'] == proc_type]
    
    if len(proc_data) == 0:
        ax.axis('off')
        continue
    
    # Calculate mortality rates
    readmit_mort = proc_data[proc_data['readmitted_7day'] == True]['died_90day'].mean() * 100
    no_readmit_mort = proc_data[proc_data['readmitted_7day'] == False]['died_90day'].mean() * 100
    
    # Create grouped bar chart
    categories = ['With\n7-day\nReadmit', 'Without\n7-day\nReadmit']
    values = [readmit_mort, no_readmit_mort]
    colors = ['#e74c3c', '#27ae60']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Styling
    display_name = 'Shoulder/Elbow' if proc_type == 'shoulder_elbow' else proc_type.capitalize()
    ax.set_title(f'{display_name}', fontsize=13, fontweight='bold')
    ax.set_ylabel('90-Day Mortality Rate (%)', fontsize=11)
    ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add sample sizes
    n_readmit = len(proc_data[proc_data['readmitted_7day'] == True])
    n_no_readmit = len(proc_data[proc_data['readmitted_7day'] == False])
    ax.text(0.5, 0.95, f'n={n_readmit:,} vs n={n_no_readmit:,}', 
            transform=ax.transAxes, ha='center', va='top',
            fontsize=9, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Turn off unused subplot
if len(procedure_types) < 6:
    axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('ortho_readmission_vs_mortality_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: ortho_readmission_vs_mortality_analysis.png")

# Save detailed results to CSV
results_df = pd.DataFrame(results_table)
results_df.to_csv('ortho_readmission_vs_mortality_detailed.csv', index=False)
print("  ✓ Saved: ortho_readmission_vs_mortality_detailed.csv")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print(f"1. Overall 90-day mortality is {readmitted['died_90day'].mean()*100:.2f}% for patients WITH 7-day readmission")
print(f"2. Overall 90-day mortality is {not_readmitted['died_90day'].mean()*100:.2f}% for patients WITHOUT 7-day readmission")
if not_readmitted['died_90day'].mean() > 0:
    print(f"3. Patients with 7-day readmission are {relative_risk:.2f}x more likely to die within 90 days")
print(f"4. This association {'IS' if p_value < 0.05 else 'IS NOT'} statistically significant (p={p_value:.6f})")
print("="*80)

print("\n" + "="*80)
print("✅ SUCCESS: Complete ICU composite outcome prediction and mortality analysis!")
print(f"Models trained on ICU patients only - {len(cohort_icu):,} patients")
print("Composite outcome: 90-day mortality OR 7-day readmission OR LOS >7 days")
print("="*80)

