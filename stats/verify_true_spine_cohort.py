"""
Verify True Spine Surgery Cohort - Excluding Hip/Knee Replacements
"""

import pandas as pd
import numpy as np
import os

print("=" * 80)
print("VERIFYING TRUE SPINE SURGERY COHORT")
print("=" * 80)
print()

# Set paths
mimic_path = '/Users/brandoing/Desktop/gitlab/datathon_2025/physionet.org/files/mimiciv/3.1/hosp'

print("[1/3] Loading data...")
procedures = pd.read_csv(os.path.join(mimic_path, 'procedures_icd.csv.gz'), compression='gzip')
d_icd_procedures = pd.read_csv(os.path.join(mimic_path, 'd_icd_procedures.csv.gz'), compression='gzip')
admissions = pd.read_csv(os.path.join(mimic_path, 'admissions.csv.gz'), compression='gzip')

print("✓ Loaded data")
print()

print("[2/3] Analyzing procedure codes...")

# Merge to get descriptions
procs_with_desc = procedures.merge(
    d_icd_procedures[['icd_code', 'long_title']], 
    on='icd_code', 
    how='left'
)

# Current filter (INCORRECT)
spine_icd9 = ['03', '810', '813', '816']
spine_icd10_body = ['0R', '0S']  # <- 0S includes hip/knee!
spine_icd10_ops = ['B', 'G', 'N', 'T', 'Q', 'S', 'H', 'J', 'P', 'R', 'U', 'W']

def is_spine_surgery_OLD(code):
    if pd.isna(code):
        return False
    code = str(code)
    if len(code) <= 3:
        return code[:2] in spine_icd9 or code[:3] in spine_icd9
    else:
        return (code[:2] in spine_icd10_body and 
                code[2] in spine_icd10_ops)

procs_with_desc['is_spine_OLD'] = procs_with_desc['icd_code'].apply(is_spine_surgery_OLD)

# Get sample of what the OLD filter captures
print("SAMPLE OF PROCEDURES CAPTURED BY OLD FILTER:")
print("-" * 80)
old_spine = procs_with_desc[procs_with_desc['is_spine_OLD']]
sample_procs = old_spine.groupby('long_title').size().sort_values(ascending=False).head(20)
print(sample_procs)
print()

# Count hip/knee in old filter
hip_knee_count = old_spine[
    old_spine['long_title'].str.contains('Hip|Knee|hip|knee', case=False, na=False)
].shape[0]
total_old = old_spine.shape[0]
print(f"OLD FILTER STATS:")
print(f"  Total procedures: {total_old}")
print(f"  Hip/Knee procedures: {hip_knee_count} ({hip_knee_count/total_old*100:.1f}%)")
print()

# NEW CORRECTED FILTER
def is_spine_surgery_NEW(code, title):
    """
    True spine surgery filter:
    - ICD-9: Codes starting with 03 (spinal canal), 810, 813, 816
    - ICD-10: 0R (upper joints - cervical/thoracic) with spine operations
              0S (lower joints - lumbar/sacral) with spine operations
              BUT exclude hip/knee based on body part character and title
    """
    if pd.isna(code):
        return False
    
    code = str(code)
    
    # ICD-9
    if len(code) <= 3:
        return code[:2] in spine_icd9 or code[:3] in spine_icd9
    
    # ICD-10
    if len(code) >= 7:
        body_system = code[:2]  # 0R or 0S
        operation = code[2]      # B, G, N, etc.
        body_part = code[3]      # Specific vertebrae or joint
        
        # Must be upper or lower joints
        if body_system not in ['0R', '0S']:
            return False
        
        # Must be spine-related operation
        if operation not in spine_icd10_ops:
            return False
        
        # Exclude hip/knee based on body part character
        # Hip: body_part in ['9', 'A', 'B', 'E'] (hip joints)
        # Knee: body_part in ['C', 'D'] (knee joints)
        if body_part in ['9', 'A', 'B', 'C', 'D', 'E']:
            # Double-check with title to avoid false exclusions
            if pd.notna(title) and any(word in str(title).lower() for word in ['hip', 'knee', 'ankle', 'foot']):
                return False
        
        # Include if:
        # 0R (upper joints): cervical/thoracic vertebrae
        # 0S (lower joints): lumbar/sacral vertebrae (NOT hip/knee)
        
        # Verify with title as final check
        if pd.notna(title):
            title_lower = str(title).lower()
            # Must contain spine-related words
            spine_words = ['vertebra', 'vertebral', 'spine', 'spinal', 'cervical', 
                          'thoracic', 'lumbar', 'sacral', 'coccygeal', 'fusion']
            has_spine_word = any(word in title_lower for word in spine_words)
            
            # Must NOT contain non-spine joint words
            non_spine_words = ['hip', 'knee', 'ankle', 'foot', 'shoulder', 
                              'elbow', 'wrist', 'hand', 'finger']
            has_non_spine = any(word in title_lower for word in non_spine_words)
            
            return has_spine_word and not has_non_spine
        
        return True
    
    return False

procs_with_desc['is_spine_NEW'] = procs_with_desc.apply(
    lambda row: is_spine_surgery_NEW(row['icd_code'], row['long_title']), 
    axis=1
)

print("SAMPLE OF PROCEDURES CAPTURED BY NEW FILTER:")
print("-" * 80)
new_spine = procs_with_desc[procs_with_desc['is_spine_NEW']]
sample_procs_new = new_spine.groupby('long_title').size().sort_values(ascending=False).head(20)
print(sample_procs_new)
print()

# Count hip/knee in new filter
hip_knee_count_new = new_spine[
    new_spine['long_title'].str.contains('Hip|Knee|hip|knee', case=False, na=False)
].shape[0]
total_new = new_spine.shape[0]
print(f"NEW FILTER STATS:")
print(f"  Total procedures: {total_new}")
print(f"  Hip/Knee procedures: {hip_knee_count_new} ({hip_knee_count_new/total_new*100 if total_new > 0 else 0:.1f}%)")
print()

print("[3/3] Comparing cohort sizes...")
print()

# Get elective admissions
elective_admissions = admissions[admissions['admission_type'] == 'ELECTIVE']

# OLD cohort
old_hadm = set(procedures[procedures['icd_code'].apply(is_spine_surgery_OLD)]['hadm_id'])
old_elective = elective_admissions[elective_admissions['hadm_id'].isin(old_hadm)]

# NEW cohort
new_spine_hadm = set(procs_with_desc[procs_with_desc['is_spine_NEW']]['hadm_id'])
new_elective = elective_admissions[elective_admissions['hadm_id'].isin(new_spine_hadm)]

print("COHORT COMPARISON:")
print("-" * 80)
print(f"OLD Filter (including hip/knee):")
print(f"  Total elective admissions: {len(old_elective)}")
print()
print(f"NEW Filter (true spine only):")
print(f"  Total elective admissions: {len(new_elective)}")
print()
print(f"Difference: {len(old_elective) - len(new_elective)} admissions excluded")
print(f"  ({(len(old_elective) - len(new_elective))/len(old_elective)*100:.1f}% of original cohort)")
print()

# Show examples of excluded procedures
print("EXAMPLES OF EXCLUDED PROCEDURES (non-spine):")
print("-" * 80)
excluded = procs_with_desc[procs_with_desc['is_spine_OLD'] & ~procs_with_desc['is_spine_NEW']]
excluded_titles = excluded.groupby('long_title').size().sort_values(ascending=False).head(15)
for title, count in excluded_titles.items():
    print(f"  {count:4d} | {title}")
print()

print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()
print("✓ Use NEW filter to exclude hip/knee/other joint replacements")
print("✓ This ensures analysis focuses on true spine surgeries")
print("✓ Re-run all analyses with corrected cohort definition")
print()
print("The current published results likely include non-spine orthopedic surgeries.")
print("This must be corrected before publication!")

