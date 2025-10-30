import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the data
filepath = '/Users/alanmach/Downloads/ChronicIllness.csv'
df = pd.read_csv(filepath)

print("="*80)
print("COMPREHENSIVE CROHN'S DISEASE EDA")
print("="*80)

print("\n" + "="*80)
print("1. INITIAL DATA OVERVIEW")
print("="*80)

print(f"\nTotal records in dataset: {len(df):,}")
print(f"Total unique users: {df['user_id'].nunique():,}")
print(f"Date range: {df['checkin_date'].min()} to {df['checkin_date'].max()}")

print("\n--- Column Information ---")
print(df.dtypes)

print("\n--- Missing Values ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))

print("\n--- Trackable Types Distribution (Full Dataset) ---")
print(df['trackable_type'].value_counts())

print("\n" + "="*80)
print("2. CROHN'S DISEASE SPECIFIC DATA")
print("="*80)

# Filter for Crohn's disease
crohns = df[(df['trackable_type'] == 'Condition') & 
            (df['trackable_name'] == "Crohn's disease")].copy()

print(f"\nTotal Crohn's disease check-ins: {len(crohns):,}")
print(f"Unique users with Crohn's disease: {crohns['user_id'].nunique():,}")

# Convert date
crohns['checkin_date'] = pd.to_datetime(crohns['checkin_date'])
crohns['year'] = crohns['checkin_date'].dt.year
crohns['month'] = crohns['checkin_date'].dt.month
crohns['day_of_week'] = crohns['checkin_date'].dt.day_name()
crohns['quarter'] = crohns['checkin_date'].dt.quarter

print(f"Date range for Crohn's: {crohns['checkin_date'].min()} to {crohns['checkin_date'].max()}")

print("\n" + "="*80)
print("3. CROHN'S DISEASE SEVERITY ANALYSIS")
print("="*80)

# Convert trackable_value to numeric
crohns['severity'] = pd.to_numeric(crohns['trackable_value'], errors='coerce')

print("\n--- Severity Distribution ---")
print(crohns['severity'].value_counts().sort_index())
print(f"\nMean severity: {crohns['severity'].mean():.2f}")
print(f"Median severity: {crohns['severity'].median():.2f}")
print(f"Std deviation: {crohns['severity'].std():.2f}")

severity_counts = crohns['severity'].value_counts().sort_index()
print("\n--- Severity Interpretation ---")
print("0 (Not active): {:,} ({:.1f}%)".format(
    severity_counts.get(0, 0), 
    (severity_counts.get(0, 0) / len(crohns) * 100)))
print("1 (Mild): {:,} ({:.1f}%)".format(
    severity_counts.get(1, 0), 
    (severity_counts.get(1, 0) / len(crohns) * 100)))
print("2 (Moderate): {:,} ({:.1f}%)".format(
    severity_counts.get(2, 0), 
    (severity_counts.get(2, 0) / len(crohns) * 100)))
print("3 (Severe): {:,} ({:.1f}%)".format(
    severity_counts.get(3, 0), 
    (severity_counts.get(3, 0) / len(crohns) * 100)))
print("4 (Extremely active): {:,} ({:.1f}%)".format(
    severity_counts.get(4, 0), 
    (severity_counts.get(4, 0) / len(crohns) * 100)))

print("\n" + "="*80)
print("4. DEMOGRAPHIC ANALYSIS")
print("="*80)

print("\n--- Sex Distribution ---")
sex_counts = crohns['sex'].value_counts()
print(sex_counts)
print("\nWith percentages:")
for sex, count in sex_counts.items():
    print(f"{sex}: {count:,} ({count/len(crohns)*100:.1f}%)")

print("\n--- Country Distribution ---")
country_counts = crohns['country'].value_counts()
print(country_counts.head(10))

print("\n--- Age Distribution ---")
age_stats = crohns['age'].describe()
print(age_stats)
print(f"\nAge range: {crohns['age'].min():.0f} to {crohns['age'].max():.0f}")

# Age groups
crohns['age_group'] = pd.cut(crohns['age'], 
                              bins=[0, 18, 30, 40, 50, 60, 100],
                              labels=['0-18', '19-30', '31-40', '41-50', '51-60', '60+'])
print("\n--- Age Groups ---")
print(crohns['age_group'].value_counts().sort_index())

print("\n" + "="*80)
print("5. TEMPORAL PATTERNS")
print("="*80)

print("\n--- Check-ins by Year ---")
print(crohns['year'].value_counts().sort_index())

print("\n--- Check-ins by Month ---")
print(crohns['month'].value_counts().sort_index())

print("\n--- Check-ins by Day of Week ---")
print(crohns['day_of_week'].value_counts())

print("\n--- Average Severity by Year ---")
severity_by_year = crohns.groupby('year')['severity'].agg(['mean', 'count'])
print(severity_by_year)

print("\n--- Average Severity by Month ---")
severity_by_month = crohns.groupby('month')['severity'].agg(['mean', 'count'])
print(severity_by_month)

print("\n--- Average Severity by Day of Week ---")
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
severity_by_dow = crohns.groupby('day_of_week')['severity'].mean().reindex(day_order)
print(severity_by_dow)

print("\n" + "="*80)
print("6. USER-LEVEL ANALYSIS")
print("="*80)

# Check-ins per user
checkins_per_user = crohns.groupby('user_id').size()
print("\n--- Check-ins per User Statistics ---")
print(checkins_per_user.describe())

print(f"\nUsers with 1 check-in: {(checkins_per_user == 1).sum()}")
print(f"Users with 2-5 check-ins: {((checkins_per_user >= 2) & (checkins_per_user <= 5)).sum()}")
print(f"Users with 6-10 check-ins: {((checkins_per_user >= 6) & (checkins_per_user <= 10)).sum()}")
print(f"Users with >10 check-ins: {(checkins_per_user > 10).sum()}")

# Average severity per user
avg_severity_per_user = crohns.groupby('user_id')['severity'].mean()
print("\n--- Average Severity per User ---")
print(avg_severity_per_user.describe())

# Duration of tracking per user
user_duration = crohns.groupby('user_id')['checkin_date'].agg(['min', 'max'])
user_duration['days_tracked'] = (user_duration['max'] - user_duration['min']).dt.days
print("\n--- Tracking Duration per User (days) ---")
print(user_duration['days_tracked'].describe())

print("\n" + "="*80)
print("7. SEVERITY BY DEMOGRAPHICS")
print("="*80)

print("\n--- Average Severity by Sex ---")
severity_by_sex = crohns.groupby('sex')['severity'].agg(['mean', 'std', 'count'])
print(severity_by_sex)

print("\n--- Average Severity by Country (Top 10) ---")
severity_by_country = crohns.groupby('country')['severity'].agg(['mean', 'std', 'count'])
severity_by_country = severity_by_country.sort_values('count', ascending=False).head(10)
print(severity_by_country)

print("\n--- Average Severity by Age Group ---")
severity_by_age = crohns.groupby('age_group')['severity'].agg(['mean', 'std', 'count'])
print(severity_by_age)

# Correlation between age and severity
age_severity_corr = crohns[['age', 'severity']].corr().iloc[0, 1]
print(f"\nCorrelation between age and severity: {age_severity_corr:.3f}")

print("\n" + "="*80)
print("8. ASSOCIATED DATA FOR CROHN'S PATIENTS")
print("="*80)

# Get all user_ids with Crohn's
crohns_users = crohns['user_id'].unique()

# Filter full dataset for these users
crohns_user_data = df[df['user_id'].isin(crohns_users)].copy()

print(f"\nTotal records for Crohn's users: {len(crohns_user_data):,}")
print(f"Average records per Crohn's user: {len(crohns_user_data) / len(crohns_users):.1f}")

print("\n--- Trackable Types for Crohn's Patients ---")
trackable_dist = crohns_user_data['trackable_type'].value_counts()
print(trackable_dist)

# Most common symptoms
print("\n--- Top 20 Symptoms for Crohn's Patients ---")
symptoms = crohns_user_data[crohns_user_data['trackable_type'] == 'Symptom']
symptom_counts = symptoms['trackable_name'].value_counts().head(20)
print(symptom_counts)

# Most common treatments
print("\n--- Top 20 Treatments for Crohn's Patients ---")
treatments = crohns_user_data[crohns_user_data['trackable_type'] == 'Treatment']
treatment_counts = treatments['trackable_name'].value_counts().head(20)
print(treatment_counts)

# Most common tags
print("\n--- Top 20 Tags for Crohn's Patients ---")
tags = crohns_user_data[crohns_user_data['trackable_type'] == 'Tag']
tag_counts = tags['trackable_name'].value_counts().head(20)
print(tag_counts)

# Other conditions
print("\n--- Top 10 Other Conditions for Crohn's Patients ---")
other_conditions = crohns_user_data[
    (crohns_user_data['trackable_type'] == 'Condition') & 
    (crohns_user_data['trackable_name'] != "Crohn's disease")
]
condition_counts = other_conditions['trackable_name'].value_counts().head(10)
print(condition_counts)

print("\n" + "="*80)
print("9. DATA QUALITY ASSESSMENT")
print("="*80)

print("\n--- Crohn's Dataset Completeness ---")
print(f"Records with age: {crohns['age'].notna().sum():,} ({crohns['age'].notna().sum()/len(crohns)*100:.1f}%)")
print(f"Records with sex: {crohns['sex'].notna().sum():,} ({crohns['sex'].notna().sum()/len(crohns)*100:.1f}%)")
print(f"Records with country: {crohns['country'].notna().sum():,} ({crohns['country'].notna().sum()/len(crohns)*100:.1f}%)")
print(f"Records with severity: {crohns['severity'].notna().sum():,} ({crohns['severity'].notna().sum()/len(crohns)*100:.1f}%)")

print("\n--- Unusual Values ---")
print(f"Severity values outside 0-4 range: {((crohns['severity'] < 0) | (crohns['severity'] > 4)).sum()}")
print(f"Age values < 0 or > 120: {((crohns['age'] < 0) | (crohns['age'] > 120)).sum()}")

# Duplicate check-ins
crohns['date_only'] = crohns['checkin_date'].dt.date
duplicates = crohns.groupby(['user_id', 'date_only']).size()
print(f"\nUsers with multiple Crohn's check-ins on same day: {(duplicates > 1).sum()}")

print("\n" + "="*80)
print("10. SUMMARY STATISTICS")
print("="*80)

summary = {
    'Total Crohn\'s Check-ins': len(crohns),
    'Unique Users': crohns['user_id'].nunique(),
    'Date Range': f"{crohns['checkin_date'].min().date()} to {crohns['checkin_date'].max().date()}",
    'Average Severity': f"{crohns['severity'].mean():.2f}",
    'Most Common Severity': crohns['severity'].mode()[0] if len(crohns['severity'].mode()) > 0 else 'N/A',
    'Average Age': f"{crohns['age'].mean():.1f}",
    'Most Common Sex': crohns['sex'].mode()[0] if len(crohns['sex'].mode()) > 0 else 'N/A',
    'Most Common Country': crohns['country'].mode()[0] if len(crohns['country'].mode()) > 0 else 'N/A',
    'Avg Check-ins per User': f"{checkins_per_user.mean():.1f}",
    'Avg Tracking Duration (days)': f"{user_duration['days_tracked'].mean():.1f}"
}

print("\n--- Key Summary Metrics ---")
for key, value in summary.items():
    print(f"{key}: {value}")

print("\n" + "="*80)
print("EDA COMPLETE")
print("="*80)

# Save filtered Crohn's data
output_path = '/Users/alanmach/Downloads/crohns_disease_data.csv'
crohns.to_csv(output_path, index=False)
print(f"\nFiltered Crohn's disease data saved to: {output_path}")