#!/usr/bin/env python3
"""
Setup Verification Script for MIMIC-IV Datathon Analysis

Run this script to verify your environment is set up correctly
before running the main analysis notebook.

Usage:
    python verify_setup.py
"""

import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.7+"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.7+)")
        return False

def check_packages():
    """Check if required packages are installed"""
    print("\nChecking required packages...")
    required_packages = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical operations',
        'matplotlib': 'Plotting',
        'seaborn': 'Statistical visualization',
        'jupyter': 'Interactive notebooks',
    }
    
    all_installed = True
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {package:15s} - {description}")
        except ImportError:
            print(f"  ✗ {package:15s} - MISSING ({description})")
            all_installed = False
    
    return all_installed

def check_data_files():
    """Check if MIMIC-IV data files are present"""
    print("\nChecking MIMIC-IV data files...")
    
    data_root = Path('physionet.org/files/mimiciv/3.1')
    hosp_path = data_root / 'hosp'
    icu_path = data_root / 'icu'
    
    if not data_root.exists():
        print(f"  ✗ Data directory not found: {data_root}")
        return False
    
    print(f"  ✓ Data directory found: {data_root}")
    
    # Check critical files
    critical_files = [
        ('hosp/patients.csv.gz', 'Patient demographics'),
        ('hosp/admissions.csv.gz', 'Hospital admissions'),
        ('hosp/procedures_icd.csv.gz', 'Surgical procedures'),
        ('hosp/diagnoses_icd.csv.gz', 'Diagnoses'),
        ('hosp/d_icd_procedures.csv.gz', 'Procedure dictionary'),
        ('icu/icustays.csv.gz', 'ICU stays'),
        ('icu/chartevents.csv.gz', 'Vital signs'),
        ('icu/inputevents.csv.gz', 'Medications & fluids'),
        ('icu/outputevents.csv.gz', 'Fluid outputs'),
    ]
    
    all_present = True
    for filename, description in critical_files:
        filepath = data_root / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename:40s} ({size_mb:>7.1f} MB) - {description}")
        else:
            print(f"  ✗ {filename:40s} - MISSING ({description})")
            all_present = False
    
    return all_present

def check_notebook():
    """Check if the main notebook exists"""
    print("\nChecking analysis notebook...")
    notebook_path = Path('01_explore_elective_spine_surgery.ipynb')
    
    if notebook_path.exists():
        print(f"  ✓ Notebook found: {notebook_path}")
        return True
    else:
        print(f"  ✗ Notebook not found: {notebook_path}")
        return False

def main():
    """Run all verification checks"""
    print("=" * 70)
    print("MIMIC-IV Datathon Setup Verification")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Required Packages", check_packages()),
        ("MIMIC-IV Data Files", check_data_files()),
        ("Analysis Notebook", check_notebook()),
    ]
    
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} - {check_name}")
    
    all_passed = all(passed for _, passed in checks)
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 SUCCESS! Your environment is ready.")
        print("\nNext steps:")
        print("  1. Run: jupyter notebook 01_explore_elective_spine_surgery.ipynb")
        print("  2. In Jupyter: Cell → Run All")
        print("  3. Wait for analysis to complete (~10-20 minutes)")
        return 0
    else:
        print("\n⚠️  SETUP INCOMPLETE - Please fix the issues above.")
        print("\nCommon fixes:")
        print("  • Missing packages: pip install -r requirements.txt")
        print("  • Missing data: Ensure MIMIC-IV CSVs are in physionet.org/files/mimiciv/3.1/")
        print("  • Missing notebook: Make sure you're in the correct directory")
        return 1

if __name__ == "__main__":
    sys.exit(main())

