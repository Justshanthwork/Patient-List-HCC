"""
HCC Patient Curation List — Real-World Evidence Study
=====================================================
Disease  : Hepatocellular Carcinoma (HCC), ICD-10 C22.0
Data     : Integra Connect PrecisionQ platform
Author   : prashanth.jain

Global Inclusion / Exclusion Criteria
--------------------------------------
INCLUDE:
  - Confirmed HCC diagnosis (ICD-10 C22.0)
  - Initiated 1L systemic treatment OR TACE procedure on/after Jul 1, 2022
  - Age ≥ 18 at HCC diagnosis

EXCLUDE:
  - Patients treated for another primary cancer during study period (other_primary = 1)
  - Patients with clinical trial participation during study period (procedure code S99)

Priority Groups
---------------
Priority 1 — Has structured LOT1 systemic treatment or TACE in LOT/procedure table
             starting on or after Jul 1, 2022
Priority 2 — C22.0 diagnosis on or after Jul 1, 2022, NO structured LOT on record.
             Included to assess whether TACE procedure occurred post-diagnosis
             (HCC Emerald 1 alignment)

Output
------
lot_output/HCC_curation_patient_list_YYYYMMDD.xlsx   (Priority 1 + 2, with priority column)
lot_output/df_lot_use_cat.xlsx                       (full LOT table for Priority 1 patients)
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import sys
import types
import datetime
from pathlib import Path

import pandas as pd

# comorbidipy references a removed pandas internal — patch before importing
import pandas.errors
_compat = types.ModuleType("pandas.core.common")
_compat.SettingWithCopyWarning = pd.errors.SettingWithCopyWarning
sys.modules["pandas.core.common"] = _compat
from comorbidipy import comorbidity  # noqa: E402

from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(r'C:\Users\prashanth.jain\Desktop\Projects\HCC')

PATHS = {
    'data'     : BASE / 'data',
    'output'   : BASE / 'output',
    'inter'    : BASE / 'intermediate',
    'interqa'  : BASE / 'intermediateqa',
    'knowledge': BASE / 'knowledge',
    'rules'    : BASE / 'rules',
    'demogr'   : BASE / 'output' / 'demogr',
    'outcomes' : BASE / 'output' / 'outcomes',
    'lot'      : BASE / 'output' / 'lot',
}

for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# ── Study parameters ───────────────────────────────────────────────────────────
STUDY_START = '2022-07-01'
STUDY_END   = '2026-01-31'
HCC_CODE    = 'C22.0'
MIN_AGE     = 18

# Categories with too few patients to analyse separately — collapsed to 'other'
RARE_CATEGORIES = {
    'CTLA4', 'PD-1/PD-L1_VEGF_IV_chemo', 'PD-1/PD-L1_CTLA4_TKI',
    'TKI_chemo', 'VEGF_IV', 'PD-1/PD-L1_TKI_chemo', 'other',
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_csv(directory: Path, keyword: str, date_cols: list = None) -> pd.DataFrame:
    """
    Find, header-normalise, and load a PrecisionQ CSV.

    Locates the first CSV in `directory` whose name contains `keyword`,
    lowercases its header row in-place (PrecisionQ exports UPPERCASE columns),
    then loads it into a DataFrame with optional date parsing.

    Raises FileNotFoundError if no matching file is found.
    """
    matches = [f for f in directory.iterdir() if keyword in f.name and f.suffix == '.csv']
    if not matches:
        raise FileNotFoundError(f"No CSV containing '{keyword}' found in {directory}")
    path = matches[0]

    # Lowercase header in-place (idempotent — skips if already lowercase)
    with open(path, 'r', newline='') as f:
        header = f.readline()
        body   = f.read()
    if header != header.lower():
        with open(path, 'w', newline='') as f:
            f.write(header.lower() + body)

    df = pd.read_csv(path, header=0, low_memory=False)
    for col in (date_cols or []):
        df[col] = pd.to_datetime(df[col])
    return df


def replace_biosimilars(regimen: str) -> str:
    """
    Collapse biosimilar suffixes to the reference molecule name.

    PrecisionQ appends a dash-suffix per biosimilar variant (e.g. bevacizumab-bvzr).
    This function strips those so all variants count together.

    'atezolizumab,bevacizumab-bvzr' -> 'atezolizumab,bevacizumab'
    'durvalumab,tremelimumab-actl'  -> 'durvalumab,tremelimumab'
    """
    drugs = {d.strip().split('-')[0] for d in regimen.split(',')}
    if any('trastuzumab' in d for d in drugs):
        return 'trastuzumab'
    return ','.join(sorted(drugs))


def categorize_regimen(regimen: str) -> str:
    """
    Map a biosimilar-standardised regimen to an AZ treatment category.

    Priority order: PD-1/PD-L1 > TKI > VEGF_IV > CTLA4 > chemo > other.
    See module docstring for full category descriptions.
    """
    PD1   = {'nivolumab', 'pembrolizumab', 'atezolizumab', 'durvalumab',
              'dostarlimab', 'dostarlimab-gxly', 'toripalimab'}
    CTLA4 = {'ipilimumab', 'tremelimumab', 'tremelimumab-actl'}
    TKI   = {'lenvatinib', 'sorafenib', 'cabozantinib', 'regorafenib',
              'selpercatinib', 'larotrectinib', 'entrectinib'}
    VEGF  = {'ramucirumab', 'bevacizumab'}
    CHEMO = {'fluorouracil', 'gemcitabine', 'capecitabine', 'oxaliplatin', 'cisplatin'}

    drugs = {d.strip() for d in regimen.split(',')}
    io    = bool(drugs & PD1)
    ctla4 = bool(drugs & CTLA4)
    tki   = bool(drugs & TKI)
    vegf  = bool(drugs & VEGF)
    chemo = bool(drugs & CHEMO)

    if io:
        if drugs <= PD1:           return 'PD-1/PD-L1_mono'
        if ctla4 and vegf and tki: return 'PD-1/PD-L1_CTLA4_VEGF_IV_TKI'
        if ctla4 and vegf:         return 'PD-1/PD-L1_CTLA4_VEGF_IV'
        if ctla4 and tki:          return 'PD-1/PD-L1_CTLA4_TKI'
        if ctla4:                  return 'PD-1/PD-L1_CTLA4'
        if vegf and tki:           return 'PD-1/PD-L1_VEGF_IV_TKI'
        if vegf:                   return 'PD-1/PD-L1_VEGF_IV_chemo' if chemo else 'PD-1/PD-L1_VEGF_IV'
        if tki:                    return 'PD-1/PD-L1_TKI_chemo'     if chemo else 'PD-1/PD-L1_TKI'
        if chemo:                  return 'PD-1/PD-L1_chemo'
        return 'PD-1/PD-L1_other'

    if tki:   return 'TKI_chemo'     if chemo else 'TKI'
    if vegf:  return 'VEGF_IV_chemo' if chemo else 'VEGF_IV'
    if ctla4: return 'CTLA4'
    if chemo: return 'chemo'
    return 'other'


# ── Pipeline ───────────────────────────────────────────────────────────────────

def load_exclusion_list(base_dir: Path) -> set:
    """
    Load the already-curated patient list and return their combined_div_mpi_ids as a set.

    Searches for an Excel file whose name contains 'HCC General Disease Curation List'
    in the base project directory. Patients in this file are excluded from the new list
    because they have already been sent for curation.
    """
    matches = [f for f in base_dir.iterdir()
               if 'HCC General Disease Curation List' in f.name and f.suffix == '.xlsx']
    if not matches:
        print("[warn] No 'HCC General Disease Curation List' file found — no pre-curated exclusions applied")
        return set()

    path = matches[0]
    df_excl = pd.read_excel(path, dtype=str)

    # Use combined_div_mpi_id as the primary exclusion key (division-specific)
    if 'combined_div_mpi_id' not in df_excl.columns:
        raise KeyError(f"'combined_div_mpi_id' column not found in {path.name}. "
                       f"Columns present: {df_excl.columns.tolist()}")

    excl_ids = set(df_excl['combined_div_mpi_id'].dropna())
    print(f"[excl] Pre-curated exclusion list loaded: {len(excl_ids):,} patients  ({path.name})")
    return excl_ids


def apply_global_exclusions(
    df_demogr: pd.DataFrame,
    df_disease_c220: pd.DataFrame,
    df_procedure: pd.DataFrame,
    already_curated_ids: set,
) -> pd.DataFrame:
    """
    Apply global inclusion/exclusion criteria and return the eligible demographics table.

    Criteria applied:
    - Must have C22.0 diagnosis in disease table
    - Age ≥ 18 at HCC diagnosis
    - No clinical trial participation during study period (no S99 procedure code)
    - Not already in the pre-curated exclusion list (HCC General Disease Curation List)
    """
    # Patients with a clinical trial procedure code (S99) on/after study start
    trial_ids = set(
        df_procedure.loc[
            df_procedure['procedure_source_code'].str.contains('S99', na=False) &
            (df_procedure['date_event'] >= STUDY_START),
            'mpi_id'
        ]
    )

    mask = (
        df_demogr['mpi_id'].isin(df_disease_c220['mpi_id']) &          # C22.0 confirmed
        (df_demogr['age_dx'] >= MIN_AGE) &                              # age ≥ 18
        (~df_demogr['mpi_id'].isin(trial_ids)) &                        # no clinical trial
        (~df_demogr['combined_div_mpi_id'].isin(already_curated_ids))   # not already curated
    )

    return df_demogr[mask]


def build_priority1(
    df_lot: pd.DataFrame,
    df_procedure: pd.DataFrame,
    eligible_ids: set,
) -> tuple:
    """
    Identify Priority 1 patients and their LOT data.

    Priority 1 = patients who initiated 1L systemic treatment OR a TACE/TAE
    procedure on or after Jul 1, 2022.

    TACE is captured from two sources:
    - LOT table : regimen string contains 'tace' or 'tae'
    - Procedure table : procedure name contains 'tace' or 'tae'

    Returns
    -------
    p1_ids    : set of mpi_ids qualifying as Priority 1
    df_lot_p1 : all LOT rows for Priority 1 patients (LOT1 through LOT-n)
    """
    in_window = df_lot['start_date'].between(STUDY_START, STUDY_END)
    in_cohort = df_lot['mpi_id'].isin(eligible_ids)
    is_lot1   = df_lot['no_div_lot'] == 1

    # 1a. Systemic LOT1 patients (non-TACE regimens in LOT table)
    systemic_ids = set(df_lot.loc[
        is_lot1 & in_window & in_cohort &
        ~df_lot['regimen'].str.lower().str.contains('tace|tae', na=False),
        'mpi_id'
    ])

    # 1b. TACE captured as LOT1 regimen (e.g. 'doxorubicin,tace', 'tace')
    tace_lot_ids = set(df_lot.loc[
        is_lot1 & in_window & in_cohort &
        df_lot['regimen'].str.lower().str.contains('tace|tae', na=False),
        'mpi_id'
    ])

    # 1c. TACE captured in procedure table (not in LOT at all)
    tace_proc_ids = set(df_procedure.loc[
        df_procedure['procedure'].str.lower().str.contains('tace|tae', na=False) &
        (df_procedure['date_event'] >= STUDY_START) &
        (df_procedure['date_event'] <= STUDY_END) &
        df_procedure['mpi_id'].isin(eligible_ids),
        'mpi_id'
    ])

    p1_ids = systemic_ids | tace_lot_ids | tace_proc_ids

    # Pull all LOT rows (LOT1–n) for Priority 1 patients
    df_lot_p1 = df_lot[df_lot['mpi_id'].isin(p1_ids)].copy()

    # Standardise and categorise regimens
    df_lot_p1['regimen_biosimilar'] = df_lot_p1['regimen'].apply(replace_biosimilars)
    df_lot_p1['treatment_category'] = df_lot_p1['regimen_biosimilar'].apply(categorize_regimen)
    df_lot_p1['treatment_category'] = df_lot_p1['treatment_category'].where(
        ~df_lot_p1['treatment_category'].isin(RARE_CATEGORIES), other='other'
    )

    return p1_ids, df_lot_p1


def build_priority2(
    df_demogr: pd.DataFrame,
    df_disease_c220: pd.DataFrame,
    df_lot: pd.DataFrame,
    eligible_ids: set,
    p1_ids: set,
) -> pd.DataFrame:
    """
    Identify Priority 2 patients.

    Priority 2 = eligible patients with C22.0 diagnosis on/after Jul 1, 2022
    who have NO structured LOT on record and are not already in Priority 1.
    Included to assess whether TACE occurred post-diagnosis (HCC Emerald 1 alignment).

    Returns
    -------
    df_p2 : demographics rows for Priority 2 patients
    """
    # Patients diagnosed on/after study start
    diagnosed_in_window = set(
        df_disease_c220.loc[df_disease_c220['diag_date'] >= STUDY_START, 'mpi_id']
    )

    # Patients with ANY LOT record (to exclude from P2)
    has_lot = set(df_lot['mpi_id'])

    p2_ids = (eligible_ids & diagnosed_in_window) - has_lot - p1_ids

    df_p2 = df_demogr[df_demogr['mpi_id'].isin(p2_ids)].copy()
    return df_p2


def build_curation_list(
    df_lot_p1: pd.DataFrame,
    df_p2_demogr: pd.DataFrame,
    df_disease_c220: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine Priority 1 and Priority 2 into a single curation patient list.

    Priority 1 rows come from the LOT1-only view (one row per patient).
    Priority 2 rows are built from demographics + diagnosis date (no LOT data).

    Returns
    -------
    pd.DataFrame with columns:
        mpi_id, division_mask, combined_div_mpi_id, priority,
        hcc_diagnosis_date, regimen, treatment_category,
        lot1_start_date, lot1_end_date, lot1_duration_months, metastatic
    """
    diag_lookup = df_disease_c220.set_index('combined_div_mpi_id')['diag_date']

    # ── Priority 1 ─────────────────────────────────────────────────────────────
    df_lot1 = (
        df_lot_p1[df_lot_p1['no_div_lot'] == 1]
        .merge(df_disease_c220[['combined_div_mpi_id', 'diag_date']],
               on='combined_div_mpi_id', how='left')
        [[
            'mpi_id', 'division_mask', 'combined_div_mpi_id',
            'diag_date', 'regimen_biosimilar', 'treatment_category',
            'start_date', 'end_date', 'duration_months', 'metastatic',
        ]]
        .rename(columns={
            'diag_date':          'hcc_diagnosis_date',
            'start_date':         'lot1_start_date',
            'end_date':           'lot1_end_date',
            'duration_months':    'lot1_duration_months',
            'regimen_biosimilar': 'regimen',
        })
        .assign(priority=1)
    )

    # ── Priority 2 ─────────────────────────────────────────────────────────────
    df_p2 = (
        df_p2_demogr[['mpi_id', 'division_mask', 'combined_div_mpi_id']]
        .assign(
            hcc_diagnosis_date  = lambda d: d['combined_div_mpi_id'].map(diag_lookup),
            regimen             = None,
            treatment_category  = None,
            lot1_start_date     = None,
            lot1_end_date       = None,
            lot1_duration_months= None,
            metastatic          = None,
            priority            = 2,
        )
    )

    col_order = [
        'division_mask',
        'combined_div_mpi_id',
        'priority',
        'mpi_id',
        'hcc_diagnosis_date',
        'regimen',
        'treatment_category',
        'lot1_start_date',
        'lot1_end_date',
        'lot1_duration_months',
        'metastatic',
    ]

    curation = (
        pd.concat([df_lot1, df_p2], ignore_index=True)
        .sort_values(['priority', 'mpi_id'])
        .reset_index(drop=True)
        [col_order]
    )
    return curation


def main():
    # ── Load ───────────────────────────────────────────────────────────────────
    data = PATHS['data']
    df_demogr    = load_csv(data, 'DEMOGRAPHICS')
    df_lot       = load_csv(data, 'LOT_',          date_cols=['start_date', 'end_date'])
    df_disease   = load_csv(data, 'DISEASE_',      date_cols=['diag_date'])
    df_procedure = load_csv(data, 'PROCEDURE',      date_cols=['date_event'])
    # Reserved for future analysis sections (comorbidity scoring, visit sequencing):
    # df_comorb         = load_csv(data, 'COMORB',        date_cols=['cond_st_date', 'cond_end_date'])
    # df_visit          = load_csv(data, 'VISIT',          date_cols=['visit_date'])
    # df_diseasehistory = load_csv(data, 'DISEASEHISTORY', date_cols=['diag_date'])

    # ── Base cohort (C22.0) ────────────────────────────────────────────────────
    df_disease_c220 = df_disease[df_disease['cancer_code'] == HCC_CODE]
    print(f"[base] C22.0 in disease table: {df_disease_c220['mpi_id'].nunique():,}")

    # ── Load pre-curated exclusion list ────────────────────────────────────────
    already_curated_ids = load_exclusion_list(BASE)

    # ── Global exclusions ──────────────────────────────────────────────────────
    df_eligible = apply_global_exclusions(df_demogr, df_disease_c220, df_procedure, already_curated_ids)
    eligible_ids = set(df_eligible['mpi_id'])
    print(f"[excl] After age/trial/pre-curated exclusions: {len(eligible_ids):,}")

    # ── Priority 1 ─────────────────────────────────────────────────────────────
    p1_ids, df_lot_p1 = build_priority1(df_lot, df_procedure, eligible_ids)
    print(f"[P1]   Priority 1 patients (LOT1 systemic or TACE): {len(p1_ids):,}")

    # ── Priority 2 ─────────────────────────────────────────────────────────────
    df_p2 = build_priority2(df_demogr, df_disease_c220, df_lot, eligible_ids, p1_ids)
    print(f"[P2]   Priority 2 patients (no LOT, Dx ≥ {STUDY_START}): {len(df_p2):,}")

    # ── Save LOT table for Priority 1 (full LOT1-n) ───────────────────────────
    df_lot_p1.to_excel(PATHS['lot'] / 'df_lot_use_cat.xlsx', index=False)

    # ── Build and save curation list ───────────────────────────────────────────
    df_curation = build_curation_list(df_lot_p1, df_p2, df_disease_c220)
    today    = datetime.date.today().strftime('%Y%m%d')
    out_path = PATHS['lot'] / f'HCC_curation_patient_list_{today}.xlsx'
    p1 = df_curation[df_curation['priority'] == 1]
    p2 = df_curation[df_curation['priority'] == 2]

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        p1.to_excel(writer, sheet_name='Priority 1', index=False)
        p2.to_excel(writer, sheet_name='Priority 2', index=False)

    print(f"\n[✓] Curation list saved → {out_path.name}")
    print(f"    Priority 1 : {len(p1):,} patients  (sheet: 'Priority 1')")
    print(f"    Priority 2 : {len(p2):,} patients  (sheet: 'Priority 2')")
    print(f"    Total      : {len(df_curation):,} patients")

    return df_lot_p1, df_p2, df_curation


if __name__ == '__main__':
    df_lot_p1, df_p2, df_curation = main()
