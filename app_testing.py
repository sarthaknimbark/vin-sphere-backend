import pandas as pd
from pathlib import Path
import glob
import re
from collections import defaultdict

project_root = Path(__file__).resolve().parent
INPUT_CSV = project_root / 'lookup_clean_translated.csv'
DEP_DIR   = project_root / 'Valuation' / 'data' / 'drivearabia_car_depreciation_valuation' / 'depreciated_2'

# ── Load dep CSVs ─────────────────────────────────────────────────────────────
csv_files = [f for f in glob.glob(str(DEP_DIR / '*_dep.csv'))]
valid_dfs = []
for f in csv_files:
    try:
        df = pd.read_csv(f)
        if not df.empty:
            valid_dfs.append(df)
    except Exception:
        pass

all_dep = pd.concat(valid_dfs, ignore_index=True) if valid_dfs else pd.DataFrame(columns=['model_slug', 'year'])
all_dep['model_slug'] = all_dep['model_slug'].astype(str).str.strip().str.lower()
all_dep['year'] = pd.to_numeric(all_dep['year'], errors='coerce')

# Global set for exact full-slug checks
dep_set = set(zip(all_dep['model_slug'], all_dep['year']))

# ── Pre-indexing for massive performance gains ────────────────────────────────
# Group slugs by year into native Python collections to avoid slow pandas row filtering inside loops
dep_by_year = defaultdict(list)
for slug, yr in zip(all_dep['model_slug'], all_dep['year']):
    if pd.notna(yr):
        dep_by_year[int(yr)].append(slug)

# Pre-compute stripped slugs per year for step 3 (normalization)
dep_stripped_by_year = {
    yr: [re.sub(r'[\s\-]+', '', s) for s in slugs]
    for yr, slugs in dep_by_year.items()
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def to_slug(text: str) -> str:
    """Lowercase + spaces → hyphens (same as make_slug/model_slug in app.py)."""
    return re.sub(r'\s+', '-', str(text).strip().lower())

def strip_slug(text: str) -> str:
    """Remove ALL hyphens and spaces — used for normalised comparison."""
    return re.sub(r'[\s\-]+', '', str(text).strip().lower())

# Noise suffixes that appear in lookup CSV but not in dep slugs
_NOISE = re.compile(
    r'\b(\d+\.\d+[lL]?|\d+[lL]\b|4x4|4wd|awd|rwd|fwd|'
    r'at\b|mt\b|cvt\b|turbo|diesel|petrol|hybrid|phev|ev|'
    r'series|edition|plus|pro|max|ultra|sport\+?|type-?[a-z]|'
    r'v4|v6|v8|v10|v12)\b',
    re.I)

def clean_model(model: str) -> str:
    """Strip spec suffixes that dep CSVs typically don't include."""
    return _NOISE.sub('', model).strip(' -')

# ── Core coverage check — mirrors app.py find_matches_in_df ──────────────────
def is_covered(make: str, model: str, year_val) -> bool:
    if year_val is None or year_val not in dep_by_year:
        return False

    make_s = to_slug(make)
    year_slugs = dep_by_year[year_val]
    year_stripped_slugs = dep_stripped_by_year[year_val]

    # FIX: restrict candidate slugs to THIS make only. Previously step 2/3/C
    # scanned every brand's slugs for the year, so short model names like
    # "x5" falsely matched unrelated brands (lexus-lx570, infiniti-qx50,
    # mg-rx5, changan-x5...) and made real coverage gaps invisible.
    if make_s:
        make_year_slugs = [s for s in year_slugs if s.startswith(f"{make_s}-")]
        make_year_stripped_slugs = [
            re.sub(r'[\s\-]+', '', s) for s in make_year_slugs
        ]
    else:
        make_year_slugs = year_slugs
        make_year_stripped_slugs = year_stripped_slugs

    # Helper: check ONE model variant
    def check_model_str(m: str) -> bool:
        m_slug  = to_slug(m)          # e.g. "land-cruiser-200"
        m_norm  = strip_slug(m)       # e.g. "landcruiser200"
        full    = f"{make_s}-{m_slug}" if make_s else m_slug

        # 1. Exact full slug check
        if (full, year_val) in dep_set:
            return True

        # 2. Dep slug contains full model slug (same make only)
        if any(m_slug in s for s in make_year_slugs):
            return True

        # 3. Normalised (strips hyphens) — catches "c-200" vs "c200" (same make only)
        if len(m_norm) >= 3:
            if any(m_norm in s for s in make_year_stripped_slugs):
                return True

        return False

    # A) Try original model
    if check_model_str(model):
        return True

    # B) Try model with noise stripped ("SPORTAGE 4X4" → "SPORTAGE")
    cleaned = clean_model(model)
    if cleaned and cleaned.upper() != model.upper():
        if check_model_str(cleaned):
            return True

    # C) Progressive prefix — remove last word one at a time (same make only)
    words = to_slug(model).split('-')
    for n in range(len(words) - 1, 0, -1):
        prefix_model = '-'.join(words[:n])
        if len(prefix_model) < 3:
            continue
        full_prefix = f"{make_s}-{prefix_model}" if make_s else prefix_model
        if any(full_prefix in s for s in make_year_slugs):
            return True

    return False

# ── Main scan ─────────────────────────────────────────────────────────────────
print("Reading lookup CSV...")
lookup = pd.read_csv(INPUT_CSV)
missing_with_chassis    = set()
missing_without_chassis = set()

print(f"Processing {len(lookup)} rows...")
for i, row in enumerate(lookup.itertuples(index=False)):
    make  = str(getattr(row, 'make',  '')).strip().upper()
    model = str(getattr(row, 'model', '')).strip().upper()
    trim  = str(getattr(row, 'trim',  '')).strip().upper()
    chassis = str(getattr(row, 'chassisNumber', '')) \
              if hasattr(row, 'chassisNumber') else ''
    
    # Safely extract chassis string representation
    chassis = str(chassis).strip().upper() if pd.notna(chassis) else ''

    try:
        year_val = int(pd.to_numeric(getattr(row, 'year', None)))
    except (ValueError, TypeError):
        year_val = None

    if is_covered(make, model, year_val):
        continue

    display_year = year_val if year_val is not None else 'N/A'
    missing_with_chassis.add((make, model, trim, display_year, chassis))
    missing_without_chassis.add((make, model, trim, display_year))

    if (i + 1) % 5000 == 0:
        print(f'{i + 1}/{len(lookup)} rows evaluated.')

# ── Write output ──────────────────────────────────────────────────────────────
print("Writing reports to disk...")

with open('missing_full_records.txt', 'w', encoding='utf8') as f:
    f.write('=== ALL CARS MISSING DEPRECIATION VALUES (WITH CHASSIS) ===\n')
    f.write(f'{"MAKE":<20} | {"MODEL":<25} | {"TRIM":<20} | {"YEAR":<6} | CHASSIS\n')
    f.write('-' * 90 + '\n')
    for mk, md, tr, yr, ch in sorted(missing_with_chassis):
        f.write(f'{mk:<20} | {md:<25} | {tr:<20} | {yr:<6} | {ch}\n')

with open('missing_unique_make_model.txt', 'w', encoding='utf8') as f:
    f.write('=== UNIQUE MISSING DEPRECIATION VALUES (WITHOUT CHASSIS) ===\n')
    f.write(f'{"MAKE":<20} | {"MODEL":<25} | {"TRIM":<20} | {"YEAR":<6}\n')
    f.write('-' * 75 + '\n')
    for mk, md, tr, yr in sorted(missing_without_chassis):
        f.write(f'{mk:<20} | {md:<25} | {tr:<20} | {yr:<6}\n')

print(f'Done! Found {len(missing_without_chassis)} unique vehicle definitions missing.')