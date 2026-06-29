import pandas as pd
from pathlib import Path
import glob
import re

project_root = Path(__file__).resolve().parent
INPUT_CSV = project_root / 'Data' / 'lookup_data_clean_translated.csv'
DEP_DIR = project_root / 'Valuation' / 'data' / 'drivearabia_car_depreciation_valuation' / 'depreciated_2'

# Clean up: Filter out empty or unreadable CSVs before concatenating to avoid the FutureWarning
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

dep_set = set(zip(all_dep['model_slug'], all_dep['year']))

lookup = pd.read_csv(INPUT_CSV)

missing_with_chassis = set()
missing_without_chassis = set()

print(f"Processing {len(lookup)} rows...")

for i, row in enumerate(lookup.itertuples(index=False)):
    make = str(getattr(row, 'make', '')).strip().lower()
    model = str(getattr(row, 'model', '')).strip().lower().replace(' ', '-')
    trim = str(getattr(row, 'trim', '')).strip().upper()
    chassis = str(getattr(row, 'chassisNumber', '')).strip().upper()
    
    try:
        year_val = int(pd.to_numeric(getattr(row, 'year', None)))
    except (ValueError, TypeError):
        year_val = None

    if make and not model.startswith(f"{make}-"):
        slug = f"{make}-{model}"
    else:
        slug = model

    if (slug, year_val) in dep_set:
        continue

    year_matches = all_dep[all_dep['year'] == year_val]
    
    # FIX: Added regex=False here to prevent regex evaluation errors
    substring_match = year_matches['model_slug'].str.contains(model, case=False, na=False, regex=False).any()

    if substring_match:
        continue

    display_year = year_val if year_val is not None else 'N/A'
    display_make = make.upper()
    display_model = model.replace('-', ' ').upper()

    missing_with_chassis.add((display_make, display_model, trim, display_year, chassis))
    missing_without_chassis.add((display_make, display_model, trim, display_year))

    if (i + 1) % 5000 == 0:
        print(f'{i + 1}/{len(lookup)} rows evaluated.')

# Write results out...
with open('missing_full_records.txt', 'w', encoding='utf8') as f:
    f.write('=== ALL CARS MISSING DEPRECIATION VALUES (WITH CHASSIS) ===\n')
    f.write(f'{"MAKE":<20} | {"MODEL":<25} | {"TRIM":<20} | {"YEAR":<6} | {"CHASSIS"}\n')
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