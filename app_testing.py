import pandas as pd
from pathlib import Path
import glob

project_root = Path(__file__).resolve().parent
INPUT_CSV = project_root / 'lookup_data.csv'
# DEP_DIR = project_root / 'Valuation' / 'data' / 'drivearabia_car_depreciation_valuation' / 'depreciated'
DEP_DIR = project_root / 'Valuation' / 'data' / 'drivearabia_car_depreciation_valuation'

# load all depreciation files once
# all_dep = pd.concat([pd.read_csv(f) for f in glob.glob(str(DEP_DIR/'*_dep.csv'))], ignore_index=True)
all_dep = pd.read_csv(str(DEP_DIR / 'car_dep.csv'))
all_dep['model_slug'] = all_dep['model_slug'].astype(str).str.upper()

lookup = pd.read_csv(INPUT_CSV)

# Changed to sets to ensure unique entries for both combinations
missing_with_chassis = set()
missing_without_chassis = set()

for i, row in lookup.iterrows():
    make = str(row.get('make', '')).strip().upper()
    model = str(row.get('model', '')).strip().upper().replace(' ', '-')
    trim = str(row.get('trim', '')).strip().upper()  # Extracted trim
    year = pd.to_numeric(row.get('year'), errors='coerce')
    chassis = str(row.get('chassisNumber', '')).strip().upper()

    slug = f'{make.lower()}-{model.lower()}'

    matches = all_dep[(all_dep['model_slug'] == slug.upper())]

    if pd.notna(year):
        matches = matches[matches['year'] == int(year)]

    if matches.empty:
        clean_model = model.replace('-', ' ')
        display_year = int(year) if pd.notna(year) else 'N/A'
        
        # Add to both unique sets
        missing_with_chassis.add((make, clean_model, trim, display_year, chassis))
        missing_without_chassis.add((make, clean_model, trim, display_year))

    if (i + 1) % 1000 == 0:
        print(f'{i + 1}/{len(lookup)}')

# File 1: All fields (Make, Model, Trim, Year, Chassis)
with open('missing_full_records.txt', 'w', encoding='utf8') as f:
    f.write('=== ALL CARS MISSING DEPRECIATION VALUES (WITH CHASSIS) ===\n')
    f.write(f'{"MAKE":<20} | {"MODEL":<25} | {"TRIM":<20} | {"YEAR":<6} | {"CHASSIS"}\n')
    f.write('-' * 90 + '\n')
    for mk, md, tr, yr, ch in sorted(missing_with_chassis):
        f.write(f'{mk:<20} | {md:<25} | {tr:<20} | {yr:<6} | {ch}\n')

# File 2: All fields EXCEPT Chassis (Make, Model, Trim, Year)
with open('missing_unique_make_model.txt', 'w', encoding='utf8') as f:
    f.write('=== UNIQUE MISSING DEPRECIATION VALUES (WITHOUT CHASSIS) ===\n')
    f.write(f'{"MAKE":<20} | {"MODEL":<25} | {"TRIM":<20} | {"YEAR":<6}\n')
    f.write('-' * 75 + '\n')
    for mk, md, tr, yr in sorted(missing_without_chassis):
        f.write(f'{mk:<20} | {md:<25} | {tr:<20} | {yr:<6}\n')

print('Done')