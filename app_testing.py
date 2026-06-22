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

unique_missing=set()
full_missing=[]

for i,row in lookup.iterrows():
    make = str(row.get('make','')).strip().upper()
    model = str(row.get('model','')).strip().upper().replace(' ','-')
    year = pd.to_numeric(row.get('year'), errors='coerce')
    chassis = str(row.get('chassisNumber','')).strip().upper()

    slug = f'{make.lower()}-{model.lower()}'

    matches = all_dep[(all_dep['model_slug']==slug.upper())]

    if pd.notna(year):
        matches = matches[matches['year']==int(year)]

    if matches.empty:
        unique_missing.add((make,model.replace('-',' ')))
        full_missing.append((make,model.replace('-',' '),chassis))

    if (i+1)%1000==0:
        print(f'{i+1}/{len(lookup)}')

with open('missing_unique_make_model_1.txt','w',encoding='utf8') as f:
    f.write('=== CARS MISSING DEPRECIATION VALUES ===\n')
    for m1,m2 in sorted(unique_missing):
        f.write(f'{m1:<20} | {m2}\n')

with open('missing_full_records_1.txt','w',encoding='utf8') as f:
    f.write('=== ALL CARS MISSING DEPRECIATION VALUES ===\n')
    for m1,m2,ch in full_missing:
        f.write(f'{m1:<20} | {m2:<30} | {ch}\n')

print('Done')