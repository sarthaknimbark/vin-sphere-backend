import glob
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent
INPUT_CSV = project_root / 'lookup_clean_translated.csv'
DEP_DIR = project_root / 'Valuation' / 'data' / 'drivearabia_car_depreciation_valuation'
# DEP_DIR = project_root / 'Valuation' / 'data' / 'drivearabia_car_depreciation_valuation' / 'depreciated_2'

def normalize_trim(val):
    if pd.isna(val):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(val).upper())

# ── 1. PRE-INDEX DEPRECIATION DATA FOR MASSIVE SPEED ─────────────────────────
print("Pre-indexing depreciation lookup files into memory...")
csv_files = glob.glob(str(DEP_DIR / '*_dep_2.csv'))

# Schema: dep_index[filename][(model_slug, year_val)] = list of rows (dict format)
dep_index = defaultdict(lambda: defaultdict(list))

for f in csv_files:
    try:
        filename = Path(f).name
        df = pd.read_csv(f)
        if df.empty:
            continue
            
        # Standardize matching columns on load
        df["model_slug"] = df["model_slug"].astype(str).str.strip().str.lower()
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["trim_key"] = df["trim_name"].apply(normalize_trim)
        
        # Convert to dictionary records for lightning-fast hash lookups
        for row in df.to_dict(orient="records"):
            yr = row["year"]
            if pd.notna(yr):
                dep_index[filename][(row["model_slug"], int(yr))].append(row)
    except Exception as e:
        print(f"Skipping corrupt file {f}: {e}")

def get_fast_depreciated_values(result: dict) -> dict:
    """Uses hash-map lookups instead of slow Pandas filtering."""
    make = str(result.get("make", "")).strip().lower()
    model = str(result.get("model", "")).strip().lower().replace(" ", "-")
    
    if make and not model.startswith(f"{make}-"):
        model_slug = f"{make}-{model}"
    else:
        model_slug = model

    try:
        year_val = int(pd.to_numeric(result.get("year")))
    except Exception:
        return {}

    trim = str(result.get("trim", "")).strip().upper()
    trim_key = normalize_trim(trim)
    matched_values = {}

    # Query the pre-built index
    for filename, slug_map in dep_index.items():
        # Step 1: Exact model slug match
        candidate_rows = slug_map.get((model_slug, year_val), [])
        
        # Step 2: Fallback substring match (only if exact match yielded nothing)
        if not candidate_rows:
            candidate_rows = [
                row for (s, y), rows in slug_map.items() 
                if y == year_val and model in s
                for row in rows
            ]
            
        if not candidate_rows:
            continue

        # Step 3: Trim matching logic
        best_rows = []
        if trim_key:
            # Exact trim key match
            exact_rows = [r for r in candidate_rows if r["trim_key"] == trim_key]
            if exact_rows:
                best_rows = exact_rows
            else:
                # Substring trim match
                contains_rows = [r for r in candidate_rows if trim_key in r["trim_key"] or r["trim_key"] in trim_key]
                if contains_rows:
                    best_rows = contains_rows
                else:
                    # Fuzzy match fallback
                    max_score = 0.82
                    for r in candidate_rows:
                        if r["trim_key"]:
                            score = SequenceMatcher(None, trim_key, r["trim_key"]).ratio()
                            if score > max_score:
                                max_score = score
                                best_rows = [r]
                            elif score == max_score:
                                best_rows.append(r)

        if not best_rows:
            best_rows = candidate_rows

        # Extract median depreciated value from best matching rows
        valid_vals = [r["depreciated_value"] for r in best_rows if pd.notna(r.get("depreciated_value"))]
        if valid_vals:
            valid_vals.sort()
            mid = len(valid_vals) // 2
            median_val = (valid_vals[mid] + valid_vals[~mid]) / 2
            matched_values[filename] = float(median_val)
            
    return matched_values


# ── 2. PROCESS LOOKUP AND LOG VARIANCE ────────────────────────────────────────
print("Reading lookup CSV...")
lookup = pd.read_csv(INPUT_CSV)

variance_records_with_chassis = set()
variance_records_without_chassis = set()

print(f"Evaluating {len(lookup)} rows for price variances...")
for i, row in enumerate(lookup.itertuples(index=False)):
    make = str(getattr(row, 'make', '')).strip().upper()
    model = str(getattr(row, 'model', '')).strip().upper()
    trim = str(getattr(row, 'trim', '')).strip().upper()
    chassis = str(getattr(row, 'chassisNumber', '')) if hasattr(row, 'chassisNumber') else ''
    chassis = str(chassis).strip().upper() if pd.notna(chassis) else 'N/A'

    try:
        year_val = int(pd.to_numeric(getattr(row, 'year', None)))
        display_year = str(year_val)
    except (ValueError, TypeError):
        year_val = None
        display_year = 'N/A'

    try:
        lookup_price = float(pd.to_numeric(getattr(row, 'price', None)))
    except (ValueError, TypeError):
        continue

    if lookup_price <= 0 or pd.isna(lookup_price):
        continue

    vehicle_payload = {"make": make, "model": model, "trim": trim, "year": year_val}
    dep_values = get_fast_depreciated_values(vehicle_payload)

    if dep_values:
        for filename, dep_price in dep_values.items():
            display_filename = filename.rsplit('.', 1)[0]
            variance = (dep_price - lookup_price) / lookup_price

            if abs(variance) > 0.05:
                var_pct_str = f"{variance * 100:+.2f}%"
                variance_records_with_chassis.add(
                    (make, model, trim, display_year, chassis, f"{lookup_price:,.0f}", f"{dep_price:,.0f}", var_pct_str, display_filename)
                )
                variance_records_without_chassis.add(
                    (make, model, trim, display_year, f"{lookup_price:,.0f}", f"{dep_price:,.0f}", var_pct_str, display_filename)
                )

    if (i + 1) % 1000 == 0:
        print(f'{i + 1}/{len(lookup)} rows processed.')

# ── 3. WRITE LOGS ─────────────────────────────────────────────────────────────
print("Writing variance reports...")
Path('Data').mkdir(parents=True, exist_ok=True)

with open('variance_full_records.txt', 'w', encoding='utf8') as f:
    f.write('=== CARS WITH PRICE VARIANCE GAP > 5% (WITH CHASSIS) ===\n')
    f.write(f'{"MAKE":<15} | {"MODEL":<20} | {"TRIM":<15} | {"YEAR":<5} | {"LOOKUP":<10} | {"DEP PRICE":<10} | {"GAP":<9} | {"SRC_FILE":<20} | CHASSIS\n')
    f.write('-' * 135 + '\n')
    for mk, md, tr, yr, ch, lp, dp, vp, fn in sorted(variance_records_with_chassis):
        f.write(f'{mk:<15} | {md:<20} | {tr:<15} | {yr:<5} | {lp:<10} | {dp:<10} | {vp:<9} | {fn:<20} | {ch}\n')

with open('variance_unique_make_model.txt', 'w', encoding='utf8') as f:
    f.write('=== UNIQUE CAR DEFINITIONS WITH PRICE VARIANCE GAP > 5% (WITHOUT CHASSIS) ===\n')
    f.write(f'{"MAKE":<15} | {"MODEL":<20} | {"TRIM":<15} | {"YEAR":<5} | {"LOOKUP":<10} | {"DEP PRICE":<10} | {"GAP":<9} | SRC_FILE\n')
    f.write('-' * 115 + '\n')
    for mk, md, tr, yr, lp, dp, vp, fn in sorted(variance_records_without_chassis):
        f.write(f'{mk:<15} | {md:<20} | {tr:<15} | {yr:<5} | {lp:<10} | {dp:<10} | {vp:<9} | {fn}\n')

print(f'Done! Logged {len(variance_records_without_chassis)} unique mismatches.')