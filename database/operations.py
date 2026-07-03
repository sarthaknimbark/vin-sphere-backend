import datetime
import pandas as pd
from .connection import get_db_connection

def lookup_vin_in_master_dataset(file_path: str, vin: str) -> dict:
    vin_upper = vin.upper().strip()
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chassisnumber, year, make, model, trim, bodytype, regionalspec, cylinders, color, origin, weightinkg, noofpassengers
                FROM master_data
                WHERE UPPER(TRIM(chassisnumber)) = %s
                LIMIT 1;
            """, (vin_upper,))
            row = cur.fetchone()
            if row:
                return {
                    "chassisNumber": row[0],
                    "year": row[1],
                    "make": row[2],
                    "model": row[3],
                    "trim": row[4],
                    "bodyType": row[5],
                    "regionalSpec": row[6],
                    "cylinders": row[7],
                    "color": row[8],
                    "origin": row[9],
                    "weightInKg": row[10],
                    "noOfPassengers": row[11]
                }
    except Exception as e:
        print(f"Error reading master dataset from DB: {e}")
    finally:
        if conn:
            conn.close()
    return None

def log_prediction(vin: str, length: int, source: str, result: dict):
    from zoneinfo import ZoneInfo
    try:
        mumbai_time = datetime.datetime.now(ZoneInfo("Asia/Kolkata"))
    except Exception:
        mumbai_time = datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30)
    
    mumbai_time = mumbai_time.replace(microsecond=0)

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO prediction_logs (
                    timestamps, length, source, chassisnumber, year, make, model, trim, bodytype, regionalspec, cylinders, color, origin, weightinkg, noofpassengers
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                mumbai_time,
                length,
                source,
                vin,
                result.get("year", ""),
                result.get("make", ""),
                result.get("model", ""),
                result.get("trim", ""),
                result.get("body_type", ""),
                result.get("regional_spec", ""),
                result.get("cylinders", ""),
                result.get("color", ""),
                result.get("origin", ""),
                result.get("weight", ""),
                result.get("no_of_passengers", "")
            ))
            conn.commit()
    except Exception as e:
        print(f"Error logging prediction in DB: {e}")
    finally:
        if conn:
            conn.close()

def safe_read_pending_predictions() -> pd.DataFrame:
    conn = None
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("""
            SELECT 
                chassisnumber AS vin, 
                year, 
                make, 
                model, 
                trim, 
                bodytype AS body_type, 
                regionalspec AS regional_spec, 
                cylinders, 
                color, 
                origin, 
                weightinkg AS weight, 
                noofpassengers AS no_of_passengers
            FROM pending_predictions;
        """, conn)
        df["timestamp"] = "UNKNOWN"
        return df
    except Exception as e:
        print(f"Error reading pending predictions from DB: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def save_pending_prediction(vin: str, length: int, result: dict):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # 1. If it already exists in master_data, don't add to pending queue
            cur.execute("SELECT 1 FROM master_data WHERE UPPER(TRIM(chassisnumber)) = %s LIMIT 1;", (vin.upper().strip(),))
            if cur.fetchone():
                return

            # 2. If it already exists in pending_predictions, don't duplicate
            cur.execute("SELECT 1 FROM pending_predictions WHERE UPPER(TRIM(chassisnumber)) = %s LIMIT 1;", (vin.upper().strip(),))
            if cur.fetchone():
                return
            
            cur.execute("""
                INSERT INTO pending_predictions (
                    chassisnumber, year, make, model, trim, bodytype, regionalspec, cylinders, color, origin, weightinkg, noofpassengers
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                vin.upper().strip(),
                result.get("year", ""),
                result.get("make", ""),
                result.get("model", ""),
                result.get("trim", ""),
                result.get("body_type", ""),
                result.get("regional_spec", ""),
                result.get("cylinders", ""),
                result.get("color", ""),
                result.get("origin", ""),
                result.get("weight", ""),
                result.get("no_of_passengers", "")
            ))
            conn.commit()
    except Exception as e:
        print(f"Error saving pending prediction in DB: {e}")
    finally:
        if conn:
            conn.close()

def approve_pending_prediction(vin: str, updated_attrs: dict) -> bool:
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            vin_upper = vin.upper().strip()
            
            # 1. Insert/update in master_data
            cur.execute("""
                INSERT INTO master_data (
                    chassisnumber, year, make, model, trim, bodytype, regionalspec, cylinders, color, origin, weightinkg, noofpassengers
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chassisnumber) DO UPDATE SET
                    year = EXCLUDED.year,
                    make = EXCLUDED.make,
                    model = EXCLUDED.model,
                    trim = EXCLUDED.trim,
                    bodytype = EXCLUDED.bodytype,
                    regionalspec = EXCLUDED.regionalspec,
                    cylinders = EXCLUDED.cylinders,
                    color = EXCLUDED.color,
                    origin = EXCLUDED.origin,
                    weightinkg = EXCLUDED.weightinkg,
                    noofpassengers = EXCLUDED.noofpassengers;
            """, (
                vin_upper,
                updated_attrs.get("year", ""),
                updated_attrs.get("make", ""),
                updated_attrs.get("model", ""),
                updated_attrs.get("trim", ""),
                updated_attrs.get("body_type", ""),
                updated_attrs.get("regional_spec", ""),
                updated_attrs.get("cylinders", ""),
                updated_attrs.get("color", ""),
                updated_attrs.get("origin", ""),
                updated_attrs.get("weight", ""),
                updated_attrs.get("no_of_passengers", "")
            ))
            
            # 2. Insert/update in approved_predictions
            cur.execute("""
                INSERT INTO approved_predictions (
                    chassisnumber, year, make, model, trim, bodytype, regionalspec, cylinders, color, origin, weightinkg, noofpassengers
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chassisnumber) DO UPDATE SET
                    year = EXCLUDED.year,
                    make = EXCLUDED.make,
                    model = EXCLUDED.model,
                    trim = EXCLUDED.trim,
                    bodytype = EXCLUDED.bodytype,
                    regionalspec = EXCLUDED.regionalspec,
                    cylinders = EXCLUDED.cylinders,
                    color = EXCLUDED.color,
                    origin = EXCLUDED.origin,
                    weightinkg = EXCLUDED.weightinkg,
                    noofpassengers = EXCLUDED.noofpassengers;
            """, (
                vin_upper,
                updated_attrs.get("year", ""),
                updated_attrs.get("make", ""),
                updated_attrs.get("model", ""),
                updated_attrs.get("trim", ""),
                updated_attrs.get("body_type", ""),
                updated_attrs.get("regional_spec", ""),
                updated_attrs.get("cylinders", ""),
                updated_attrs.get("color", ""),
                updated_attrs.get("origin", ""),
                updated_attrs.get("weight", ""),
                updated_attrs.get("no_of_passengers", "")
            ))
            
            # 3. Remove the record from pending_predictions
            cur.execute("DELETE FROM pending_predictions WHERE UPPER(TRIM(chassisnumber)) = %s;", (vin_upper,))
            
            conn.commit()
            return True
    except Exception as e:
        print(f"Error approving pending prediction: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def reject_pending_prediction(vin: str) -> bool:
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM pending_predictions WHERE UPPER(TRIM(chassisnumber)) = %s;", (vin.upper().strip(),))
            conn.commit()
            return True
    except Exception as e:
        print(f"Error rejecting pending prediction: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()
