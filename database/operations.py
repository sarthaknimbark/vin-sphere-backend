import datetime
import pandas as pd
from .connection import get_db_connection

def lookup_vin_in_master_dataset(vin: str) -> dict:
    vin_upper = vin.upper().strip()
    length = len(vin_upper)
    table_name = "master_data"
    if length in (10, 11, 12):
        table_name = f"master_data_{length}"
        
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            query = f"""
                SELECT chassisnumber, year, make, model, trim, bodytype, regionalspec, cylinders, color, origin, weightinkg, noofpassengers
                FROM {table_name}
                WHERE UPPER(TRIM(chassisnumber)) = %s
                LIMIT 1;
            """
            cur.execute(query, (vin_upper,))
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
                noofpassengers AS no_of_passengers,
                created_at,
                id
            FROM pending_predictions
            ORDER BY id DESC;
        """, conn)
        if not df.empty and "created_at" in df.columns:
            df["timestamp"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
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
            vin_upper = vin.upper().strip()
            table_name = "master_data"
            if len(vin_upper) in (10, 11, 12):
                table_name = f"master_data_{len(vin_upper)}"

            # 1. If it already exists in master_data, don't add to pending queue
            query = f"SELECT 1 FROM {table_name} WHERE UPPER(TRIM(chassisnumber)) = %s LIMIT 1;"
            cur.execute(query, (vin_upper,))
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
            table_name = "master_data"
            if len(vin_upper) in (10, 11, 12):
                table_name = f"master_data_{len(vin_upper)}"
            
            # 1. Insert/update in master_data
            query_master = f"""
                INSERT INTO {table_name} (
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
            """
            cur.execute(query_master, (
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

def update_pending_prediction(vin: str, updated_attrs: dict) -> bool:
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            vin_upper = vin.upper().strip()
            cur.execute("""
                UPDATE pending_predictions SET
                    year = %s,
                    make = %s,
                    model = %s,
                    trim = %s,
                    bodytype = %s,
                    regionalspec = %s,
                    cylinders = %s,
                    color = %s,
                    origin = %s,
                    weightinkg = %s,
                    noofpassengers = %s
                WHERE UPPER(TRIM(chassisnumber)) = %s;
            """, (
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
                updated_attrs.get("no_of_passengers", ""),
                vin_upper
            ))
            conn.commit()
            return True
    except Exception as e:
        print(f"Error updating pending prediction: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


def get_approved_predictions_count() -> int:
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM approved_predictions;")
            return cur.fetchone()[0]
    except Exception as e:
        print(f"Error getting approved predictions count: {e}")
        return 0
    finally:
        if conn:
            conn.close()


def fetch_training_data_by_length(length: int) -> pd.DataFrame:
    """
    Fetches all training data for a specific chassis length from the PostgreSQL database.
    """
    conn = None
    try:
        conn = get_db_connection()
        if length in (10, 11, 12):
            table_name = f"master_data_{length}"
            query = f"""
                SELECT 
                    chassisnumber AS "chassisNumber",
                    year AS "year",
                    make AS "make",
                    model AS "model",
                    trim AS "trim",
                    bodytype AS "bodyType",
                    regionalspec AS "regionalSpec",
                    cylinders AS "cylinders",
                    color AS "color",
                    origin AS "origin",
                    weightinkg AS "weightInKg",
                    noofpassengers AS "noOfPassengers"
                FROM {table_name};
            """
        elif length == 17:
            query = """
                SELECT 
                    chassisnumber AS "chassisNumber",
                    year AS "year",
                    make AS "make",
                    model AS "model",
                    trim AS "trim",
                    bodytype AS "bodyType",
                    regionalspec AS "regionalSpec",
                    cylinders AS "cylinders",
                    color AS "color",
                    origin AS "origin",
                    weightinkg AS "weightInKg",
                    noofpassengers AS "noOfPassengers"
                FROM master_data
                WHERE LENGTH(chassisnumber) = 17;
            """
        else:
            raise ValueError(f"Unsupported chassis length: {length}")
            
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error fetching training data by length: {e}")
        raise
    finally:
        if conn:
            conn.close()

