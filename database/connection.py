import os
import psycopg2
import streamlit as st
from pathlib import Path

# Resolve project root relative to this file (database/connection.py)
database_dir = Path(__file__).resolve().parent
project_root = database_dir.parent

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=project_root / ".env")
except ImportError:
    pass

def get_db_connection():
    import os

    try:
        if "postgres" in st.secrets:
            secrets_params = dict(st.secrets["postgres"])
            secrets_params.setdefault("connect_timeout", 3)
            return psycopg2.connect(**secrets_params)
    except Exception:
        pass

    try:
        import json
        conn_params_str = os.environ.get("DB_CONN_PARAMS")
        if conn_params_str:
            conn_params = json.loads(conn_params_str)
            conn_params.setdefault("connect_timeout", 3)
            return psycopg2.connect(**conn_params)
    except Exception:
        pass

    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5432"),
        database=os.environ.get("PGDATABASE", "postgres"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "postgres"),
        connect_timeout=3
    )

def init_db():
    conn = None
    try:
        conn = get_db_connection()
    except Exception as e:
        print(f"Database connection check error: {e}")
    finally:
        if conn:
            conn.close()
