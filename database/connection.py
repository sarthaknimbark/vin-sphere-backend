import os
import psycopg2
import streamlit as st
from pathlib import Path

# Resolve project root relative to this file (database/connection.py)
database_dir = Path(__file__).resolve().parent
project_root = database_dir.parent

def get_db_connection():
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=project_root / ".env")
    except ImportError:
        pass

    try:
        if "postgres" in st.secrets:
            return psycopg2.connect(**st.secrets["postgres"])
    except Exception:
        pass

    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5432"),
        database=os.environ.get("PGDATABASE", "postgres"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "postgres")
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
