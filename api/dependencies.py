from fastapi import HTTPException, status
from database.connection import get_db_connection

def get_db():
    """FastAPI dependency to retrieve a database connection."""
    try:
        conn = get_db_connection()
        yield conn
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection error: {str(e)}"
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass
