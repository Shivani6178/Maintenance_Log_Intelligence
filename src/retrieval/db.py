import psycopg2
from psycopg2.extensions import connection as PGConnection
from pgvector.psycopg2 import register_vector
import os
from dotenv import load_dotenv

load_dotenv()


def get_connection() -> PGConnection:
    """Open a new Postgres connection with pgvector type support registered."""
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    register_vector(conn)
    return conn