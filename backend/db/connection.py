import os
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

BASE_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = BASE_DIR.parent
PROJECT_ROOT = BACKEND_ROOT.parent

# Load environment variables from the backend .env first, then fall back to the repo root.
load_dotenv(BACKEND_ROOT / '.env', override=False)
load_dotenv(PROJECT_ROOT / '.env', override=False)


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _normalize_database_url(raw_url: str) -> str:
    if raw_url.startswith("postgresql://"):
        raw_url = raw_url.replace("postgresql://", "postgresql+psycopg://", 1)

    if raw_url.startswith("postgresql+psycopg://"):
        parsed = urlsplit(raw_url)
        if parsed.username is not None:
            username = quote(parsed.username, safe="")
            password = quote(parsed.password or "", safe="")
            netloc = f"{username}:{password}@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
    return raw_url


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return _normalize_database_url(database_url)

    host = _env("POSTGRES_HOST", "localhost")
    port = _env("POSTGRES_PORT", "5432")
    db = _env("POSTGRES_DB", "medassist")
    user = _env("POSTGRES_USER", "postgres")
    password = _env("POSTGRES_PASSWORD", "root@123")

    url = URL.create(
        drivername="postgresql+psycopg",
        username=user,
        password=password,
        host=host,
        port=int(port),
        database=db,
    )
    return str(url)


# Only set search_path for local development, not for Neon (pooler doesn't support it)
connect_args = {}
if not os.getenv("DATABASE_URL"):
    connect_args = {"options": "-c search_path=medassist"}

engine = create_engine(
    get_database_url(),
    pool_pre_ping=True,
    connect_args=connect_args,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def test_connection() -> bool:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return True

