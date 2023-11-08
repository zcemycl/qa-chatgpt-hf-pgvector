import os
from typing import Iterator

from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

db_url = (
    os.environ["DB_URL"]
    if "DB_URL" in os.environ
    else "postgresql://postgres:postgres@localhost/postgres"
)

engine = None


def make_engine() -> Engine:
    global engine
    if engine is None:
        engine = create_engine(db_url)
    return engine


def get_session() -> Iterator[Session]:
    session = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=make_engine(),
        class_=Session,
        expire_on_commit=False,
    )
    with session() as session:
        yield session
        session.close()
