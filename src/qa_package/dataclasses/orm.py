from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Column
from sqlalchemy.orm import declarative_base

Base = declarative_base()
metadata = Base.metadata


class record(Base):
    __tablename__ = "record"
    id = Column("id", BigInteger, primary_key=True)  # article_id is biginteger
    factors = Column("factors", Vector(1536))  # vector dimension = 1536
