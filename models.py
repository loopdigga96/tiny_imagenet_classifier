from sqlalchemy import Column, Integer, Float, String, func, DateTime
from sqlalchemy.types import PickleType

from .database import Base


class Requests(Base):
    __tablename__ = "requests"

    id = Column(Integer, primary_key=True, index=True)
    # stored in milliseconds
    infer_time = Column(Integer, unique=False, index=False)
    infer_result = Column(PickleType)
    predicted_class = Column(Integer)
    confidence = Column(Float)
    client_host = Column(String)
    time = Column(DateTime)


