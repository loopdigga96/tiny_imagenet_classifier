from typing import List, Optional
from datetime import datetime

from pydantic import BaseModel


class Requests(BaseModel):
    id: Optional[int]
    infer_time: int
    infer_result: List
    predicted_class: int
    confidence: float
    client_host: str
    time: datetime

    class Config:
        orm_mode = True
