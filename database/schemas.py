from typing import List, Optional, Dict

from pydantic import BaseModel


class Requests(BaseModel):
    id: Optional[int]
    infer_time: int
    infer_result: List
    predicted_class: int
    confidence: float
    client_host: str

    class Config:
        orm_mode = True
