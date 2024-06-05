from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    id: str
    content: List[str]
    

class PredictionResult(BaseModel):
    id: str
    label: str
    prob: float
    
class PredictResponse(BaseModel):
    id: str
    processing_time: float
    request_time: float
    results: List[PredictionResult]

class SetupModelRequest(BaseModel):
    model_name: str
    domain: str