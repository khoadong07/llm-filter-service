from pydantic import BaseModel
from typing import List

class PredictContent(BaseModel):
    id: str
    content: str

class PredictRequest(BaseModel):
    domain: str
    lang_detect: bool
    contents: List[PredictContent]
    

class PredictionResult(BaseModel):
    id: str
    label: str
    prob: float
    
class PredictResponse(BaseModel):
    processing_time: float
    request_time: float
    results: List[PredictionResult]

class SetupModelRequest(BaseModel):
    model_name: str
    domain: str