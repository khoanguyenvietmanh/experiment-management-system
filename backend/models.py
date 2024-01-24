
from typing import Any, Optional, Union, Dict, List
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class TrainApiData(BaseModel):
    drop_out: float
    learning_rate: float
    epochs: int


class GridSearchAPiData(BaseModel):
    params_dict: Dict[str, Any]


class Job_Train(BaseModel):
    uid: str
    type: str
    status: str
    params: Dict[str, Any]
    accuracy: float
    run_time: float

class Job_GridSearch(BaseModel):
    uid: str
    type: str
    status: str
    params: Dict[str, Any]
    best_accuracy: float
    best_params: Dict[str, Any]
    run_time: float
    

    