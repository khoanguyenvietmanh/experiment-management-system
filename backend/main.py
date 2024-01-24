from ml.train import train, grid_search

from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from celery.result import AsyncResult
from ml.train import app as celery_app
from backend.models import TrainApiData, GridSearchAPiData, Job_Train, Job_GridSearch

import backend.schemas as schemas
from backend.database import sessionlocal, engine
from sqlalchemy.orm import Session

app = FastAPI()

schemas.Base.metadata.create_all(bind=engine)

def get_db():
    db = sessionlocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/jobs_train")
async def fetch_jobs(db: Session = Depends(get_db)):
    jobs = db.query(schemas.Job_Train).all()
    return jobs

@app.get("/jobs_gridsearch")
async def fetch_jobs(db: Session = Depends(get_db)):
    jobs = db.query(schemas.Job_GridSearch).all()
    return jobs

@app.post("/train", status_code=201)
def run_task(data: TrainApiData, db: Session = Depends(get_db)):
    drop_out = data.drop_out
    epochs = data.epochs
    learning_rate = data.learning_rate

    params = {"drop_out": drop_out, "epochs": epochs, "learning_rate": learning_rate}

    current_job = db.query(schemas.Job_Train).filter(schemas.Job_Train.params == params).first()

    if current_job:
        return JSONResponse({"task_id": "Already existed !!!"})

    task = train.delay(drop_out, epochs, learning_rate)

    job = schemas.Job_Train(
        uid=str(task.id),
        type="Train",
        params=params
    )

    db.add(job)
    db.commit()

    return JSONResponse({"task_id": task.id})

@app.post("/grid_search", status_code=201)
def run_task(data: GridSearchAPiData, db: Session = Depends(get_db)):
    params_dict = data.params_dict

    current_job = db.query(schemas.Job_GridSearch).filter(schemas.Job_GridSearch.params == params_dict).first()

    if current_job:
        return JSONResponse({"task_id": "Already existed !!!"})

    task = grid_search.delay(params_dict)

    job = schemas.Job_GridSearch(
        uid=str(task.id),
        type="Grid Search",
        params=params_dict
    )

    db.add(job)
    db.commit()

    return JSONResponse({"task_id": task.id})

@app.get("/jobs/{task_id}")
def get_status(task_id):
    task_result = AsyncResult(id=task_id, app=celery_app)
    result = {
        "task_id": task_id,
        "task_status": task_result.status,
        "task_result": task_result.result,
    }

    return JSONResponse(result)

@app.post("/update_train")
def update(job: Job_Train, db: Session = Depends(get_db)):
    current_job = db.query(schemas.Job_Train).filter(schemas.Job_Train.uid == job.uid).first()

    current_job.status = job.status
    current_job.accuracy = job.accuracy
    current_job.run_time = job.run_time
    
    db.commit()

    return JSONResponse({"task_id": job.uid})


@app.post("/update_gridsearch")
def update(job: Job_GridSearch, db: Session = Depends(get_db)):
    current_job = db.query(schemas.Job_GridSearch).filter(schemas.Job_GridSearch.uid == job.uid).first()

    current_job.status = job.status
    current_job.best_accuracy = job.best_accuracy
    current_job.best_params = job.best_params
    current_job.run_time = job.run_time
    
    db.commit()

    return JSONResponse({"task_id": job.uid})
