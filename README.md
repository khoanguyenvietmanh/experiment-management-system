# Fastapi + streamlit

Project Structure
```
.
├── backend --> implement and deploy APIs for FastAPI Server
│   ├── database.py --> initialize sqlite database
│   ├── Dockerfile
│   ├── main.py --> generate database and implement all needed APIs 
│   ├── models.py --> define data format for each request
│   ├── requirements.txt
│   └── schemas.py --> define all job tables
├── docker-compose.yml
├── frontend --> implement User Interface 
│   ├── Dockerfile
│   ├── requirements.txt
│   └── streamlit_main.py --> implement Streamlit UI for user to interact
├── __init__.py
├── ml --> define model and implement training process
│   ├── data.py --> load MNIST data
│   ├── Dockerfile
│   ├── models.py --> define architecture of models
│   ├── train.py --> implement training process
│   └── utils.py
├── README.md
└── requirements.txt
```

Setup env and install all necessary modules.
```bash
pip install -r requirements.txt
```

Make sure you install the redis by docker or by following command in your local machine
```bash
sudo apt-get install redis-server
```
# Start app
Go in the root dir and run these

Celery Worker
```bash
celery -A ml.train worker --loglevel=INFO
```

Streamlit
```bash
streamlit run frontend/streamlit_main.py
```

FastAPI 
```
uvicorn backend.main:app --reload
```

- FastApi: http://localhost:8000/docs
- Streamlit: http://localhost:8501/


## Docker
```bash
docker-compose up -d --build
```

