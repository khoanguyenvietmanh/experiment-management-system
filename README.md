# Fastapi + streamlit

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

