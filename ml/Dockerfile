FROM python:3.9-slim

RUN mkdir /ml
RUN pip install celery==5.2.7
RUN pip install redis==4.5.4
RUN pip install torch
RUN pip install torchvision
RUN pip install tqdm
RUN pip install scikit-learn

COPY ./ml /ml

CMD ["celery", "-A", "ml.train", "worker", "--loglevel=info"]


