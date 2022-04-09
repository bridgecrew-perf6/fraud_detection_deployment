FROM python:3.6
COPY model.py .
COPY train.py .
COPY requirements.txt .
ADD data ./data
RUN pip install -r requirements.txt