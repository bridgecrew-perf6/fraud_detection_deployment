FROM python:3.6
COPY app.py .
COPY model.py .
COPY train.py .
COPY requirements.txt .
ADD data ./data
ADD pickles ./pickles
RUN pip install -r requirements.txt