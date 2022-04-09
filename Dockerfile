FROM python:3.6
COPY app.py .
COPY model.py .
COPY requirements.txt .
# ADD pickles ./pickles
RUN pip install -r requirements.txt