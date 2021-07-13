FROM python:3.8-slim

WORKDIR /app

RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get install -y libhdf5-dev
RUN apt-get install -y pkg-config

COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

CMD [ "python3", "app.py"]