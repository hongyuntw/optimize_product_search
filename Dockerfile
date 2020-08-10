FROM python:3.6
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD pip install -r requirements.txt && python api.py 