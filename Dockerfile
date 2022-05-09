FROM python:3.6-slim-buster

ENV FLASK_ENV development
ENV FLASK_APP app/app.py

# COPY ./


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y


# ENTRYPOINT ["./gunicorn.sh"]
COPY . .

CMD gunicorn --bind 0.0.0.0:5000 app:app --timeout 600 --log-level debug