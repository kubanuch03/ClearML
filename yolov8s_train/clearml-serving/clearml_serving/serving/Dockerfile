# 1. Берем легкий базовый образ
FROM python:3.11-slim-bullseye

ENV LC_ALL=C.UTF-8
WORKDIR /app

RUN pip3 install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY clearml_serving /app/clearml_serving

EXPOSE 8080
ENTRYPOINT ["/app/clearml_serving/serving/entrypoint.sh"]