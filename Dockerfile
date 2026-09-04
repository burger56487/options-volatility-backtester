FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt /app/requirements-api.txt
RUN pip install --no-cache-dir -r /app/requirements-api.txt

COPY . /app

ENV APP_DB_PATH=/data/app.db
VOLUME ["/data"]

EXPOSE 8000
CMD ["uvicorn", "src.api.run_server:application", "--host", "0.0.0.0", "--port", "8000"]
