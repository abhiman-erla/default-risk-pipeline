# MLflow tracking server with a Postgres backend store and proxied artifact serving.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    "mlflow>=2.16,<3" \
    "psycopg2-binary>=2.9" \
    "boto3>=1.34"

EXPOSE 5000

# command is supplied by docker-compose.yml
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
