
FROM python:3.12-slim

WORKDIR /app

COPY requirements-container.txt .

RUN pip install --no-cache-dir -r requirements-container.txt

COPY app.py .
COPY model/ ./model/
COPY templates/ ./templates/
COPY static/ ./static/

EXPOSE 5000

CMD ["python", "app.py"]
