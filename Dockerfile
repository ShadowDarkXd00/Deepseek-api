FROM mcr.microsoft.com/playwright/python:v1.49.0-jammy

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py shadow.py guide.html index.html remote.html connect.html import.html ./

ENV DATA_DIR=/data
CMD ["python", "server.py"]
