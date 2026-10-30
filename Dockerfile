FROM mcr.microsoft.com/playwright/python:v1.49.0-jammy

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py shadow.py media.py integrations.py agent.py anthropic_api.py agent.html guide.html index.html remote.html connect.html import.html ./

ENV DATA_DIR=/data
CMD ["python", "server.py"]
