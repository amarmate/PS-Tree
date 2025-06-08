# Dockerfile
FROM python:3.10-slim

# Erstelle einen Arbeitsordner
WORKDIR /app

# Installiere virtualenv
RUN pip install --upgrade pip && pip install virtualenv
RUN apt-get update && apt-get install git

# Erstelle eine virtuelle Umgebung
RUN virtualenv venv

# Aktiviere die virtuelle Umgebung beim Containerstart
CMD ["/bin/bash"]