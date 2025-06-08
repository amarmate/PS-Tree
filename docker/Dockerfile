# Dockerfile
FROM python:3.10-slim

# Erstelle einen Arbeitsordner
WORKDIR /app

# Installiere virtualenv
RUN apt-get update && apt-get install -y git build-essential gfortran tmux \
    && rm -rf /var/lib/apt/lists/*


RUN git clone https://github.com/amarmate/PS-Tree.git
WORKDIR /app/PS-Tree
RUN pip install --upgrade pip
RUN pip install numpy==1.22.1
RUN pip install -r requirements_dev.txt
RUN pip install -r requirements.txt


# Aktiviere die virtuelle Umgebung beim Containerstart
CMD ["/bin/bash"]