FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt update \
 && apt install -y software-properties-common \
                   build-essential gfortran git tmux \
                   pkg-config libopenblas-dev liblapack-dev \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt update \
 && apt install -y python3.10 python3.10-venv python3.10-dev openssh-server \
 && rm -rf /var/lib/apt/lists/*

# 1) Venv anlegen und Pip upgraden
RUN python3.10 -m venv venv \
 && . venv/bin/activate \
 && pip install --upgrade pip

# 2) setuptools downgraden, damit glmnet-Setup läuft
RUN . venv/bin/activate \
 && pip install "setuptools<60.0.0" wheel

# 3) Repo klonen und Dependencies installieren
SHELL ["/bin/bash", "-lc"]
RUN . venv/bin/activate \
 && git clone --depth 1 https://github.com/amarmate/PS-Tree.git /app/PS-Tree \
 && cd PS-Tree \
 && pip install numpy \
 && pip install -r requirements.txt

RUN . venv/bin/activate \ 
 && pip install sympy cython \
 && pip install mlflow scikit-optimize


COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 22
EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]

