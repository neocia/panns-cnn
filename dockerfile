# Base: CUDA runtime para GPU + Ubuntu 22.04
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Evita prompts do apt
ENV DEBIAN_FRONTEND=noninteractive

# Diretório de trabalho
WORKDIR /workspace

# Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    git wget ffmpeg libsndfile1 build-essential \
 && rm -rf /var/lib/apt/lists/*

# Atualiza pip
RUN python3 -m pip install --upgrade pip

# Copia o código e o requirements.txt para dentro do container
COPY . /workspace

# Instala dependências do Python a partir do requirements.txt
RUN pip install -r requirements.txt

# Comando padrão ao iniciar o container
CMD ["bash"]
