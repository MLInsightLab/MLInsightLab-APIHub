FROM python:3.12-slim

# Update software and install dependencies
RUN apt update && \
    apt upgrade -y && \
    apt install -y curl ca-certificates build-essential && \
    apt autoremove -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Install docker command line tools
RUN curl -fsSL https://get.docker.com | sh

# Change the workdir
WORKDIR /code

# Copy and install python requirements
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade pip uv && \
    uv pip install --system --no-cache-dir --upgrade -r /code/requirements.txt

# Copy and install main files

COPY ./db_utils.py /code/db_utils.py
COPY ./global_variables.py /code/global_variables.py
COPY ./main.py /code/main.py
COPY ./model_utils.py /code/model_utils.py
COPY ./templates.py /code/templates.py

# Add group mlil and add root to that group
RUN groupadd -g 1004 mlil
RUN adduser root mlil

