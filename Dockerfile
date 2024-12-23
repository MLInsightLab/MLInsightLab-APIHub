FROM python:3.12

# Update software
RUN apt update && apt upgrade -y && apt autoremove -y

# Change the workdir
WORKDIR /code

# Copy and install python requirements
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy and install main files
COPY ./templates.py /code/templates.py
COPY ./db_utils.py /code/db_utils.py
COPY ./utils.py /code/utils.py
COPY ./main.py /code/main.py

# Add group mlil and add root to that group
RUN groupadd -g 1004 mlil
RUN adduser root mlil

# Expose necessary port
EXPOSE 4488
