#!/bin/bash

# Update package lists
sudo apt update

# Install required packages
sudo apt install -y wget zip unzip gunicorn python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools

# Install virtualenv
sudo apt install python3-venv

python3 -m venv .venv

source .venv/bin/activate

pip3 install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

wget https://github.com/maksymalist/JunkJudge/releases/download/v1.0/models.zip -O models.zip && \
    unzip -d models.zip && \
    rm models.zip

sudo ufw allow 5000

gunicorn --bind 0.0.0.0:5000 wsgi:app