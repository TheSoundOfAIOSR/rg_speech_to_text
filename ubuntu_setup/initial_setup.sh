#!/bin/bash
sudo apt-get update && apt-get install -y build-essential libsndfile1 ffmpeg python3-venv python3-dev libportaudio2 portaudio19-dev python3-pyaudio

python3 -m venv venv
venv/bin/python -m pip install --upgrade pip setuptools
venv/bin/python -m pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
venv/bin/python -m pip install -r requirements-linux.txt
venv/bin/python -m pip install git+https://github.com/NVIDIA/NeMo.git@r1.0.0rc1#egg=nemo_toolkit[all]
