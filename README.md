# rg_speech_to_text
Research Group Speech To Text

This repository will be used to experiment/test new approaches before they are fit into the common codebase. 

## Installation
This repo can be used in native Windows 10, native Ubuntu, Mac.
Some features related to language models requires Linux. For Windows 10 users there is a possibility to use Ubuntu under WSL2.

### Environment preparation

For users of WSL2, we provide a helper setup script available in `wsl2_setup` directory.
That scripts will create a dedicated wsl instance, which then can be used for this project.
Depending on which Windows version we have, we distinguish the following:
 - Windows Build >= 20150 have WSL2 with GPU access, therefore it can be installed CUDA as in native Linux.
 - Windows Build >= 21376 have WSL2 with WSLg, PulseAudio server integrated to communicate with host OS audio and can run graphical Linux apps.
 - Windows Build < 20150 have WSL2 which is sufficient to run in CPU mode.
With minimal effort, WSL2 of any version listed above can give us an Ubuntu 20.04 LTS environment in which the OS specific setup is exactly the same as in native Ubuntu 20.04 LTS.

### Installing via pip
- Download and Install python (recommend 3.8)
- Create a virtual environment using `python -m venv env_name`
- enable created environment `env_path\Scripts\activate`
- Update pip and setuptools using `python -m pip install --upgrade pip setuptools`
- Install PyTorch 1.7.1 from https://pytorch.org/get-started/locally/  with `torchvision` and `torchaudio`
    `pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`
- install required dependencies `pip install -r requirements[-linux|-win].txt`

### Installing via conda
- Download and install miniconda
- Create a new virutal environment using `conda create --name env_name python==3.8`
- enable create environment `conda activate env_name`
- Install PyTorch `conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch`
- install required dependencies `pip install -r requirements[-linux].txt` 
  For Windows platform it is enough requirements.txt and in addition install the audio driver from conda, like the following:
  ```
  conda install -c conda-forge python-sounddevice
  conda install pyaudio
  ```
  The reason for this different install path is [explained here.](setup/audio_setup.md)  

### Installing TheSoundOfAIOSR's rg_speech_to_text from source

```
bash
git clone https://github.com/TheSoundOfAIOSR/rg_speech_to_text
cd rg_speech_to_text
pip3 install .
```
