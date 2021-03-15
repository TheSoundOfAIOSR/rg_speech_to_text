# rg_speech_to_text
Research Group Speech To Text

This repository will be used to experiment/test new approaches before they are fit into the common codebase. 

## Installation
### Environment preparation

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
- install required dependencies `pip install -r requirements[-linux|-win].txt`

NeMo toolkit 1.0.0rc1 currently is supported in Linux only and will be installed by
```
python -m pip install git+https://github.com/NVIDIA/NeMo.git@r1.0.0rc1#egg=nemo_toolkit[all]
```
For Windows environment the only option remain WSL2 with GPU support as described in `setup/stt/gpu/environment-setup.md`

### Installing TheSoundOfAIOSR's rg_speech_to_text from source

```
bash
git clone https://github.com/TheSoundOfAIOSR/rg_speech_to_text
cd rg_speech_to_text
pip3 install .
```
