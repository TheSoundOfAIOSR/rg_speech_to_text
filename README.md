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
- Install PyTorch 1.8.0 from https://pytorch.org/get-started/locally/  with `torchvision` and `torchaudio`
- install required dependencies `pip install -r requirements[-linux|-win].txt`

### Installing via conda
- Download and install miniconda
- Create a new virutal environment using `conda create --name env_name python==3.8`
- enable create environment `conda activate env_name`
- Install PyTorch `conda install pytorch torchaudio cudatoolkit=11.1 -c pytorch`  with `torchvision` and `torchaudio`
- install required dependencies `pip install -r requirements[-linux|-win].txt`

NeMo toolkit 1.0.0r1 currently is supported in Linux only and will be installed by
```
python -m pip install git+https://github.com/NVIDIA/NeMo.git@r1.0.0r1#egg=nemo_toolkit[all]
```

### Installing TheSoundOfAIOSR's rg_speech_to_text from source

```
bash
git clone https://github.com/TheSoundOfAIOSR/rg_speech_to_text
cd rg_speech_to_text
pip3 install .
```
 