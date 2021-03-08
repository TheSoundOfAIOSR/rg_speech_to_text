# rg_speech_to_text
Research Group Speech To Text

This repository will be used to experiment/test new approaches before they are fit into the common codebase. 

## Installation

### Installing via pip
- Download and Install python (recommend 3.8)
- Create a virtual environment using `python -m venv env_name`
- enable created environment `env_path\Scripts\activate`
- Install PyTorch `pip install torch==1.8.0+cu102 torchaudio===0.8.0 -f https://download.pytorch.org/whl/torch_stable.html`
- install required dependencies `pip install -r requirements.txt`

### Installing via conda
- Download and install miniconda
- Create a new virutal environment using `conda create --name env_name python==3.8`
- enable create environment `conda activate env_name`
- Install PyTorch `conda install pytorch torchaudio cudatoolkit=11.1 -c pytorch`
- install required dependencies `pip install -r requirements.txt`

### Installing from source

```
bash
git clone https://github.com/TheSoundOfAIOSR/rg_speech_to_text
cd rg_speech_to_text
pip3 install .
```
