# rg_speech_to_text
Research Group Speech To Text

This repository will be used to experiment/test new approaches before they are fit into the common codebase. 

## Installation
### Environment preparation

For working TensorFlowASR and/or NeMo toolkits.

Create an environment with pip and setuptools
```
python -m pip install --upgrade pip setuptools
```

Install `tensorflow>=2.4.1`, depending on the GPU availability, install `tensorflow-gpu` also.

Install `torch>=1.7.1` with `torchvision` and `torchaudio`.
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Then add the requirements.txt to your environment.

NeMo toolkit 1.0.0b4 will be installed by
```
python -m pip install git+https://github.com/NVIDIA/NeMo.git@r1.0.0b4#egg=nemo_toolkit[all]
```

### Installing TheSoundOfAIOSR's rg_speech_to_text from source

```
bash
git clone https://github.com/TheSoundOfAIOSR/rg_speech_to_text
cd rg_speech_to_text
pip3 install .
```
 