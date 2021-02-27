# Overview
Below are outlined the environment setup and requirements.

# Choise of operating system
On Linux we have access to the latest development in Deep Learning frameworks and it's associated libraries.
Professional audio equipements instead are available more easily in Windows or Mac.
Mac lacks in GPU required for Deep Learning.
As a conclusion, the setup is good to be supported in Linux and Windows 10.

In Linux, especially Ubuntu 20.04 LTS, the support for CUDA and all the libraries is the easiest. One just need to follow
the required documentations.
In Windows 10, we have good support for CUDA, latest TensorFlow and PyTorch, but some of the external modules for ASR, 
such as CTC decoders, requires not just GCC compiler, but also uses some linux headers, which practically is excluding
native Windows 10 platform, if we use these modules. Most likely these problematic dependencies are required for training
and eventually not for inference. Therefor we still keep the environment ready to be tested on Windows 10. Therefore below 
we detailed setup instructions for TensorFlow with native Windows 10 GPU support.

# Python with Jupyter Notebook
Has to be installed Python 3.8.
Has to be updated `pip`.

# TensorFlow
We have to use at least TensorFlow 2.3, but we identified that tensorflow-text in 2.3 for Windows 10 is broken, therefore we took the latest TensorFlow version at of today.
TensorFlow 2.4.1 or newer, has to be installed as described in [GPU support page](https://www.tensorflow.org/install/gpu)
There are described some native software requirements, such as NVIDIA GPU driver, CUDA Toolki, etc.

The quick test of the `tensorflow-gpu` setup is possible by running the following python code:
```
import tensorflow as tf
tf.test.gpu_device_name()
```

## Native Windows 10 support.
When CUDA Toolkit for Windows 10 were installed from `cuda_11.2.1_461.09_win10.exe` and `cudnn-11.2-windows-x64-v8.1.0.77.zip`, 
the above test code will fail with the error message regarding missing `cusolver64_10.dll`. In the installer installed a newer version of this file, but looks like this release it was not tested when was released, because some other DLL-s was linked to this older version, from CUDA 10, instead of CUDA 11.
To fix that, we have to download `cuda_10.2.89_441.22_win10.exe` as part of CUDA Tookit 10 and then open with 7zip, to find a `cusolver\bin` folder with `cusolver64_10.dll` and  `cusolverMg64_10.dll` files. These two files has to be copied to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin` folder. Finally script above will show no more error. 

## Windows 10 support with WSL2
WSL2 with GPU support requires Windows 10 Build 20150 or higher. At the time of this writing, this version is available only
in Insider's Program. [Here is a detailed description](http://aka.ms/GPUinWSL) and [a video also](https://www.youtube.com/watch?v=PdxXlZJiuxA)
To use this preview, you'll need to [register for the Windows Insider Program](https://insider.windows.com/getting-started/#register). Once you do, follow [these instuctions](https://insider.windows.com/getting-started/#install) to install the latest Insider build. When choosing your settings, ensure you're selecting the [Dev Channel](https://docs.microsoft.com/en-us/windows-insider/flight-hub/#active-development-builds-of-windows-10).
In addition to WSL2, also [Docker Desktop WSL 2 backend](https://docs.docker.com/docker-for-windows/wsl/) makes sense to setup.

The installation details of CUDA Toolkit within WSL2 is available at the download site. The main difference compared to a normal Ubuntu installation is that WSL2 doesn't require to be installed an NVidia video driver, but on the Windows 10 host a special driver will be downloaded as described in the documentation.

# python-sounddevice
`python-sounddevice` is a decent multi-platform library which wraps most of the important sound drivers.
The [installation process description can be found here](https://python-sounddevice.readthedocs.io/en/0.4.1/installation.html).

## Native Windows 10 support.
If you are using Windows, you can alternatively install one of the packages provided at https://www.lfd.uci.edu/~gohlke/pythonlibs/#sounddevice. The PortAudio library (with ASIO support) is included in the package and you can get the rest of the dependencies on the same page. In Windows, ASIO is a standard for professional audio, therefore is adviced to install a package which already contains the necessary ASIO interface also. 
Since we are using Python 3.8, for Windows 10 we have to choose `sounddevice‑0.4.1‑cp38‑cp38‑win_amd64.whl` for 64 bit. After downloading the Wheel file, by `pip install <filename>`.

## Windows 10 support with WSL2
For WSL2 on Windows 10, audio subsystem setup will require a little bit of extra effort.

For the host machine, is required to install a Pulse Audio server, that will communicate with the real audio interfaces.
The Pulse Audio build for Windows, supports ASIO drivers also, therefore the professional audio setups can be integrated.
A short description what is required in order to make audio working within WSL2 or docker container [is available here](https://discourse.ubuntu.com/t/getting-sound-to-work-on-wsl2/11869). Eventually the server address of the host is not written well here, therefore is presented what you need to have in `~/.bashrc`
```
# Get the IP Address of the Windows 10 Host and use it in Environment.
HOST_IP=$(host `hostname` | grep -oP '(\s)\d+(\.\d+){3}' | tail -1 | awk '{ print $NF }' | tr -d '\r')
export DISPLAY=$HOST_IP:0.0
export PULSE_SERVER=tcp:$HOST_IP
```
In WSL2 is required to install `sudo apt install libportaudio2`.
In order to use the bridging between the host and WSL2 container, don't forget to start the pulseaudio server in host. The audio is routed through network connection between host and the container.