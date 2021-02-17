# Overview
Below are outlined the environment setup and requirements.
The setup has to work in Linux, Windows 10.

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
When CUDA Toolkit for Windows 10 were installed from `cuda_11.2.1_461.09_win10.exe` and `cudnn-11.2-windows-x64-v8.1.0.77.zip`, 
the above test code will fail with the error message regarding missing `cusolver64_10.dll`. In the installer installed a newer version of this file, but looks like this release it was not tested when was released, because some other DLL-s was linked to this older version, from CUDA 10, instead of CUDA 11.
To fix that, we have to download `cuda_10.2.89_441.22_win10.exe` as part of CUDA Tookit 10 and then open with 7zip, to find a `cusolver\bin` folder with `cusolver64_10.dll` and  `cusolverMg64_10.dll` files. These two files has to be copied to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin` folder. Finally script above will show no more error. 

# python-sounddevice
`python-sounddevice` is a decent multi-platform library which wraps most of the important sound drivers.
The [installation process description can be found here](https://python-sounddevice.readthedocs.io/en/0.4.1/installation.html).

If you are using Windows, you can alternatively install one of the packages provided at https://www.lfd.uci.edu/~gohlke/pythonlibs/#sounddevice. The PortAudio library (with ASIO support) is included in the package and you can get the rest of the dependencies on the same page. In Windows, ASIO is a standard for professional audio, therefore is adviced to install a package which already contains the necessary ASIO interface also. 
Since we are using Python 3.8, for Windows 10 we have to choose `sounddevice‑0.4.1‑cp38‑cp38‑win_amd64.whl` for 64 bit. After downloading the Wheel file, by `pip install <filename>`.