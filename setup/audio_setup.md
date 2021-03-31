
In native Windows 10, we identified an issue.
It is [a known issue](https://github.com/spatialaudio/python-sounddevice/issues/256) and currently only a few peoples experienced it.

The symthoms is that the application which imports sounddevice or PyAudio and are running in native Windows 10, the underlaying PortAudio causes to exit python without throwing any exceptions.

Here is how to check in a simple python repl. After installing the sounddevice, execute the following:
```
import sounddevice as sd
```
or in case of PyAudio
```
import pyaudio
pa = pyaudio.PyAudio()
```
While executing these simple imports, python repl quickly exit without any message.

This issue seems to appear to some upgraded Windows 10 with a pre-release version which will be available as final release in April- May 2021.
In this situation, if you have this issue, you must use miniconda based setup.
PyPI package of both audio drivers which causes this issue, they have Conda distributions which seems to not have this issue. Because of that, instead of pip venv, you should create conda environment where the audio driver has to be installed like below:
```
conda install -c conda-forge python-sounddevice

# optionally pyaudio if we still have in the requirements
conda install pyaudio
```

If the issue is indeed related to a pre-relese Windows version, than there is a high chance that when this version came out officially, the changes it introduces will render useless those apps which use the current PyPI distributions. If somehow will be fixed until than, then we will remove this notice also.