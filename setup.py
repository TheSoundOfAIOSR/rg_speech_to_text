# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fr:
    requirements = fr.read().splitlines()

setuptools.setup(
    name="RG_SPEECH_TO_TEXT",
    version="0.1",
    author="openSource Team",
    author_email="-",
    description="Speech to Text module for Sound design & Tweaking using latent space interpolation and text description Research project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheSoundOfAIOSR/rg_speech_to_text",
    packages=setuptools.find_packages(include=["TheSoundOfAIOSR*"]),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    python_requires='>=3.8',
)