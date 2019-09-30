FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 as base

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        pkg-config \
        software-properties-common \
        unzip \
        vim \
        libjpeg-dev \
        libpng-dev \
        git \
        screen \
        htop \
        && rm -rf /var/lib/apt/lists/*


# install python 3.7
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.7 \
    python3.7-distutils \
    python3.7-dev

RUN apt-get install wget
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.7 get-pip.py

RUN pip3.7 install -U pip
RUN pip3.7 install -U setuptools

RUN ln -s /usr/bin/python3.7 /usr/bin/python 
RUN ln -s /usr/bin/pip3.7 /usr/bin/pip 
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . /app

