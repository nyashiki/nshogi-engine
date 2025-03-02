FROM nvcr.io/nvidia/tensorrt:24.04-py3

RUN apt-get update && apt-get install -y \
    clang-15 \
    vim \
    python3-pip \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100

RUN pip3 install flask flask_socketio
