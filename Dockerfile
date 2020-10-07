FROM pytorch/pytorch

### Dependenices, some of which we might not need.
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    gcc \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    libglfw3 \
    patchelf \
    libopenmpi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

### Install Mujoco
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 \
    && rm mujoco.zip

### TODO: make sure to have a mjkey in the repo!
COPY ./mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

### Clone repo with submodules pulled using HTTPS
RUN git clone https://github.com/ameesh-shah/carl.git && \
    cd carl && \
	git submodule init && \
	git config submodule.learning_to_adapt.url https://github.com/iclavera/learning_to_adapt.git && \
	git config submodule.gym-duckietown.url https://github.com/jesbu1/gym-duckietown.git && \
	git submodule update --init --recursive

### Add Linux dependencies
RUN apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf libopenmpi-dev

### Dependencies for mujoco-py
RUN pip install glfw>=1.4.0 && \
    pip install numpy>=1.11 && \
    pip install Cython>=0.27.2 && \
    pip install imageio>=2.1.2 && \
    pip install cffi>=1.10 && \
    pip install fasteners~=0.15 && \
    pip install lockfile

### Dependencies for CARL
RUN pip install --ignore-installed -r carl/requirements.txt

WORKDIR carl/

### Env for CARL
RUN cd gym-duckietown && \
	git checkout 144cdf1b5355571e534c4efe92935d5927be58c7 && \
	pip install -e .
RUN cd learning_to_adapt && \
	git checkout 4ba064431e68edafaa211e7dcf91704c533b4188 && \
	pip install -e .

RUN bash
