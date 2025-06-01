# CUDA 12.1, Python 3.10 이미지 사용
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 비대화 모드 설정 (tzdata 입력 방지용)
ENV DEBIAN_FRONTEND=noninteractive

# 시스템 패키지 설치 + tzdata 시간대 설정
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    ffmpeg libavutil-dev libavcodec-dev libavformat-dev libswscale-dev \
    libgl1-mesa-glx libosmesa6-dev libglew-dev libglfw3 libglfw3-dev \
    patchelf curl git unzip cmake gcc g++ python3-pip python3-dev \
 && ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime \
 && dpkg-reconfigure --frontend noninteractive tzdata \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --upgrade pip setuptools wheel

# 주요 패키지 설치 (gym → gymnasium으로 대체)
RUN pip install \
    jupyterlab \
    pandas>=1.2 \
    scikit-learn>=0.22 \
    opencv-python>=4.2 \
    pyyaml>=5.1 \
    yacs>=0.1.6 \
    einops>=0.3 \
    tensorboard \
    psutil \
    tqdm \
    matplotlib \
    simplejson \
    fvcore \
    av \
    numpy==1.26.4 \
    scipy==1.12.0 \
    mujoco==2.3.7 \
    mujoco-py==2.1.2.14 \
    gymnasium \
    flax==0.7.5 \
    dm_control==1.0.14 \
    brax==0.0.16 \
    imageio \
    tfp-nightly==0.20.0.dev20230524 \
    wandb \
    ml_collections

# JAX 설치 (CUDA 12용)
RUN pip install --upgrade "jax[cuda12_pip]==0.4.19" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

CMD ["/bin/bash"]
