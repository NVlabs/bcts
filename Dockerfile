FROM nvcr.io/nvidian/pytorch:20.12-py3 as base
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install ffmpeg libsm6 libxext6 libxrender-dev -y

RUN pip install atari_py
RUN pip install wandb

RUN git clone https://gitlab+deploy-token-1211491:7q_xkJ1C2Bdnr2tP9iQX@gitlab.com/galdl20/bcts.git
RUN git clone -b 1.3 https://github.com/Kaixhin/Rainbow.git 
RUN wget -i bcts/pretrained_paths.txt -P bcts/pretrained 
RUN cd bcts && python main.py
