FROM nvcr.io/nvidian/pytorch:20.12-py3 as base
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install ffmpeg libsm6 libxext6 libxrender-dev -y

RUN pip install atari_py
RUN pip install wandb

RUN git clone https://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@github.com/galdl/bcts.git
RUN git clone -b 1.3 https://github.com/Kaixhin/Rainbow.git 
RUN wget -i bcts/pretrained_paths.txt -P bcts/pretrained 
RUN cd bcts && python main.py
