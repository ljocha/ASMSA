# 3.10 is the last to support torch 1.12 we need due to gromacs
FROM jupyter/base-notebook:python-3.10

ENV DEBIAN_FRONTEND=noninteractive 
ENV TZ=Europe/Prague

USER root
RUN apt update && apt install -y git g++ libz-dev procps curl wget docker.io && curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && install -m 755 kubectl /usr/local/bin && apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /opt/ASMSA && chown -R jovyan /opt/ASMSA
RUN adduser jovyan docker

USER jovyan
RUN pip install 'tensorflow[and-cuda]'
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN pip install git+https://github.com/onnx/tensorflow-onnx 

#COPY 3142.patch /tmp
#RUN . /opt/bin/activate && cd /usr/local/lib/python$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")/dist-packages && patch -p1 </tmp/3142.patch

RUN cd /tmp && git clone --single-branch -b k8s https://github.com/ljocha/GromacsWrapper.git && pip install ./GromacsWrapper && rm -rf GromacsWrapper

COPY IMAGE prepare.ipynb tune.ipynb train.ipynb md.ipynb /opt/ASMSA/
COPY md.mdp.template *.mdp /opt/ASMSA/
COPY tuning.py tuning.sh start-notebook.sh /usr/local/bin/
# WORKDIR /home/jovyan

COPY --chown=jovyan gmx-wrap2.sh /usr/local/bin/gmx
RUN mkdir /tmp/asmsa
COPY --chown=jovyan README.md setup.cfg pyproject.toml /tmp/asmsa/
COPY --chown=jovyan src /tmp/asmsa/src
RUN pip3 install /tmp/asmsa

# RUN . /opt/bin/activate && pip3 install dfo-ls
