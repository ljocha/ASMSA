FROM python:3.9-slim as builder

ENV DEBIAN_FRONTEND=noninteractive 
ENV TZ=Europe/Prague

RUN apt update && apt install -y wget git g++ libz-dev tini procps && apt clean && rm -rf /var/lib/apt/lists/*

COPY start-notebook.sh /usr/local/bin/

RUN useradd -m -u 1000 jovyan
RUN mkdir -p /opt && chown jovyan /opt

USER jovyan
WORKDIR /opt

# torch & torchvision must be compatible with gromacs container
RUN python -m venv . 
RUN . bin/activate && pip install \
dict_hash \
numpy \
tensorflow \
jupyter-server-proxy \
tensorboard \
torch==1.12.1 \
torchvision==0.13.1 \
mdtraj \
keras_tuner \
jupyterhub \
jupyterlab \
matplotlib \
"ipywidgets<8" \
nglview \
networkx \
sympy onnx2torch \
&& rm -r /home/jovyan/.cache

RUN wget https://raw.githubusercontent.com/tensorflow/tensorboard/e59ca8d45746f459d797f4e69377eda4433e1624/tensorboard/notebook.py -O - > /opt/lib/python3.9/site-packages/tensorboard/notebook.py

RUN . bin/activate && pip install git+https://github.com/onnx/tensorflow-onnx && rm -r /home/jovyan/.cache

RUN . bin/activate && jupyter-nbextension enable nglview --py 

ARG DIST=asmsa-0.0.2.tar.gz

COPY dist/$DIST /tmp
RUN . bin/activate && pip3 install /tmp/$DIST && rm -r /home/jovyan/.cache

RUN . bin/activate && pip3 install kubernetes dill && rm -r /home/jovyan/.cache

USER root
RUN apt update && apt install -y curl && curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && install -m 755 kubectl /usr/local/bin && apt clean && rm -rf /var/lib/apt/lists/*

USER jovyan

RUN mkdir /opt/ASMSA
COPY IMAGE prepare.ipynb tune.ipynb train.ipynb md.ipynb tune_demo.ipynb /opt/ASMSA/
COPY md.mdp.template *.mdp /opt/ASMSA/
COPY tuning.py tuning.sh /usr/local/bin/
WORKDIR /home/jovyan

ENTRYPOINT ["tini", "-g", "--"]
CMD ["start-notebook.sh"]
