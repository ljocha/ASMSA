FROM python:3.8-slim as builder

ARG DIST=asmsa-0.0.1.tar.gz
ENV DEBIAN_FRONTEND=noninteractive 
ENV TZ=Europe/Prague

RUN apt update && apt install -y git g++ libz-dev procps

WORKDIR /opt
# torch & torchvision must be compatible with gromacs container
RUN python -m venv . 
COPY requirements.txt /tmp/requirements.txt
RUN . bin/activate && pip install -r /tmp/requirements.txt
RUN . bin/activate && pip install git+https://github.com/onnx/tensorflow-onnx 
RUN . bin/activate && jupyter-nbextension enable nglview --py 

COPY dist/$DIST /tmp
RUN . bin/activate && pip3 install /tmp/$DIST 

# select tensorflow gpu image as base
FROM tensorflow/tensorflow:latest-gpu 

RUN apt update && apt install -y wget g++ libz-dev tini procps && apt clean && rm -rf /var/lib/apt/lists/*

# apply tensorboard fix
RUN wget https://raw.githubusercontent.com/tensorflow/tensorboard/e59ca8d45746f459d797f4e69377eda4433e1624/tensorboard/notebook.py -O - > /usr/local/lib/python3.8/dist-packages/tensorboard/notebook.py 

RUN useradd -m -u 1000 jovyan
RUN mkdir -p /opt && chown jovyan /opt

COPY --from=builder /opt /opt
COPY --from=builder /usr/local /usr/local

USER root
RUN apt update && apt install -y curl && curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && install -m 755 kubectl /usr/local/bin && apt clean && rm -rf /var/lib/apt/lists/*

USER jovyan

RUN mkdir /opt/ASMSA
COPY IMAGE prepare.ipynb tune.ipynb train.ipynb md.ipynb /opt/ASMSA/
COPY md.mdp.template *.mdp /opt/ASMSA/
COPY tuning.py tuning.sh start-notebook.sh /usr/local/bin/
WORKDIR /home/jovyan

# use tensorflow from base image
ENV PYTHONPATH="/usr/local/lib/python3.8/dist-packages/"

ENTRYPOINT ["tini", "-g", "--"]
CMD ["start-notebook.sh"]
