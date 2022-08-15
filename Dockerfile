FROM nvcr.io/nvidia/tensorflow:22.02-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive 
ENV TZ=Europe/Prague

RUN apt update
RUN apt install -y python3-notebook python3-pip

RUN pip3 install jupyterhub mdtraj matplotlib jupyter-server-proxy

WORKDIR /home/jovyan
ENV HOME /home/jovyan

COPY . /home/jovyan
RUN useradd -m -u 1000 jovyan
RUN chown -R 1000:1000 /home/jovyan

RUN apt-get update
RUN apt install -y krb5-user sshfs
COPY krb5.conf /etc

RUN apt-get install -y inotify-tools


ENTRYPOINT ["./start.sh"]
