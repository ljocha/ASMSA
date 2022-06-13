FROM nvcr.io/nvidia/tensorflow:22.02-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive 
ENV TZ=Europe/Prague

RUN apt update
RUN apt install -y python3-notebook
RUN apt install -y python3-pip

RUN pip3 install jupyterhub

RUN pip3 install mdtraj matplotlib

WORKDIR /home/jovyan
ENV HOME /home/jovyan

COPY . /home/jovyan
RUN useradd -m -u 1000 jovyan
RUN chown -R 1000:1000 /home/jovyan

RUN apt install -y krb5-user sshfs
COPY krb5.conf /etc

RUN apt-get install -y inotify-tools
RUN apt-get install -y rsync
COPY monitor.sh /tmp

# ENTRYPOINT ["/tmp/monitor.sh", "&"]
ENTRYPOINT []
