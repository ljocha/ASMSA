#!/bin/bash

pvc_path=$(df --output=target | grep home)

rsync -avz ${pvc_path}/jovyan /home

while inotifywait -r -e modify,create,delete,move /home/jovyan; do
    rsync -avz /home/jovyan $pvc_path
done
