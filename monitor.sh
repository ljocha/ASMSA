#!/bin/bash

pvc_path=$(df --output=target | grep home)

if [ -z "$(ls -A ${pvc_path})" ]; then
    rsync -avz ${pvc_path} /home/jovyan
fi

while inotifywait -r -e modify,create,delete,move /home/jovyan; do
    rsync -avz /home/jovyan $pvc_path
done
