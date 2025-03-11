#!/bin/bash

set -e

# set persist directory based on HOSTNAME
# which should always be unique
identificator=$HOSTNAME
pvc_path=$(df --output=target | grep home)
persist_folder="${pvc_path}/${identificator}/"

# create persist directory if it's the first run
if [ ! -d "$persist_folder" ]; then
  mkdir $persist_folder
fi

rsync -avz $persist_folder /home/jovyan

function pvc_sync_daemon {
    while inotifywait -r -e modify,create,delete,move /home/jovyan; do
        rsync -az --delete /home/jovyan/ $persist_folder
    done
}


pvc_sync_daemon &
exec "$@"
