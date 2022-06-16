#!/bin/bash

set -e

identificator=$JUPYTERHUB_SERVER_NAME
pvc_path=$(df --output=target | grep home)
persist_folder=${pvc_path}/${identificator}
rsync -avz $persist_folder /home

function pvc_sync_daemon {
    while inotifywait -r -e modify,create,delete,move /home/jovyan; do
        rsync -az --delete /home/jovyan $persist_folder
    done
}


pvc_sync_daemon &
exec "$@"
