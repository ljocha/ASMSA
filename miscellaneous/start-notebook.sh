#!/bin/bash

source /opt/bin/activate

cd /home/jovyan
# mv ASMSA ASMSA.$(date | tr ' ' -)

[ -f README.md ] || cp /opt/ASMSA/README-hub.md README.md

# (cd /opt && tar cf - ASMSA) | tar xf -
# cd ASMSA

#--SingleUserNotebookApp.default_url=/lab --SingleUserNotebookApp.max_body_size=6291456000 
exec jupyterhub-singleuser --ip 0.0.0.0 "$@"
