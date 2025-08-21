#!/bin/bash

if [ -f /opt/bin/activate ]; then
	source /opt/bin/activate
else
	source /home/jovyan/bin/activate
fi

exec "$@"
