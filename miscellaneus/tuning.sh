#!/bin/bash

. /opt/bin/activate

exec /usr/local/bin/tuning.py "$@"
