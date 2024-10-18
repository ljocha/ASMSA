#!/bin/bash

for d in "$@"; do
	pushd $d
	cat - <<EOF | gmx energy -f min.edr -o min.xvg 
Potential
Dih.-Rest.
EOF
	popd
done
