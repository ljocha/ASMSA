#!/bin/bash

for i in "$@"; do
	pushd min_$i

	ln ../min.mdp .

	gmx pdb2gmx -f min.pdb -o min.gro -p min.top -water tip3p -ff amber94 &&
	gmx editconf -f min.gro -o min-box.gro -c -d 2.0 -bt dodecahedron &&
	cp dihre.itp posre.itp &&
	gmx grompp -f min.mdp -c min-box.gro -p min.top -o min.tpr &&
	gmx mdrun -deffnm min &&

    	popd
done
