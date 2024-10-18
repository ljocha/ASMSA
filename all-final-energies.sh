./final-energies.sh $(seq 0 255) >log.1 2>log.1 &
./final-energies.sh $(seq 256 511) >log.2 2>log.2 &
./final-energies.sh $(seq 512 767) >log.3 2>log.3 &
./final-energies.sh $(seq 768 1023) >log.4 2>log.4 &
./final-energies.sh $(seq 1024 1279 ) >log.5 2>log.5 &
./final-energies.sh $(seq 1280 1535) >log.6 2>log.6 &
./final-energies.sh $(seq 1536 1791 ) >log.7 2>log.7 &
./final-energies.sh $(seq 1792 2047 ) >log.8 2>log.8 &
