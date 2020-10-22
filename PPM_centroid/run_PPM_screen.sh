#!/bin/env bash
source /cds/home/s/seaberg/setup_python.sh

cd /cds/home/s/seaberg/Commissioning_Tools/PPM_centroid

if [ $# -eq 1 ]; then
    IMAGER=$1
else
    IMAGER="IM1L0"
fi

python run_interface.py -c $IMAGER &
