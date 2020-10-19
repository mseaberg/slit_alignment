#!/bin/env bash
source /reg/neh/home/seaberg/setup_python.sh

cd /reg/neh/home/seaberg/Commissioning_Tools/PPM_centroid

if [ $# -eq 1 ]; then
    IMAGER=$1
else
    IMAGER="IM1L0"
fi

python run_interface.py -c $IMAGER &
