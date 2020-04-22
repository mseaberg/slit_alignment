#!/bin/env bash
source /reg/g/pcds/pyps/conda/py36env.sh
cd /reg/g/pcds/l2si-commissioning/ppm/screen_files
python launch.py &
/reg/neh/home/zlentz/epics/ioc/tst/gigECam/current/build/iocBoot/ioc-tst-gige-ppm/edm-ioc-tst-gige-ppm.cmd &
