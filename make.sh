#!/bin/bash
make clean
make >build_output.txt 2>&1
source /opt/intel/sgxsdk/environment
./app
