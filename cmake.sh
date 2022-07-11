#!/bin/bash
set -x
source ~/anaconda3/etc/profile.d/conda.sh&&conda activate myenv &&rm -rf build && cmake -DFAISS_ENABLE_GPU=OFF -DCMAKE_BUILD_TYPE=Release -B build .