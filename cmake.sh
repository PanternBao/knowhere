#!/bin/bash
set -x
source ~/intel/oneapi/setvars.sh intel64
source ~/miniconda3/etc/profile.d/conda.sh  &&rm -rf build && cmake -DFAISS_OPT_LEVEL=avx2  -DFAISS_ENABLE_GPU=OFF -DCMAKE_BUILD_TYPE=Release -B build .