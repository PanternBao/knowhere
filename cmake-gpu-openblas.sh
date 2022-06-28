#!/bin/bash
set -x
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
rm -rf build faiss/python/swigfaiss_avx2.swig
which nvcc
nvcc -V
#use mkl
#source ~/intel/oneapi/setvars.sh intel64 >/dev/null
source ~/miniconda3/etc/profile.d/conda.sh >/dev/null &&rm -rf build && cmake  -DFAISS_OPT_LEVEL=avx2 -DBLAS_LIBRARIES=/opt/OpenBLAS/lib/libopenblas.so  -DFAISS_ENABLE_GPU=ON -DCMAKE_BUILD_TYPE=Release -B build .