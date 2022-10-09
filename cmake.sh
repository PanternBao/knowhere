#!/bin/bash
set -x
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
rm -rf build faiss/python/swigfaiss_avx2.swig
#use mkl
#source ~/intel/oneapi/setvars.sh intel64
source ~/miniconda3/etc/profile.d/conda.sh  &&rm -rf build && cmake -DFAISS_OPT_LEVEL=avx2 -DBLA_VENDOR=Intel10_64_dyn -DMKL_LIBRARIES=$CONDA_PREFIX/lib -DFAISS_ENABLE_GPU=OFF -DCMAKE_BUILD_TYPE=Release -B build .