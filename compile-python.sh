#!/bin/bash
set -x
export LD_PRELOAD=/home/dcy/intel/oneapi/mkl/latest/lib/intel64/libmkl_def.so.2:/home/dcy/intel/oneapi/mkl/latest/lib/intel64/libmkl_avx2.so.2:/home/dcy/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so.2:/home/dcy/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.so.2:/home/dcy/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_thread.so.2:/home/dcy/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so:/home/dcy/intel/oneapi/mkl/latest/lib/intel64/libmkl_sequential.so.2
source ~/miniconda3/etc/profile.d/conda.sh&&\
make VERBOSE=1 -C build -j faiss_avx2 && make VERBOSE=1 -C build -j swigfaiss swigfaiss_avx2 && (cd build/faiss/python && python setup.py install) \
 && (cd build/faiss/python && python setup.py build) \
 && export PYTHONPATH="/home/dcy/miniconda3/lib/python3.8/site-packages/"\
 && pytest -v -s $1