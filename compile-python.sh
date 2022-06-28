#!/bin/bash
set -x
#export LD_PRELOAD=~/intel/oneapi/mkl/latest/lib/intel64/libmkl_def.so.2:~/intel/oneapi/mkl/latest/lib/intel64/libmkl_avx2.so.2:~/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so.2:~/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.so.2:~/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_thread.so.2:~/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so:~/intel/oneapi/mkl/latest/lib/intel64/libmkl_sequential.so.2
export PYTHONPATH="~/miniconda3/lib/python3.8/site-packages/"
rm -rf "${PYTHONPATH}faiss*"
source ~/miniconda3/etc/profile.d/conda.sh&&\
make -C build -j faiss_avx2 && make  -C build -j swigfaiss swigfaiss_avx2 && (cd build/faiss/python && python setup.py install) \
 && (cd build/faiss/python && python setup.py build) \
 && (
  if [ -n "$1" ];then
     pytest -v -s "$1"
  fi
     )