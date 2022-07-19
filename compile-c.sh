#!/bin/bash
set -x
export LD_PRELOAD=/home/dcy/intel/oneapi/mkl/latest/lib/intel64/libmkl_def.so.2:/home/dcy/intel/oneapi/mkl/latest/lib/intel64/libmkl_avx2.so.2:/home/dcy/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so.2:/home/dcy/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.so.2:/home/dcy/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_thread.so.2:/home/dcy/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so:/home/dcy/intel/oneapi/mkl/latest/lib/intel64/libmkl_sequential.so.2
source ~/miniconda3/etc/profile.d/conda.sh && echo "$LD_LIBRARY_PATH"\
make -C build -j faiss_avx2 \
 && make -C build -j $1 \
 && ./build/demos/$1