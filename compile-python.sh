#!/bin/bash
set -x
source ~/anaconda3/etc/profile.d/conda.sh&&conda activate myenv&&\
make -C build -j faiss && make -C build -j swigfaiss && (cd build/faiss/python && python setup.py install) \
 && (cd build/faiss/python && python setup.py build) \
 && export PYTHONPATH="/root/anaconda3/envs/myenv/lib/python3.8/site-packages/"\
 && pytest -v -s $1