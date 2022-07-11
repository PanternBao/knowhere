#!/bin/bash
set -x
source ~/anaconda3/etc/profile.d/conda.sh&&conda activate myenv&&\
make -C build -j faiss && make -C build install \
 && make -C build demo_ivfpq_indexing\
 && ./build/demos/demo_ivfpq_indexing