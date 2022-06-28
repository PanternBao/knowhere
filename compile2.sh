set -x
make -C build -j faiss && make -C build install \
 && make -C build demo_ivfpq_indexing\
 && ./build/demos/demo_ivfpq_indexing