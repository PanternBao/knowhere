# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""this is a basic test script for simple indices work"""
from __future__ import absolute_import, division, print_function
# no unicode_literals because it messes up in py2
import time

import numpy as np
import unittest
import faiss
import tempfile
import os
import re
import warnings


from common_faiss_tests import get_dataset, get_dataset_2

def pytest_keyboard_interrupt(excinfo):
    SystemExit(0)


class EvalIVFPQAccuracy(unittest.TestCase):


    def pytest_keyboard_interrupt(excinfo):
        SystemExit(0)
    def test_IndexIVFPQ(self):
        d = 128
        nb = 40000
        nt = 40000
        nq = 100
        m=8
        nlist=1
        nprobe=1
        topk=10


        (xt, xb, xq) = get_dataset_2(d, nt, nb, nq)

        gt_index = faiss.IndexFlatL2(d)
        gt_index.add(xb)
        D, gt_nns = gt_index.search(xq, 1)

        self.fun1(d, gt_nns, m, nlist, nprobe, topk, xb, xq, xt)
        print("ok")
        self.fun2(d, gt_nns, m, nlist, nprobe, topk, xb, xq, xt)
        self.fun1(d, gt_nns, m, nlist, 2048, topk, xb, xq, xt)
        self.fun2(d, gt_nns, m, nlist, 2048, topk, xb, xq, xt)
        self.fun1(d, gt_nns, m, nlist, 1024, topk, xb, xq, xt)
        self.fun2(d, gt_nns, m, nlist, 1024, topk, xb, xq, xt)

    def fun1(self, d, gt_nns, m, nlist, nprobe, topk, xb, xq, xt):
        coarse_quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(coarse_quantizer, d, nlist, m, 8)
        gpu_config = faiss.GpuClonerOptions()
        gpu_config.useFloat16 = False
        gpu_config.usePrecomputed = False
        res = faiss.StandardGpuResources()  # use a single GPU
        index = faiss.index_cpu_to_gpu(res, 0, index, gpu_config)
        # index.cp.min_points_per_centroid = 5    # quiet warning
        index.train(xt)
        index.add(xb)
        index.nprobe = nprobe
        time_start = time.time()
        D, nns = index.search(xq, topk)
        time_end = time.time()
        n_ok = (nns == gt_nns).sum()
        nq = xq.shape[0]
        print(f"gpu:n_ok:{n_ok},nq:{nq},time:{time_end - time_start}")

    def fun2(self, d, gt_nns, m, nlist, nprobe, topk, xb, xq, xt):
        coarse_quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(coarse_quantizer, d, nlist, m, 8)
        # index.cp.min_points_per_centroid = 5    # quiet warning
        index.train(xt)
        index.add(xb)
        index.nprobe = nprobe
        time_start = time.time()
        D, nns = index.search(xq, topk)
        time_end = time.time()
        n_ok = (nns == gt_nns).sum()
        nq = xq.shape[0]
        print(f"cpu:n_ok:{n_ok},nq:{nq},time:{time_end - time_start}")
