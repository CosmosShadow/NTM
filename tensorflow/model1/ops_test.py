"""Tests for ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.ops import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

from ops import *

class SmoothCosineSimilarityTest(test_util.TensorFlowTestCase):
    def testSmoothCosineSimilarity(self):
        m = constant_op.constant(
            [[1,2,3],
             [2,2,2],
             [3,2,1],
             [0,2,4]], dtype=np.float32)
        v = constant_op.constant([2,2,2], dtype=np.float32)
        for use_gpu in [True, False]:
            with self.test_session(use_gpu=use_gpu):
                sim = smooth_cosine_similarity(m, v).eval()
                self.assertAllClose(sim, [0.92574867671153,
                                           0.99991667361053,
                                           0.92574867671153,
                                           0.77454667246876])

class CircularConvolutionTest(test_util.TensorFlowTestCase):
    def testCircularConvolution(self):
        v = constant_op.constant([1,2,3,4,5,6,7], dtype=tf.float32)
        k = constant_op.constant([0,0,1], dtype=tf.float32)
        for use_gpu in [True, False]:
            with self.test_session(use_gpu=use_gpu):
                cir_conv = circular_convolution(v, k).eval()
                self.assertAllEqual(cir_conv, [7,1,2,3,4,5,6])

class BinaryCrossEntropyTest(test_util.TensorFlowTestCase):
    def testBinaryCrossEntropy(self):
        logits = np.array([0,1,0,1,0], dtype=np.float32)
        targets = np.array([0,1,1,1,1], dtype=np.float32)
        for use_gpu in [True, False]:
            with self.test_session(use_gpu=use_gpu):
                loss = binary_cross_entropy_with_logits(logits, targets).eval()
                self.assertAllClose(loss, 11.052408446371)

if __name__ == "__main__":
    googletest.main()






