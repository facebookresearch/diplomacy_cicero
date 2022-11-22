#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Code here is originally from:
https://github.com/diplomacy/research/blob/master/diplomacy_research/models/layers/graph_convolution.py#L28

License of that code reproduced here:
# ==============================================================================
# Copyright 2019 - Philip Paquette
#
# NOTICE:  Permission is hereby granted, free of charge, to any person obtaining
#   a copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#   sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
# ==============================================================================
"""
import numpy as np


def preprocess_adjacency(adjacency_matrix):
    """ Symmetrically normalize the adjacency matrix for graph convolutions.
        :param adjacency_matrix: A NxN adjacency matrix
        :return: A normalized NxN adjacency matrix
    """
    # Computing A^~ = A + I_N
    adj = adjacency_matrix
    adj_tilde = adj + np.eye(adj.shape[0])

    # Calculating the sum of each row
    sum_of_row = np.array(adj_tilde.sum(1))

    # Calculating the D tilde matrix ^ (-1/2)
    d_inv_sqrt = np.power(sum_of_row, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)

    # Calculating the normalized adjacency matrix
    norm_adj = adj_tilde.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return np.array(norm_adj, dtype=np.float32)
