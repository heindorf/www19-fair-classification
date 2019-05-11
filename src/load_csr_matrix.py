# -----------------------------------------------------------------------------
# WWW 2019 Debiasing Vandalism Detection Models at Wikidata
#
# Copyright (c) 2019 Stefan Heindorf, Yan Scholten, Gregor Engels, Martin Potthast
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

import bz2

import numpy as np
from scipy.sparse import csr_matrix


def load_csr_matrix(path):
    data = load_ndarray_from_bz2(path + '_data.csv.bz2')
    indices = load_ndarray_from_bz2(path + '_indices.csv.bz2')
    indptr = load_ndarray_from_bz2(path + '_indptr.csv.bz2')
    shape = load_ndarray_from_bz2(path + '_shape.csv.bz2')

    return csr_matrix((data, indices, indptr), shape)


def load_ndarray_from_bz2(fname):
    dbuf = bz2.BZ2File(fname).read().split(b',')
    return np.array(dbuf, dtype='int32')
