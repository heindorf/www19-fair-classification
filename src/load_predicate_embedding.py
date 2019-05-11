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

import collections
import logging

import numpy as np
import pandas as pd

from scipy.sparse import vstack

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Binarizer

from load_csr_matrix import load_csr_matrix
from transformers import FrequencyTransformer

FILE_ITEM_PREDICATES = '../../data/item-properties/item-properties.bz2'

PATH_FEATURES = '../../data/features/'
PATH_TRAIN = PATH_FEATURES + 'training/'    # noqa
PATH_VAL   = PATH_FEATURES + 'validation/'  # noqa
PATH_TEST  = PATH_FEATURES + 'test/'        # noqa


def load_matrices():
    matrices = collections.OrderedDict()

    path = PATH_TRAIN + 'embeddings/'
    matrices['X_S_train'] = path + 'subjectOut'
    matrices['X_P_train'] = path + 'predicate'
    matrices['X_O_train'] = path + 'objectIn'
    matrices['X_OO_train'] = path + 'objectOut'

    path = PATH_VAL + 'embeddings/'
    matrices['X_S_val'] = path + 'subjectOut'
    matrices['X_P_val'] = path + 'predicate'
    matrices['X_O_val'] = path + 'objectIn'
    matrices['X_OO_val'] = path + 'objectOut'

    path = PATH_TEST + 'embeddings/'
    matrices['X_S_test'] = path + 'subjectOut'
    matrices['X_P_test'] = path + 'predicate'
    matrices['X_O_test'] = path + 'objectIn'
    matrices['X_OO_test'] = path + 'objectOut'

    for key, X in matrices.items():
        logging.debug('load {}...'.format(key))
        matrices[key] = load_csr_matrix(X)

    meta = collections.OrderedDict()
    meta['n_train'] = matrices['X_O_train'].shape[0]
    meta['n_val'] = matrices['X_O_val'].shape[0]
    meta['n_test'] = matrices['X_O_test'].shape[0]

    data = collections.OrderedDict()

    data['X_S_all'] = vstack([
        matrices['X_S_train'],
        matrices['X_S_val'],
        matrices['X_S_test']
    ])

    data['X_P_all'] = vstack([
        matrices['X_P_train'],
        matrices['X_P_val'],
        matrices['X_P_test']
    ])

    data['X_O_all'] = vstack([
        matrices['X_O_train'],
        matrices['X_O_val'],
        matrices['X_O_test']
    ])

    data['X_OO_all'] = vstack([
        matrices['X_OO_train'],
        matrices['X_OO_val'],
        matrices['X_OO_test']
    ])

    meta['X_S_all'] = np.array(
        ['S' + str(p) for p in range(data['X_S_all'].shape[1])])
    meta['X_P_all'] = np.array(
        ['P' + str(p) for p in range(data['X_P_all'].shape[1])])
    meta['X_O_all'] = np.array(
        ['O' + str(p) for p in range(data['X_O_all'].shape[1])])
    meta['X_OO_all'] = np.array(
        ['OO' + str(p) for p in range(data['X_OO_all'].shape[1])])

    return data, meta


def binarize_features(data):
    encoder = Binarizer(threshold=0.5, copy=False)

    for key, X in data.items():
        data[key] = encoder.fit_transform(X)


def select_item_predicates_at_end_of_training_set(data, meta):
    item_predicates = pd.read_csv(FILE_ITEM_PREDICATES, header=None)
    item_predicates = item_predicates.values.flatten()

    def _remove_attribute_predicates_from_X(X):
        # mask = np.zeros((1, data['X_object_pred_all'].shape[1]))
        # mask[0, item_predicates] = 1
        # return X.multiply(mask).tocsr()
        return X.tocsc()[:, item_predicates].tocsr()

    for key, X in data.items():
        logging.debug(key)
        data[key] = _remove_attribute_predicates_from_X(X)
        meta[key] = meta[key][item_predicates]


def count_nonzero(X, _):
    return np.asarray((X != 0).sum(axis=0)).ravel()


def select_features(
        data, meta, y, slice_fit, score_func=count_nonzero, k=100):
    if y is None:
        rand_X = next(iter(data.values()))
        y = np.zeros(rand_X[slice_fit].shape[0])

    logging.debug(slice_fit)

    for key in data:
        logging.debug(data[key].shape)
        selector = SelectKBest(score_func=score_func, k=k)
        selector = selector.fit(data[key][slice_fit], y[slice_fit])
        data[key] = selector.transform(data[key])
        meta[key] = meta[key][selector.get_support()]


def frequency_encoding(data, slice_fit):
    # slice_fit = slice(0, n_train + n_val)

    # convert to DataFrame
    df_freq = pd.DataFrame()
    df_freq['subjectPredEmbedFrequency'] = rows_to_str(data['X_S_all'])
    df_freq['objectPredEmbedFrequency'] = rows_to_str(data['X_O_all'])
    df_freq['objectOutPredEmbedFrequency'] = rows_to_str(data['X_OO_all'])

    transformer = FrequencyTransformer()
    transformer = transformer.fit(
        df_freq[['subjectPredEmbedFrequency']][slice_fit])
    df_freq[['subjectPredEmbedFrequency']] = transformer.transform(
        df_freq[['subjectPredEmbedFrequency']])

    transformer = FrequencyTransformer()
    transformer = transformer.fit(
        df_freq[['objectPredEmbedFrequency']][slice_fit])
    df_freq[['objectPredEmbedFrequency']] = transformer.transform(
        df_freq[['objectPredEmbedFrequency']])

    transformer = FrequencyTransformer()
    transformer = transformer.fit(
        df_freq[['objectOutPredEmbedFrequency']][slice_fit])
    df_freq[['objectOutPredEmbedFrequency']] = transformer.transform(
        df_freq[['objectOutPredEmbedFrequency']])

    return df_freq


def rows_to_str(array):
    rows = array.tolil().rows
    for i in range(len(rows)):
        rows[i] = ','.join(str(elem) for elem in rows[i])
    return rows


# ---------------------------------------------------------
# Internal Functions
# ---------------------------------------------------------

def _get_slice_fit(meta, use_test_set):
    if use_test_set:
        return slice(0, meta['n_train'] + meta['n_val'])
    else:
        return slice(0, meta['n_train'])
