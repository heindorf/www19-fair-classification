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

import abc
import logging

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import clone


_logger = logging.getLogger()


########################################################################
# Multiple Instance Learning
########################################################################
class BaseMultipleInstanceClassifier(BaseEstimator):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fit(self, g, X, y):
        pass

    @abc.abstractmethod
    def predict_proba(self, g, X):
        pass


class SingleInstanceClassifier(BaseMultipleInstanceClassifier):
    def __init__(self, base_estimator, agg_func='mean', window=None):
        self.agg_func = agg_func  # name of aggregation function
        self.base_estimator = base_estimator
        self.proba = None
        self.window = window

    def fit(self, g, X, y, sample_weight=None):
        self.base_estimator = clone(self.base_estimator)
        self.base_estimator.fit(X, y, sample_weight=sample_weight)
        self.proba = None

    def set_proba(self, proba):
        self.proba = proba

    # g contains the group ids
    def predict_proba(self, g, X):
        # Determines the aggregation function (e.g., mean, max, min, ...)
        if len(g) != len(X):
            raise Exception(
                'g and X should have same lengh')

        # Has user explicitly specified proba?
        # (to save some computational time)
        if self.proba is not None:
            proba = self.proba  # use stored proba and ignore X
        else:
            # if proba has not been explicitly set,
            # use base_estimator to compute it
            proba = self.base_estimator.predict_proba(X)[:, 1]

        if self.agg_func == 'cummean':
            agg_proba = self._cummean_proba(g, proba, self.window)
        else:
            raise Exception('Unknown function name: ' + str(self.agg_func))

        return agg_proba

    @staticmethod
    def _cummean_proba(group, proba, window):
        sum_result = group_reduce_lookahead(group, proba, np.add, window)
        count_result = group_reduce_lookahead(
            group, np.asarray([1] * len(proba)), np.add, window)
        result = sum_result / count_result
        _logger.debug(
            'Average group length per revision: ' +
            str(np.sum(count_result) / len(proba)))

        return result


class SimpleMultipleInstanceClassifier(BaseMultipleInstanceClassifier):
    def __init__(self, base_estimator, trans_func='min_max', window=None):
        self.trans_func = trans_func  # name of aggregation function
        self.base_estimator = base_estimator
        self.window = window

    def fit(self, g, X, y, sample_weight=None):
        self.base_estimator = clone(self.base_estimator)
        _logger.debug('transforming...')
        _, trans_X, trans_y = self._cummin_cummax_trans_func(
            g, X, y, self.window)
        _logger.debug('transforming...done.')

        _logger.debug('fitting...')
        self.base_estimator.fit(trans_X, trans_y, sample_weight=sample_weight)
        _logger.debug('fitting...done.')

    def predict_proba(self, g, X):
        # transformation into 'group space'
        trans_g, trans_X, _ = self._cummin_cummax_trans_func(
            g, X, None, self.window)

        # prediction in 'group space'
        trans_proba = self.base_estimator.predict_proba(trans_X)

        if self.trans_func == 'cummin_cummax':
            proba = trans_proba[:, 1]  # result already in 'instance space'

        return proba

    @classmethod
    def _cummin_cummax_trans_func(cls, g, X, y, window):
        _logger.debug('lookahead maximum...')
        max_X = group_reduce_lookahead(g, X, np.maximum, window)

        _logger.debug('lookahead minimum...')
        min_X = group_reduce_lookahead(g, X, np.minimum, window)

        _logger.debug('concatenate...')
        result_X = np.concatenate([max_X, min_X], axis=1)
        del(max_X)  # free memory
        del(min_X)  # free memory

        _logger.debug('ascontiguous...')
        result_X = np.ascontiguousarray(result_X)

        return g, result_X, y


class CombinedMultipleInstanceClassifier(BaseMultipleInstanceClassifier):
    def __init__(self, base_estimator1, base_estimator2):
        self.base_estimator1 = base_estimator1
        self.base_estimator2 = base_estimator2

    def fit(self, g, X, y, sample_weight=None):
        self.base_estimator1 = clone(self.base_estimator1)
        self.base_estimator2 = clone(self.base_estimator2)
        self.base_estimator1.fit(g, X, y, sample_weight)
        self.base_estimator2.fit(g, X, y, sample_weight)
        self.base_estimator1_proba = None
        self.base_estimator2_proba = None

    def set_proba(self, base_estimatro1_proba, base_estimator2_proba):
        self.base_estimator1_proba = base_estimatro1_proba
        self.base_estimator2_proba = base_estimator2_proba

    def predict_proba(self, g, X):
        if self.base_estimator1_proba is None:
            base_estimator1_proba = self.base_estimator1.predict_proba(g, X)
        else:
            base_estimator1_proba = self.base_estimator1_proba

        if self.base_estimator2_proba is None:
            base_estimator2_proba = self.base_estimator2.predict_proba(g, X)
        else:
            base_estimator2_proba = self.base_estimator2_proba

        proba = self.average_proba(
            base_estimator1_proba, base_estimator2_proba)
        return proba

    # Averages the scores of two classifiers
    @staticmethod
    def average_proba(prob1, prob2):
        tmp = pd.DataFrame()
        tmp['prob1'] = prob1
        tmp['prob2'] = prob2

        avg_proba = np.ascontiguousarray(tmp.mean(axis=1).values)

        return avg_proba


########################################################################
# Online Transformers
########################################################################
class StreamGroupReduceTransformer:
    """Operates on streams of (g,v) pairs where g denotes a group and v a value.

    Reduces the stream within every group by applying the two-parameter
    function func.
    """

    def __init__(self, func):
        self.func = func
        self.d = {}

    def partial_fit(self, g, v):
        if g in self.d:
            self.d[g] = self.func(self.d[g], v)
        else:
            self.d[g] = v
        return self.d[g]

    def transform(self, g):
        return self.d[g]


def group_reduce_lookahead(g, X, func, lookahead):
    """Apply function func cumulatively while looking ahead."""
    if lookahead > len(g):
        lookahead = len(g)  # unlimited lookahead

    result = [np.nan] * len(g)

    transformer = StreamGroupReduceTransformer(func)

    for i in range(len(g) + lookahead - 1):
        if i < len(g):
            # add current element to lookahead data structure
            cur_g = g[i]
            cur_v = X[i]
            transformer.partial_fit(cur_g, cur_v)

        prev_i = i - lookahead + 1
        if prev_i >= 0:
            # compute result
            prev_g = g[prev_i]
            result[prev_i] = transformer.transform(prev_g)

    result = np.asarray(result)
    return result
