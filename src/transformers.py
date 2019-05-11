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

import numpy as np
import pandas as pd

from numpy.core import getlimits
from sklearn.base import TransformerMixin


class BooleanImputer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, **fit_params):
        result = X.astype(np.float32)
        result = result.fillna(0.5)

        return pd.DataFrame(result)


class CumFrequencyTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumption: X is ordered by revisionId
        grouped = X.groupby(by=list(X.columns))
        result = grouped.cumcount() + 1

        result = result.to_frame()

        return result


class FrequencyTransformer(TransformerMixin):
    """Transforms categorical features to a numeric value (frequency).

    Given a data frame with columns (C1, C2, ..., Cn), computes for each
    unique tuple (c1, c2, ..., cn), how often it appears in the data frame.

    For example, it counts how many revisions were done with this predicate
    on the training set (one column C1='predicate').
    """
    def __init__(self):
        self.__frequencies = None

    def fit(self, X, y=None):
        self.__frequencies = X.groupby(by=list(X.columns)).size()
        self.__frequencies.name = 'frequencies'
        return self

    def transform(self, X):
        result = X.join(self.__frequencies, on=list(X.columns), how='left')

        # all other frequencies are at least 1
        result = result['frequencies'].fillna(0)
        result = result.to_frame()

        return result


class InfinityImputer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X

        for column in X.columns:
            datatype = result.loc[:, column].dtype.type
            limits = getlimits.finfo(datatype)

            result.loc[:, column].replace(np.inf, limits.max, inplace=True)
            result.loc[:, column].replace(-np.inf, limits.min, inplace=True)

        return result


class LogTransformer(TransformerMixin):
    """Compute the formula sign(X)*ceil(log2(|X|+1))"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X

        sign = result.apply(np.sign)

        result = result.apply(np.absolute)
        result = result + 1
        result = result.apply(np.log2)
        result = result.apply(np.ceil)
        result = sign * result

        result = result.fillna(0)
        return result


class MedianImputer(TransformerMixin):
    def __init__(self):
        self.__median = None

    def fit(self, X, y=None):
        self.__median = X.median()
        return self

    def transform(self, X):
        result = X.fillna(self.__median)
        result = InfinityImputer().fit_transform(result)

        return result


class MinusOneImputer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        result = X.fillna(-1)
        result = InfinityImputer().fit_transform(result)

        return result


class EqualsTransformer(TransformerMixin):
    def __init__(self, value):
        self.__value = value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # value is assumed to be a tuple
        result = [True] * len(X)

        for i in range(len(self.__value)):
            result = result & self.isNanEqual(X.iloc[:, i], self.__value[i])

        result = pd.DataFrame(result)
        result.columns = [str(self.__value)]

        return result

    @staticmethod
    def isNanEqual(a, b):
        result = ((a == b) | (pd.isnull(a) & pd.isnull(b)))
        return result
