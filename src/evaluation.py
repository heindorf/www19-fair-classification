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
import warnings

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


N_CALIBRATION = 10


class PrefitClassifier:
    def __init__(self, proba):
        self.proba = proba.reshape(-1, 1)
        self.classes_ = np.array([0, 1])
        pass

    def predict_proba(self, X):
        result = np.hstack([1 - self.proba, self.proba])
        return result


def calibrate_scores_(y, proba, sample_weight=None):
    proba = proba.reshape(-1, 1)
    clf = CalibratedClassifierCV(
        PrefitClassifier(proba), method='isotonic', cv='prefit')
    clf.fit(proba, y, sample_weight)
    result = clf.predict_proba(proba)[:, 1]
    return result


def calibrate_scores(y, proba, sample_weight=None):
    # one calibration is not sufficient for a stable result
    for i in range(N_CALIBRATION):
        proba = calibrate_scores_(y, proba, sample_weight)
    return proba


def evaluate_proba_performance(y, proba, sample_weight=None, index=''):
    if isinstance(index, str):
        index = [index]

    result = collections.OrderedDict()
    result['n_samples'] = y.shape[0]
    result['n_positive'] = int(y.sum())
    result['ACC'] = accuracy_score(
        y, proba >= 0.5, sample_weight=sample_weight)

    if y.sum() > 0 and y.sum() < len(y):
        result['PR'] = average_precision_score(
            y, proba, sample_weight=sample_weight)
        result['ROC'] = roc_auc_score(
            y, proba, sample_weight=sample_weight)

    result = pd.DataFrame(result, index=index)

    return result


def evaluate_proba_bias(y, protected, proba, calibrate=True,
                        sample_weight=None, index=''):
    if calibrate and sample_weight is not None:
        warnings.warn(
            'The options calibrate and sample_weight should not be used' +
            'together. The calibration should be done before sampling.')

    if isinstance(index, str):
        index = [index]

    if calibrate:
        # print('calibrating scores...')
        proba = calibrate_scores(
            y, proba, sample_weight=sample_weight)

    # print('filtering benign edits...')
    benign = ~y.astype(bool)

    # print('computing metrics...')
    if sample_weight is None:
        protected_mean = np.mean(proba[protected])
        non_protected_mean = np.mean(proba[~protected])
        benign_protected_mean = np.mean(proba[benign & protected])
        benign_non_protected_mean = np.mean(proba[benign & ~protected])
    else:
        protected_mean = np.average(
            proba[protected],
            weights=sample_weight[protected])
        non_protected_mean = np.average(
            proba[~protected],
            weights=sample_weight[~protected])
        benign_protected_mean = np.average(
            proba[benign & protected],
            weights=sample_weight[benign & protected])
        benign_non_protected_mean = np.average(
            proba[benign & ~protected],
            weights=sample_weight[benign & ~protected])

    benign_score_diff = benign_protected_mean - benign_non_protected_mean
    benign_score_ratio = benign_protected_mean / benign_non_protected_mean

    result = collections.OrderedDict()
    result['n_samples'] = y.shape[0]
    result['p_mean'] = protected_mean
    result['np_mean'] = non_protected_mean
    result['bp_mean'] = benign_protected_mean
    result['bnp_mean'] = benign_non_protected_mean
    result['score_diff'] = benign_score_diff
    result['score_ratio'] = benign_score_ratio

    result = pd.DataFrame(result, index=index)

    return result


def evaluate_proba_performance_bias(
        y, protected, proba, calibrate=True, sample_weight=None, index=''):
    metrics_performance = evaluate_proba_performance(
        y, proba, sample_weight, index)
    metrics_bias = evaluate_proba_bias(
        y, protected, proba, calibrate, sample_weight, index)
    metrics = pd.concat([metrics_performance, metrics_bias],
                        axis=1, keys=['Performance', 'Bias'])
    return metrics
