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

import logging
import re
import warnings

import numpy as np
import pandas as pd

from transformers import BooleanImputer
from transformers import FrequencyTransformer
from transformers import LogTransformer
from transformers import MedianImputer
from transformers import MinusOneImputer

FILE_WDVD_FEATURES       = '../../data/features/wdvd_features.csv.bz2'  # noqa

PATH_WDVD_TRUTH = '../../data/external/wdvc-2016/'
FILE_WDVD_TRUTH_TRAIN    = PATH_WDVD_TRUTH + 'training/wdvc16_truth.csv.bz2'            # noqa
FILE_WDVD_TRUTH_VAL      = PATH_WDVD_TRUTH + 'validation/wdvc16_2016_03_truth.csv.bz2'  # noqa
FILE_WDVD_TRUTH_TEST     = PATH_WDVD_TRUTH + 'test/wdvc16_2016_05_truth.csv.bz2'     # noqa

TRAINING_SET_START   =  33066010  # First revision on May 1, 2013, UTC      # noqa
SELECTION_SET_START  = 287589578  # First revision on January 1, 2016, UTC  # noqa
VALIDATION_SET_START = 308612956  # First revision on March 1, 2016, UTC    # noqa
TEST_SET_START       = 328006162  # First revision on May 1, 2016, UTC      # noqa
TAIL_SET_START       = 352684962  # First revision on July 1, 2016, UTC     # noqa

TRUE_FALSE_MAPPING = {'T': True, 'F': False}


def load_df_wdvd(use_test_set=False, nrows=None):
    logging.info('load_df_wdvd...')
    logging.debug('use_test_set={}'.format(use_test_set))
    logging.debug('nrows={}'.format(nrows))

    if use_test_set:
        # compatible to wsdmcup17 feature preprocessing
        slice_fit = slice(0, TEST_SET_START)
    else:
        slice_fit = slice(0, VALIDATION_SET_START)

    logging.debug('read_features...')
    df_wdvd_features = read_features(nrows)

    logging.debug('read_truth...')
    df_wdvd_truth = read_truth(nrows)

    logging.debug('merging...')
    df_wdvd = df_wdvd_features.merge(
        df_wdvd_truth, left_index=True, right_index=True)

    logging.debug('processing...')
    rename_columns(df_wdvd)
    convert_bool_columns(df_wdvd)
    convert_datetime_columns(df_wdvd)

    convert_property_column(df_wdvd)
    convert_itemValue_column(df_wdvd)

    logging.debug('transforming...')
    transform_df(slice_fit, df_wdvd)
    convert_dtype64_columns(df_wdvd)

    return df_wdvd


USECOLS = [
    'revisionId',
    'revisionSessionId',
    'itemId',
    'userId',
    'timestamp',
    'contentType',
    'commentTail',

    'englishLabel',
    'englishDescription',
    'englishAliases',
    'englishSitelink',
    'property',
    'itemValue',
    'literalValue',

    # Character features
    'alphanumericRatio',
    'asciiRatio',
    'bracketRatio',
    'digitRatio',
    'latinRatio',
    'longestCharacterSequence',
    'lowerCaseRatio',
    'nonLatinRatio',
    'punctuationRatio',
    'upperCaseRatio',
    'whitespaceRatio',

    # Word features
    'badWordRatio',
    'containsLanguageWord',
    'containsURL',
    'languageWordRatio',
    'longestWord',
    'lowerCaseWordRatio',
    'proportionOfLanguageAdded',  # in ORES but not WDVD
    'proportionOfLinksAdded',
    'proportionOfQidAdded',
    'upperCaseWordRatio',

    # Statement features
    'dataType',

    # User features
    'cumUserUniqueItems',
    'isPrivilegedUser',
    'isRegisteredUser',
    'userCityName',
    'userContinentCode',
    'userCountryCode',
    'userCountyName',
    'userName',
    'userSecondsSinceFirstRevision',
    'userSecondsSinceFirstRevisionRegistered',
    'userRegionCode',
    'userTimeZone',

    # Misc user features
    'isAdminUser',
    'isAdvancedUser',
    'isBotUser',
    'isCuratorUser',

    # Item features
    'logCumItemUniqueUsers',
    'isHuman',
    'isLivingPerson',

    # Revision features
    'commentCommentSimilarity',
    'commentLabelSimilarity',
    'commentLength',
    'commentSitelinkSimilarity',
    'commentTailLength',
    'isLatinLanguage',  # used by ORES but not WDVD
    'param1',  # used by ORES, also known as changeCount
    'param3',
    'param4',
    'positionWithinSession',
    'revisionAction',
    'revisionLanguage',
    'revisionPrevAction',
    'revisionSubaction',
    'revisionTags',

    # Misc content features
    'numberOfAliases',
    'numberOfBadges',
    'numberOfDescriptions',
    'numberOfLabels',
    'numberOfProperties',
    'numberOfQualifiers',
    'numberOfReferences',
    'numberOfSitelinks',
    'numberOfStatements',
]

DTYPE = {
    # Meta features
    'revisionId': np.int32,
    'revisionSessionId': np.int32,
    'itemId': np.int32,
    'userId': np.int32,
    'timestamp': str,
    'contentType': 'category',
    'commentTail': 'category',

    # Character features
    'alphanumericRatio': np.float32,
    'asciiRatio': np.float32,
    'bracketRatio': np.float32,
    'digitRatio': np.float32,
    'latinRatio': np.float32,
    'longestCharacterSequence': np.float32,
    'lowerCaseRatio': np.float32,
    'nonLatinRatio': np.float32,
    'punctuationRatio': np.float32,
    'upperCaseRatio': np.float32,
    'whitespaceRatio': np.float32,

    # Word features
    'badWordRatio': np.float32,
    'containsLanguageWord': str,  # convert to bool later
    'containsURL': str,           # convert to bool later
    'languageWordRatio': np.float32,
    'longestWord': np.float32,
    'lowerCaseWordRatio': np.float32,
    'proportionOfLanguageAdded': np.float32,  # in ORES, but not WDVD
    'proportionOfLinksAdded': np.float32,
    'proportionOfQidAdded': np.float32,
    'upperCaseWordRatio': np.float32,

    # Sentence features
    'commentCommentSimilarity': np.float32,
    'commentLabelSimilarity': np.float32,
    'commentSitelinkSimilarity': np.float32,
    'commentTailLength': np.float32,

    # Statement features
    'literalValue': str,
    'itemValue': str,     # convert to float later (there are some exceptions)
    'property': str,      # convert to float later (starts with P
    'dataType': 'category',

    # User features
    'isRegisteredUser': str,  # convert to bool later
    'isPrivilegedUser': str,  # convert to bool later
    'cumUserUniqueItems': np.int32,

    'userCityName': 'category',
    'userCountryCode': 'category',
    'userCountyName': 'category',
    'userContinentCode': 'category',
    'userName': 'category',
    'userRegionCode': 'category',
    'userTimeZone': 'category',

    'userSecondsSinceFirstRevision': np.int32,
    'userSecondsSinceFirstRevisionRegistered': np.int32,

    # Misc user features
    'param3': 'category',
    'param4': 'category',
    'isAdminUser': str,      # convert to bool later
    'isAdvancedUser': str,   # convert to bool later
    'isBotUser': str,        # convert to bool later
    'isCuratorUser': str,    # convert to bool later

    # Item features
    'logCumItemUniqueUsers': np.int32,
    'isHuman': str,  # used by ORES but not WDVD
    'isLivingPerson': str,  # used by ORES but not WDVD

    # Misc item features
    'numberOfLabels': np.int32,
    'numberOfDescriptions': np.int32,
    'numberOfAliases': np.int32,
    'numberOfStatements': np.int32,
    'numberOfSitelinks': np.int32,
    'numberOfQualifiers': np.int32,
    'numberOfReferences': np.int32,
    'numberOfBadges': np.int32,
    'numberOfProperties': np.int32,

    # Revision features
    'commentLength': np.float32,
    'isLatinLanguage': str,
    'param1': np.float32,
    'positionWithinSession': np.int32,
    'revisionAction': 'category',
    'revisionLanguage': 'category',
    'revisionPrevAction': 'category',
    'revisionSubaction': 'category',
    'revisionTags': 'category',
    'revisionHashTag': 'category',

    # Misc revision features
    'bytesIncrease': np.float32,
    'revisionSize': np.int32,
    'timeSinceLastRevision': np.float32,
}


def read_features(nrows):
    # see https://stackoverflow.com/q/40659212
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        df_wdvd_features = pd.read_csv(
            FILE_WDVD_FEATURES,
            index_col=0,
            quotechar='"',
            low_memory=True,
            keep_default_na=False,
            na_values=['', u'\ufffd'],
            dtype=DTYPE,
            usecols=USECOLS,
            engine='c',
            nrows=nrows
        )
    return df_wdvd_features


def read_truth(nrows):
    # see https://stackoverflow.com/q/40659212
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        df_truth_train = pd.read_csv(
            FILE_WDVD_TRUTH_TRAIN,
            index_col=0,
            nrows=nrows)
        df_truth_val = pd.read_csv(
            FILE_WDVD_TRUTH_VAL,
            index_col=0,
            nrows=None)
        df_truth_test = pd.read_csv(
            FILE_WDVD_TRUTH_TEST,
            index_col=0,
            nrows=None)
    return pd.concat([df_truth_train, df_truth_val, df_truth_test])


def convert_bool_columns(df):
    bool_columns = [
        'containsLanguageWord', 'containsURL', 'isAdminUser', 'isAdvancedUser',
        'isBotUser', 'isCuratorUser', 'isHuman', 'isLatinLanguage',
        'isLivingPerson', 'isPrivilegedUser', 'isRegisteredUser',
        'rollbackReverted', 'undoRestoreReverted']

    for column in bool_columns:
        df[column] = df[column].map(TRUE_FALSE_MAPPING)


def convert_datetime_columns(df):
    df['timestamp'] = pd.to_datetime(
        df['timestamp'], format='%Y-%m-%dT%H:%M:%SZ', utc=True)

    return df


def convert_property_column(df):
    def func(x):
        # is not nan?
        if x == x:
            return int(re.search('P(\d+)', x).group(1))
        else:
            return x

    df['property'] = df['property'].apply(func)


def convert_dtype64_columns(df):
    for column in df:
        if df[column].dtype == np.float64:
            df[column] = df[column].astype(np.float32)
        elif df[column].dtype == np.int64:
            df[column] = df[column].astype(np.int32)


def convert_itemValue_column(df):
    def func(x):
        # is not nan?
        if x == x:
            match = re.search('(\d+)', x)
            if match is not None:
                return int(match.group(1))
            else:
                return np.nan
        else:
            return np.nan

    df['itemValue'] = df['itemValue'].apply(func)


def rename_columns(df):
    mapping = {'REVISION_ID': 'revisionId',
               'ROLLBACK_REVERTED': 'rollbackReverted',
               'UNDO_RESTORE_REVERTED': 'undoRestoreReverted',
               'param1': 'changeCount'}
    df.rename(columns=mapping, copy=False, inplace=True)
    return df


def transform(slice_fit, df_feature, transformer):
    df_fit = df_feature.loc[slice_fit]
    transformer.fit(df_fit)
    return transformer.transform(df_feature)


def transform_df(slice_fit, df):
    logging.debug('meta features...')
    df['itemFreq']                  = transform(slice_fit, df[['itemId']]                   , FrequencyTransformer())  # noqa
    df['logItemFreq']               = transform(slice_fit, df[['itemFreq']]                 , LogTransformer())        # noqa
    df['logItemId']                 = transform(slice_fit, df[['itemId']]                   , LogTransformer())        # noqa

    logging.debug('character features...')
    df['alphanumericRatio']         = transform(slice_fit, df[['alphanumericRatio']]        , MedianImputer())         # noqa
    df['asciiRatio']                = transform(slice_fit, df[['asciiRatio']]               , MedianImputer())         # noqa
    df['bracketRatio']              = transform(slice_fit, df[['bracketRatio']]             , MedianImputer())         # noqa
    df['digitRatio']                = transform(slice_fit, df[['digitRatio']]               , MedianImputer())         # noqa
    df['latinRatio']                = transform(slice_fit, df[['latinRatio']]               , MedianImputer())         # noqa
    df['longestCharacterSequence']  = transform(slice_fit, df[['longestCharacterSequence']] , MinusOneImputer())       # noqa
    df['lowerCaseRatio']            = transform(slice_fit, df[['lowerCaseRatio']]           , MedianImputer())         # noqa
    df['nonLatinRatio']             = transform(slice_fit, df[['nonLatinRatio']]            , MedianImputer())         # noqa
    df['punctuationRatio']          = transform(slice_fit, df[['punctuationRatio']]         , MedianImputer())         # noqa
    df['upperCaseRatio']            = transform(slice_fit, df[['upperCaseRatio']]           , MedianImputer())         # noqa
    df['whitespaceRatio']           = transform(slice_fit, df[['whitespaceRatio']]          , MedianImputer())         # noqa

    logging.debug('word features...')
    df['badWordRatio']              = transform(slice_fit, df[['badWordRatio']]             , MedianImputer())         # noqa
    df['languageWordRatio']         = transform(slice_fit, df[['languageWordRatio']]        , MedianImputer())         # noqa
    df['longestWord']               = transform(slice_fit, df[['longestWord']]              , MinusOneImputer())       # noqa
    df['lowerCaseWordRatio']        = transform(slice_fit, df[['lowerCaseWordRatio']]       , MedianImputer())         # noqa
    df['upperCaseWordRatio']        = transform(slice_fit, df[['upperCaseWordRatio']]       , MedianImputer())         # noqa

    logging.debug('sentence features...')
    df['commentCommentSimilarity']  = transform(slice_fit, df[['commentCommentSimilarity']] , MinusOneImputer())       # noqa
    df['commentLabelSimilarity']    = transform(slice_fit, df[['commentLabelSimilarity']]   , MinusOneImputer())       # noqa
    df['commentSitelinkSimilarity'] = transform(slice_fit, df[['commentSitelinkSimilarity']], MinusOneImputer())       # noqa
    df['commentTailLength']         = transform(slice_fit, df[['commentTailLength']]        , MinusOneImputer())       # noqa

    logging.debug('statement features...')
    df['literalValueFreq']          = transform(slice_fit, df[['literalValue']]             , FrequencyTransformer())  # noqa
    df['itemValueFreq']             = transform(slice_fit, df[['itemValue']]                , FrequencyTransformer())  # noqa
    df['logItemValue']              = transform(slice_fit, df[['itemValue']]                , LogTransformer())        # noqa
    df['propertyFreq']              = transform(slice_fit, df[['property']]                 , FrequencyTransformer())  # noqa
    df['dataTypeFreq']              = transform(slice_fit, df[['dataType']]                 , FrequencyTransformer())  # noqa

    df['userCityFreq']              = transform(slice_fit, df[['userCityName']]             , FrequencyTransformer())  # noqa
    df['userCountryFreq']           = transform(slice_fit, df[['userCountryCode']]          , FrequencyTransformer())  # noqa
    df['userCountyFreq']            = transform(slice_fit, df[['userCountyName']]           , FrequencyTransformer())  # noqa
    df['userContinentFreq']         = transform(slice_fit, df[['userContinentCode']]        , FrequencyTransformer())  # noqa
    df['userFreq']                  = transform(slice_fit, df[['userName']]                 , FrequencyTransformer())  # noqa
    df['userRegionFreq']            = transform(slice_fit, df[['userRegionCode']]           , FrequencyTransformer())  # noqa
    df['userTimeZoneFreq']          = transform(slice_fit, df[['userTimeZone']]             , FrequencyTransformer())  # noqa

    logging.debug('revision features...')
    df['commentLength']             = transform(slice_fit, df[['commentLength']]            , MinusOneImputer())       # noqa
    df['isLatinLanguage']           = transform(slice_fit, df[['isLatinLanguage']]          , BooleanImputer())        # noqa
    df['revisionActionFreq']        = transform(slice_fit, df[['revisionAction']]           , FrequencyTransformer())  # noqa
    df['revisionLanguageFreq']      = transform(slice_fit, df[['revisionLanguage']]         , FrequencyTransformer())  # noqa
    df['revisionPrevActionFreq']    = transform(slice_fit, df[['revisionPrevAction']]       , FrequencyTransformer())  # noqa
    df['revisionSubactionFreq']     = transform(slice_fit, df[['revisionSubaction']]        , FrequencyTransformer())  # noqa
    df['revisionTagsFreq']          = transform(slice_fit, df[['revisionTags']]             , FrequencyTransformer())  # noqa
