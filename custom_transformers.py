import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.exceptions import NotFittedError

from sklearn.base import BaseEstimator, TransformerMixin # for creating custom transformers based on sklearn linrary
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(logging.Formatter(fmt='[%(asctime)s: %(funcName)s: %(levelname)s] %(message)s'))
logger.addHandler(handler)


#=====================================================

class DFColumnsDropper(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None) -> None:
        #super().__init__()
        if type(columns) != type(list()):
            self.columns = [columns]
        else:
            self.columns = columns


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        if self.columns is None:
            return X
        cols_to_drop = []
        for col in self.columns:
            if col in X.columns:
                cols_to_drop.append(col)

        return X.drop(columns=cols_to_drop)

#=====================================================

class DFValuesReplacer(BaseEstimator, TransformerMixin):
    """
    """
    def __init__(self, replaces=None) -> None:
        """
        replaces should be the dict {'column_name': {value_to_replace: value_to_replace_with}}
        """
        self.replaces = replaces # replaces should be in form {'column1': {'value_to_find': 'value_to_replace_with'}, ...}
        pass


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        if self.replaces is None or self.replaces == {}:
            return X

        X_transformed = X.copy()
        #for c, val in X_transformed.columns:
        X_transformed = X_transformed.replace(self.replaces)
        return X_transformed

#=====================================================

class DFWoeEncoder(BaseEstimator, TransformerMixin):
    """
    Implementation of Weight-of-Evidence (WoE) encoder
    WoE is nuanced view towards the relationship between a categorical independent variable and a dependent variable
    The mathematical definition of Weight of Evidence is the natural log of the odds ratio:
    WoE = ln ( %_of_positive / %_of_negative )
    WoE is supervised encoder and it has to be fitted before used
    """
    def __init__(self) -> None:
        #super().__init__()
        pass


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        return X

#=====================================================

class DFCrossFeaturesImputer(BaseEstimator, TransformerMixin):
    """
    For example, I want to fill in missing education level. But I know corresponding job (which is 'housemaid')
    I want to get the most frequent education level among all who have a 'housemaid' job 
    I'm going to pass a dict {'education': 'job'} => fill column 'education' with most frequent values for value in job
    """
    def __init__(self, cross_features=None, strategy='most_frequent', nan_equiv=np.nan) -> None: #, logger=logger
        if cross_features is None:
            raise('NoImputeObject')
        self.nan_equiv = nan_equiv
        self.strategy = strategy
        self.cross_features = cross_features
        self._imputers = None
        self._is_fitted = False
        #self._logger = logger
        pass


    def fit(self, X, y=None):
        if self.cross_features is None:
            #self._logger.debug('No columns passed')
            return self
        self._imputers = {}
        # target_feat is column which contains missing values
        #self._logger.debug('Internal imputers initialized. Beginning loop through column pairs')
        for target_feat, base_feat in self.cross_features.items():
            #self._logger.debug('\tBeginning training imputer for `{}` using `{}` feature'.format(target_feat, base_feat))
            self._imputers[target_feat] = {}
            # crosstab's 1st arg - rows, 2nd - columns
            #ct = pd.crosstab(df['education'], df['job'])
            try:
                is_nan = np.isnan(self.nan_equiv)
            except TypeError:
                #self._logger.debug('\tCaptured TypeError while checking if `nan_equiv` is np.nan => is_nan = False')
                is_nan = False
            if not is_nan:
                #self._logger.debug('\tCreating crosstable using explicit filter')
                ct = pd.crosstab(X[X[target_feat]!=self.nan_equiv][target_feat], X[X[base_feat]!=self.nan_equiv][base_feat])
            else:
                #self._logger.debug('\tCreating crosstable without explicit filter')
                ct = pd.crosstab(X[target_feat], X[base_feat])
            # define most frequent values for each base_feat (in example - for each job possible)
            #self._logger.debug('\tCrosstable:')
            #self._logger.debug(ct)
            #self._logger.debug('\tDefining imputing values in `{}`'.format(base_feat))
            for base_val in ct.columns:
                # calculate max value for the column
                max_freq = ct[base_val].max()
                #self._logger.debug('\t\tValue: {}, max frequency = {}'.format(base_val, max_freq))
                #if base_val != self.nan_equiv:
                # index (education value) corresponding to most_frequent value
                self._imputers[target_feat][base_val] = ct.index[ct[base_val]==max_freq][0] #get 1st value as max even if there are several equal values
        self._is_fitted = True
        return self


    def _map_values(self, x, target_feat, base_feat, map_values):
        repl_val = map_values.get(x[base_feat], None)
        if repl_val is not None:
            x[target_feat] = repl_val
        return x


    def transform(self, X, y=None):
        if not self._is_fitted:
            raise('NotFittedError')
        try:
            is_nan = np.isnan(self.nan_equiv)
        except TypeError:
            #self._logger.debug('\tCaptured TypeError while checking if `nan_equiv` is np.nan => is_nan = False')
            is_nan = False
        X_mod = X.copy()
        for target_feat, base_feat in self.cross_features.items():
            map_values = self._imputers[target_feat]
            if is_nan:
                m = X_mod[target_feat].isna()
            else:
                m = X_mod[target_feat]==self.nan_equiv
            X_mod.loc[m, :] = X_mod.loc[m, :].apply(self._map_values, axis=1, args=(target_feat, base_feat, map_values))
        return X_mod
        