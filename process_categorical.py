import pdb
import numpy as np
import pandas as pd
from collections import defaultdict
from statsmodels.stats.proportion import proportions_ztest
from sklearn.preprocessing import OneHotEncoder, Imputer


#  ------- Categorize each cat variable ---------------
# From looking at summaries in the notebook
var_types = {}

# Straightforward. These guys have a handful of categories, with a mix of frequencies, and no obvious interpretation
var_types['straightforward'] = [1, 5, 226, 232, 283, 305, 325, 352, 353, 354, 466, 467, 1934]

# Virtually all the same  value (and no dicernable difference in outcome)
var_types['useless'] = [8, 9, 10, 11, 12, 43, 44, 196, 202, 216, 222, 229, 239]

# Virtually all the same value, but with a difference in outcome
var_types['almost_useless'] = [214, 230]

# DATES
# Note that many of these have up to 90+% Missing
# At least the formats look the same
var_types['dates'] = [73, 75, 156, 157, 158, 159, 166, 167, 168, 169, 176, 177, 178, 179, 204, 217]

# Thinly spread - needs work
# 200 - cities
# 237, 274 - states
var_types['thin'] = [200, 237, 274, 342, 404]

# Convert to actual column names
for key, val in var_types.iteritems():
    var_types[key] = ['VAR_%04.f' % v for v in val]
# -----------------------------------------------------------

class CategoryEncoder(object):
    '''
    Perform one-hot-encoding of categorical variables.
    Keep track of the actual levels for each variable
    '''

    def __init__(self):
        pass

    def fit(self, df):
        '''
        Input: DataFrame containing categorical features to be encoded
        Output: (DataFrame of encodings, Dictionary of encodings)
        '''

        # Make sure variable types are all strings
        df = df.replace(np.nan, 'NA')
        df = df.apply(lambda col: col.astype(str))

        # Use a dictionary to store the feature values
        self.feature_key = {}
        for col in df:
            uniq_values = df[col].unique()
            self.feature_key[col] = defaultdict(lambda: len(uniq_values), [(val, i) for i, val in enumerate(uniq_values)])

        # Make new dataframe containing value labels
        new_df = self.labels_to_numbers(df)

        # One Hot encodings
        self.encoding = OneHotEncoder()
        self.encoding.fit(new_df)


    def labels_to_numbers(self, df):
            new = [df[col].apply(lambda val: self.feature_key[col][val]) for col in df]
            return pd.DataFrame(new).T

    def transform(self, df):
        '''
        One hot encoding of df
        '''
        # Make sure variable types are all strings
        df = df.replace(np.nan, 'NA')
        df = df.apply(lambda col: col.astype(str))
        new_df = self.labels_to_numbers(df)
        return pd.DataFrame(self.encoding.transform(new_df).toarray(), index=df.index)

class ConvertZScore(object):

    def __init__(self, thresh):
        self.thresh = thresh
        self.to_bin = {}

    def get_pvals(self, col, target):

        target_prop = target.mean()  # Baseline
        level_counts = col.value_counts()  # counts

        # Bin category levels that account for less than <thresh> of the total data
        to_bin = level_counts.index[level_counts < self.thresh]
        col[col.isin(to_bin)] = 'Other'

        # Record the bins
        self.to_bin[col.name] = to_bin

        # Get a p-value for each proportion
        df = pd.concat([col, target], axis=1)
        agg = df.groupby(col.name)['target'].aggregate({
                'count': lambda x: x.sum(),
                'nobs': lambda x: x.count()
        })
        zscores = agg.apply(lambda x: proportions_ztest(x['count'], x['nobs'], target_prop)[0], axis=1)

        return defaultdict(int, zscores)

    def fit(self, df, target):
        '''
        Only apply this to the training set
        '''

        # Get a dictionary of zscores for each column
        self.score_dicts = {}
        for col in df:
            self.score_dicts[col] = self.get_pvals(df[col], target)

    def transform(self, df):
        '''
        Convert each of the categorical columns that are sliced too thin
        '''
        for col in df:
            df.loc[df[col].isin(self.to_bin[col]), col] = 'Other'
            df[col] = df[col].apply(lambda x: self.score_dicts[col][x])
        return df


class ImputeMissing(object):

    def __init__(self, vartype='date'):
        self.vartype = vartype

    def fit(self, df):
        if self.vartype == 'date':
            self.fit_date(df)
        elif self.vartype == 'numeric':
            self.fit_numeric(df)
        else:
            print 'Must specify vartype'

    def fit_date(self, df):
        pass

    def fit_numeric(self, df):
        pass

    def transform(self, df):
        pass


def convert_dates(date_col):
    '''
    Convert a date column to date format
    '''
    return pd.to_datetime(date_col, format='%d%b%y:%H:%M:%S')


def convert_all_date_columns(df):
    '''
    Convert all date columns
    '''
    date_df = df[var_types['dates']].apply(convert_dates)
    return date_df.astype(int)

def prep_geography(df):
    '''
    Input is the entire DataFrame.
    Output is just the engineered columns
    '''
    # Just need to check if the states are equal
    states = ['VAR_0237', 'VAR_0274']
    out = pd.Series(df[states[0]] == df[states[1]])
    out.name = 'states_equal'
    return pd.DataFrame(out)

def engineer_dates(df):
    # Take differences
    diff_df = [('-'.join([col1, col2]), df[col1] - df[col2]) for col1 in df for col2 in df if col1 < col2]
    diff_df = pd.DataFrame.from_dict(dict(diff_df))

    # Min/max/num present
    summary_df = pd.DataFrame({
        'max_date': df.apply(np.max, axis=1),
        'min_date' : df.apply(np.min, axis=1),
        'num_missing': df.apply(lambda row: row.isnull().sum(), axis=1)
        })
    summary_df['date_range'] = summary_df['max_date'] - summary_df['min_date']

    return pd.concat([summary_df, diff_df],axis=1).astype(int)


def prep_categorical(df, train_idx, test_idx):

    # Convert all dates
    date_df = convert_all_date_columns(df)

    # Get engineered vars
    eng_date_df = engineer_dates(date_df)
    eng_geo_df = prep_geography(df)

    # Train/test split
    train = df.loc[train_idx, :]
    test = df.loc[test_idx, :]

    # Thinly sliced columns
    zconv = ConvertZScore(thresh = 200)
    zconv.fit(train[var_types['thin']], train['target'])
    thin_train_df = zconv.transform(train[var_types['thin']])
    thin_test_df = zconv.transform(test[var_types['thin']])

    # One-hot encoding of the rest
    enc = CategoryEncoder()
    enc.fit(train[var_types['straightforward']])
    ohe_train = enc.transform(train[var_types['straightforward']])
    ohe_test = enc.transform(test[var_types['straightforward']])

    # Combine into train/test dataframes
    train = pd.concat([
        date_df.loc[train_idx],
        eng_date_df.loc[train_idx],
        eng_geo_df.loc[train_idx],
        thin_train_df,
        ohe_train], axis=1)
    test = pd.concat([
        date_df.loc[test_idx],
        eng_date_df.loc[test_idx],
        eng_geo_df.loc[test_idx],
        thin_test_df,
        ohe_test], axis=1)
    #
    # # Impute Missing
    # imp = Imputer()
    # train = imp.fit_transform(train)
    # test = imp.transform(test)

    return train, test
