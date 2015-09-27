# process_numeric.py
import pdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer


def prep_numeric(df, train_idx, test_idx):

    # Remove features of int_df that only have a single value
    num_unique = df.apply(lambda x: x.nunique())
    df = df[num_unique[num_unique > 1].index]
    num_missing = df.apply(lambda x: x.isnull().sum())
    df = df[num_missing[num_missing < df.shape[0]].index]
    features = df.columns

    # Replace 'weird' values with Missing
    thresholds = df.apply(np.median) * 10
    df = df.apply(lambda col: custom_replace(col, thresholds[col.name], 500))
    # pdb.set_trace()
    # Split into train/test
    train = df.loc[train_idx]
    test = df.loc[test_idx]

    # # Impute missing values
    # imp = Imputer(strategy='mean')
    # train = imp.fit_transform(train)
    # test = imp.transform(test)

    train = train.fillna(0)
    test = test.fillna(0)

    return train, test

def custom_replace(x, val_thresh, cnt_thresh):
    '''
    Replace values in x that
        (a) are larger than thresh and
        (b) occur more than cnt_thresh times
    '''
    large = x.loc[x > val_thresh]
    cnts = large.value_counts()
    if len(cnts) >= 1000:
        return x
    to_replace = cnts.loc[cnts > cnt_thresh].index
    if len(to_replace)>0:
        return x.replace(to_replace, np.nan)
    return x
