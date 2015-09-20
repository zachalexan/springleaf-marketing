import numpy as np
import pandas as pd
from collections import defaultdict
from statsmodels.stats.proportion import proportions_ztest


'''
This is a long, messy script for processing each of the 51 categorical variables
'''

#------- Categorize each cat variable ---------------
# From looking at summaries in the notebook
var_types = defaultdict(list)

# Straightforward. These guys have a handful of categories, with a mix of frequencies, and no obvious interpretation
var_types['straightforward'] = [1, 5, 226, 232, 283, 305, 325, 342, 352, 353, 354, 466, 467, 1934]

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
#-----------------------------------------------------------


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
    return date_df

def convert_thin(thin_col, target, thresh):
    '''
    Assign impact scores to each level of the 'thin' columns

    return a series with the same index as thin_col, but with the values replaced by z-scores.
    '''

    # Temporary dataframe to store local data
    temp_df = pd.DataFrame({'thin' : thin_col,
                            'target' : target})

    target_prop = temp_df.target.mean()  # Baseline
    level_counts = temp_df.thin.value_counts()

    # Bin category levels that account for less than <thresh> of the total data
    to_bin = level_counts.index[level_counts < thresh]
    temp_df['thin'][temp_df.thin.isin(to_bin)] = 'Other'

    # Get a p-value for each proportion
    agg = temp_df.groupby('thin')['target'].aggregate({
        'count' : lambda x: x.sum(),
        'nobs' : lambda x: x.count()
    })
    tscores = agg.apply(lambda x: proportions_ztest(x['count'], x['nobs'], target_prop)[0], axis=1)

    # Replace levels in the original
    temp_df = temp_df.replace({'thin': dict(tscores)})
    return temp_df['thin']

def convert_all_thin_vars(df, thresh, target):
    '''
    Convert each of the categorical columns that are sliced too thin
    '''
    tscore_df = df[var_types['thin']].apply(lambda x: convert_thin(x, target, thresh))

    return tscore_df


def engineer_dates(df):
    # Take differences
    diff_df = [('-'.join([col1,col2]), df[col1] - df[col2]) for col1 in df for col2 in df if not col1==col2]

    # Min/max/num present


    
    return pd.DataFrame.from_dict(dict(diff_df))



if __name__ == 'main':

    # Load data
    train = pd.read_csv('data/train.csv')
    train = train.set_index('ID')

    # dates
    date_df = convert_all_date_columns(train)

    # Geography

    # Thin columns

    # Dummy variables

    # Run model

    pass
