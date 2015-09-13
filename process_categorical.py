import numpy as np
import pandas as pd
from collections import defaultdict

##############################################################
# This is a long, messy script for processing each of the 51 categorical variables
#############################################################


#------- Categorize each cat variable ---------------
# From looking at summaries in the notebook
var_types = defaultdict(list)

# Straightforward. These guys have a handful of categories, with a mix of frequencies, and no obvious interpretation
var_types['straightforward'] = [1, 5, 226, 232, 283, 305, 325, 342, 352, 353, 354, 466, 467, 1934]


# Virtually all the same  value (and no dicernable difference in outcome)
var_types['useless'] = [8, 9, 10, 11, 12, 43, 44, 196, 202, 216, 222, 229, 239]


# Virtually all the same value, but with a difference in outcome
var_types['almost_useless'] = [214, 230]


# Dates
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
    date_df = df[var_types['dates']].apply(convert_dates)
    return date_df

def convert_thin(thin_col, thresh):
    '''
    Assign impact scores to each level of the 'thin' columns
    '''
    pass



if __name__ == 'main':
    pass
