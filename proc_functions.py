# proc_functions.py

import numpy as np
import pandas as pd

def get_summary(col, targ):
    cnts = col.value_counts(normalize=True)
    num_unique = len(cnts)
    top_freq = cnts.values[0]
    top_item = cnts.index[0]
    fraction_missing = col.isnull().mean()
    first_few = ', '.join(list(col.unique()
                               .astype(str)[:max(num_unique, 3)]))

    # Get target frequency spread
    temp = pd.concat([col, targ],axis=1)
    grouped = temp.groupby(col.name)['target'].mean()
    freq_range = grouped.max() - grouped.min()


    return pd.Series(
        {'NumUnique' : num_unique,
         'Top Freq' : top_freq,
         'Top Item' : top_item,
         'Fraction Missing' : fraction_missing,
         'Examples' : first_few,
         'Freq Range' : freq_range
        })
