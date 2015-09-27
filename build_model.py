# build_model.py
import pandas as pd
import numpy as np
from process_numeric import prep_numeric
from process_categorical import prep_categorical
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


if __name__ == '__main__':

    # Load data
    train = pd.read_csv('data/train.csv').set_index('ID')
    target = train.pop('target')
    test = pd.read_csv('data/test.csv').set_index('ID')
    train_idx = train.index
    test_idx = test.index
    df = pd.concat([train, test])

    # Process numeric
    print 'process numeric'
    dtypes = df.dtypes
    numeric_vars = df.loc[:, (dtypes=='int64') | (dtypes=='float64')]
    num_train, num_test = prep_numeric(numeric_vars, train_idx, test_idx)

    # Process categorical
    print 'process categorical'
    df['target'] = 0
    df.target[train.index] = target
    cat_train, cat_test = prep_categorical(df, train_idx, test_idx)

    # Global train/test
    print 'global train/test'
    X_train = pd.concat([num_train, cat_train], axis=1)
    X_test = pd.concat([num_test, cat_test], axis=1)
    y_train = target

    # Model
    print 'build model'
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    pred = gbc.predict_proba(X_test)[:,1]

    # Write results
    results = pd.DataFrame({
        'ID': X_test.index,
        'target': pred})
    results.to_csv('data/submission.csv', index=False)
