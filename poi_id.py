#!/usr/bin/python

import pickle
import numpy as np
from pandas import DataFrame

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import SelectFpr, f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier

from tester import dump_classifier_and_data

"""
Data Structure Definition
"""
# Feature names - email data
F_EMAIL = ['to_messages', 'from_messages',  'from_poi_to_this_person',
           'from_this_person_to_poi', 'shared_receipt_with_poi']
# Feature names - finance data
F_FINANCE = ['salary', 'bonus', 'long_term_incentive', 'deferred_income',
             'deferral_payments', 'loan_advances', 'other', 'expenses',
             'director_fees', 'total_payments',
             'exercised_stock_options', 'restricted_stock',
             'restricted_stock_deferred', 'total_stock_value']
# All features
F_ALL = F_EMAIL + F_FINANCE
# All column names
COLUMNS = ['poi'] + ['email_address'] + F_EMAIL + F_FINANCE
# Data type of all columns
DTYPE = [bool] + [str] + list(np.repeat(float, 19))

"""
Data Preparation
"""
# Load the dictionary containing the dataset
with open("./input/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
# Convert dictionary into DataFrame
df = DataFrame.from_dict(data_dict, orient='index')
# Reorder columns
df = df.ix[:, COLUMNS]
# Convert data type
for i in xrange(len(COLUMNS)):
    df[COLUMNS[i]] = df[COLUMNS[i]].astype(DTYPE[i], raise_on_error=False)
# Assign 0 to NaN in finance features
# (Assuming that "-" in enron61702insiderpay.pdf means 0.)
for f in F_FINANCE:
    df.loc[df[f].isnull(), f] = 0


# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
# Task 2: Remove outliers
# Task 3: Create new feature(s)
"""
Data Modification (based on outlier confirmation)
"""
# Remove invalid data points
df = df[df.index != 'TOTAL']
df = df[df.index != 'THE TRAVEL AGENCY IN THE PARK']

# Miss-alignment of columns
df.loc['BELFER ROBERT', F_FINANCE] = \
    [0, 0, 0, -102500, 0, 0, 0, 3285,
     102500, 3285, 0, 44093, -44093, 0]
df.loc['BHATNAGAR SANJAY', F_FINANCE] = \
    [0, 0, 0, 0, 0, 0, 0, 137864, 0, 137864,
     15456290, 2604490, -2604490, 15456290]

"""
Feature Engineering - Ratio of Email
"""
df['recieved_from_poi_ratio'] = \
    df['from_poi_to_this_person'] / df['to_messages']
df['sent_to_poi_ratio'] = \
    df['from_this_person_to_poi'] / df['from_messages']
df['shared_receipt_with_poi_ratio'] = \
    df['shared_receipt_with_poi'] / df['to_messages']

# Update column definition
F_EMAIL_NEW = ['recieved_from_poi_ratio', 'sent_to_poi_ratio',
               'shared_receipt_with_poi_ratio']
F_ALL_NEW = F_ALL + F_EMAIL_NEW

"""
Log-scaling for original features
"""
for f in F_ALL:
    df[f] = [np.log(abs(v)) if v != 0 else 0 for v in df[f]]


# Task 4: Try a varity of classifiers
# Task 5: Tune your classifier to achieve better than .3 precision and recall

# Based on my assessment observed in ./output/result_all.txt,
# the best five models are chosen to be tested by tester.py.
# Please look at ./output/result_final.txt for the test result of these five.
# I chose the fourth from the top as the my final model because it has the
# best f1 score.

#pipe = make_pipeline(
#          Imputer(axis=0, copy=True, missing_values='NaN',
#                  strategy='median', verbose=0),
#          PCA(copy=True, n_components=12, whiten=True),
#          LogisticRegression(C=1, class_weight='balanced', dual=False,
#                             fit_intercept=True, intercept_scaling=0.6,
#                             max_iter=100, multi_class='ovr', n_jobs=-1,
#                             penalty='l2', random_state=None,
#                             solver='liblinear', tol=0.0001, verbose=0,
#                             warm_start=False))
#pipe = make_pipeline(
#          Imputer(axis=0, copy=True, missing_values='NaN',
#                  strategy='median', verbose=0),
#          SVC(C=1, cache_size=200, class_weight='balanced', coef0=0.0,
#              decision_function_shape='ovo', degree=3, gamma='auto',
#              kernel='linear', max_iter=-1, probability=False,
#              random_state=20160308, shrinking=False, tol=0.001,
#              verbose=False))
#pipe = make_pipeline(
#          Imputer(axis=0, copy=True, missing_values='NaN',
#                  strategy='median', verbose=0),
#          PCA(copy=True, n_components=18, whiten=True),
#          SVC(C=1, cache_size=200, class_weight='balanced', coef0=0.0,
#              decision_function_shape='ovo', degree=3, gamma='auto',
#              kernel='linear', max_iter=-1, probability=False,
#              random_state=20160308, shrinking=False, tol=0.001,
#              verbose=False))
pipe = make_pipeline(
          Imputer(axis=0, copy=True, missing_values='NaN',
                  strategy='median', verbose=0),
          ExtraTreesClassifier(bootstrap=False, class_weight='balanced',
                               criterion='gini', max_depth=None,
                               max_features='sqrt', max_leaf_nodes=None,
                               min_samples_leaf=3, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=30,
                               n_jobs=-1, oob_score=False,
                               random_state=20160308, verbose=0,
                               warm_start=False))
#pipe = make_pipeline(
#          Imputer(axis=0, copy=True, missing_values='NaN',
#                  strategy='median', verbose=0),
#          SelectFpr(alpha=0.05, score_func=f_classif),
#          ExtraTreesClassifier(bootstrap=False, class_weight='balanced',
#                               criterion='gini', max_depth=None,
#                               max_features='sqrt', max_leaf_nodes=None,
#                               min_samples_leaf=3, min_samples_split=2,
#                               min_weight_fraction_leaf=0.0, n_estimators=30,
#                               n_jobs=-1, oob_score=False,
#                               random_state=20160308, verbose=0,
#                               warm_start=False))

# Task 6: Dump your classifier, dataset, and features_list
dump_classifier_and_data(pipe, df.to_dict(orient='index'), ['poi'] + F_ALL_NEW)
