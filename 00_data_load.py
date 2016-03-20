import pickle
import numpy as np
from pandas import DataFrame

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
