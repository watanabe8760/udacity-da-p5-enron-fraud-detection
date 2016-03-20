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
