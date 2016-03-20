import seaborn as sns
import matplotlib.pyplot as plt


"""
Exploratory Analysis
"""
print df.shape
print df.isnull().sum() / df.shape[0]
print df.describe()
print df.corr()

# Heapmap of correlation
sns.heatmap(df.corr(), square=True, annot=True, fmt='.2f')
plt.xticks(rotation=90)
plt.yticks(rotation=0)

# Scatter plot of high correlation combinations
plt.scatter(df['to_messages'], df['shared_receipt_with_poi'])        # 0.85
plt.scatter(df['loan_advances'], df['total_payments'])               # 0.97
plt.scatter(df['exercised_stock_options'], df['total_stock_value'])  # 0.96

# Confirm number of zero which might cause high correlation
for i in xrange(len(F_EMAIL)):
    print F_EMAIL[i], sum(df[F_EMAIL[i]] == 0)
for i in xrange(len(F_FINANCE)):
    print F_FINANCE[i], sum(df[F_FINANCE[i]] == 0)

print sum(df['to_messages'] == 0)
print sum(df['shared_receipt_with_poi'] == 0)
print sum(df['loan_advances'] == 0)
print sum(df['total_payments'] == 0)
print sum(df['exercised_stock_options'] == 0)
print sum(df['total_stock_value'] == 0)

# Histogram
for i in xrange(len(F_ALL)):
    plt.subplot(4, 5, i+1)
    plt.hist(df[F_ALL[i]].dropna(), bins=20)
    plt.title(F_ALL[i])

# Histogram (log scaled)
for i in xrange(len(F_ALL)):
    plt.subplot(4, 5, i+1)
    plt.hist(df[F_ALL[i]].dropna(), bins=20, log=True)
    plt.title(F_ALL[i])


"""
Exploratory Analysis for new Email Features
"""
# Histogram
for i in xrange(len(F_EMAIL_NEW)):
    plt.subplot(1, 3, i+1)
    plt.hist(df[F_EMAIL_NEW[i]].dropna(), bins=20)
    plt.title(F_EMAIL_NEW[i])

# Histogram (log scaled)
for i in xrange(len(F_EMAIL_NEW)):
    plt.subplot(1, 3, i+1)
    plt.hist(df[F_EMAIL_NEW[i]].dropna(), bins=20, log=True)
    plt.title(F_EMAIL_NEW[i])
