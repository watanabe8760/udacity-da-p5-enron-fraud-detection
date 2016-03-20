def check_test_set(model, x, y):
    """
    Test data set evaluation utility.
    Premise:
        1. model is already fitted.
        2. x, y are test data set.
    """
    y_pred = model.predict(x)
    print crosstab(y, y_pred, rownames=['Actual'], colnames=['Predicted'])
    print ''
    print classification_report(y, y_pred), '\n'
    return {'precision': round(precision_score(y, y_pred), 3),
            'recall': round(recall_score(y, y_pred), 3),
            'f1': round(f1_score(y, y_pred), 3)}

"""
Pipelines
"""
# Original features
pipe_org = make_pipeline(Imputer(strategy='median', axis=0),
                         GaussianNB())
# Features selected by p-values
pipe_fpr = make_pipeline(Imputer(strategy='median', axis=0),
                         SelectFpr(f_classif, alpha=0.05),
                         GaussianNB())
# Features selected by PCA
n_pca = 14
pipe_pca = make_pipeline(Imputer(strategy='median', axis=0),
                         PCA(n_components=n_pca, whiten=True),
                         GaussianNB())

"""
Training and Cross Validation
"""
sss = StratifiedShuffleSplit(df['poi'], n_iter=10,
                             test_size=0.3, random_state=20160305)
result_org = DataFrame(columns=['precision', 'recall', 'f1'])
result_fpr = DataFrame(columns=['precision', 'recall', 'f1'])
result_pca = DataFrame(columns=['precision', 'recall', 'f1'])
for train_idx, test_idx in sss:
    # Separate data into training and test sets
    x_train = df.iloc[train_idx][F_ALL_NEW]
    x_test = df.iloc[test_idx][F_ALL_NEW]
    y_train = df.iloc[train_idx]['poi']
    y_test = df.iloc[test_idx]['poi']
    # Fit models
    pipe_org.fit(x_train, y_train)
    pipe_fpr.fit(x_train, y_train)
    pipe_pca.fit(x_train, y_train)
    # Evaluate models and store results
    result_org = result_org.append(check_test_set(pipe_org, x_test, y_test),
                                   ignore_index=True)
    result_fpr = result_fpr.append(check_test_set(pipe_fpr, x_test, y_test),
                                   ignore_index=True)
    result_pca = result_pca.append(check_test_set(pipe_pca, x_test, y_test),
                                   ignore_index=True)

for result in [result_org, result_fpr, result_pca]:
    print '[mean]'
    print np.round(np.mean(result, axis=0), 3)
    print '[std]'
    print np.round(np.std(result, axis=0), 3)
    print ''


"""
[Test] whether imputation in pipeline works as expected
"""
def impute_expected(x_train, x_test):
    """
    Expected imputation functionality in pipeline.
    """
    imputer = Imputer(strategy='median', axis=0)
    imputer.fit(x_train)
    x_train = DataFrame(imputer.transform(x_train),
                        index=x_train.index.values,
                        columns=x_train.columns.values)
    x_test = DataFrame(imputer.transform(x_test),
                       index=x_test.index.values,
                       columns=x_test.columns.values)
    return x_train, x_test

clf = GaussianNB()
result_tst = DataFrame(columns=['precision', 'recall', 'f1'])
for train_idx, test_idx in sss:
    # Separate data into training and test sets
    x_train = df.iloc[train_idx][F_ALL_NEW]
    x_test = df.iloc[test_idx][F_ALL_NEW]
    y_train = df.iloc[train_idx]['poi']
    y_test = df.iloc[test_idx]['poi']
    # Clean up features
    x_train, x_test = impute_expected(x_train, x_test)
    # Fit model
    clf.fit(x_train, y_train)
    # Evaluate model and store result
    result_tst = result_tst.append(check_test_set(clf, x_test, y_test),
                                   ignore_index=True)

# Comfirm if the results are indentical
print result_org.equals(result_tst)


"""
[Test] whether SelectFpr in pipeline works as expected
"""
def SelectFpr_expected(x_train, x_test):
    """
    Expected SelectFpr functionality in pipeline.
    """
    select_fpr = SelectFpr(f_classif, alpha=0.05)
    select_fpr.fit(x_train, y_train)
    x_train_fpr = select_fpr.transform(x_train)
    x_test_fpr = select_fpr.transform(x_test)
    print 'Number of features selected:', sum(select_fpr.pvalues_ < 0.05)
    return x_train_fpr, x_test_fpr

clf = GaussianNB()
result_tst = DataFrame(columns=['precision', 'recall', 'f1'])
for train_idx, test_idx in sss:
    # Separate data into training and test sets
    x_train = df.iloc[train_idx][F_ALL_NEW]
    x_test = df.iloc[test_idx][F_ALL_NEW]
    y_train = df.iloc[train_idx]['poi']
    y_test = df.iloc[test_idx]['poi']
    # Clean up features
    x_train, x_test = impute_expected(x_train, x_test)
    x_train, x_test = SelectFpr_expected(x_train, x_test)
    # Fit model
    clf.fit(x_train, y_train)
    # Evaluate model and store result
    result_tst = result_tst.append(check_test_set(clf, x_test, y_test),
                                   ignore_index=True)

# Comfirm if the results are indentical
print result_fpr.equals(result_tst)


"""
[Test] whether PCA in pipeline works as expected
"""
def PCA_expected(x_train, x_test):
    pca = PCA(n_components=n_pca, whiten=True)
    pca.fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)
    print 'Variance explained:', sum(pca.explained_variance_ratio_[xrange(n_pca)])
    return x_train_pca, x_test_pca

clf = GaussianNB()
result_tst = DataFrame(columns=['precision', 'recall', 'f1'])
for train_idx, test_idx in sss:
    # Separate data into training and test sets
    x_train = df.iloc[train_idx][F_ALL_NEW]
    x_test = df.iloc[test_idx][F_ALL_NEW]
    y_train = df.iloc[train_idx]['poi']
    y_test = df.iloc[test_idx]['poi']
    # Clean up features
    x_train, x_test = impute_expected(x_train, x_test)
    x_train, x_test = PCA_expected(x_train, x_test)
    # Fit model
    clf.fit(x_train, y_train)
    # Evaluate model and store result
    result_tst = result_tst.append(check_test_set(clf, x_test, y_test),
                                   ignore_index=True)

# Comfirm if the results are indentical
print result_pca.equals(result_tst)
