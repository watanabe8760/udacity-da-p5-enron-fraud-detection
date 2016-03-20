"""
Pipelines
"""
# Original features
pipe_org = make_pipeline(Imputer(strategy='median'),
                         LogisticRegression())
# Features selecttion
pipe_fpr = make_pipeline(Imputer(strategy='median'),
                         SelectFpr(),
                         LogisticRegression())
# PCA
pipe_pca = make_pipeline(Imputer(strategy='median'),
                         PCA(),
                         LogisticRegression())

"""
Parameters
"""
param_clf = {'logisticregression__penalty': ['l1', 'l2'],
             'logisticregression__C': [10 ** i for i in xrange(-4, 5)],
             'logisticregression__fit_intercept': [True],
             'logisticregression__intercept_scaling':
                 [0.5 + 0.1 * i for i in xrange(0, 11)],
             'logisticregression__class_weight': ['balanced'],
             'logisticregression__solver': ['liblinear'],
             'logisticregression__multi_class': ['ovr'],
             'logisticregression__n_jobs': [-1]}
param_fpr = {'selectfpr__score_func': [f_classif],
             'selectfpr__alpha': [0.05, 0.07, 0.09]}
param_pca = {'pca__n_components': list(xrange(10, 20, 2)),
             'pca__whiten': [True]}

"""
Grid Search and Evaluation
"""
find_best_model(pipe_org, param_clf, '07_logistic_regression')
find_best_model(pipe_fpr, dict(param_fpr.items() + param_clf.items()),
                '07_logistic_regression_fpr')
find_best_model(pipe_pca, dict(param_pca.items() + param_clf.items()),
                '07_logistic_regression_pca')
