"""
Pipelines
"""
# Original features
pipe_org = make_pipeline(Imputer(strategy='median'),
                         LinearSVC())
# Features selecttion
pipe_fpr = make_pipeline(Imputer(strategy='median'),
                         SelectFpr(),
                         LinearSVC())
# PCA
pipe_pca = make_pipeline(Imputer(strategy='median'),
                         PCA(),
                         LinearSVC())

"""
Parameters
"""
param_clf = {'linearsvc__C': [10 ** i for i in xrange(-3, 4)],
             'linearsvc__loss': ['hinge', 'squared_hinge'],
             'linearsvc__penalty': ['l2'],
             'linearsvc__tol': [1e-3],
             'linearsvc__multi_class': ['ovr'],
             'linearsvc__fit_intercept': [True],
             'linearsvc__intercept_scaling':
                 [0.5 + 0.1 * i for i in xrange(0, 11)],
             'linearsvc__class_weight': ['balanced'],
             'linearsvc__random_state': [20160308],
             'linearsvc__max_iter': [-1]}
param_fpr = {'selectfpr__score_func': [f_classif],
             'selectfpr__alpha': [0.05, 0.07, 0.09]}
param_pca = {'pca__n_components': list(xrange(10, 20, 2)),
             'pca__whiten': [True]}

"""
Grid Search and Evaluation
"""
find_best_model(pipe_org, param_clf, '10_linear_svm')
find_best_model(pipe_fpr, dict(param_fpr.items() + param_clf.items()),
                '10_linear_svm_fpr')
find_best_model(pipe_pca, dict(param_pca.items() + param_clf.items()),
                '10_linear_svm_pca')
