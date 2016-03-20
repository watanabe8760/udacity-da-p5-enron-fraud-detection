"""
Pipelines
"""
# Original features
pipe_org = make_pipeline(Imputer(strategy='median'),
                         SVC())
# Features selecttion
pipe_fpr = make_pipeline(Imputer(strategy='median'),
                         SelectFpr(),
                         SVC())
# PCA
pipe_pca = make_pipeline(Imputer(strategy='median'),
                         PCA(),
                         SVC())

"""
Parameters
"""
param_clf = [{'svc__kernel': ['linear', 'rbf'],
              'svc__C': [10 ** i for i in xrange(-3, 4)],
              'svc__gamma': ['auto'],
              'svc__coef0': [0.0],
              'svc__probability': [False],
              'svc__shrinking': [False, True],
              'svc__tol': [1e-3],
              'svc__class_weight': ['balanced'],
              'svc__max_iter': [-1],
              'svc__decision_function_shape': ['ovo'],
              'svc__random_state': [20160308]},
             {'svc__kernel': ['poly'],
              'svc__C': [10 ** i for i in xrange(-3, 4)],
              'svc__degree': [2, 3],
              'svc__gamma': ['auto'],
              'svc__coef0': [0.0, 0.1, 0.3, 0.7, 1.0],
              'svc__probability': [False],
              'svc__shrinking': [False, True],
              'svc__tol': [1e-3],
              'svc__class_weight': ['balanced'],
              'svc__max_iter': [-1],
              'svc__decision_function_shape': ['ovo'],
              'svc__random_state': [20160308]},
             {'svc__kernel': ['sigmoid'],
              'svc__C': [10 ** i for i in xrange(-3, 4)],
              'svc__gamma': ['auto'],
              'svc__coef0': [0.0, 0.1, 0.3, 0.7, 1.0],
              'svc__probability': [False],
              'svc__shrinking': [False, True],
              'svc__tol': [1e-3],
              'svc__class_weight': ['balanced'],
              'svc__max_iter': [-1],
              'svc__decision_function_shape': ['ovo'],
              'svc__random_state': [20160308]}]
param_fpr = {'selectfpr__score_func': [f_classif],
             'selectfpr__alpha': [0.05, 0.07, 0.09]}
param_pca = {'pca__n_components': list(xrange(10, 20, 2)),
             'pca__whiten': [True]}

"""
Grid Search and Evaluation
"""
find_best_model(pipe_org, param_clf, '09_svm')
find_best_model(pipe_fpr, dict(param_fpr.items() + param_clf[0].items()),
                '09_svm_fpr_1')
find_best_model(pipe_fpr, dict(param_fpr.items() + param_clf[1].items()),
                '09_svm_fpr_2')
find_best_model(pipe_fpr, dict(param_fpr.items() + param_clf[2].items()),
                '09_svm_fpr_3')
find_best_model(pipe_pca, dict(param_pca.items() + param_clf[0].items()),
                '09_svm_pca_1')
find_best_model(pipe_pca, dict(param_pca.items() + param_clf[1].items()),
                '09_svm_pca_2')
find_best_model(pipe_pca, dict(param_pca.items() + param_clf[2].items()),
                '09_svm_pca_3')
