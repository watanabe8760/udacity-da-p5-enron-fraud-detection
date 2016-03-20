"""
Pipelines
"""
# Original features
pipe_org = make_pipeline(Imputer(strategy='median'),
                         GradientBoostingClassifier())
# Features selecttion
pipe_fpr = make_pipeline(Imputer(strategy='median'),
                         SelectFpr(),
                         GradientBoostingClassifier())
# PCA
pipe_pca = make_pipeline(Imputer(strategy='median'),
                         PCA(),
                         GradientBoostingClassifier())

"""
Parameters
"""
param_clf = {'gradientboostingclassifier__loss': ['deviance', 'exponential'],
             'gradientboostingclassifier__learning_rate': [0.01, 0.05,  0.1],
             'gradientboostingclassifier__n_estimators': [75, 100, 125],
             'gradientboostingclassifier__max_depth': [None],
             'gradientboostingclassifier__min_samples_split': [2, 3, 4],
             'gradientboostingclassifier__min_samples_leaf': [1, 2, 3],
             'gradientboostingclassifier__min_weight_fraction_leaf': [0.0],
             'gradientboostingclassifier__subsample': [0.6, 0.8, 1.0],
             'gradientboostingclassifier__max_features': ['sqrt'],
             'gradientboostingclassifier__max_leaf_nodes': [None],
             'gradientboostingclassifier__init': [None],
             'gradientboostingclassifier__random_state': [20160310],
             'gradientboostingclassifier__presort': ['auto']}
param_fpr = {'selectfpr__score_func': [f_classif],
             'selectfpr__alpha': [0.05]}
param_pca = {'pca__n_components': list(xrange(14, 18, 2)),
             'pca__whiten': [True]}

"""
Grid Search and Evaluation
"""
find_best_model(pipe_org, param_clf, '14_gradient_boosting')
find_best_model(pipe_fpr, dict(param_fpr.items() + param_clf.items()),
                '14_gradient_boosting_fpr')
find_best_model(pipe_pca, dict(param_pca.items() + param_clf.items()),
                '14_gradient_boosting_pca')
