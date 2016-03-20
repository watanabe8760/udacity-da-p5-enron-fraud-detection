"""
Pipelines
"""
# Original features
pipe_org = make_pipeline(Imputer(strategy='median'),
                         RandomForestClassifier())
# Features selecttion
pipe_fpr = make_pipeline(Imputer(strategy='median'),
                         SelectFpr(),
                         RandomForestClassifier())
# PCA
pipe_pca = make_pipeline(Imputer(strategy='median'),
                         PCA(),
                         RandomForestClassifier())

"""
Parameters
"""
param_clf = {'randomforestclassifier__n_estimators': [8, 9, 10, 11, 12],
             'randomforestclassifier__criterion': ['gini', 'entropy'],
             'randomforestclassifier__max_features': ['sqrt'],
             'randomforestclassifier__max_depth': [None],
             'randomforestclassifier__min_samples_split': [2, 3, 4],
             'randomforestclassifier__min_samples_leaf': [1, 2, 3],
             'randomforestclassifier__max_leaf_nodes': [None],
             'randomforestclassifier__bootstrap': [True],
             'randomforestclassifier__oob_score': [True, False],
             'randomforestclassifier__n_jobs': [-1],
             'randomforestclassifier__random_state': [20160308],
             'randomforestclassifier__class_weight':
                 ['balanced', 'balanced_subsample']}
param_fpr = {'selectfpr__score_func': [f_classif],
             'selectfpr__alpha': [0.05]}
param_pca = {'pca__n_components': list(xrange(14, 18, 2)),
             'pca__whiten': [True]}

"""
Grid Search and Evaluation
"""
find_best_model(pipe_org, param_clf, '11_random_forest')
find_best_model(pipe_fpr, dict(param_fpr.items() + param_clf.items()),
                '11_random_forest_fpr')
find_best_model(pipe_pca, dict(param_pca.items() + param_clf.items()),
                '11_random_forest_pca')
