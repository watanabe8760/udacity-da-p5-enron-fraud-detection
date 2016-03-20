"""
Pipelines
"""
# Original features
pipe_org = make_pipeline(Imputer(strategy='median'),
                         AdaBoostClassifier())
# Features selecttion
pipe_fpr = make_pipeline(Imputer(strategy='median'),
                         SelectFpr(),
                         AdaBoostClassifier())
# PCA
pipe_pca = make_pipeline(Imputer(strategy='median'),
                         PCA(),
                         AdaBoostClassifier())

"""
Parameters
"""
# Discrete SAMME AdaBoost adapts based on errors in predicted class labels.
# Real SAMME.R uses the predicted class probabilities.
param_clf = {'adaboostclassifier__base_estimator': [DecisionTreeClassifier()],
             'adaboostclassifier__n_estimators': [50, 100, 150],
             'adaboostclassifier__learning_rate': [0.5, 0.8, 1.0, 1.2, 1.5],
             'adaboostclassifier__algorithm': ['SAMME', 'SAMME.R'],
             'adaboostclassifier__random_state': [20160308]}
param_fpr = {'selectfpr__score_func': [f_classif],
             'selectfpr__alpha': [0.05, 0.07, 0.09]}
param_pca = {'pca__n_components': list(xrange(10, 20, 2)),
             'pca__whiten': [True]}

"""
Grid Search and Evaluation
"""
find_best_model(pipe_org, param_clf, '13_adaboost')
find_best_model(pipe_fpr, dict(param_fpr.items() + param_clf.items()),
                '13_adaboost_fpr')
find_best_model(pipe_pca, dict(param_pca.items() + param_clf.items()),
                '13_adaboost_pca')
