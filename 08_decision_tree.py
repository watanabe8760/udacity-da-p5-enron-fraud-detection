"""
Pipelines
"""
# Original features
pipe_org = make_pipeline(Imputer(strategy='median'),
                         DecisionTreeClassifier())
# Features selecttion
pipe_fpr = make_pipeline(Imputer(strategy='median'),
                         SelectFpr(),
                         DecisionTreeClassifier())
# PCA
pipe_pca = make_pipeline(Imputer(strategy='median'),
                         PCA(),
                         DecisionTreeClassifier())

"""
Parameters
"""
param_clf = {'decisiontreeclassifier__criterion': ['gini', 'entropy'],
             'decisiontreeclassifier__splitter': ['best', 'random'],
             'decisiontreeclassifier__max_features':
                 [0.2 * i for i in xrange(1, 6)],
             'decisiontreeclassifier__max_depth': [None],
             'decisiontreeclassifier__min_samples_split': [2, 3, 4],
             'decisiontreeclassifier__min_samples_leaf': [1, 2, 3],
             'decisiontreeclassifier__max_leaf_nodes': [None],
             'decisiontreeclassifier__class_weight': ['balanced'],
             'decisiontreeclassifier__random_state': [20160308],
             'decisiontreeclassifier__presort': [True]}
param_fpr = {'selectfpr__score_func': [f_classif],
             'selectfpr__alpha': [0.05, 0.07, 0.09]}
param_pca = {'pca__n_components': list(xrange(10, 20, 2)),
             'pca__whiten': [True]}

"""
Grid Search and Evaluation
"""
find_best_model(pipe_org, param_clf, '08_decision_tree')
find_best_model(pipe_fpr, dict(param_fpr.items() + param_clf.items()),
                '08_decision_tree_fpr')
find_best_model(pipe_pca, dict(param_pca.items() + param_clf.items()),
                '08_decision_tree_pca')
