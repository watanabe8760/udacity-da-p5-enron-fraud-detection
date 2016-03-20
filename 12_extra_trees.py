"""
Pipelines
"""
# Original features
pipe_org = make_pipeline(Imputer(strategy='median'),
                         ExtraTreesClassifier())
# Features selecttion
pipe_fpr = make_pipeline(Imputer(strategy='median'),
                         SelectFpr(),
                         ExtraTreesClassifier())
# PCA
pipe_pca = make_pipeline(Imputer(strategy='median'),
                         PCA(),
                         ExtraTreesClassifier())

"""
Parameters
"""
param_clf = [{'extratreesclassifier__n_estimators': [10, 20, 30, 40, 50],
              'extratreesclassifier__criterion': ['gini', 'entropy'],
              'extratreesclassifier__max_features': ['sqrt'],
              'extratreesclassifier__max_depth': [None],
              'extratreesclassifier__min_samples_split': [2, 3, 4],
              'extratreesclassifier__min_samples_leaf': [1, 2, 3],
              'extratreesclassifier__min_weight_fraction_leaf': [0.0],
              'extratreesclassifier__max_leaf_nodes': [None],
              'extratreesclassifier__bootstrap': [True],
              'extratreesclassifier__oob_score': [True],
              'extratreesclassifier__n_jobs': [-1],
              'extratreesclassifier__random_state': [20160308],
              'extratreesclassifier__class_weight':
                  ['balanced', 'balanced_subsample']},
             {'extratreesclassifier__n_estimators': [10, 20, 30, 40, 50],
              'extratreesclassifier__criterion': ['gini', 'entropy'],
              'extratreesclassifier__max_features': ['sqrt'],
              'extratreesclassifier__max_depth': [None],
              'extratreesclassifier__min_samples_split': [2, 3, 4],
              'extratreesclassifier__min_samples_leaf': [1, 2, 3],
              'extratreesclassifier__min_weight_fraction_leaf': [0.0],
              'extratreesclassifier__max_leaf_nodes': [None],
              'extratreesclassifier__bootstrap': [False],
              'extratreesclassifier__oob_score': [False],
              'extratreesclassifier__n_jobs': [-1],
              'extratreesclassifier__random_state': [20160308],
              'extratreesclassifier__class_weight':
                  ['balanced', 'balanced_subsample']}]
param_fpr = {'selectfpr__score_func': [f_classif],
             'selectfpr__alpha': [0.05]}
param_pca = {'pca__n_components': list(xrange(14, 18, 2)),
             'pca__whiten': [True]}


"""
Grid Search and Evaluation
"""
find_best_model(pipe_org, param_clf, '12_extra_trees')
find_best_model(pipe_fpr, dict(param_fpr.items() + param_clf[1].items()),
                '12_extra_trees_fpr')
find_best_model(pipe_pca, dict(param_pca.items() + param_clf[1].items()),
                '12_extra_trees_pca')
