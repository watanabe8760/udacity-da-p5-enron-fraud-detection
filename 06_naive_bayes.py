"""
Pipelines
"""
# Original features
pipe_org = make_pipeline(Imputer(strategy='median'),
                         GaussianNB())
# Features selected by p-values
pipe_fpr = make_pipeline(Imputer(strategy='median'),
                         SelectFpr(),
                         GaussianNB())
# Features selected by PCA
pipe_pca = make_pipeline(Imputer(strategy='median'),
                         PCA(),
                         GaussianNB())

"""
Parameters
"""
param_fpr = {'selectfpr__score_func': [f_classif],
             'selectfpr__alpha': [0.05, 0.06, 0.07, 0.08, 0.09]}
param_pca = {'pca__n_components': list(xrange(10, 20, 2)),
             'pca__whiten': [True]}

"""
Grid Search and Evaluation
"""
find_best_model(pipe_org, {}, '06_naive_bayes')
find_best_model(pipe_fpr, param_fpr, '06_naive_bayes_fpr')
find_best_model(pipe_pca, param_pca, '06_naive_bayes_pca')
