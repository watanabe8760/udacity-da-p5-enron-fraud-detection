"""
[Confirmation] Which features are selected by p-values based on FPR test?
"""
fpr = SelectFpr(f_classif, alpha=0.05)
fpr.fit(Imputer(strategy='median').fit_transform(df[F_ALL_NEW]), df['poi'])
features = [(f, round(s, 3), round(p, 4))
            for f, s, p in zip(F_ALL_NEW, fpr.scores_, fpr.pvalues_)
            if p < 0.05]
features = DataFrame.from_records(features, index='feature',
                                  columns=['feature', 'score', 'pvalue'])
features.sort_values(by='score', ascending=False, inplace=True)
print features


"""
[Confirmation] What are the variances captured by PCA components?
"""
pca = PCA(copy=True, n_components=16, whiten=True)
pca.fit(Imputer(strategy='median').fit_transform(df[F_ALL_NEW]), df['poi'])
print np.sort(np.round(pca.explained_variance_ratio_, 3))[::-1]
print sum(pca.explained_variance_ratio_)
