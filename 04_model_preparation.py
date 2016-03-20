"""
Library
"""
# Vizualization
import seaborn as sns
import matplotlib.pyplot as plt
# Feature Transformation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import SelectFpr, f_classif
from sklearn.decomposition import PCA
# Cross Validation
from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
# Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier,\
                             GradientBoostingClassifier, RandomForestClassifier
# Evaluation
from pandas import crosstab
from sklearn.metrics import classification_report, f1_score,\
                            precision_score, recall_score

# Define features and label
X = df[F_ALL_NEW]
y = df['poi']

# Splitter for grid search
splitter = StratifiedShuffleSplit(y, n_iter=10, random_state=201603)
# Splitter for evaluation
splitter_ = StratifiedShuffleSplit(y, n_iter=100, random_state=42)


def save_dist(name, f1, precision, recall):
    # Set outline
    plt.figure(figsize=(18, 6))
    plt.suptitle(name)
    # Plot f1 scores
    plt.subplot(1, 3, 1)
    ax1 = sns.distplot(f1, bins=10,
                       color=sns.color_palette('muted')[0], norm_hist=True)
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 10)
    ax1.set_xlabel('f1')
    ax1.set_ylabel('density')
    ax1.vlines(np.mean(f1), 0, 10, color=sns.color_palette('dark')[0])
    # Plot precision scores
    plt.subplot(1, 3, 2)
    ax1 = sns.distplot(precision, bins=10,
                       color=sns.color_palette('muted')[1], norm_hist=True)
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 10)
    ax1.set_xlabel('precision')
    ax1.set_ylabel('density')
    ax1.vlines(np.mean(precision), 0, 10, color=sns.color_palette('dark')[1])
    # Plot recall scores
    plt.subplot(1, 3, 3)
    ax1 = sns.distplot(recall, bins=10,
                       color=sns.color_palette('muted')[2], norm_hist=True)
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 10)
    ax1.set_xlabel('recall')
    ax1.set_ylabel('density')
    ax1.vlines(np.mean(recall), 0, 10, color=sns.color_palette('dark')[2])
    # Save plots
    plt.savefig('./fig/' + name + '.png')
    plt.close()


def evaluate(model, name):
    """
    Evaluates model by cross validation.
    """
    # Get scores through cross validation
    score_f1 = cross_val_score(model, X, y, scoring='f1', cv=splitter_)
    score_pr = cross_val_score(model, X, y, scoring='precision', cv=splitter_)
    score_re = cross_val_score(model, X, y, scoring='recall', cv=splitter_)
    # Save image of score distributions
    save_dist(name, score_f1, score_pr, score_re)
    # Compute mean and std of each score
    result = DataFrame(index=['f1', 'precision', 'recall'],
                       columns=['mean', 'std'])
    result.loc['f1', 'mean'] = np.mean(score_f1)
    result.loc['precision', 'mean'] = np.mean(score_pr)
    result.loc['recall', 'mean'] = np.mean(score_re)
    result.loc['f1', 'std'] = np.std(score_f1)
    result.loc['precision', 'std'] = np.std(score_pr)
    result.loc['recall', 'std'] = np.std(score_re)
    print model
    print result


def find_best_model(pipe, param, name):
    """
    Finds the best estimator for the pipeline by grid search.
    """
    model = GridSearchCV(pipe, param, scoring='f1', cv=splitter)
    model.fit(X, y)
    # Evaluate the best estimator
    evaluate(model.best_estimator_, name)
    return model.best_estimator_
