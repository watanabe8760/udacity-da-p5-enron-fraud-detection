# Enron Fraud Detection

## 1. Abstract

The purpose of this project is to figure out how well machine learning algorithms can indentify Person Of Interest (POI) who were indicted, settled without admitting guilt or testified in exchange for immunity in [Enron scandal](https://en.wikipedia.org/wiki/Enron_scandal).

## 2. Envrionment

```
Python 2.7.11 |Anaconda 2.3.0 (64-bit)| (default, Dec  7 2015, 14:10:42) [MSC v.1500 64 bit (AMD64)]

> conda list
numpy                     1.10.1                   py27_0    defaults
pandas                    0.17.1              np110py27_0    defaults
scikit-learn              0.17                np110py27_1    defaults
seaborn                   0.6.0               np110py27_0    defaults
```

## 3. Files

```
│   00_data_load.py             
│   01_exploratory_analysis_1.py
│   02_preprocess.py            
│   03_exploratory_analysis_2.py
│   04_model_preparation.py     
│   05_trial.py                 
│   06_naive_bayes.py           
│   07_logistic_regression.py   
│   08_decision_tree.py         
│   09_svm.py                   
│   10_linear_svm.py            
│   11_random_forest.py         
│   12_extra_trees.py           
│   13_adaboost.py              
│   14_gradient_boosting.py     
│   15_feature_review.py        
│   poi_id.py                   : Creates and saves final model
│   README.md                   
│   tester.py                   : Tests final model
│
├───fig                         : Stores figures of evaluation results in 06 ~ 14.
│
├───input
│       enron61702insiderpay.pdf            : Financial data source
│       enron61702insiderpay_to_text_1.txt  : Data scraped data from pdf (former half)
│       enron61702insiderpay_to_text_2.txt  : Data scraped data from pdf (latter half)
│       final_project_dataset.pkl           : Initial data set
│
└───output
        my_classifier.pkl   : Final model
        my_dataset.pkl      : Data set used for final model
        my_feature_list.pkl : Feature list used for final model
        result_all.txt      : Results of grid search and evaluation in 06 ~ 14
        result_final.txt    : Test result of tester.py
```

## 4. Data

The initial data is stored in final_project_dataset.pkl which has been developed by [Katie Malone](https://civisanalytics.com/team/katie-malone/) who was an intern of Udacity and built [online machine learning course](https://www.udacity.com/course/intro-to-machine-learning--ud120) during that time. Note, she developed the initial dataset for the sake of learning so that some inconsistencies might be included intentionally and the data validation and cleaning is a part of this project. The initial data set has 146 rows (people) and 21 columns (features).

### 4.1. Features

The data consists of three types of features, label, email and finance.

```
[Label]
1. poi : Indicates who is POI. (1=POI, 0=Non-POI)

[Email]
2. email_address
3. to_messages            : Number of emails the person sent.
4. from_messages          : Number of emails the person received.
5. from_poi_to_this_person: Number of emails the person received from POI.
6. from_this_person_to_poi: Number of emails the person sent to POI.
7. shared_receipt_with_poi: Number of emails the person received with POI.

[Finance]
8. salary
9. bonus
10. long_term_incentive
11. deferred_income
12. deferral_payments
13. loan_advances
14. other
15. expenses
16. director_fees
17. total_payments
18. exercised_stock_options
19. restricted_stock
20. restricted_stock_deferred
21. total_stock_value
```

For the definition of finacial features, please refer to the data source.

### 4.2. Data Source

Each type of features comes from different sources that are available in public.

[Email]

(Original) https://www.cs.cmu.edu/~./enron/  
(Processed) https://github.com/udacity/ud120-projects/tree/master/final_project/emails_by_address

The original data is the famous Enron email dataset. That was processed by Katie and the count from the processed data is used for those features. Since some people do not have email data, email features for those people became NaN.

[Finance]

http://news.findlaw.com/hdocs/docs/enron/enron61702insiderpay.pdf

The features were scraped from the first half of the pdf file. The way of scraping Katie performed is unknown so I did some validation through outlier confirmation. As a result, some unsuccessfully scraped data points were detected. For the details, please refer to next section. Since "-" is interpreted as zero, no NaN is observed in financial features.


### 4.3. Preprosess

Before applying machine learning algorithms, the following four preporsesses are applied. These works are observed in 02_preprocess.py.

#### Invalid data points removal

Through the outlier confirmation in 01_exploratory_analysis_1.py, it was noticed that the total line in the pdf file is included as a single data point. And when I checked the total line in the pdf file, it was also noticed that there is a line which does not seem to express a personal data point named "THE TRAVEL AGENCY IN THE PARK" above the total line. I judged those two data points as invalid in terms of the purpose of analysis and removed from the data set.

#### Miss alignment adjustment

Again through the outlier confirmation in 01_exploratory_analysis_1.py, some irregular numbers were observed in a certain features. By comparing the irregularities with the data source, it was noticed that the scraping from the pdf file was not performed successfully for two lines, "BELFER ROBERT" and "BHATNAGAR SANJAY". Lack of "-" in the line seems to cause miss alignment so those miss alignments were modified.

#### Feature engineering

Intuitively speaking, the number of emails sent to or received from POI seems to be very important features to figure out who is POI. But at the same time, the absolute number might not a good indicator because how often the person emails changes the meaning of absolute number. So I decided to create ratio features as follows.

```
22. recieved_from_poi_ratio      : = from_poi_to_this_person / to_messages
23. sent_to_poi_ratio            : = from_this_person_to_poi / from_messages
24. shared_receipt_with_poi_ratio: = shared_receipt_with_poi / to_messages
```

#### Log scaling

When a distibution of each feature was observed by histogram, it was noticed that most of them are highly skewed. To make it more balanced or close to the normal distribution, I applied log scaling to all the features except the three I engineered. The reasons I did this are:

 - Some models assume normal distribution of feature distributions. If features are normally distributed, model performance would be better.
 - Computational power of training is less required for some models so the training time would be also less.

Note, since exponential function can reverse scaled features to the original, this transformation does not lose any information. So I applied log scaling to the original features and did not engineer new log-scaled features.


## 5. Modeling

As expressed in each file name, nine machine learning algorithms are tested.

1. Naive Bayes
2. Logistic Regression
3. Decision Tree
4. Support Vector Machine
5. Linear Support Vector Machine
6. Random Forest
7. Extra Trees
8. Adaboost
9. Gradient Boosting

Combining imputation and feature selection by pipeline, the following three paterns are tested for every algorithm. (In a sense PCA is a combination of transformation and feature selection.)

```
1. Imputation -> Model Training (Using all features except email_address)
2. Imputation -> Feature Selection by p-values -> Model Training
3. Imputation -> Principal Component Analysis -> Model Training
```

Imputation by median was applied to all the email features in all paterns. The reason I chose median for imputation is that median seems to be more plausible than mean to represent "the center" since distributions of those features are highly skewed.

During the training grid search was performed to find the best parameters for each model through cross validation.


### 6. Feature Importance

To quickly have a sense about what features contribute to the models, the table below shows importance scores (F-values) and P-values. These were computed based on the entire data set. (See the work in 15_feature_review.py.) 

```
feature                         score  pvalue

other                          17.451  0.0001
sent_to_poi_ratio              16.023  0.0001
expenses                       13.371  0.0004
bonus                          11.766  0.0008
salary                          9.510  0.0025
deferred_income                 7.707  0.0062
from_poi_to_this_person         7.488  0.0070
total_payments                  7.020  0.0090
from_this_person_to_poi         6.993  0.0091
shared_receipt_with_poi_ratio   6.638  0.0110
total_stock_value               6.191  0.0140
restricted_stock                5.908  0.0163
shared_receipt_with_poi         5.678  0.0185
long_term_incentive             4.939  0.0278
to_messages                     3.067  0.0821
restricted_stock_deferred       2.737  0.1002
director_fees                   2.392  0.1242
loan_advances                   2.063  0.1531
recieved_from_poi_ratio         1.888  0.1716
exercised_stock_options         0.108  0.7429
from_messages                   0.006  0.9362
deferral_payments               0.000  0.9884
```

Note, since I used cross validation in the actual training process, the score and p-value in every training set would be different from the above depending on which part of data is held out as validation set. So the table above just provides the general idea.

A good news is that two of engineered features (*_ratio) are working very well as expected.


## 6. Summary of Result

The following table shows __F1 score__ of the best model in each pattern. 

```
|           Classifier             |      Feature Selection       |
|                                  | Original | P-value |   PCA   |
|----------------------------------|:--------:|:-------:|:-------:|
| 1. Naive Bayes                   |   0.374  |  0.387  |  0.244  |
| 2. Logistic Regression           |   0.404  |  0.428  |  0.472  |
| 3. Decision Tree                 |   0.327  |  0.346  |  0.255  |
| 4. Support Vector Machine        |   0.495  |  0.462  |  0.510  |
| 5. Linear Support Vector Machine |   0.000  |  0.000  |  0.000  |
| 6. Random Forest                 |   0.322  |  0.370  |  0.181  |
| 7. Extra Trees                   |   0.514  |  0.503  |  0.268  |
| 8. Adaboost                      |   0.218  |  0.279  |  0.208  |
| 9. Gradient Boosting             |   0.320  |  0.290  |  0.105  |
```

These results were taken by `evaluate` in 04_model_preparation.py, which does the same evaluation as tester.py does with less number of cross validation. The best five patterns are carried to the final evaluation by tester.py. The details of result are seen in result_all.txt.


[Note]

1. The threshold of p-value was grid searched for each model. As a result, all the best models used 0.05. (For some computationally expensive models, only 0.05 was applied from the beginning.)
2. For PCA, the number of components used for modeling was grid searched. The number of components varies from 10 to  18 depending on the model.


## 7. Q&A

### Summarize the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?

The goal of this project is to apply as many classification machine learning algorithms as possible and to see which outperforms. The data used is small (144 rows / 22 columns) though, it's still a high dimentional space so it's really hard for human beings to figure out the decision boundary of POI accurately by any heuristic approch. Machine learning can explore this high dimentional space efficiently based on their own algorithms and this might be able to develop a model which specifies the decision boundary. Since I do not know which algorithm performs better, it is encouraged to try as many algorithms as possible.

As explained in the previous section (4. Data), the original data has 146 rows and 21 columns, but as a result of modification 144 rows and 23 columns are used to train model. In terms of manual feature selection, only email address which is the only string feature is removed. Note, one column is the label which is the target of prediction, so there are 22 columns to be used as features.

[rows]

* Original (146) - Invalid (2) = 144

[columns]

* Original (21) + Engineered (3) - Email Adress (1) = 23 (1 label + 22 features)


### What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? Explain what feature you tried to make, and the rationale behind it. In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.

As mentioned, only email address is removed manually and all the other numerical features are brought to next stage and log scaling was applied to the original features to mitigate skewness. To retain as many features as possible makes sense since automated feature selections are performed as a part of pipeline in the training phase.

The final model I chose is Extra Trees classifier and no feature selection was applied aside from imputation. The reason why the best model does not require any future selection seems to be that the classifier utilizes the power of randamization which can deal bias-variance problems well.


### What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?

For the brief summary, please refer to the section 5 "Summary of Result".


### What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm?

Parameter tuning is a way to optimize the efficiency of training process. If the best parameters are chosen, the training process is faster (not always) and the model to be developed has better predictive power. 

For my final model Extra Trees classifier, the parameters are mainly for the following.

1. Specification of decision tree
2. Measurement of error
3. Number of decision trees used

I chose the range of parameters based on the default values and some trials. For the parameters of decision tree specification, I intended to control it by number of data points in a leaf or split because I felt that it's not a good idea to control it by depth of tree or number of leaves for a small data set.


### What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?

Validation is a way to check the reliability of model by keeping a validation data set out from a training data set. If an entire data set is used to train model, the model might overfit to the training data set. If overfitting is the case, the predictive power of model is weak, which means that the model failed to generize the phenomenon well.

In my analysis, ten holds stratified cross validation with randomization was performed for grid search and one hundred holds stratified cross validation with randomization was performed for evaluation . This method can randomly create holds with balancing the percentage of label in the hold out. Since the data set is small and the ratio of 1 (POI) and 0 (Non-POI) is unbalanced, it is important to keep the ratio in each hold.


### Give at least 2 evaluation metrics and your average performance for each of them. Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.

I'd like to present precision and recall from the result of tester.py (result_final.txt). Precision and recall are common metrics for classification problems. They can give us false positive rate and false negative rate respectively. These metrics are useful when false positive rate or false negative rate more matters than the other depending on the context of analysis.

For example, let me compare two results in result_final.txt.

```
[Logistic Regression with PCA]
     Accuracy: 0.76413
    Precision: 0.33878
       Recall: 0.80800
           F1: 0.47740
           F2: 0.63273
    Total predictions: 15000
       True positives: 1616
      False positives: 3154
      False negatives:  384
       True negatives: 9846

[Extra Trees]
     Accuracy: 0.85767
    Precision: 0.47385
       Recall: 0.61150
           F1: 0.53394
           F2: 0.57792
    Total predictions: 15000
       True positives: 1223
      False positives: 1358
      False negatives:  777
       True negatives: 11642
```

In terms of F1 score which is the weighted average of precision and recall, Extra Trees (ET) outperforms Logistic Regression (LR). But if you only look at recall, LR is doing much better than ET, which means that LR is better at detecting POI while it contains more false positive. In other words, LR can detect more POI than ET by sacrificing its precision. For some problems, it makes sense to weigh precision or recall for a sake of analysis though, I chose F1 score to decide the final model because I thought that both are equally important for this problem.


## 8. References

* Enron scandal, Wikipedia - https://en.wikipedia.org/wiki/Enron_scandal
* A look at those involved in the Enron scandal, USA Today - http://usatoday30.usatoday.com/money/industries/energy/2005-12-28-enron-participants_x.htm
* The Immortal Life of the Enron E-mails, MIT Technology Review - https://www.technologyreview.com/s/515801/the-immortal-life-of-the-enron-e-mails/
* Katie Malone, Civic Analytics - https://civisanalytics.com/team/katie-malone/
* Sci-kit and Regression Summary, Stack Overflow - http://stackoverflow.com/questions/26319259/sci-kit-and-regression-summary
* False positive rate, Wikipedia - https://en.wikipedia.org/wiki/False_positive_rate
* False discovery rate, Wikipedia - https://en.wikipedia.org/wiki/False_discovery_rate
* Precision and recall, Wikipedia - https://en.wikipedia.org/wiki/Precision_and_recall
* Python sklearn.feature_selection.f_classif Examples - http://www.programcreek.com/python/example/85917/sklearn.feature_selection.f_classif
* Why do we need to normalize data before analysis, Cross Validated - http://stats.stackexchange.com/questions/69157/why-do-we-need-to-normalize-data-before-analysis
* Perform feature normalization before or within model validation?, Cross Validated - http://stats.stackexchange.com/questions/77350/perform-feature-normalization-before-or-within-model-validation
* How should the interquartile range be calculated in Python?, Stack Overflow - http://stackoverflow.com/questions/27472330/how-should-the-interquartile-range-be-calculated-in-python
* scikit learn svc coef0 parameter range, Stack Overflow - http://stackoverflow.com/questions/21390570/scikit-learn-svc-coef0-parameter-range
* What is a good range of values for the svm.SVC() hyperparameters to be explored via GridSearchCV()?, Stack Overflow - http://stackoverflow.com/questions/26337403/what-is-a-good-range-of-values-for-the-svm-svc-hyperparameters-to-be-explored
* Imputation before or after splitting into train and test?, Cross Validated - http://stats.stackexchange.com/questions/95083/imputation-before-or-after-splitting-into-train-and-test
* Is there a rule-of-thumb for how to divide a dataset into training and validation sets?, Stack Overflow - http://stackoverflow.com/questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio
* What is the difference between test set and validation set?, Cross Validated - http://stats.stackexchange.com/questions/19048/what-is-the-difference-between-test-set-and-validation-set
* Python - What is exactly sklearn.pipeline.Pipeline?, Stack Overflow - http://stackoverflow.com/questions/33091376/python-what-is-exactly-sklearn-pipeline-pipeline
* How can I use a custom feature selection function in scikit-learn's pipeline, Stack Overflow - http://stackoverflow.com/questions/25250654/how-can-i-use-a-custom-feature-selection-function-in-scikit-learns-pipeline
* Pipelining: chaining a PCA and a logistic regression, scikit learn - http://scikit-learn.org/stable/auto_examples/plot_digits_pipe.html
* EnsembleVoteClassifier, Sebastian Raschka - http://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/
* Implementing a Weighted Majority Rule Ensemble Classifier in scikit-learn, Sebastian Raschka - http://sebastianraschka.com/Articles/2014_ensemble_classifier.html
* Markdown Tables Generator - http://www.tablesgenerator.com/markdown_tables
* How to save a Seaborn plot into a file, Stack Overflow - http://stackoverflow.com/questions/32244753/how-to-save-a-seaborn-plot-into-a-file
* matplotlib.axes, matplotlib - http://matplotlib.org/api/axes_api.html
* Color Palettes in Seaborn, Chris Albon - http://chrisalbon.com/python/seaborn_color_palettes.html
* Seaborn distplot y-axis normalisation wrong ticklabels, Stack Overflow - http://stackoverflow.com/questions/32274865/seaborn-distplot-y-axis-normalisation-wrong-ticklabels
