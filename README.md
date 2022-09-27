# *Finding the Best ML Classifier*
## Evaluating ML Classifiers for Loan Case Prediction

**In this repository, you can check out the Machine Learning classifiers I built to predict whether a loan case will be paid off or not. Find out more in the following sections!**

___


## Background
In this project, I loaded a historical dataset from previous loan applications, cleaned the data, and applied different classification algorithms on the data in order to predict whether a loan case will be paid off or not.

I used the following algorithms to build my models:
- k-Nearest Neighbor
- Decision Tree
- Support Vector Machine
- Logistic Regression

And I used the following metrics to evaluate my models in my report:
- Jaccard Index
- F1-score
- Log Loss


---


## Technologies & Usage
This project leverages Python 3.9, Numpy, Matplotlib, Scikit-Learn, Pandas, and Seaborn with the following requirements and dependencies:
- import itertools
- import numpy as np
- import matplotlib.pyplot as plt
- from matplotlib.ticker import NullFormatter
- import pandas as pd
- import numpy as np
- import matplotlib.ticker as ticker
- from sklearn import preprocessing
- %matplotlib inline
- import warnings
- import seaborn as sns
- from sklearn.model_selection import train_test_split
- from sklearn.neighbors import KNeighborsClassifier
- from sklearn import metrics
- from sklearn.tree import DecisionTreeClassifier
- import sklearn.tree as tree
- from sklearn import svm
- from sklearn.linear_model import LogisticRegression
- from sklearn.metrics import confusion_matrix, jaccard_score, f1_score, logg_loss
