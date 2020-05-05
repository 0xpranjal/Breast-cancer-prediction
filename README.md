# Breast-cancer-detection
Breast cancer detection using 4 different models i.e. Logistic Regression, KNN, SVM and Decision Tree Machine Learning models and optimising them for even a better accuracy. This project is started with the goal use machine learning algorithms and learn how to optimize the tuning params and also and hopefully to help some diagnoses.

# Problem Statement
This project focuses in investigating the probability of predicting the chances of breast cancer from the given   characteristics of breast mass computed from dataset. This project will examine the data available and attempt to predict the possibility that a breast cancer. To achieve this goal, the following steps are identified:
•	Download the breast cancer data from UCI repository
•	Familiarize with the data by looking at its shape, the relations between variables, their possible correlations, and other attributes of the dataset. 
•	Preprocess data if needed
•	Split the data into testing and training samples
•	Employ various classifiers (K-neighbors, Decision trees, SVC and Random Forest classifier) to predict the data with different sets of training samples (100, 200, 300, and 400). 
•	Once the best predicting model is identified, will reduce the training set in size to see what is the limit for this classifier to best predict these data.
•	Compare the best identified classifier with evaluation metric stated at the beginning of the project.
•	Write conclusions. 

# Project Overview
According to the Centers for Disease Control and Prevention (CDC) breast cancer is the most common type of cancer for women regardless of race and ethnicity (CDC, 2016). Around 220,000 women are diagnosed with breast cancer each year in the United States (CDC, 2016). Although we may not be aware of all the factors contributing in developing breast cancer, certain attributes such as family history, age, obesity, alcohol and tobacco use have been identified from research studies on this topic (DeSantis, Ma, Bryan, & Jemal, 2014). Breast images procedures such as mammography have been found to be quite effective in early identification cases of breast cancer (Ball, 2012).  When breast images procedures are not utilized, patients can find out late about their diagnosis to be able to treat it.  Similar work on attempting to find the best way to predict the type of cancer based on images of mammograms has identified Support Vector Machine as the best predictor after tuning parameters.


### Libraries used
```python
import numpy as np #for linear algebra
import pandas as pd #for chopping, processing
import csv #for opening csv files
%matplotlib inline 
import matplotlib.pyplot as plt #for plotting the graphs
from scipy import stats #for statistical info
from time import time

from sklearn import tree
from sklearn.model_selection import train_test_split # to split the data in train and test
from sklearn.model_selection import KFold # for cross validation
from sklearn.grid_search import GridSearchCV  # for tuning parameters
from sklearn import metrics  # for checking the accuracy 

#Classifiers 

from sklearn import svm #for Support Vector Machines
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , confusion_matrix #for Logistic regression
from sklearn.svm import SVC # for support vector classifier
from sklearn.neighbors import NearestNeighbors #for nearest neighbor classifier
from sklearn.neighbors import KNeighborsClassifier # for K neighbor classifier
from sklearn.tree import DecisionTreeClassifier #for decision tree classifier
from sklearn.ensemble import RandomForestClassifier #for Random Forest
```

###How to
To run the scripts you just type:
```python
python script_name.py
```
As result of execution the reached accuracy will print

#### Dataset and Inputs
The characteristics of the cell nuclei have been captured in the images and a classification methods which uses linear programming to construct a decision line. The dataset is published by Kaggle and taken from the University of California Irvine (UCI) machine learning repository.  The data is taken from the Breast Cancer Wisconsin Center. It includes ten (10) attributes taken from each cell nucleus as well as ID and the diagnosis (M=malignant, B=benign).  The dataset has 570 cases and 31 variables.  
* the dataset can be found [here](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
