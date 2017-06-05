import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
import clean_data
reload (clean_data)
from clean_data import CleanedData
import featurize
reload (featurize)

train_path = 'data/full_data_one_row_trainby_user.csv'
test_path = 'data/full_data_one_row_testby_user.csv'

columns_to_leave = ['selfPrep1','experienceInYears1','interviewsDonePriorToThisOne1','likable1','communication1','asInterviewer1','problemSolving1','codingSkills1','hiring1']
cols_to_dummify = []

train_data = CleanedData(train_path)
X_train,y_train = train_data.fit_data(columns_to_leave,cols_to_dummify,'hiring1')

test_data = CleanedData(test_path)
X_test,y_test = test_data.fit_data(columns_to_leave,cols_to_dummify,'hiring1')

X_train = featurize.scale(X_train)
X_test = featurize.scale(X_test)

X_train = sm.add_constant(X_train)
model = sm.OLS(y_train,X_train)
results = model.fit()
