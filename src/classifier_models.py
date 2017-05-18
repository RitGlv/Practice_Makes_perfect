import clean_data
reload (clean_data)
import predictors
reload (predictors)
import featurize
reload (featurize)
from predictors import Predictors
from clean_data import CleanedData
from sklearn.tree import DecisionTreeRegressor , DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor , RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor , GradientBoostingClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

'''
Run first classification to include total interview info as row
'''
train_path = 'data/full_data_one_row_train.csv'
test_path = 'data/full_data_one_row_test.csv'
#
# cols_for_first_iteration = ['selfPrep1','totalMatch1','experienceInYears1','interviewsDonePriorToThisOne1','selfPrep2','experienceInYears2','interviewsDonePriorToThisOne2','similarity']
# cols_to_dummify = []

cols_for_first_iteration = ['selfPrep1','totalMatch1','status1','degree1','experienceInYears1','studyArea1','interviewsDonePriorToThisOne1','selfPrep2','status2','degree2','experienceInYears2','studyArea2','interviewsDonePriorToThisOne2']
cols_to_dummify = ['status1','degree1','studyArea1','status2','degree2','studyArea2']
train_data = CleanedData(train_path)
X_train,y_train = train_data.fit_data(cols_for_first_iteration,cols_to_dummify,'totalMatch1')

test_data = CleanedData(test_path)
X_test,y_test = test_data.fit_data(cols_for_first_iteration,cols_to_dummify,'totalMatch1')

X_train = featurize.scale(X_train)
X_test = featurize.scale(X_test)

y_train_class = map(lambda y: int(y<8), y_train)
y_test_class = map(lambda y: int(y<8), y_test)

classifier_dict = {'Decision_Tree':DecisionTreeClassifier(),'Random_Forest':RandomForestClassifier(n_estimators=50),'GD_boost':GradientBoostingClassifier(max_depth = 6,learning_rate=10)}#,'SVM':SVC()}
clasiffiers = Predictors(X_train,y_train_class,X_test,y_test_class,classifier_dict)
clasiffiers.fit_regressors(classifier=True)

'''
Classifier model for only first interview
'''
# train_path = 'data/full_data_one_row_train_only_first.csv'
# test_path = 'data/full_data_one_row_test_only_first.csv'
#
# # cols_for_first_iteration = ['selfPrep1','totalMatch1','experienceInYears1','interviewsDonePriorToThisOne1','selfPrep2','experienceInYears2','interviewsDonePriorToThisOne2','similarity']
# # cols_to_dummify = []
#
# cols_for_first_iteration = ['totalMatch1','experienceInYears1','selfPrep2','experienceInYears2','interviewsDonePriorToThisOne2','similarity']
# cols_to_dummify = []
# train_data = CleanedData(train_path)
# X_train,y_train = train_data.fit_data(cols_for_first_iteration,cols_to_dummify,'totalMatch1')
#
# test_data = CleanedData(test_path)
# X_test,y_test = test_data.fit_data(cols_for_first_iteration,cols_to_dummify,'totalMatch1')
#
# X_train['experience'] = np.abs(np.sqrt(X_train.experienceInYears1)-np.sqrt(X_train.experienceInYears2))
# X_train.pop('experienceInYears1')
# # X_train.pop('experienceInYears2')
# # X_train = featurize.scale(X_train)
# # X_test = featurize.scale(X_test)
#
# y_train_class = map(lambda y: int(y<8), y_train)
# y_test_class = map(lambda y: int(y<8), y_test)
#
# classifier_dict = {'Decision_Tree':DecisionTreeClassifier(),'Random_Forest':RandomForestClassifier(),'GD_boost':GradientBoostingClassifier(),'KNN':KNeighborsClassifier(weights='distance')}
# clasiffiers = Predictors(X_train,y_train_class,X_test,y_test_class,classifier_dict)
# clasiffiers.fit_regressors(classifier=True)
