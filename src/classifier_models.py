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

'''
Run first classification to include total interview info as row
'''
train_path = 'data/full_data_one_row_train.csv'
test_path = 'data/full_data_one_row_test.csv'

cols_for_first_iteration = ['selfPrep1','totalMatch1','status1','degree1','experienceInYears1','studyArea1','interviewsDonePriorToThisOne1','selfPrep2','status2','degree2','experienceInYears2','studyArea2','interviewsDonePriorToThisOne2','samecountry1']
cols_to_dummify = ['status1','degree1','studyArea1','status2','degree2','studyArea2','samecountry1']
train_data = CleanedData(train_path)
X_train,y_train = train_data.fit_data(cols_for_first_iteration,cols_to_dummify,'totalMatch1')

test_data = CleanedData(test_path)
X_test,y_test = test_data.fit_data(cols_for_first_iteration,cols_to_dummify,'totalMatch1')

X_train = featurize.scale(X_train)
X_test = featurize.scale(X_test)

y_train_class = map(lambda y: int(y<9), y_train)
y_test_class = map(lambda y: int(y<9), y_test)

classifier_dict = {'Decision_Tree':DecisionTreeClassifier(),'Random_Forest':RandomForestClassifier(),'GD_boost':GradientBoostingClassifier()}
clasiffiers = Predictors(X_train,y_train_class,X_test,y_test_class,classifier_dict)
clasiffiers.fit_regressors(classifier=True)
