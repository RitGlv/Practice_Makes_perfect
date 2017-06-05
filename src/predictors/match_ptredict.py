import clean_data
reload (clean_data)
import predictors
reload (predictors)
import featurize
reload (featurize)
from predictors import Predictors
from clean_data import CleanedData
from sklearn.tree import DecisionTreeRegressor , DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor , RandomForestClassifier, GradientBoostingRegressor , GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import Lasso,Ridge, LogisticRegression
from sklearn.metrics import jaccard_similarity_score
import numpy as np
from create_db import rename_cols
import pandas as pd


'''
Run regression based on
'''
train_path = 'data/full_data_one_row_train.csv'
test_path = 'data/full_data_one_row_test.csv'

def fit_data(cols,path):
    df_train = rename_cols(pd.read_csv(path))
    features_df = featurize.create_features_matrix(df_train,df_train.columns.values,categorical=True,expertise_col='experienceAreas1')
    #create y as the min match score between user1 and user2
    y_train = featurize.create_y_by_min(df_train,'match1','match2')
    y_class = featurize.good_match_bool(df_train)['good_match']

    X_train = features_df[cols]
    return X_train,y_train,y_class,df_train

columns_for_train = ['country1','degree1','status1','"ML"', '"DataScience"', '"TestEng"','"DevOps"','"Android"', '"Security"','"Backend"', '"Kernel"','"iOS"', '"EngMgr"', '"Big Data"', '"Frontend"']

X_train,y_train,y_train_class,df_train = fit_data(columns_for_train,train_path)
X_test,y_test,y_test_class,df_test = fit_data(columns_for_train,test_path)

regressors_dict = {'Lasso': Lasso(alpha=0.1),'Ridge':Ridge(), 'RF':RandomForestRegressor(), 'GB':GradientBoostingRegressor()}
regressors = Predictors(X_train,y_train,X_test,y_test,regressors_dict)
regressors.fit_regressors()

# {'GB': array([ 1.25701629,  1.26938458,  1.32176152]),
#  'Lasso': array([ 1.25496643,  1.25544299,  1.32065415]),
#  'RF': array([ 1.44261372,  1.45953509,  1.50777398])}

'''
Add diff columns
'''
def add_deltas(df_new,df,col1,col2):
    name = col1 + '_diff'
    df_new[name] = np.abs(df[col1] - df[col2])

columns_for_train = ['country1','degree1','status1','"ML"', '"DataScience"', '"TestEng"','"DevOps"','"Android"', '"Security"','"Backend"', '"Kernel"','"iOS"', '"EngMgr"', '"Big Data"', '"Frontend"']

X_train = featurize.dummify(X_train[columns_for_train],columns_for_train)
X_test = featurize.dummify(X_test[columns_for_train],columns_for_train)

cols_for_diff = [['questionDiff1','questionDiff2'],['selfPrep1','selfPrep2'],['experienceInYears1','experienceInYears2'],['interviewsDonePriorToThisOne1','interviewsDonePriorToThisOne2']]

for pair in cols_for_diff:
    add_deltas(X_train,df_train,pair[0],pair[1])
    add_deltas(X_test,df_test,pair[0],pair[1])

regressors_dict = {'Lasso': Lasso(alpha=0.1),'Ridge':Ridge(alpha = 10, solver = 'lsqr'), 'RF':RandomForestRegressor(), 'GB':GradientBoostingRegressor(learning_rate=0.001, n_estimators=600)}

regressors_diff = Predictors(X_train,y_train,X_test,y_test,regressors_dict)
regressors_diff.fit_regressors()
#
# {'GB': array([ 1.25592965,  1.27183922,  1.31985975]),
#  'Lasso': array([ 1.25020716,  1.25092507,  1.32065415]),
#  'RF': array([ 1.39064453,  1.38887214,  1.39293263]),
#  'Ridge': array([ 1.23889062,  1.24599787,  1.29214907])}

# '''
# Add Similarity Metric
# '''
# def similarity_score(df,cols):
#     df['jaccard'] = df[cols].sum(axis=1)*1.0/df[cols].count(axis=1)
#
# columns_for_train = ['degree1','status1','"ML"', '"DataScience"', '"TestEng"','"DevOps"','"Android"', '"Security"','"Backend"', '"Kernel"','"iOS"', '"EngMgr"']
#
# X_train = X_train[columns_for_train]
# X_test = X_test[columns_for_train]
#
# similarity_score(X_train,columns_for_train)
# similarity_score(X_test,columns_for_train)
#
# regressors_sim = Predictors(X_train,y_train,X_test,y_test,regressors_dict)
# regressors_sim.fit_regressors()
#
# classifier_dict = {'Ada':AdaBoostClassifier(n_estimators=10, learning_rate=10),'lg':LogisticRegression(C=10)}
# classifiers = Predictors(X_train,y_train_class,X_test,y_test_class,classifier_dict)
# classifiers.fit_regressors(classifier=True)
#
# {'GB': array([ 1.24113419,  1.24384472,  1.30394457]),
#  'Lasso': array([ 1.25020716,  1.25092507,  1.32065415]),
#  'RF': array([ 1.47006846,  1.47019722,  1.51609719]),
#  'Ridge': array([ 1.23730288,  1.24511884,  1.29091601])}
