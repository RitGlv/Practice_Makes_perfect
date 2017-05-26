from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, f1_score, recall_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

class Predictors(object):

    def __init__(self,X_train,y_train,X_test,y_test,regressor_dict):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.regressors = regressor_dict
        self.fitted_regressors = None
        self.scores = None
        self.test_scores = None
        self.best_model = None
        self.rmse = None
        self.importances = None
        self.prediction = None
        self.f1 = None

    def fit_regressors(self,classifier=False):
        '''
        Apply fit on selected regressors
        '''
        self.fitted_regressors = {name : model.fit(self.X_train,self.y_train) for name,model in self.regressors.iteritems()}
        self.validate(classifier=classifier)
        self.find_best_model()
        self.test_error(classifier=classifier)
        # self.get_feature_importances()

    def validate(self,classifier=False):
        '''
        Input: dictionary with regressor name as key and fitter model as value
        Get cross val scores (mse) to choose the best model
        '''
        if classifier:
            self.scores  = {name : cross_val_score(model,self.X_train,self.y_train,scoring="roc_auc")*-1 for name,model in self.fitted_regressors.iteritems()}
        else:
            self.scores  = {name : cross_val_score(model,self.X_train,self.y_train,scoring="neg_mean_squared_error")*-1 for name,model in self.fitted_regressors.iteritems()}

    def find_best_model(self,classifier=False):
        '''
        Determine which is the best model
        '''
        mean_scores = {name : score.mean() for name,score in self.scores.iteritems()}
        if classifier:
            self.best_model = max(mean_scores,key = mean_scores.get)
        self.best_model = min(mean_scores,key = mean_scores.get)

    def test_error(self,classifier=False):
        '''
        Check test scores for best model, using MSE.
        Save state for rmse for best model
        '''
        self.prediction = self.fitted_regressors[self.best_model].predict(self.X_test)
        if classifier:
            self.f1 = roc_auc_score(self.prediction,self.y_test)
        else:
            self.rmse = np.sqrt(mean_squared_error(self.prediction,self.y_test))

    def test_error_base(self,classifier=False):
        '''
        Check test scores for best model, using MSE.
        Save state for rmse for best model
        '''
        if classifier:
            prediction = np.ones(len(self.y_test))
            f1 = roc_auc_score(prediction,self.y_test)
            return f1

        prediction = np.ones(len(self.y_test))*(self.y_train.mean())
        rmse = np.sqrt(mean_squared_error(prediction,self.y_test))
        return rmse

    def get_feature_importances(self):
        '''
        return the most importent fetures for predicting the target
        '''
        importances = self.fitted_regressors[self.best_model].feature_importances_
        indices = np.argsort(importances)[::-1]
        self.importances = self.X_train.columns[indices]
        return importances,indices

    def plot_feature_importances(self,no_of_features):
        importances,indices = self.get_feature_importances()
        plt.bar(range(no_of_features), importances[indices][:no_of_features],color="r", align="center")
        plt.xticks(range(no_of_features),self.importances[:no_of_features])
        plt.show()
