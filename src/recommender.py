import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import featurize

class UserSimilarityRecommender(object):
    '''
    Creates a matrix with recommendation scores based on fature similarity
    Future imporivement - take rating into consideration to make further similarity based recommendations (take max rated user, recommend similar user by characteristics)
    '''

    def __init__(self):
        self.ratings = None
        self.user_features_matrix = None

    def fit(self):
        pass

    def predict_one(self):
        pass

    def predict_all(self):
        pass

    def fill_scores(self):
        pass

class UserMatrix(self):

    def __init__(self):
        self.user_feature_mat = None

    def fit(self,user_df):
        pass
