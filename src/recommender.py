import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import featurize

class SimilarityRecommender(object):
    '''
    Creates a matrix with recommendation scores based on content boosted collaborative filtering.
    The final recommendation is based on user-user and item-item similarity
    Currently works with static info, future: incorporate feature change over time
    '''

    def __init__(self,features_df,user_ids):
        self.ratings = None
        self.sim_matrix = None
        #user_ids should be a concat of id and intervie_no to treat users differently at different points in time
        self.user_matrix = np.asarray(user_ids)
        self.features = features_df

    def fit(self):
        pass

    def predict_one(self):
        pass

    def get_most_similar_users(self,user):
        '''
        Ranked list of the most similar users to the requested user
        User defined as a row in the sim_matrix
        Treat users at different point of time as different users
        '''
        user_location = self.user_matrix[0]== user
        similar_users = self.sim_matrix[user_location]
        similar_users_index = similar_users.argsort[::-1]
        ranked_users = self.user_matrix[similar_users_index]
        return ranked_users

    def get_ratings_matrix(self):
        '''
        Get a matrix with all of the users matching scores
        Considering two options = symmetric matching (weighted mean? 0.75(user_rating)+0.25(other_user_rating)) or individual
        '''
        pass

    def get_similarity_score(self,metric='eucalidian'):
        '''
        Calculates similarity between every 2 users
        '''
        self.sim_matrix = pairwise_distances(self.user_matrix,metric=metric)
