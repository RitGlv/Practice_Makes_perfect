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

    def __init__(self):
        self.ratings = None
        self.user_matrix = None

    def fit(self):
        pass

    def predict_one(self):
        pass

    def get_ratings_matrix(self):
        '''
        Get a matrix with all of the users matching scores
        '''
        pass
    def get_similarity_score(self,metric='eucalidian'):
        '''
        Calculates similarity between every 2 users
        '''
        self.sim_matrix = pairwise_distances(self.user_matrix)
