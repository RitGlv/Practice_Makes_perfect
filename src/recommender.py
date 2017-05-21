import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import featurize
reload (featurize)

class SimilarityRecommender(object):
    '''
    Creates a matrix with recommendation scores based on content boosted collaborative filtering.
    The final recommendation is based on user-user and item-item similarity
    Currently works with static info, future: incorporate feature change over time
    '''

    def __init__(self,features_df,ratings_df):
        #ratings_df = processed matrix of match rating per interview
        self.ratings = None
        self.sim_matrix = None
        #features_df = processed matrix of features per user,assumes userId as index
        self.features = features_df
        self.baseline = None
        self.recommended = []

    def fit(self):
        self.get_ratings_matrix()
        self.get_similarity_score()

    def predict_one(self,user,n):
        '''
        Returns a list pf top N matched users
        '''
        n_most_similar = self.get_most_similar_users(user,n)
        for similar_user in n_most_similar:
            matched = np.asarray([self.ratings.index==similar_user]).argmax()
            self.recommended.append(get_most_similar_users(matched,n))

    def get_most_similar_users(self,user,n):
        '''
        Ranked list of the most similar users to the requested user
        User defined as a row in the sim_matrix
        Treat users at different point of time as different users
        '''
        sorted_indices=np.argsort(self.sim_matrix[self.feafeatures.index==user])
        n_most_similar= sorted_similarity[0][:n]
        return n_most_similar

    def get_ratings_matrix(self,index='userId1', columns='matched_user', values='totalMatch1'):
        '''
        Get a matrix with all of the users matching scores
        '''
        self.match_matrix = self.ratings.pivot(index=index, columns=columns, values=values).fillna(-1)

    def get_similarity_score(self,metric='eucalidian'):
        '''
        Calculates similarity between every 2 users
        '''
        self.sim_matrix = pairwise_distances(self.features_df,metric=metric)

    def baseline_model_random_choice(self):
        #to set random choice

    def model_eval(self,users):
        '''
        Asses model based on AUC for different n for recommendation
        '''
        # #pseudocode
        # for user in users:
        #     for i in range(self.ratings[user])
        pass


if __name__=="__main__":
    path = 'data/full_data_one_row_trainby_userwith_matched_user.csv'
    df = pd.read_csv(path)
    min_df = df[['userId1','matched_user','totalMatch1','match1']]
    with_match_type = featurize.good_match_bool(min_df)
    new = featurize.dataframe_for_matrix(with_match_type)
    piv = new.pivot(index='userId1', columns='matched_user', values='totalMatch1').fillna(0)
