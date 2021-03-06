import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import featurize
reload (featurize)
import decomposition
reload (decomposition)
from decomposition import decomposed
import plots
reload (plots)
from plots import plot_pca
import matplotlib.pyplot as plt

class SimilarityRecommender(object):
    '''
    Creates a matrix with recommendation scores based on content boosted collaborative filtering.
    The final recommendation is based on user-user and item-item similarity
    Currently works with static info, future: incorporate feature change over time
    '''

    def __init__(self,features_df,ratings_df):
        #ratings_df = processed matrix of match rating per interview
        self.ratings = ratings_df
        self.sim_matrix = None
        #features_df = processed matrix of features per user,assumes userId as index
        self.features = features_df
        self.baseline = None
        self.recommended = []
        self.false_positive_users = []
        self.true_positive_users = []
        self.all_recommendations = []
        self.count=0

    def fit(self):
        self.get_ratings_matrix()
        self.get_similarity_score()

    def predict_one(self,user,n):
        '''
        Returns a list pf top N matched users
        '''
        self.recommended = []
        n_most_similar = self.get_most_similar_users(user,n)
        for similar_user in n_most_similar:
            if np.asarray(self.match_matrix.iloc[similar_user]).max():
                matched = np.asarray(self.match_matrix.iloc[similar_user]).argmax()
                matched_id = self.match_matrix.index[matched]
                most_similar = self.get_most_similar_users(matched_id,n)
                for m in most_similar:
                    self.recommended.append(self.match_matrix.index[m])
        self.recommended = set(self.recommended)

    def get_most_similar_users(self,user,n):
        '''
        Ranked list of the most similar users to the requested user
        User defined as a row in the sim_matrix
        Treat users at different point of time as different users
        '''
        sorted_indices=np.argsort(self.sim_matrix[self.features.index==user])
        n_most_similar= sorted_indices[0][1:(n+1)]
        return n_most_similar

    def get_ratings_matrix(self,index='userId1', columns='matched_user', values='good_match'):
        '''
        Get a matrix with all of the users matching scores
        '''
        self.match_matrix = self.ratings.pivot(index=index, columns=columns, values=values).fillna(-1)

    def get_similarity_score(self,metric='euclidean'):
        '''
        Calculates similarity between every 2 users
        '''
        self.sim_matrix = pairwise_distances(self.features,metric=metric)

    def model_eval(self,n):
        '''
        Asses model based on AUC for different n for recommendation
        Predict all
        n=2,3,5,10
        '''
        self.eval_mat = np.zeros(self.sim_matrix.shape)*-1.0
        for user in self.match_matrix.index:
            self.predict_one(user,n)
            for predicted_match in self.recommended:
                self.eval_mat[self.match_matrix.index==user][0][self.match_matrix.index==predicted_match]=1
                if self.match_matrix[self.match_matrix.index==predicted_match][user][0] == 0:
                    self.false_positive_users.append((user,predicted_match))
                elif self.match_matrix[self.match_matrix.index==predicted_match][user][0] == 1:
                    self.true_positive_users.append((user,predicted_match))
            self.all_recommendations.append((user,self.recommended))
            self.count+=1

if __name__=="__main__":

    '''
    Load data for all interview match rating
    '''

    path = 'data/full_data_one_row_swap_idsby_userwith_matched_user.csv'
    df_for_rating = pd.read_csv(path)

    #crate dataframe for match rating matrix
    min_df = df_for_rating[['userId1','matched_user','totalMatch1','match1']]
    with_match_type = featurize.good_match_bool(min_df)
    interview_rating = featurize.dataframe_for_matrix(with_match_type)

    train_path = 'data/full_data_one_row_swap_idsby_user.csv'
    df = pd.read_csv(train_path).set_index('userId1')
    df['experienceInYears1'] = np.sqrt(df['experienceInYears1'])

    #columns to leave in the static inforamtion(pre_interview) grouped user dataframe
    cols_to_leave = ['selfPrep1', 'experienceAreas1','experienceInYears1','degree1', 'status1','studyArea1']
    categories = ['degree1','status1','studyArea1']
    
    pca = decomposed(df)
    pca.fit(cols_to_leave,categories,6)
    df_pca = pd.DataFrame(pca.X_pca).set_index(pca.processed.index)
    sim = SimilarityRecommender(df_pca,interview_rating)
    sim.fit()
