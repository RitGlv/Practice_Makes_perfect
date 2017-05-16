import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances

def dummify(df,cols,constant_and_drop=False):
    '''
    Input: list of columns to dumiify.
    Updates the transformed df state
    '''
    df = pd.get_dummies(df, columns=cols, drop_first=constant_and_drop)
    if constant_and_drop:
        const = np.full(len(df), 1)
        df['constant'] = const
    return df

def churned(df,today,timedelta):
    '''
    If (max_interview == interviewsDonePriorToThisOne) & (timedelta_since_last_interview>30)
    then churn == True
    '''
    df['churned'] = (today - df['interviewTime'])>timedelta

def stopped_after_one(df,user_col,columns_to_max):
    '''
    Input: dataframe, name of user_col, columns to present max value for. in this case, interviewsDonePriorToThisOne (will choose the case when 0)
    Tag user if stopped after only one interview
    '''
    max_interviews = df.groupby(user_col).max()[columns_to_max]
    only_1 = max_interviews[max_interviews==0].index.values
    df['stopped_after_one'] = map(lambda x: x in only_1,df[user_col])
    df['max_interview'] = map(lambda x: max_interviews[x],df[user_col])
    return df

def scale(df):
    '''
    Scaling features (columns) to a range between 0-1
    depends on min and max
    '''
    for col in df:
        if df[col].dtype == np.int64:
            max_col = df[col].max()
            min_col = df[col].min()
            df[col] = df[col].apply(lambda x: (x-min_col)*1.0/(max_col-min_col))
    return df

def is_us(df):
    '''
    Add a bool column to indicate whather the user if from US or not
    '''
    #change to country similarity?
    df['is_us'] = map(lambda x: 'United States' in x, df['country'])
    return df

def add_similarity_score(df,user1,user2):
    '''
    Add a column for similarity score
    '''
    pass

def same_columns(df,col1,col2):
    df['same'+col1] = df[col1]==df[col2]
    return df

def create_features_matrix(df,cols_to_keep,to_dummify = False,cols_to_dummify = None):
    '''
    Create a dataframe with all users and their scaled features for calculating similarity, using Interview information (meaning paired information).
    The output is a similarity score per matched interview
    '''
    features_df = df[cols_to_keep]
    if dummify:
        features_df = dummify(features_df,cols_to_dummify)
    features_df = scale(features_df)
    all_matrix = np.asarray(features_df)
    #add split df to create "2" vectors
    features_matrix = all_matrix[:,1:]
    #interview ID
    id_matrix = all_matrix[:,0]
    return features_matrix, id_matrix

def create_similarity_per_interview(user1_features,user2_fetures,metric = 'cosine'):
    '''
    Get both users features vector and calculate the similarity using chosen distance metric
    '''
    similarity = pairwise_distances(user1_features,user2_fetures,metric = metric)
    return similarity


if __name__=="__main__":
    cols_for_first_iteration = ['selfPrep','match','status','languageChanged','degree','questionChanged','experienceInYears','studyArea','nps','interviewsDonePriorToThisOne','is_us']
    cols_to_dummify = ['status','languageChanged','degree','questionChanged','studyArea','is_us']
