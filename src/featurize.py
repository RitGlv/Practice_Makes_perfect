import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cosine

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

# def same_columns(df,col1,col2):
#     df['same'+col1] = df[col1]==df[col2]
#     return df

def create_features_matrix(df,cols_to_keep,categorical = False):
    '''
    Create a dataframe with all users and their scaled features for calculating similarity, using Interview information (meaning paired information).
    Assumes simmetry in no of columns per each user
    First split then dummify
    '''
    features_df = df[cols_to_keep]
    no_of_cols = features_df.shape[1]
    split1 = features_df.iloc[:,0:(no_of_cols/2)]
    split2 = features_df.iloc[:,(no_of_cols/2):no_of_cols]
    if categorical:
        similarity_df = pd.DataFrame()
        split2.columns = list(split1.columns.values)
        for col in list(split1.columns.values):
            similarity_df[col] = split1[col] == split2[col]
        return np.asarray(similarity_df)
    split1 = scale(split1)
    split2 = scale(split2)
    return np.asarray(split1), np.asarray(split2)

def create_similarity_per_interview(df,cols_to_keep,metric = 'cosine',categorical=False):
    '''
    Get both users features vector and calculate the similarity using chosen distance metric
    The output is a similarity score per matched interview
    '''

    copy = df.copy()
    if categorical:
        features = create_features_matrix(df,cols_to_keep,categorical)
        copy['similarity'] = map(lambda x: sum(x)*1.0/len(x),features)
        return copy

    user1,user2 = create_features_matrix(df,cols_to_keep)
    similarity_list = [cosine(user1[i],user2[i]) for i in range(len(user1))]
    copy['similarity'] = similarity_list

    return copy

def create_characteristic():
    '''
    Aggregate the user's inforamtion up to the point of the interview
    '''
    pass

def get_experience_area(df,col):
    '''
    Get the unique set of areas of expertise out of the data.
    Input: dataframe
    Iterates over all the dataframe rows, the updates the set based in the string within each row
    '''
    expertise_set = set(df[col][0])
    for ex in df[col]:
        expertise_list = ex.strip('[]')
        expertise_list = expertise_list.split(',')
        expertise_set.update(expertise_list)
    return expertise_set

def df_with_expertise(df,col):
    '''
    Get dataframe and return a new dataframe with dummified columns for expertise
    '''
    df[col] = map(lambda x: "" if x=='[]' else x,df[col])
    expertise = get_experience_area(df,col)
    for ex in expertise:
        df[ex] = map(lambda x: int(ex in x),df[col])
    df = df.rename(columns = {"":"Expertise_not_mentioned"})
    return df

def good_match_bool(df):
    df['good_match'] = (df.totalMatch1>7) & (np.abs((df.totalMatch1-2*df.match1))<3)
    return df

def dataframe_for_matrix(df):
    '''
    Remove duplicates - cases when users were matched more that once
    '''
    df['both'] = df['userId1']+df['matched_user']
    new = df.drop_duplicates('both')
    return new

def feturized_by_unique_user(df):
    '''
    Get a DataFrame with features, return the grouped df with mean
    Assumues userId has been set as index
    '''
    grouped_df = df.groupby(level=0).mean()
    return grouped_df

if __name__=="__main__":
    cols_to_categorical = ['education1','country1','degree1','status1','studyArea1','education2','country2','degree2','status2','studyArea2']
