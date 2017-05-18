
from pandas.io.json import json_normalize
import pandas as pd

'''
Variuous options to crate mid process csv tables to analyze
'''
def flatten_json(collection,key):
    '''
    Create a flattened DF from a nested JSON.
    Input: collection (cursor) and a key to flatten
    '''
    unwind_key = "$"+key
    cursor = list(collection.aggregate([{ "$unwind" : unwind_key }]))
    df = json_normalize(cursor[0][key])
    for index in xrange(len(cursor)-1):
        df2 = json_normalize(cursor[index+1][key])
        df = df.append(df2)
    return df

def drop_cols_and_save(df_cleaned,filename,cols):
'''
Clean data (remove columns) after flattenning to create minimal initial table containing just numerical features and item ids
'''
    df_cleaned = df_cleaned.drop(cols,axis=1)
    df_cleaned.to_csv(filename)

def user_user_score_dataframe(path,cols):
    '''
    Input: filepath to a simplified csv table with interview_id, user1 and 2
    ids and scores
    return a dataframe for each user, their match and score for this specific match
    '''
    df = pd.read_csv(path)
    new_df = pd.DataFrame()
    for i in xrange(len(df)):
        new_df = new_df.append(pd.DataFrame(list(df[['interviewId','feedbacks.0.giver.userId','feedbacks.1.giver.userId','feedbacks.0.giver.nps']].loc[i])).transpose())
        new_df = new_df.append(pd.DataFrame(list(df[['interviewId','feedbacks.1.giver.userId','feedbacks.0.giver.userId','feedbacks.1.giver.nps']].loc[i])).transpose())
    new_df.columns = ['Interview_id','user_id','mtached_user_id','nps']
    return new_df

def crete_minimal_user_table(path,cols,path_to_save):
    minimal_user_info = user_user_score_dataframe(path)
    minimal_user_info.to_csv('table_2_users_nps.csv')

if __name__=="__main__":
    cols_to_drop = ['giver.userInfo.address.city','giver.userInfo.address.country','giver.userInfo.address.state','giver.userInfo.education','giver.userInfo.experienceAreas','giver.genderInfo.country','giver.genderInfo.firstName','giver.genderInfo.lastName','giver.userInfo.address.autocomplete']
