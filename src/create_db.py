import pymongo
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
'''
Connecting to mongodb to access data
'''
def connect(db_name,collection_name,mongo_uri=None):
    '''
    Create a mongodb connection
    '''
    mc = pymongo.MongoClient(mongo_uri)
    db = mc[db_name]
    collection = db[collection_name]
    return collection

def split_and_save(path):
    '''
    create test train split and save test data in a different file
    '''
    df = pd.read_csv(path).set_index('interviewId')
    path_name = path.split('.')[0]
    train_data, test_data = train_test_split(df,test_size=0.2,random_state=1)

    test_data.to_csv(path_name+'_test.csv')
    train_data.to_csv(path_name+'_train.csv')

def create_first_interview(paths,cols):
    '''
    Create files with only first interview data
    '''
    for path in paths:
        path_name = path.split('.')[0]
        df = pd.read_csv(path).set_index('interviewId')
        df = df[(df[cols[0]]==0) | (df[cols[1]]==0)]
        df.to_csv(path_name + '_only_first.csv')

    '''
    This will be use for preparing one data set with de-duped user info as a row
    '''

def load_dataframes(path):
    '''
    Loading file into dataframe, merging dataframes into one
    Assuming same column order between dataframes
    Assuming symmetry between columns
    '''
    df = pd.read_csv(path)
    df = rename_cols(df)
    no_of_cols = df.shape[1]
    df_user_1 = df.iloc[:,0:(no_of_cols/2)]
    df_user_2 = df.iloc[:,(no_of_cols/2):no_of_cols]
    df_user_2.columns = df_user_1.columns.values
    df_all = df_user_1.append(df_user_2).set_index('userId1')
    df_all.to_csv(path.split('.')[0]+'by_user.csv')


def rename_cols(df):
    '''
    Chenging column names to the last part in the name,
    indicating what the column really means
    '''
    no_of_cols = df.shape[1]
    first_cols = [col.split('.')[-1] for col in df.columns.values][0:(no_of_cols/2)]
    cols = [col+str(x) for x in range(1,3) for col in first_cols]
    df.columns = cols
    return df



if __name__=="__main__":
    collection_name = 'interviews'
    db_name = 'project'
    collection = connect(db_name,collection_name)
    split_and_save('data/full_data_one_row.csv')
    train_path = 'data/full_data_one_row_train.csv'
    test_path = 'data/full_data_one_row_test.csv'
    cols = ['feedbacks.0.giver.interviewsDonePriorToThisOne','feedbacks.1.giver.interviewsDonePriorToThisOne']
    create_first_interview([train_path,test_path],cols)
