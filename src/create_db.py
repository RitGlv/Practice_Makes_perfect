import pymongo
import pandas as pd
from sklearn.model_selection import train_test_split
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


if __name__=="__main__":
    collection_name = 'interviews'
    db_name = 'project'
    collection = connect(db_name,collection_name)
    split_and_save('data/full_data_one_row.csv')
