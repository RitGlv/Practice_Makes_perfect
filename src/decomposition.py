from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
import pandas as pd
import featurize

class decomposed(object):

    def __init__(self,df):
        self.data = df
        self.processed = None
        self.pca = None
        self.X_pca = None

    def process(self,categories,columns_to_drop):
        '''
        Get dataframe with only user information, columns with categorial data
        Save satet of processed data
        '''
        self.data = featurize. df_with_expertise(self.data,'experienceAreas1')
        self.data = self.data[columns_to_leave]
        self.data = featurize.dummify(self.data,categories)
        #scale

    def fit(self,cols_to_leave,categories,n_components):
        self.process(categories)
        self.pca = PCA(n_components = n_components)
        self.X_pca = pca.fit_transform(self.processed)

if __name__=="__main__":
    train_path = 'data/full_data_one_row_trainby_user.csv'
    test_path = 'data/full_data_one_row_testby_user.csv'
