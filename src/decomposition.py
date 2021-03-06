from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
import pandas as pd
import featurize
reload (featurize)

class decomposed(object):

    def __init__(self,df):
        self.data = df
        self.processed = None
        self.pca = None
        self.X_pca = None

    def process(self,categories,columns_to_leave):
        '''
        Get dataframe with only user information, columns with categorial data
        Save satet of processed data
        '''
        self.data = self.data[columns_to_leave]
        self.processed = featurize. df_with_expertise(self.data,'experienceAreas1')
        self.processed.pop('experienceAreas1')
        self.processed = featurize.dummify(self.processed,categories)
        self.processed = featurize.feturized_by_unique_user(self.processed)
        #scale

    def fit(self,cols_to_leave,categories,n_components):
        self.process(categories,cols_to_leave)
        self.pca = PCA(svd_solver='full',n_components = n_components)
        self.X_pca = self.pca.fit_transform(self.processed)

    def assess(self):
        '''
        Check the amount of explained variance within the components
        '''
        var = self.pca.explained_variance_ratio_
        return sum(var)

if __name__=="__main__":
    train_path = 'data/full_data_one_row_trainby_user.csv'
    test_path = 'data/full_data_one_row_testby_user.csv'
    df = pd.read_csv(train_path).set_index('userId1')
    users_df = df[['interviewsDonePriorToThisOne1','totalMatch1']]
    cols_to_leave = ['selfPrep1', 'experienceAreas1','experienceInYears1','degree1', 'status1','studyArea1','interviewsDonePriorToThisOne1','likable1','hiring1','communication1','asInterviewer1','problemSolving1','codingSkills1']

    categories = ['degree1','status1','studyArea1']

    pca = decomposed(df)
    pca.fit(cols_to_leave,categories,3)
