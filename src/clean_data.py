import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import featurize

class CleanedData(object):

    def __init__(self,path):
        self.df_all = pd.read_csv(path)
        self.df_transformed = None
        self.y = None

    def rename_cols(self,df):
        '''
        Chenging column names to the last part in the name,
        indicating what the column really means
        '''
        no_of_cols = df.shape[1]
        first_cols = [col.split('.')[-1] for col in df.columns.values][0:(no_of_cols/2)]
        cols = [col+str(x) for x in range(1,3) for col in first_cols]
        df.columns = cols
        return df

    def time_to_datetime(self):
        '''
        Convert str time indicator to DateTime
        '''
        for user in range(1,3):
            self.df_all['interviewTime'+str(user)] = pd.DatetimeIndex(self.df_all['interviewTime'+str(user)])

    def df_with_needed_cols(self,cols):
        '''
        Input: list of columns to leave in the DataFrame
        modifies self.df_transformed for future use to featurize the data
        '''
        self.df_transformed = self.df_all[cols]

    def fit_data(self,cols_to_leave,cols_to_dummify,target,other_target_to_remove=None,constant_and_drop=False):
        '''
        Method using to creat training data, including train/test split.
        For new data, use "data_clean"
        '''
        self.df_all = self.rename_cols(self.df_all)
        self.df_all = featurize.same_columns(self.df_all,'country1','country2')
        # self.df_all = featurize.is_us(self.df_all)
        self.time_to_datetime()
        self.df_with_needed_cols(cols_to_leave)
        self.df_transformed = featurize.dummify(self.df_transformed,cols_to_dummify)
        self.create_y(target,other_target_to_remove)
        return self.df_transformed, self.y


    def create_y(self,target,other_target_to_remove):
        self.y = self.df_transformed.pop(target)
        if other_target_to_remove:
            self.df_transformed.pop(other_target_to_remove)
    #
    # def load_dataframes(self):
    #     '''
    #     Loading file into dataframe, merging dataframes into one
    #     Assuming same column order between dataframes
    #     '''
    #     self.df_user_1 = pd.read_csv(self.paths[0])
    #     self.df_user_2 = pd.read_csv(self.paths[1])
    #     self.df_user_2.columns = self.df_user_1.columns.values
    #     self.df_all = self.df_user_1.append(self.df_user_2)
    #     self.df_user_1 = self.rename_cols(self.df_user_1)
    #     self.df_user_2 = self.rename_cols(self.df_user_2)
    #
    #
    # def create_data(self):
    #     self.load_dataframes()
    #     self.df_all = self.rename_cols(self.df_all)
    #     X_train,X_test = train_test_split(self.df_all,test_size=0.2,random_state=1)
    #     self.df_all = X_train
    #     return X_train, X_test
