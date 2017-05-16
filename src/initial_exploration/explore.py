import pandas as pd
import matplotlib.pyplot as plt

def crosstabs(df,col1,col2,count=False):
    '''
    Create a crosstab view to see proportion differences in DataFrame df for columns 1 and 2.
    Will present the precentage unless count is True
    '''
    cross = pd.crosstab(df[col1],df[col2])
    precentage = cross.div(cross.sum(axis=1),axis=0)
    return precentage

if __name__=="__main__":
    df = pd.read_csv('flattened_data.csv')
    describe = df.describe()
    correlations = df.corr()
    #some general stats:
    #check relationship between nps and no.of interviews:
    no_interviews_nps = df.groupby('giver.interviewsDonePriorToThisOne').mean()['giver.nps'].sort_values

    #distribution of no. of interviews done:
    interviews_done = df.groupby('giver.interviewsDonePriorToThisOne').count()['giver.nps'].sort_values()

    #percentage of users to continue to the next interview with respect to the first:
    interviews_done_p = interviews_done/interviews_done[0]
