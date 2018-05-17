# %% ENV
import pandas as pd

# %% DESCRIPTION
# data InOut
"""
collection of function to handle dataframe
"""

# %% FUNCTIONS

def df_printGeneralInfo(df):
    '''
    :param df: the dataframe the use
    print general info about the dataframe
    '''
    print('---Dataset---')
    print('-> Info')
    print(df.info())
    print("=============================================================")
    print('-> Describe')
    print(df.describe(include = "all"))
    print("=============================================================")
    print('-> Top 5')
    print(df.head(5))
    print("=============================================================")
    print('-> Bottom 5')
    print(df.tail(5))
    print("=============================================================")
    
    
def df_checkMissing(df):
    '''
    :param df: the dataframe the use
    check missing data, only by column for now
    '''
    #counts all null cells in a row
    total = df.isnull().sum().sort_values(ascending=False) 
    #sees what percent of the data is null
    percent = ((df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)*100)
    #combines the two matrixies
    missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent']) 
    print(missing_data)
    return missing_data
    
    
def df_splitDataset(df,validation_step):
    '''
    :param df: the dataframe the use
    :param validation step: how many row to cut at the end of the df
    take a df and cut in two dfs 0 to validation_step to train set and the
    other part to validation setp
    '''
    return df[len(df)-validation_step:], df[:len(df)-validation_step]

    
    