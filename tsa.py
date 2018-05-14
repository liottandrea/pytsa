# %% ENV
# import psycopg2
import pyodbc as pod
import pandas as pd
import json
import os

# %% DESCRIPTION
# data InOut
"""
collection of function to handle time series and analyse them
"""


def df_col2Series(df, series_column):
    '''
    :param df: df to use
    :param series_column: colum to extract
    take a columns from a dataframe and create a Series
    using the index as date 
    '''
    # take a column as value
    # and create a time series
    y = pd.Series(
        data=df[series_column].values, name=series_column, index=df.index)
    y.sort_index(inplace=True)
    return y
