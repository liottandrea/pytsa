# %% ENV
# import psycopg2
import pyodbc as pod
import pandas as pd
import pandas.io.sql as psql
import json
import os

# %% DESCRIPTION
# data InOut
"""
collection of function to handle csv and database tables in and out
"""

# %% FUNCTIONS

def pg_connect2db():
    '''
    # open connection to postgre server with psycopg2
    '''
    # define our connection string
    conn_string = "host='localhost' \
        dbname='mydb' \
        user='postgres' \
        password='secret'"

    # print the connection string we will use to connect
    print("Connecting to database\n%s" % conn_string)
    # get a connection, if a connect cannot be made
    # an exception will be raised here
    # conn = psycopg2.connect(conn_string)
    return conn_string



def pg_table2df(conn, table):
    '''
    :param conn: connection object to use
    :param table: table to retrive
    Retrieve table from postgre server with psycopg2
    '''
    return psql.read_sql_query("select * from %s" % table, conn)



def csv_df2Xy(csv_file, x_cols, y_col):
    '''
    :param csv_file: csv to read
    :param x_cols: cols to put in X
    :param y_col: col to put in y
    Read a csv and move the content to two tables X and y
    '''
    # read csv
    dataset = pd.read_csv(csv_file)
    # divide x and y
    X = dataset[x_cols]
    y = dataset[y_col]
    # force to dataframe in case only one column
    if isinstance(X, pd.Series):
        X = X.to_frame()
    if isinstance(y, pd.Series):
        y = y.to_frame()
    # return
    return X, y


def odbc_connectToServer(server, database):
    '''
    :param server: ssms server to connect.
    :param database: ssms database to connect.
    Makes a connction to SQL server
    Initial version simple - more parameterise later
    '''
    # load default setting
    default_list = json.load(open('./setting/config.json'))["SQL"]
    if server is None:
        server = default_list["server"]
    if database is None:
        database = default_list["database"]
    # Make connection
    sql_conn = pod.connect(
        "Driver={ODBC Driver 13 for SQL Server};\
        Server=%s;\
        Database=%s;\
        Trusted_Connection=yes;" % (server,database))
    # Return
    return sql_conn


def odbc_StoreProc2Df(stored_proc, **kwargs):
    '''
    :param stored_proc: store proc to fire
    :param **kwargs: optional args (eg server).
    Initial version simple - parameterise later
    '''
    # optional parameter
    server = kwargs.get('server', None)
    database = kwargs.get('database', None)
    # Make connection
    sql_conn = odbc_connectToServer(server, database)
    # Create the cursor
    sql_cursor = sql_conn.cursor()
    # Run store proc
    sql_cursor.execute('exec %s;' % (stored_proc))
    # Get column names fro the cursor description using a list comprehension
    column_names = [column[0] for column in sql_cursor.description]
    # Collect the results (rows or data as alist)
    sql_results = sql_cursor.fetchall()
    # Close theconnection
    sql_conn.close()
    # Convert list to pandas
    results = pd.DataFrame.from_records(sql_results, columns=column_names)
    # Return
    return results


def odbc_Query2Df(query, **kwargs):
    '''
    :param query: query to execute on tables or views.
    :param **kwargs: optional args (eg server).
    '''
    # optional parameter
    server = kwargs.get('server', None)
    database = kwargs.get('database', None)
    # Make connection
    sql_conn = odbc_connectToServer(server, database)
    # Run query
    sql_results = pd.read_sql(sql=query, con=sql_conn)
    # Close theconnection
    sql_conn.close()
    # Return
    return sql_results

def folder_listFiles2Df(folder):
    '''
    :param folder: folder to use to list files
    list all files in a folder, skip subfolder
    '''
    file_list = [
        f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))]

    df = pd.DataFrame(data = {
        'FileName': list(map((lambda x:  os.path.splitext(x)[0]), file_list)),
        'Location': [folder + "\\" +  s for s in file_list],
        'Extension':list(map((lambda x: os.path.splitext(x)[1]),file_list))
        })
    return df


# %% EXAMPLES
"""
# conn to db
db_conn = db_connect2db()
# download table
table = db_table2df(db_conn,"my_table")
# Create the sql string
sql_string = "dpretailer.uspRetrieveDPRetailerActuals
@DomainProductGroupName = '%s', @CountryISOCode = %s" % (
     'AIR CONDITIONER',
    504
)
# Grab the actuals from database
dat = odbc_StoreProc2Df(sql_string)
"""
