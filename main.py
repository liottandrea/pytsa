#%% ENV
# extra functions
import dataInOut as myio

#%% DESCR
'''
IntProj with Python POC
Simplified version of forecast process used in R
'''

#%% SETTING
country_key = 826
domain_product_group_name = "AIR CONDITIONER"

#%% INPUT
# retrieve actual
# Create the sql string
sql_string = "dpretailer.uspRetrieveDPRetailerActuals \
    @DomainProductGroupName = '%s', @CountryISOCode = %s"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     % (
    domain_product_group_name, country_key)
# fire store proc
df = myio.odbc_StoreProc2Df(sql_string)

# I think that sometime we have a problem in identify the right type
right_type = {
    "CountryISOCode": "int64",
    "DateKey": "int64",
    "DomainProductGroupName": "category",
    "DistributionTypeName": "category",
    "TotalUnitsSold": "float",
    "TotalValueLocal": "float"}

df = df.astype(right_type)

print('---Dataset---')
print('-> General Info')
print(df.info())
print("=============================================================")
print(df.describe(include = "all"))
print("=============================================================")
print(df.head(5))
print("=============================================================")
print(df.tail(5))
print("=============================================================")


#%% PREP

#%% MAIN

#%% OUTPUT
