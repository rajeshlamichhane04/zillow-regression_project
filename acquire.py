#imports
import pandas as pd
import numpy as np
import env
import os

#sql query to pull data from Zillow database
sql = """
SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt,fips
FROM properties_2017
LEFT JOIN predictions_2017
USING (parcelid)
JOIN propertylandusetype 
USING (propertylandusetypeid)
WHERE propertylandusedesc = 'Single Family Residential'
AND transactiondate BETWEEN '2017-01-01' AND '2017-12-31'
"""

#connection set ip
def conn(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


#make function to acquire from sql
def new_zillow_data():
    #read data from zillow database using crediantials 
    df = pd.read_sql(sql,conn("zillow"))
    return df



def get_zillow_data():
    #check if csv format of data is present locally
    if os.path.isfile("zillow.csv"):
        #if present locally, read that data
        df = pd.read_csv("zillow.csv", index_col = 0)
    else:
        #if not locally found, run sql querry to pull data
        df = new_zillow_data()
        #cache the data as csv locally
        df.to_csv("zillow.csv")
    return df
