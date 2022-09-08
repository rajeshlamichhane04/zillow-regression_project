#our useful imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def clean_zillow_data(df):
    #rename the columns
    df = df.rename(columns = {'bedroomcnt': 'bedroom', 'bathroomcnt':'bathroom', 'calculatedfinishedsquarefeet':'area',
       'taxvaluedollarcnt':'tax_value', 'yearbuilt':'year_built', 'fips':'county'})
    #replace county numbers with actual names
    df.county = df['county'].replace({6037:'Los Angeles',6059:'Orange',6111:'Ventura'})
    #cast bedroom to integer
    df.bedroom = df.bedroom.astype("int64")
    
    return df



def remove_outliers(df, k, col_list):
    for col in col_list:
        #get the 1st and 3rd quantiles
         q1, q3 = df[col].quantile([.25, .75]) 
        # calculate interquartile range
         iqr = q3 - q1   
        # get upper bound
         upper_bound = q3 + k * iqr   
         # get lower bound
         lower_bound = q1 - k * iqr   
        # dataframe without outliers
         df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df





def prep_zillow_data(df):
    #clean data function
    df = clean_zillow_data(df)
    #remove outlier function
    df = remove_outliers(df,1.5, ['bedroom', 'bathroom', 'area', 'tax_value', 'year_built'])
    return df


def split_zillow_data(df):
    #split dataframw into 80% train  and 20% test ,target is churn
     train_validate, test = train_test_split(df, test_size=.2, random_state=123)
     #split train further into 75% train, 25% validate
     train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    #return train,validate,test back to function
     return train, validate, test 