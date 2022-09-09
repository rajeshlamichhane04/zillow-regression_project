import pandas as pd
import numpy as np
import acquire
import prepare
import env
import os
import seaborn as sns
import matplotlib.pyplot as plt
#for hypothesis tests
import scipy.stats as stats
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression,LassoLars,TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,explained_variance_score

def show_outliers (df):
    _, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 15))
    for i, col in enumerate(df.columns[:5]):
        ax = ax.flatten()
        sns.boxplot(x = col, data=df, ax=ax[i])
        ax[i].set_title(col)


def show_lineplot(df):
    plt.figure(figsize = (10,8))
    sns.lineplot(x = df.bedroom, y = df.tax_value, label = "bedroom")
    sns.lineplot(x = df.bathroom, y = df.tax_value, label = "bathroom")
    plt.title("more bedrooms / bathrooms increases the tax value")
    plt.xlabel("bedroom / bathroom count")
    plt.legend()
    plt.show()

def run_t_test(df):
    null_hyp = 'tax value for 4 bedroom property =< tax value for 2 bedroom property'
    alt_hyp = 'tax value for 4 bedroom property > tax value for 2 bedroom property'
    alpha = 0.05
    two_bedroom_tax_value = df[df.bedroom ==2].tax_value
    four_bedroom_tax_value = df[df.bedroom == 4].tax_value
    t,p = stats.ttest_ind(two_bedroom_tax_value, four_bedroom_tax_value)
    if p < alpha:
        print("reject null hypothesis" )
        print("we conclude ", alt_hyp)
    else:
        print("fail to reject null hypothesis")
        print("we conclude", null_hyp)


def select_rfe(X,y,  n_features_to_select = 3):
    #create the model
    rfe=RFE(LinearRegression(), n_features_to_select = n_features_to_select) 
    #fit the model
    rfe.fit(X,y)
    #use get_support()
    return X.columns[rfe.get_support()]


def scale_data(train,validate,test,columns):
    #make the scaler
    scaler = MinMaxScaler()
    #fit the scaler at train data only
    scaler.fit(train[columns])
    #tranforrm train, validate and test
    train_scaled = scaler.transform(train[columns])
    validate_scaled = scaler.transform(validate[columns])
    test_scaled = scaler.transform(test[columns])
    
    # Generate a list of the new column names with _scaled added on
    scaled_columns = [col+"_scaled" for col in columns]
    
    #concatenate with orginal train, validate and test
    scaled_train = pd.concat([train.reset_index(drop = True),pd.DataFrame(train_scaled,columns = scaled_columns)],axis = 1)
    scaled_validate = pd.concat([validate.reset_index(drop = True),pd.DataFrame(validate_scaled, columns = scaled_columns)], axis = 1)
    scaled_test= pd.concat([test.reset_index(drop = True),pd.DataFrame(test_scaled,columns = scaled_columns)],axis = 1)
    
    return scaled_train,scaled_validate,scaled_test




def linear_regression(X_train,y_train,X_validate,y_validate):

    # create the model object
    lm = LinearRegression(normalize = True)
    
    # Fit the model
    lm.fit(X_train, y_train.tax_value)
    
    # Predict y on train
    y_train['tax_value_pred_lm'] = lm.predict(X_train)
    # predict validate
    y_validate['tax_value_pred_lm'] = lm.predict(X_validate)
    
    # evaluate: train rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lm) ** (1/2)

    # evaluate: validate rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lm) ** (1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    


def lasso_lars(X_train, y_train, X_validate, y_validate, alpha):
    
    # create the model object
    lars = LassoLars(alpha)

    # fit the model.
    lars.fit(X_train, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_lars'] = lars.predict(X_train)
    # predict validate
    y_validate['tax_value_pred_lars'] = lars.predict(X_validate)
    # evaluate: train rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lars)**(1/2)

    # evaluate: validate rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lars)**(1/2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    

def Tweedie_regressor(X_train, y_train, X_validate, y_validate, power, alpha):

    # create the model object
    glm = TweedieRegressor(power=power, alpha=alpha)

    # fit the model to our training data.
    glm.fit(X_train, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_glm'] = glm.predict(X_train)
    # predict validate
    y_validate['tax_value_pred_glm'] = glm.predict(X_validate)
    # evaluate: train rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_glm)**(1/2)
    # evaluate: validate rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_glm)**(1/2)

    print("RMSE for GLM using Tweedie, power=", power, " & alpha=", alpha, 
        "\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))


def polynomial_regression(X_train, y_train, X_validate, y_validate, degree):
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree= degree)
    
    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)
    

    # transform X_validate_scaled 
    X_validate_degree2 = pf.transform(X_validate)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: train rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lm2)**(1/2)

    # predict validate
    y_validate['tax_value_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: validate rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lm2)**(1/2)

    print("RMSE for Polynomial Model, degrees=", degree, "\nTraining/In-Sample: ", rmse_train, 
        "\nValidation/Out-of-Sample: ", rmse_validate)


def test_prediction(X_train,y_train,X_test,y_test,degree):
    pf = PolynomialFeatures(degree= degree)
    
    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)
    
    X_test_degree2 = pf.transform(X_test)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.tax_value)

     # predict test
    y_test['tax_value_pred_lm2'] = lm2.predict(X_test_degree2)

    # evaluate: test rmse
    rmse_test = mean_squared_error(y_test.tax_value, y_test.tax_value_pred_lm2)**(1/2)

    print("RMSE for Polynomial Model, degrees=", degree, "\ntest: ", rmse_test, "\nr^2: ", explained_variance_score(y_test.tax_value,
                                           y_test.tax_value_pred_lm2))

    print()
    plt.figure(figsize = (15,6))
    sns.lmplot(data = y_test, x= "tax_value", y="tax_value_pred_lm2",line_kws={'color':'red'} )
    plt.title("actual vs predicted tax value")
    plt.xlabel("actual tax value")
    plt.ylabel("predicated tax value")

    plt.show()