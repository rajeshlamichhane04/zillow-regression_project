#standard imports
import pandas as pd
import numpy as np
import acquire
import prepare
import env
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression,LassoLars,TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,explained_variance_score


def show_outliers (df):
    #assign number of columns and rows
    _, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 15))
    for i, col in enumerate(df.columns[:5]):
        ax = ax.flatten()
        #plot boxplot
        sns.boxplot(x = col, data=df, ax=ax[i])
        ax[i].set_title(col)


def show_lineplot(df):
    #figure size
    plt.figure(figsize = (10,8))
    #plot lineplot 
    sns.lineplot(x = df.bedroom, y = df.tax_value, label = "bedroom")
    #plot lineplot
    sns.lineplot(x = df.bathroom, y = df.tax_value, label = "bathroom")
    plt.title("more bedrooms / bathrooms increases the tax value")
    plt.xlabel("bedroom / bathroom count")
    plt.legend()
    plt.show()



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
    X_train_degree5 = pf.fit_transform(X_train)
    #transoform x_test
    X_test_degree5 = pf.transform(X_test)

    # create the model object
    lm5 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm5.fit(X_train_degree5, y_train.tax_value)

     # predict test
    y_test['tax_value_pred_lm5'] = lm5.predict(X_test_degree5)

    # evaluate: test rmse
    rmse_test = mean_squared_error(y_test.tax_value, y_test.tax_value_pred_lm5)**(1/2)

    print("RMSE for Polynomial Model, degrees=", degree, "\ntest: ", rmse_test, "\nr^2: ", explained_variance_score(y_test.tax_value,
                                           y_test.tax_value_pred_lm5))


                                

#calculate rmse using actual and baseline mean
def get_baseline(y_train,y_validate):

    #apply baseline mean to train and validate
    y_train["baseline_mean"] = y_train.tax_value.mean()
    y_validate["baseline_mean"] = y_validate.tax_value.mean()

    #apply baseline median to train and validate
    y_train["baseline_median"] =y_train.tax_value.median()
    y_validate["baseline_median"] =y_validate.tax_value.median()
    RMSE_train_mean=mean_squared_error(y_train.tax_value,y_train.baseline_mean, squared = False)
    RMSE_validate_mean=mean_squared_error(y_validate.tax_value,y_validate.baseline_mean, squared = False)

    print("RMSE using Mean on \nTrain: ", round(RMSE_train_mean,2), "\nValidate: ", round(RMSE_validate_mean,2))
    print()

    #calculate rmse using actual and baseline mean
    RMSE_train_median= mean_squared_error(y_train.tax_value,y_train.baseline_median, squared = False)
    RMSE_validate_median= mean_squared_error(y_validate.tax_value,y_validate.baseline_median, squared = False)

    print("RMSE using Median on \nTrain: ", round(RMSE_train_median,2), "\nValidate: ", round(RMSE_validate_median,2))



def visualize_test_prediction(y_train,y_test):
    plt.figure(figsize = (12,12))
    #lmplot showing regression line
    sns.lmplot(data = y_test, x = "tax_value", y = "tax_value_pred_lm5",line_kws={'color':'red'})
    #annotation on regression line
    plt.annotate("Regression line",(800000,480000),rotation = 22, color ="red")
    #plot the baseline mean
    plt.plot(y_train.tax_value,y_train.baseline_mean,alpha=1, color="black", label='_nolegend_' )
    #annototion on baseline mean
    plt.annotate("Baseline: Prediction Using Mean", (600000, 380000))
    plt.xlabel("Actual Tax value")
    plt.ylabel("Predicted Tax value")
    plt.title("Polynomial Regression model prediction on Unseen data")

    plt.show()


def heatmap(train):
    # create a dataframe of correlation values, sorted in descending order
    corr = pd.DataFrame(train.corr().abs().tax_value).sort_values(by='tax_value', ascending=False)
    # establish figure size
    plt.figure(figsize=(10,8))
    # creat the heatmap using the correlation dataframe created above
    sns.heatmap(corr, annot=True)
    # establish a plot title
    plt.title('Features\' Correlation with tax value')
    # display the plot
    plt.show()

# peek at the distribution of my target varaible
def visualise_target(y_train):
    #plot histogram
    sns.histplot(y_train.tax_value)
    #title
    plt.title('Distribution of Tax Value')
    plt.show()



#set up for 2 sample 1 tailed independent test
def t_test_bedroom(train):
    #set up null hypothesis
    null_hyp = 'tax value of 5 bedroom property <= tax value of 2 bedroom property'
    print("null hypothesis:", null_hyp)
    #set up alternate hypothesis
    alt_hyp = 'tax value of 5 bedroom property > tax value of 2 bedroom property'
    print("alternate hypothesis:", alt_hyp)
    #alpha value
    alpha = 0.05
    #subset of tax value having 2 bedrooms
    two_bedroom_tax_value = train[train.bedroom ==2].tax_value
    #subset of tax value having 5 bedrooms
    five_bedroom_tax_value = train[train.bedroom == 5].tax_value
    #t test setup
    t,p = stats.ttest_ind(two_bedroom_tax_value, five_bedroom_tax_value)
    if p/2 < alpha:
        print()
        print("alpha: ", alpha , ",", "p/2: ", p/2)
        print("reject null hypothesis" )
        print("we conclude that the",alt_hyp)
    else:
        print("t", t , "p/2: ", p/2)
        print("fail to reject null hypothesis")
        print("we conclude that the",null_hyp)


#set up for 2 sample 1 tailed test
def t_test_bathroom(train):
    #null hypothesis set up
    null_hyp = 'tax value of 4 bathroom property <= tax value of 1 bathroom property'
    print("null hypothesis:", null_hyp)
    #alternate hypothesis set up
    alt_hyp = 'tax value of 4 bathroom property > tax value of 1 bathroom property'
    print("alternate hypothesis:", alt_hyp)
    #subset of tax value with 1 bathtoom
    one_bathroom_tax_value = train[train.bathroom == 1].tax_value
    #subset of tax valie with 4 bathroom
    four_bathroom_tax_value = train[train.bathroom == 4].tax_value
    #2 sample 1 tailed t test set up
    alpha = 0.05
    t,p = stats.ttest_ind(one_bathroom_tax_value, four_bathroom_tax_value)
    if p/2 < alpha:
        print()
        print("alpha: ", alpha , ",", "p/2: ", p/2)
        print("reject null hypothesis" )
        print("we conclude ", alt_hyp)
    else:
        print()
        print("alpha: ", alpha , ",", "p/2: ", p/2)
        print("fail to reject null hypothesis")
        print("we conclude", null_hyp)