Zillow Regression Project

<hr style="border-top: 50px groove green; margin-top: 1px; margin-bottom: 1px"></hr>

## Overview

> - This repository houses a complete files for this project. These files are

> -  README.md: It contains the outline of this project
> -  Zillow_project_workbook.ipynb: This jupyter notebook consists of data science pipeline to help me build model to predict the property value.
> -  acquire.py: It consists of codes for data acquition process
> -  prepare.py: It consists of codes for data cleaning and data split
> -  viz.py: It consists of codes for functions for statistcial test, visualization, modeling and feature selection.
> -  .igitignore file: It consists of file names that I do not want to be push to git

## Project Goals

> - The goals of this project are to find key drivers of the property value for the Single Family Residential Properties, to construct a Machine Learning Regression Model that can predict the tax values of such properties and offer recommendations on how to improve future predictions.

## Project Description

> -  Zillow has a model that is designed to predict the property tax assessed values("taxvaluedollarcnt") of single family properties that had a transaction during 2017. This project aims to make improvement on the previous model. Utilizing Data Science pipelines, this project will study various features that affects the value of property, I will find out the driving factors of such value, build up a model that can beat the previous model. I will also make some recommendations based on our findings.

## Initial Questions

> -  What sort of relationship is there between number of bedroom and the tax value?
> -  What sort of relationship is there between number of bathroom and the tax value?
> -  Is there significant correlation between area of property and its value?
> -  How does tax value compare by counties throughtout the years?

## Data Dictionary

> - 1. bathroomcnt = Number of bathrooms in home including fractional bathrooms
> - 2. bedroomcnt = Number of bedrooms in home
> - 3. calculatedfinishedsquarefeet	= Calculated total finished living area of the home
> - 4. fips = Federal Information Processing Standard code
> - 5. yearbuilt = The Year the principal residence was built
> - 6. taxvaluedollarcnt = The total tax assessed value of the property


## Steps to Reproduce

> -  To clone this repo, use this command in your terminal https://github.com/rajeshlamichhane04/zillow-regression_project
> -  You will need login credentials for MySQL database hosted at data.codeup.com
> -  You will need an env.py file that contains hostname,username and password
> -  The env.py should also contain a function named get_db_url() that establishes the string value of the database url.
> -  Store that env file locally in the repository.

## The plan

> - I set up my initial questions during this phase. I made the outline of possible exploration techniques and hypothesis testing I can use.

##  Acquisition

> - I obtanied Zillow data by using SQL query via MySQL database. I saved the file locally as a csv. I used the code created at acquire.py.

## Preparation

> - I accomplished this using prepare.py file. I cleaned up the data by removing outliers and renaming column names. I also converted bedroom datatype from float to integer. I also assigned fips with thier corresponding county names. Further, I split my data into train (56%), validate(24%) and test (20%). I scaled my features before fitting them in to my models.

> ##  Exploration

> - I used only my training data for exploration. I answered my initial question using various seaborn and matplotlib visualizations and hypothesis testings. I also constructed a corelation chart. This helped me identify my features.

##  Modeling

> - First, I scaled my top features (area, bathroom and year built) using MinMaxScaler. I used RMSE using mean for the baseline and used Linear Regression(OLS), Lassor + Lars, General Linear Regression(Tweedier Regressor) and Polynomial Regression to evalute my model. My top performing model was Polynomial Regression with degree 5. It beat baseline RMSE with only 11%.

## Prediction delivery

> - I was able to create a predicted tax value using my top performing model.

## Key Takeaways and Recommendations

> - After modeling the zillow data using five features (bathrooms, bedrooms, area, year built, and fips), the Polynomial Model with degree 5 produced the best results with high RMSE values of 215107, 215571 and 218218 for train, validate and test dataset. My model beat the baseline by only 11%.There were a numbers of outliers in our data. I removed outliers using Tukey method where I lost 16% of my data. However, I still had outliers present that may have affected my RMSE from my models. I recommend exploring more features such as pool and garage. Also having data on crime rate, happiness index, household income, and school zone would be greatly benificials as they tend to affect housing market widely.
