# -*- coding: utf-8 -*-
"""
Code for my Titanic submission for Kaggle's competition


Colin Cambo
cwb35@wildcats.unh.edu
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

#Titles used for more accurately imputing ages
titles = {
    'mr.':0, #Adult man
    'mrs.':1, #Married woman
    'miss.':2, #Non-married woman
    'master':3 #Young child
}

def calculate_medians(train, test):
    """
    Takes in train & test files and outputs median age & fare for every title
    """
    
    #Reads in both train and test and appends them
    df = pd.read_csv(train).drop(['Survived'], 
                                            axis=1).append(pd.read_csv(test))
    
    
    df['Title'] = [sum([titles[title] if title in name.lower() else 0 
                    for title in titles.keys() ]) 
                    for name in df.Name.tolist()]
                        
    median_ages = {i:np.median([age for age, title in zip(df.Age, df.Title) 
                    if title==i and np.isnan(age)==False])
                    for i in range(len(titles))}
                        
    median_fare = {i:np.median([fare for fare, title in zip(df.Fare, df.Title) 
                                if title==i and np.isnan(fare)==False])
                                for i in range(len(titles))}
                                    
    return median_ages, median_fare

def extract_features(filename, median_ages, median_fare):
    """
    Reads in DataFrame & median age & fare for imputation and performs feature
    extraction
    """
    
    df = pd.read_csv(filename)
    
    df['Title'] = [sum([titles[title] if title in name.lower() else 0 
                    for title in titles.keys() ]) 
                    for name in df.Name.tolist()]
                        
    df['Name_Length'] = df.Name.apply(len)
    df['Name_Tokens'] = df.Name.apply(lambda x: len(x.split()))
    
    df['Age'] = [median_ages[title] if np.isnan(age)==True else age 
                    for title, age in zip(df.Title, df.Age)]
    
    df['Sex'] = [1 if x=='male' else 0 for x in df['Sex']]
    df['Fare'] = [median_fare[title] if np.isnan(fare)==True else fare 
                    for title, fare in zip(df.Title, df.Fare)]
    
    return df

median_age, median_fare = calculate_medians('train.csv', 'test.csv')

train = extract_features('train.csv', median_age, median_fare)
test = extract_features('test.csv', median_age, median_fare)

cols = ['Age', 'Sex', 'Fare', 'Pclass', 'SibSp', 'Title', 'Name_Length',
        'Name_Tokens']

param_grid = {
                 'max_depth' : [4,5,6,7,8],
                 'n_estimators': [100, 200, 250,300],
                 'criterion': ['gini','entropy']
                 }

cross_validation = StratifiedKFold(train['Survived'], n_folds=5)

grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, 
                    cv=cross_validation)

grid.fit(train[cols], train['Survived'])

y_pred = grid.best_estimator_.predict(test[cols])

titanic = pd.DataFrame({
        'PassengerId': test.PassengerId,
        'Survived': y_pred})

titanic.to_csv('titanic.csv', index=False)