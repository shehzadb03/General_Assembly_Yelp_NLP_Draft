# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:51:45 2015

@author: shehzadbashir
"""
# Questions for class hour
# Upload to repo issue
# Multiple Reviews/rating - how to combine? 
# Train labels have multiple entries in train_label file 




import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

#JSON to CSV

#Yelp Business
with open('yelp_business.json', 'rU') as f:
    data = [json.loads(row) for row in f]
yelp_business2 = pd.DataFrame(data)
yelp_business2.to_csv('yelp_business.csv', index=False)

#Yelp Review
with open('yelp_review.json', 'rU') as f:
    data = [json.loads(row) for row in f]

yelp_review2 = pd.DataFrame(data)

yelp_review2.to_csv('yelp_review.csv', index=False)

#Yelp Checkin
with open('yelp_checkin.json', 'rU') as f:
    data = [json.loads(row) for row in f]

yelp_checkin2 = pd.DataFrame(data)

yelp_checkin2.to_csv('yelp_checkin.csv', index=False)

#Yelp Tip
with open('yelp_tip.json', 'rU') as f:
    data = [json.loads(row) for row in f]

yelp_tip2 = pd.DataFrame(data)

yelp_tip2.to_csv('yelp_tip.csv', index=False)

#Yelp User
with open('yelp_user.json', 'rU') as f:
    data = [json.loads(row) for row in f]

yelp_user2 = pd.DataFrame(data)

yelp_user2.to_csv('yelp_user.csv', index=False)

# Read file
id_map = pd.read_csv('restaurant_ids_to_yelp_ids.csv')
yelp_business = pd.read_csv('yelp_business.csv')

# Merge/Join

data = pd.merge(id_map, yelp_business, on='business_id', how='inner')


# Group train_lables by restaurant id and sum the # of stars (violations)

train_data = pd.read_csv('train_labels.csv')

train_data2 = train_data.groupby('restaurant_id').sum()

#Join train_lables with resaurant business data with # of stars (violations) totaled
data2 = pd.merge(data, train_data2, left_on='restaurant_id', how='inner', right_index=True)

# drop columns not needed

data3 = data2.drop(['yelp_id_1', 'yelp_id_2', 'yelp_id_3', 'open', 'hours', 'type', 'id', 'state'],axis=1)

# Count null values

data3.isnull().sum()

#Plot correlation

sns.pairplot(data3)

sns.heatmap(data3.corr())

## From Review data - check correlation between rating and length of review

review = pd.read_csv('yelp_review.csv')

text_list = [1 if len(row) < 100 else 0 for row in review.text]

len(review.text)
len(text_list)

review['text_list'] = text_list

review['text_length']=[len(row) for row in review.text]

sns.pairplot(review, x_vars=['text_length'], y_vars='stars', size=6, aspect=0.7)

sns.pairplot(data3, x_vars=['stars'], y_vars='total_violations', size=6, aspect=0.7)

#Visualizations

data3.head(2)

sns.pairplot(data3, x_vars=['*', '**', '***'], y_vars='review_count', size=6, aspect=0.7, kind='reg')

data3['*'].value_counts().plot(kind='bar')
data3['*'].describe()

# Linear Regression to predict total number of violations based on stars and review count

data3['total_violations'] = data3['*'] + data3['**'] + data3['***']

feature_cols = ['review_count', 'stars']
X = data3[feature_cols]
y = data3['total_violations']

linreg = LinearRegression()
linreg.fit(X, y)

linreg.intercept_
linreg.coef_

zip(feature_cols, linreg.coef_)

# Result = one extra review is associated with .05 increase in total # of violations. 

# Train/test/split to check RMSE

def train_test_rmse(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))

train_test_rmse(X, y)

feature_cols = ['review_count']
X = data3[feature_cols]
y = data3['total_violations']


# indicator of whether restaurant had violation or not
data3['violation_indicator'] = [1 if row > 0 else 0 for row in data3['total_violations']]

## Check by key words

data3['love'] = data3.------.str.contains('love', case=False).astype(int)
data3['hate'] = data3.------.str.contains('hate', case=False).astype(int)





