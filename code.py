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
import scipy as sp
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from textblob import TextBlob, Word
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor

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

# Merge/Join business with mapping ID
data = pd.merge(id_map, yelp_business, on='business_id', how='inner')

# Load Yelp_user data and rename features
user = pd.read_csv('yelp_user.csv')
user.rename(columns={'average_stars': 'user_avg_stars', 'review_count': 'user_review_count'}, inplace=True)

# Split Boston (historinc violations) training dataset to train and test data

train_data = pd.read_csv('train_labels.csv')

#Convert date form str to date format
train_data['date'] = pd.to_datetime(train_data.date)

train_data = train_data.sort('date')
train_data.rename(columns={'*': 'minor', '**': 'major', '***': 'severe', 'id': 'inspect_id'}, inplace=True)

## Following function does not work for multiple reasons
#train=[]
#for row in train_data:
    #if train_data[train_data.date < 2014-03-26]:
        #train.append(row)

# Function to split data for training and testing by date
import datetime
filter_date = datetime.date (2014,3,26)

train=[]
test = []

for ind, row in enumerate(train_data.iterrows()):   
    y,m,d = str(train_data.date[ind]).split('-') 
    y = int(y)
    m = int(m)        
    d = int(d.split()[0])
    yelp_date = datetime.date(y,m,d)
    
    date = train_data['date'][ind]
    inspect_id = train_data['inspect_id'][ind]
    restaurant_id = train_data['restaurant_id'][ind]
    minor = train_data['minor'][ind]
    major = train_data['major'][ind]
    severe = train_data['severe'][ind]
    row_to_append = [date,inspect_id, restaurant_id, minor, major, severe]    

    if yelp_date < filter_date:
        train.append(row_to_append)        
    else:
        test.append(row_to_append)
    
# Reassigning column names
columns = ['date', 'inspect_id', 'restaurant_id', 'minor', 'major', 'severe']
train = pd.DataFrame(train)
train.columns = columns
test = pd.DataFrame(test)
test.columns = columns

# Averages minor, major, and severe violations by restaurant ID in both datasets
train = train.groupby('restaurant_id').mean()
test = test.groupby('restaurant_id').mean()

# Add column to sum violations
train['total_violations'] = train['minor'] + train['major'] + train['severe']
test['total_violations'] = test['minor'] + test['major'] + test['severe']

## Split and clean reviews file

review = pd.read_csv('yelp_review.csv')
review = review.drop('type',axis=1)

#Convert date form str to date format
review['date'] = pd.to_datetime(review.date)

train_review=[]
test_review = []

for ind, row in enumerate(review.iterrows()):   
    y,m,d = str(review.date[ind]).split('-') 
    y = int(y)
    m = int(m)        
    d = int(d.split()[0])
    review_date = datetime.date(y,m,d)

    business_id = review['business_id'][ind]
    date = review['date'][ind]
    review_id = review['review_id'][ind]
    stars = review['stars'][ind]
    text = review['text'][ind]
    user_id = review['user_id'][ind]
    votes = review['votes'][ind]
    row_to_append = [business_id, date, review_id, stars, text, user_id, votes]    

    if review_date < filter_date:
        train_review.append(row_to_append)        
    else:
        test_review.append(row_to_append)
        
# Reassigning column names and turning into DataFrame
columns = ['business_id', 'date', 'review_id', 'stars', 'text', 'user_id', 'votes']
train_review = pd.DataFrame(train_review)
train_review.columns = columns
test_review = pd.DataFrame(test_review)
test_review.columns = columns   

# Get text length for each dataset
train_review['text_length'] = [len(row)for row in train_review.text]
test_review['text_length']=[len(row) for row in test_review.text]
        
# concatenate text and average stars in yelp review files (train & test)

# Concatenate reviews by business ID on both datasets 
train_review2 = train_review.groupby(['business_id'])['text'].apply(lambda x: ', '.join(x)).reset_index()
train_review2 = pd.DataFrame(train_review2)

test_review2 = test_review.groupby(['business_id'])['text'].apply(lambda x: ', '.join(x)).reset_index()
test_review2 = pd.DataFrame(test_review2)

#Merge Boston train & test data with Yelp Business Data
train2 = train.reset_index()
train2 = pd.merge(data, train2, on='restaurant_id', how='inner')

test2 = test.reset_index()
test2 = pd.merge(data, test2, on='restaurant_id', how='inner')

# Merge training and test datasets with review (text)
train3 = pd.merge(train2, train_review2, on='business_id', how='inner')
test3 = pd.merge(test2, test_review2, on='business_id', how='inner')

#Clean version of both datasets for analysis - exclude features that won't be used
test4 = test3.drop(['yelp_id_1', 'yelp_id_2', 'yelp_id_3', 'attributes', 'full_address', 'hours', 'name', 'city', 'inspect_id', 'type', 'state'], axis=1)
train4 = train3.drop(['yelp_id_1', 'yelp_id_2', 'yelp_id_3', 'attributes', 'full_address', 'hours', 'name', 'city', 'inspect_id', 'type', 'state'], axis=1)

# Clean for exploratory without text analysis - only for training data
train4_clean = train4.drop(['restaurant_id', 'business_id', 'categories', 'neighborhoods', 'open', 'text'], axis=1)

#Get the average length of a review from train_review and test_review file
train_text1 = train_review.groupby(['business_id'])['text'].apply(lambda x: ', '.join(x)).reset_index()
train_text2 = train_review.groupby(['business_id']).text_length.mean()
train_text3 = pd.DataFrame(test123).reset_index()
train_len_text = pd.merge(train_text1, train_text3, on='business_id', how='inner')

test_text1 = test_review.groupby(['business_id'])['text'].apply(lambda x: ', '.join(x)).reset_index()
test_text2 = test_review.groupby(['business_id']).text_length.mean()
test_text3 = pd.DataFrame(test123).reset_index()
test_len_text = pd.merge(test_text1, test_text3, on='business_id', how='inner')

#combine text and length with other training data and clean
train5 = pd.merge(train4, train_len_text, on='business_id', how='inner')
train5 = train5.drop(['text_x'], axis=1)
train5.rename(columns={'text_length': 'avg_text_length'}, inplace=True)

test5 = pd.merge(test4, test_len_text, on='business_id', how='inner')
test5 = test5.drop(['text_x'], axis=1)
test5.rename(columns={'text_length': 'avg_text_length'}, inplace=True)

## Initial (baseline) analysis, exploratory analysis, and Viz

#Correlation of different variables with the number of violations
sns.pairplot(train5, x_vars=['avg_text_length', 'review_count', 'stars'], y_vars = 'total_violations', size = 5, kind='reg')
sns.pairplot(train5, x_vars=['avg_text_length'], y_vars = 'total_violations', size = 5, kind='reg')
sns.pairplot(train5, x_vars=['review_count'], y_vars = 'total_violations', size = 5, kind='reg')
sns.pairplot(train5, x_vars=['stars'], y_vars = 'total_violations', size = 5, kind='reg')

train5['total_violations'].describe()

sns.boxplot(train5['total_violations'],train5['stars'])
sns.boxplot(train5['avg_text_length'],train5['stars'])

sns.pairplot(train5, x_vars=['avg_text_length', 'review_count', 'stars'], y_vars = 'total_violations', size = 5, kind='reg')

train5.corr()
sns.heatmap(train5.corr())

#Baseline Linear Regression Model

feature_cols = ['avg_text_length', 'review_count', 'stars']
X = train5[feature_cols]
y = train5.total_violations

# instantiate and fit
linreg = LinearRegression()
linreg.fit(X, y)

# Check RMSE with cross-val
print np.sqrt(-cross_val_score(linreg, X, y, cv=10, scoring='mean_squared_error')).mean()
# 2.8510 Score


# print the coefficients
print linreg.intercept_
print linreg.coef_
zip(feature_cols, linreg.coef_)

#With just 2 features = 2.85277 - got slightly worse
feature_cols = ['review_count', 'stars']
X = train5[feature_cols]
linreg.fit(X, y)
print np.sqrt(-cross_val_score(linreg, X, y, cv=10, scoring='mean_squared_error')).mean()


## Text Analysis

#Function to get sentiment for each review
def detect_sentiment(text):
    return TextBlob(text.decode('utf-8')).sentiment.polarity

train5['sentiment'] = train5.text_y.apply(detect_sentiment)
test5['sentiment'] = train5.text_y.apply(detect_sentiment)

#Move to exploratory data part
train5.boxplot(column='sentiment', by='stars')

feature_cols=['review_count', 'stars', 'avg_text_length', 'sentiment', 'longitude', 'latitude']
X=train5[feature_cols]
print np.sqrt(-cross_val_score(linreg, X, y, cv=10, scoring='mean_squared_error')).mean()
#2.8298 - slightly improved -- added sentiment, latitude, longitude

#Dropping fields will not be used. Drop lat/long as it did not improve model much
train6 = train5.drop(['restaurant_id', 'business_id', 'categories', 'latitude', 'longitude', 'neighborhoods', 'open', 'minor', 'major', 'severe'], axis=1)
train6 = train6[['text_y', 'review_count', 'stars', 'total_violations', 'avg_text_length', 'sentiment']]
train6['text_y'] = train6['text_y'].astype(str)
print train6['text_y'].apply(lambda x: len(x))

test6 = test5.drop(['restaurant_id', 'business_id', 'categories', 'latitude', 'longitude', 'neighborhoods', 'open', 'minor', 'major', 'severe'], axis=1)
test6 = test6[['text_y', 'review_count', 'stars', 'total_violations', 'avg_text_length', 'sentiment']]
test6['text_y'] = test6['text_y'].astype(str)
print train6['text_y'].apply(lambda x: len(x))

# split the new DataFrame into training and testing sets
feature_cols=['text_y', 'review_count', 'stars', 'avg_text_length', 'sentiment']
X=train6[feature_cols]
y=train6.total_violations
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, train_size=.7)

#TF-IDF for review  
# Min_df range sets minimum # of times word appears
# Play around with diff min_df to see how it improves accuracy - the lower the # of features, the better

vect = TfidfVectorizer(stop_words='english', min_df=6)
train_dtm = vect.fit_transform(X_train[:, 0]) 
test_dtm = vect.transform(X_test[:, 0])

# cast other feature columns to float and convert to a sparse matrix
extra = sp.sparse.csr_matrix(X_train[:, 1:].astype(float))
extra.shape

# combine sparse matrices
train_dtm_extra = sp.sparse.hstack((train_dtm, extra))
train_dtm_extra.shape

# repeat for testing set
extra = sp.sparse.csr_matrix(X_test[:, 1:].astype(float))
test_dtm_extra = sp.sparse.hstack((test_dtm, extra))
test_dtm_extra.shape

# use linear regression with all features
linreg = LinearRegression()
linreg.fit(train_dtm_extra, y_train)
y_pred = linreg.predict(test_dtm_extra)
print np.sqrt(metrics.mean_squared_error(y_test, y_pred)) 
#2.62 with 1 min_df (83k features), # 2.69 with min_df of 6 (23k features)


# Notes:
# transform sparse to dense matrix - Random forest does not accept sparse matrix
# code example to transform sparse to dense = train_dtm_extra2 = train_dtm_extra.toarray()
# Random Forest n_estimator = # of trees, max_features = # of features to try
# Use train/text split, not cross-val
# Ensemble diff models by taking mean

# Transform sparse to dense matrix for randomforests
train_dtm_extra2 = train_dtm_extra.toarray()
test_dtm_extra2 = test_dtm_extra.toarray()

rfreg = RandomForestRegressor(n_estimators=40, random_state=1) 
rfreg.fit(train_dtm_extra2, y_train)
y_pred = rfreg.predict(test_dtm_extra2)
print np.sqrt(metrics.mean_squared_error(y_test, y_pred)) 
# 2.61 RMSE with n_estimator = 10
# 2.56 RMSE with n_estimator = 20
# 2.52 RMSE with n_estimator = 40

# Tuning the number of max_features
rfreg = RandomForestRegressor(n_estimators=40, max_features = 1500 ,random_state=1) 
rfreg.fit(train_dtm_extra2, y_train)
y_pred = rfreg.predict(test_dtm_extra2)
print np.sqrt(metrics.mean_squared_error(y_test, y_pred)) 
# Tried different # of max_features - 300 features was best at 2.51 RMSE
## stick with no limit on mx features since it doesn't improve accuracy that much by setting max limit

## Combine train & Test for cross-val

vect = TfidfVectorizer(stop_words='english', min_df=6)
X2 = vect.fit_transform(X.text_y) 
# cast other feature columns to float and convert to a sparse matrix
new_feature_cols = ['review_count', 'stars', 'avg_text_length', 'sentiment']
extra2 = sp.sparse.csr_matrix(train6[new_feature_cols].astype(float))
# combine sparse matrices
X3 = sp.sparse.hstack((X2, extra2))
# Transform sparse to dense matrix for randomforests
X4 = X3.toarray()
print np.sqrt(-cross_val_score(rfreg, X4, y, cv=10, scoring='mean_squared_error')).mean()
## RMSE got worse - 2.61

# Change test6 document to TF-IDF

feature_cols=['text_y', 'review_count', 'stars', 'avg_text_length', 'sentiment']
X_test=test6[feature_cols]
y_test=test6.total_violations

vect = TfidfVectorizer(stop_words='english', min_df=6)
## Fit this or only transform?
vect2 = vect.transform(X_test.text_y) 
# cast other feature columns to float and convert to a sparse matrix
test_feature_cols = ['review_count', 'stars', 'avg_text_length', 'sentiment']
extra3 = sp.sparse.csr_matrix(test6[test_feature_cols].astype(float))
# combine sparse matrices
X5 = sp.sparse.hstack((vect2, extra3))
# Transform sparse to dense matrix for randomforests
X6 = X5.toarray()

### Make prediction on test data based on the model

X_train = train6[feature_cols]
# Train the model on all training data
rfreg.fit(X4, y)
linreg.fit(X4, y)
# Make prediction on test data separated by date
rfreg_predic_violations = rfreg.predict(X6)
linreg_predic_violations = linreg.predict(X6)


## Use NB to predict probability by stars as the y variable and adding those features to the model


train6 = train6.drop('text', axis=1)

## Combine all the data and do cross-val since date is not an issue
'''all_data = train6.append(test6)

feature_cols=['review_count', 'stars', 'avg_text_length', 'sentiment']

vect = TfidfVectorizer(stop_words='english', min_df=6)
dtm = vect.fit_transform(all_data.text_y) 

# cast other feature columns to float and convert to a sparse matrix
extra = sp.sparse.csr_matrix(all_data[feature_cols].astype(float))

# combine sparse matrices
dtm_extra = sp.sparse.hstack((dtm, extra))
#Conver to array
dtm_extra_array = dtm_extra.toarray()

X=dtm_extra_array
y=all_data.total_violations
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, train_size=.7)

rfreg = RandomForestRegressor(n_estimators=40, max_features = 1500 ,random_state=1) 
rfreg.fit(X_train, y_train)
y_pred = rfreg.predict(X_test)
print np.sqrt(metrics.mean_squared_error(y_test, y_pred)) 
print np.sqrt(-cross_val_score(rfreg, X, y, cv=10, scoring='mean_squared_error')).mean()'''
### Discard "all data" file as it did not improve score

train6['stars2']=[for

import math
train6['stars2']=[round(train6.stars) for row in train6.stars]

for row in train6.stars:
    stars2.append(float(row))
stars3 = round(stars2)
    
    
    
X_train, X_test, y_train, y_test = train_test_split(train6.text_y, train6.stars, random_state=1)
train_dtm = vect.fit_transform(X_train)
test_dtm = vect.transform(X_test)
nb.fit(train_dtm, y_train)
y_pred_class = nb.predict(test_dtm)
print metrics.accuracy_score(y_test, y_pred_class)
print metrics.confusion_matrix(y_test, y_pred_class)

## Get feature importance
## Try different n_grams
## Try Gradient Boosting

## Addition exploratory for looking at most frequent appearing terms by sentiment









 
 
 
 
 
 
 
 
 



## Ignore Code below this when reviewing

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

sns.pairplot(data3, x_vars=['*', '**', '***'], y_vars='review_count', size=6, aspect=0.7, kind='reg',)

data3['*'].value_counts().plot(kind='bar')


# Linear Regression to predict total number of violations based on stars and review count



# Add feature to see if there was a violation or not to treat as classification problem
data3['violation_status'] = [1 if 'total_violations' > 0 else 0 for row in data3.total_violations]
data3.violation_status.value_counts() ## every restaurat has at least 1 violation so this feature is not useful
data3 = data3.drop('violation_status', axis=1)


## Reviews not yet pulled in with data3 - need to figure out if concatenating or not















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





