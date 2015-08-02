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

#combine text and length with other training data and clean
train5 = pd.merge(train4, train_len_text, on='business_id', how='inner')
train5 = train5.drop(['text_x'], axis=1)
train5.rename(columns={'text_length': 'avg_text_length'}, inplace=True)

# Text Analysis


















## Ignore Code below this when reviewing 

# Group train_lables by restaurant id and sum the # of stars (violations)

train_data = pd.read_csv('train_labels.csv')

#train_data2 doc sums # of stars by each restaurat and severity
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

sns.pairplot(data3, x_vars=['*', '**', '***'], y_vars='review_count', size=6, aspect=0.7, kind='reg',)

data3['*'].value_counts().plot(kind='bar')
data3['*'].describe()

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





