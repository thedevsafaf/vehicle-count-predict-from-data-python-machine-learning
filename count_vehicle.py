#vehicle count predict from the sensor data

#Step 1: Importing the pandas module for loading the data frame

# importing the pandas module for data frame
import pandas as pd

#load dataset into train variable.
train = pd.read_csv('vehicles.csv')

# display top 5 values of data set
print(train.head())

#Step 2: Define the functions for getting month, day, hours from the Timestamp (DateTime) and load it into different columns.

# functions to get all data from time stamp 

#get date
def get_dom(dt):
    return dt.day
 
# get week day
def get_weekday(dt):
    return dt.weekday()
 
# get hour
def get_hour(dt):
    return dt.hour
 
# get year
def get_year(dt):
    return dt.year
 
# get month
def get_month(dt):
    return dt.month
 
# get year day
def get_dayofyear(dt):
    return dt.dayofyear
 
# get year week
def get_weekofyear(dt):
    return dt.weekofyear

train['DateTime'] = train['DateTime'].map(pd.to_datetime)
train['date'] = train['DateTime'].map(get_dom)
train['weekday'] = train['DateTime'].map(get_weekday)
train['hour'] = train['DateTime'].map(get_hour)
train['month'] = train['DateTime'].map(get_month)
train['year'] = train['DateTime'].map(get_year)
train['dayofyear'] = train['DateTime'].map(get_dayofyear)
train['weekofyear'] = train['DateTime'].map(get_weekofyear)

print(train.head())

#Step 3: Separate the class label and store into the target variable 

# there is no use of DateTime module, so remove it
train = train.drop(['DateTime'], axis=1)

# separating class label for training the data
train1 = train.drop(['Vehicles'], axis=1)

# class label is stored in target
target = train['Vehicles']

print(train1.head())
print()
print(target.head())
print()

#Step 4: Create and train the data using Machine Learning algorithms and predict the results after testing.

#importing Random forest
from sklearn.ensemble import RandomForestRegressor

#defining the RandomForestRegressor
m1=RandomForestRegressor()
X = train1.values
Y = target
m1.fit(X,Y)

#testing
print('Prediction: ', m1.predict([[11,6,4,1,2015,11,2]]))
print()