# -*- coding: utf-8 -*-
"""
Created on Thu Dec 05 12:04:15 2018

@author: nemab
"""

import numpy as np 
import pandas as pd 
import json
from scipy.stats import kurtosis, skew
from datetime import datetime
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')

from pandas.io.json import json_normalize
import seaborn as sns 
from sklearn import model_selection, preprocessing, metrics
import os 


os.getcwd()


# 'device', 'geoNetwork', 'totals', 'trafficSource' JSON features
# Flattening json columns
json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']
def load_df(file):
    path = "" + file
    print(path + "Path")
    df = pd.read_csv(path, converters={column: json.loads for column in json_cols}, 
                     dtype={'fullVisitorId': 'str'})
    
    for column in json_cols:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df

#Running function for both files:
    
train = load_df('C:/Users/nemab/Desktop/Projects/Google Analytics Customer Prediction/train.csv')
#train.to_csv("train_clean.csv")

test = load_df('C:/Users/nemab/Desktop/Projects/Google Analytics Customer Prediction/test.csv')


#--------EDA---------

print("There are " + str(train.shape[0]) + " rows and " + str(train.shape[1]) + " raw columns in this dataset")

print("There are " + str(test.shape[0]) + " rows and " + str(test.shape[1]) + " raw columns in this dataset")

# For train
# Numerical features
num_train = train.select_dtypes(include=[np.number])
num_train.columns

# Categorical features
cat_train = train.select_dtypes(include=[np.object])
cat_train.columns

# For test

num_test = test.select_dtypes(include=[np.number])
num_test.columns

cat_test = test.select_dtypes(include=[np.object])
cat_test.columns


# There are some constant columns in the dataset which do not contribute to our response varible:
# Dropping those columns

train = train.loc[:, (train != train.iloc[0]).any()]

test = test.loc[:, (test != test.iloc[0]).any()]

# Now we have only 36 and 34 columns respectively


# Interesting thing about our data set is that we have lot of missing values
# To see the highest perecentage wrt the columns

totalmiss = cat_train.isnull().sum().sort_values(ascending=False)
percent = (cat_train.isnull().sum()/cat_train.isnull().count()).sort_values(ascending=False)*100
missingdata = pd.concat([totalmiss, percent], axis=1,join='outer', keys=['Count of Missing', ' % of Observations'])
missingdata.index.name ='Columns'
missingdata.head(10)


totalmiss = cat_test.isnull().sum().sort_values(ascending=False)
percent = (cat_test.isnull().sum()/cat_test.isnull().count()).sort_values(ascending=False)*100
missingdata = pd.concat([totalmiss, percent], axis=1,join='outer', keys=['Count of Missing', ' % of Observations'])
missingdata.index.name ='Columns'
missingdata.head(10)


# Customer's contributing to revenue

rev = train.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()
nrc = rev[rev['totals.transactionRevenue']==0]
rc = rev[rev['totals.transactionRevenue']>0]
print("The number of nonrevenue customers are ", len(nrc))
print("The number of revenue generating customers are ", len(rc))
print("the ratio of revenue generating customers are {0:0.4}%".format(len(rc)/len(rev)*100))


# This gives us the ratio to be ~2%

# Viz

lab = ['Non-revenue generating customers','Revenue generating customers']
values = [1307589,16141]

plt.axis("equal")
plt.pie(values, labels=lab, radius=1.5, autopct="%0.2f%%",shadow=True, explode=[0,0.8], colors=['lightskyblue','lightcoral'])
plt.show()



# For all non-zero transactions:

nonz = train[train['totals.transactionRevenue']>0]['totals.transactionRevenue']

nonz.describe()

sns.distplot(nonz)


# Normalizing the revenue feature into log (Asked in the competition as well)

revenue_log = train["totals.transactionRevenue"].apply(np.log1p)

#Distribution

plt.figure(figsize=(10,8))
nonz = revenue_log[revenue_log >0]
sns.distplot(nonz)

# Skewness and Kurtosis showcase that it is not common bell curve

print("The skewness of transaction ", skew(nonz))
print("The kurtosis of transaction ", kurtosis(nonz))


# Processing all date fields

# Function to extract date features
def date_pro(df):
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d") # seting the column as pd datetime
    df["_weekday"] = df['date'].weekday #extracts week-day
    df["_day"] = df['date'].day # extracts day
    df["_month"] = df['date'].month # extracts month
    df["_year"] = df['date'].year # extracts year
    df['_visitHour'] = (df['visitStartTime'].apply(lambda x: str(datetime.fromtimestamp(x).hour))).astype(int)
    
    return df 

train = date_pro(train)
test = date_pro(test)


# Making a backup here

train_copy = train.copy()
test_copy = test.copy()

# Drill down to individaul features

sns.barplot(x=train['_weekday'], y=train['totals.transactionRevenue'], data=train)

# Observations: Weeken sale is less (People tend to buy more during business hours)


#%matplotlib inline
plt.figure(figsize=(10,5))
sns.barplot(x=train['_day'], y=train['totals.transactionRevenue'], data=train)

# By each month
plt.figure(figsize=(10,5))
sns.barplot(x=train['_month'], y=train['totals.transactionRevenue'], data=train)

# By visit hour
plt.figure(figsize=(10,5))
sns.barplot(x=train['_visitHour'], y=train['totals.transactionRevenue'], data=train)

# Observation: 7pm to 9am more sales


# Date vs customer visits

srs = train.groupby('date')['fullVisitorId'].size().reset_index()

plt.figure(figsize=(12, 8))
sns.lineplot(srs['date'], srs['fullVisitorId'])

# Significant spike in Nov and Dec (mainly due to festive season)


# Funtions :: EDA

# Extract categories
import re
def cols_extract(cat):
    cat_cols = list()
    for i in train.columns: 
        a = re.findall(r'^'+cat+'.*',i)
        if a:
            cat_cols.append(a[0])
        else:
            continue
    return cat_cols

# Plot Function
    
def cat_plots(col):
    a = train.loc[:,[col, 'totals.transactionRevenue']]
    a['totals.transactionRevenue'] = a['totals.transactionRevenue'].replace(0.0, np.nan)
    
    srs = a.groupby(col)['totals.transactionRevenue'].agg(['size','count','mean'])
    srs.columns = ["count", 'count of non-zero revenue', "mean transaction value"]

    srs['total_revenue'] = srs['count of non-zero revenue']*srs['mean transaction value']
    srs = srs.sort_values(by="count", ascending=False)
    print(srs.head(10))
    plt.figure(figsize=(8, 20)) 
    plt.subplot(4,1,1)
    sns.barplot(x=srs['count'].head(10), y=srs.index[:10])
    plt.subplot(4,1,2)
    sns.barplot(x=srs['count of non-zero revenue'].head(10), y=srs.index[:10])
    plt.subplot(4,1,3)
    sns.barplot(x=srs['mean transaction value'].head(10), y=srs.index[:10])
    plt.subplot(4,1,4)
    sns.barplot(x=srs['total_revenue'].head(10), y=srs.index[:10])

# Which all category variables are worth visualizing?
    
device_cols = cols_extract('device')

train[device_cols].nunique(dropna=False)

# Only 'device.browser', 'device.deviceCategory', 'device.operatingSystem' looks good

cat_plots('device.browser')

# Obs.: Chrome brings more revenue


cat_plots('device.deviceCategory')

# Obs.: Desktop users brings more revenue


cat_plots('device.operatingSystem')

# Obs.: Mac users brings more revenue


# Let's see from geography point of view

geo_cols = cols_extract('geoNetwork')

train[geo_cols].nunique(dropna=False)

# Will look into geoNetwork.continent, geoNetwork.country, geoNetwork.subContinent , geoNetwork.networkDomain

cat_plots('geoNetwork.continent')

# Obs. : Americas best source of revenue

cat_plots('geoNetwork.country')

# Obs. : North America best source of revenue


cat_plots('geoNetwork.subContinent')
cat_plots('geoNetwork.networkDomain')


# Checking Traffic sources

traffic_cols = cols_extract('trafficSource')

train[traffic_cols].nunique(dropna=False)

cat_plots('trafficSource.source')

# Mall Googleplex generated most revenues

cat_plots('trafficSource.medium')
# Referrals v/s organic

## ---- Start Analyzing -----

# Droping unwanted features

# Categorical 

cat_cols = list()
for i in train.columns:
    if (train[i].dtype=='object' or train[i].dtype=='bool') and (not(i.startswith('total'))):
        cat_cols.append(i)

# Check unique classes
        
train[cat_cols].nunique(dropna=False)

cat_cols.remove('fullVisitorId')


# Numeric

num_cols = list()
for i in train.columns:
    if train[i].dtype not in ['object', 'bool']:
        num_cols.append(i)
        
        
num_cols.remove('date')
num_cols.remove('visitId')
num_cols.remove('visitStartTime')
num_cols.remove('_year')

# To handle missing values: 

train[num_cols].isnull().sum()

# Missing % in cat
a = train[cat_cols].isnull().sum()
b = a/len(train)*100
b

# Stratified Sampling 

# Sampling with 35% null values and rest non-zero values

#August 2016
aug2016 = train[(train.date > 2016-08-00) & (train.date < 2016-08-31 )]

aug2016.shape

aug2016RevenueNull = aug2016[aug2016.totals_transactionRevenue.isnull()]

aug2016RevenueNull.shape

aug2016Revenue = aug2016[aug2016.totals_transactionRevenue.notnull()]

aug2016Revenue.shape

aug2016RevenueisNull = aug2016RevenueNull.sample(frac=0.035)
aug2016RevenueisNull.shape

framesaug2016 = [aug2016RevenueisNull,aug2016Revenue]
aug2016frame = pd.concat(framesaug2016)
aug2016frame.shape


# Sept 2016

sept2016 = train[(train.date > 2016-09-00) & (train.date < 2016-09-30 )]

sept2016.shape

sept2016RevenueNull = sept2016[sept2016.totals_transactionRevenue.isnull()]

sept2016RevenueNull.shape

sept2016Revenue = sept2016[sept2016.totals_transactionRevenue.notnull()]

sept2016Revenue.shape

sept2016RevenueisNull = sept2016RevenueNull.sample(frac=0.035)
sept2016RevenueisNull.shape

framessept2016 = [sept2016RevenueisNull,sept2016Revenue]
sept2016frame = pd.concat(framessept2016)
sept2016frame.shape

# Oct 2016

oct2016 = train[(train.date > 2016-10-00) & (train.date < 2016-10-31 )]

oct2016.shape

oct2016RevenueNull = oct2016[oct2016.totals_transactionRevenue.isnull()]

oct2016RevenueNull.shape

oct2016Revenue = oct2016[oct2016.totals_transactionRevenue.notnull()]

oct2016Revenue.shape

oct2016RevenueisNull = oct2016RevenueNull.sample(frac=0.029)
oct2016RevenueisNull.shape

framesoct2016 = [oct2016RevenueisNull,oct2016Revenue]
oct2016frame = pd.concat(framesoct2016)
oct2016frame.shape

# Nov 2016 

nov2016 = train[(train.date > 2016-11-00) & (train.date < 2016-11-30 )]

nov2016.shape

nov2016RevenueNull = nov2016[nov2016.totals_transactionRevenue.isnull()]

nov2016RevenueNull.shape

nov2016Revenue = nov2016[nov2016.totals_transactionRevenue.notnull()]

nov2016Revenue.shape

nov2016RevenueisNull = nov2016RevenueNull.sample(frac=0.024)
nov2016RevenueisNull.shape

framesnov2016 = [nov2016RevenueisNull,nov2016Revenue]
nov2016frame = pd.concat(framesnov2016)
nov2016frame.shape

# Dec 2016 

dec2016 = train[(train.date > 2016-12-00) & (train.date < 2016-12-31 )]

dec2016.shape

dec2016RevenueNull = dec2016[dec2016.totals_transactionRevenue.isnull()]

dec2016RevenueNull.shape

dec2016Revenue = dec2016[dec2016.totals_transactionRevenue.notnull()]

dec2016Revenue.shape

dec2016RevenueisNull = dec2016RevenueNull.sample(frac=0.028)
dec2016RevenueisNull.shape

framesdec2016 = [dec2016RevenueisNull,dec2016Revenue]
dec2016frame = pd.concat(framesdec2016)
dec2016frame.shape

# Jan 2017

jan2017 = train[(train.date > 2017-01-00) & (train.date < 2017-01-31 )]

jan2017.shape

jan2017RevenueNull = jan2017[jan2017.totals_transactionRevenue.isnull()]

jan2017RevenueNull.shape

jan2017Revenue = jan2017[jan2017.totals_transactionRevenue.notnull()]

jan2017Revenue.shape

jan2017RevenueisNull = jan2017RevenueNull.sample(frac=0.044)
jan2017RevenueisNull.shape

framesjan2017 = [jan2017RevenueisNull,jan2017Revenue]
jan2017frame = pd.concat(framesjan2017)
jan2017frame.shape


# Feb 2017


feb2017 = train[(train.date > 2017-02-00) & (train.date < 2017-02-29 )]

feb2017.shape

feb2017RevenueNull = feb2017[feb2017.totals_transactionRevenue.isnull()]

feb2017RevenueNull.shape

feb2017Revenue = feb2017[feb2017.totals_transactionRevenue.notnull()]

feb2017Revenue.shape

feb2017RevenueisNull = feb2017RevenueNull.sample(frac=0.044)
feb2017RevenueisNull.shape

framesfeb2017 = [feb2017RevenueisNull,feb2017Revenue]
feb2017frame = pd.concat(framesfeb2017)
feb2017frame.shape

# March 2017


march2017 = train[(train.date > 2017-03-00) & (train.date < 2017-03-31 )]

march2017.shape

march2017RevenueNull = march2017[march2017.totals_transactionRevenue.isnull()]

march2017RevenueNull.shape

march2017Revenue = march2017[march2017.totals_transactionRevenue.notnull()]

march2017Revenue.shape

march2017RevenueisNull = march2017RevenueNull.sample(frac=0.039)
march2017RevenueisNull.shape

framesmarch2017 = [march2017RevenueisNull,march2017Revenue]
march2017frame = pd.concat(framesmarch2017)
march2017frame.shape

# April 2017


april2017 = train[(train.date > 2017-04-00) & (train.date < 2017-04-30 )]

april2017.shape

april2017RevenueNull = april2017[april2017.totals_transactionRevenue.isnull()]

april2017RevenueNull.shape

april2017Revenue = april2017[april2017.totals_transactionRevenue.notnull()]

april2017Revenue.shape

april2017RevenueisNull = april2017RevenueNull.sample(frac=0.039)
april2017RevenueisNull.shape

framesapril2017 = [april2017RevenueisNull,april2017Revenue]
april2017frame = pd.concat(framesapril2017)
april2017frame.shape


# May 2017


may2017 = train[(train.date > 2017-05-00) & (train.date < 2017-05-31 )]

may2017.shape

may2017RevenueNull = may2017[may2017.totals_transactionRevenue.isnull()]

may2017RevenueNull.shape

may2017Revenue = may2017[may2017.totals_transactionRevenue.notnull()]

may2017Revenue.shape

may2017RevenueisNull = may2017RevenueNull.sample(frac=0.038)
may2017RevenueisNull.shape

framesmay2017 = [may2017RevenueisNull,may2017Revenue]
may2017frame = pd.concat(framesmay2017)
may2017frame.shape

# June 2017


june2017 = train[(train.date > 2017-06-00) & (train.date < 2017-06-30 )]

june2017.shape

june2017RevenueNull = june2017[june2017.totals_transactionRevenue.isnull()]

june2017RevenueNull.shape

june2017Revenue = june2017[june2017.totals_transactionRevenue.notnull()]

june2017Revenue.shape

june2017RevenueisNull = june2017RevenueNull.sample(frac=0.040)
june2017RevenueisNull.shape

framesjune2017 = [june2017RevenueisNull,june2017Revenue]
june2017frame = pd.concat(framesjune2017)
june2017frame.shape

# July 2017


july2017 = train[(train.date > 2017-07-00) & (train.date < 2017-07-31 )]

july2017.shape

july2017RevenueNull = july2017[july2017.totals_transactionRevenue.isnull()]

july2017RevenueNull.shape

july2017Revenue = july2017[july2017.totals_transactionRevenue.notnull()]

july2017Revenue.shape

july2017RevenueisNull = july2017RevenueNull.sample(frac=0.038)
july2017RevenueisNull.shape

framesjuly2017 = [july2017RevenueisNull,july2017Revenue]
july2017frame = pd.concat(framesjuly2017)
july2017frame.shape


# Combined data 

GoogleFrame = [aug2016frame,
               sept2016frame,
               oct2016frame,
               nov2016frame,
               dec2016frame,
               jan2017frame,
               feb2017frame,
               march2017frame,
               april2017frame,
               may2017frame,
               june2017frame,
               july2017frame]

Googletrain = pd.concat(GoogleFrame)
Googletrain.shape

#exporting to csv
Googletrain.to_csv("Googletrain.csv")


# Correlation plot 

Googletrain.dtypes
Googletrain.corr()


                            ['channelGrouping',
                            'device_browser',
                            'device_deviceCategory',
                            'device_operatingSystem',
                            'geoNetwork_city',
                            'geoNetwork_continent',
                            'geoNetwork_country',
                            'geoNetwork_metro',
                            'geoNetwork_networkDomain',
                            'geoNetwork_region',
                            'geoNetwork_subContinent',
                            'trafficSource_adContent',
                            'trafficSource_adwordsClickInfo.adNetworkType',
                            'trafficSource_adwordsClickInfo.gclId',
                            'trafficSource_adwordsClickInfo.page',
                            'trafficSource_adwordsClickInfo.slot',
                            'trafficSource_campaign',
                            'trafficSource_keyword',
                            'trafficSource_medium',
                            'trafficSource_referralPath',
                            'trafficSource_source',
                            'trafficSource_adwordsClickInfo.isVideoAd',
                            'trafficSource_isTrueDirect',
                            'totals_transactionRevenue']

Googletrain['channelGrouping'].astype('str')
type(Googletrain['device_browser'][0])


