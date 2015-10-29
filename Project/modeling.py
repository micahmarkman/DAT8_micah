"""
Created on Thu Oct 29 04:46:09 2015
cd Documents/GeneralAssembly/Data\ Science/DAT8_micah/project

@author: micah
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score

user_activity = pd.read_csv("data/user_activity.csv")
user_activity['month_1_count'].fillna(value=0, inplace=True)
user_activity['month_6_count'].fillna(value=0, inplace=True)
user_activity_p1 = user_activity.pivot('user_id', 'activity_name', 'month_1_count')
user_activity_p1.columns = ['month1_'+col for col in user_activity_p1.columns]
user_activity_p6 = user_activity.pivot('user_id', 'activity_name', 'month_6_count')
user_activity_p6.columns = ['month6_'+col for col in user_activity_p6.columns]

user_initlogints = user_activity[['user_id', 'initlogin_ts']].drop_duplicates()
user_initlogints.set_index('user_id', inplace=True)

user_profiles = pd.read_csv("data/user_profile.csv", index_col=0)
user_profiles['phone_present'] = pd.factorize(user_profiles['phone_present'])[0]
user_profiles['phone_present'].fillna(value=0, inplace=True) 
user_profiles['biography_present'] = pd.factorize(user_profiles['biography_present'])[0]
user_profiles['biography_present'].fillna(value=0, inplace=True)


data = user_initlogints.join([user_activity_p1, user_activity_p6, user_profiles])

district_dummies = pd.get_dummies(data['district'], prefix='district', dummy_na=True)
district_dummies.drop(district_dummies.columns[len(district_dummies.columns)-1], axis=1, inplace=True)
district_dummies.shape

facility_dummies = pd.get_dummies(data['facility'], prefix='facility', dummy_na=True)
facility_dummies.drop(facility_dummies.columns[len(facility_dummies.columns)-1], axis=1, inplace=True)
facility_dummies.shape

occupation_dummies = pd.get_dummies(data['occupation'], prefix='occupation', dummy_na=True)
occupation_dummies.drop(occupation_dummies.columns[len(occupation_dummies.columns)-1], axis=1, inplace=True)
occupation_dummies.shape

data = data.join([district_dummies, facility_dummies,occupation_dummies])
data_input = data.copy(deep=True)

data['month6_consumes_basic'] = data['month6_Log in'] + data['month6_Register'] + data['month6_Seach tags']\
                               + data['month6_Search content'] + data['month6_Search users'] \
                               + data['month6_Spotlight search'] + data['month6_View']
data['month6_consumes_adv'] = data['month6_Download'] + data['month6_Follow'] + data['month6_Like'] \
                             + data['month6_Mention'] + data['month6_Send'] +data['month6_Unfollow'] \
                             + data['month6_Unlike'] + data['month6_Watch']
data['month6_contributes'] = data['month6_Aided'] + data['month6_Approve'] + data['month6_Associate'] \
                            + data['month6_Copy'] + data['month6_Create'] + data['month6_Moderate'] \
                            + data['month6_Modify'] + data['month6_Rate or Vote'] + data['month6_Resolved'] \
                           + data['month6_Validate']
data['month6_irrelevant'] = data['month6_Delete'] + data['month6_Dissociate'] + data['month6_Expire'] \
                           + data['month6_Hide'] + data['month6_Log out']+ data['month6_Move'] \
                           + data['month6_Offline'] + data['month6_Online'] + data['month6_Reject'] \
                           + data['month6_Remove watch']  + data['month6_Undelete']
#user_activity[user_activity['activity_name'] == 'Create'].hist(column='month_6_count', by='activity_name')
#data['month6_consumes_basic'].value_counts()
#data['month6_consumes_adv'].value_counts()
#data['month6_contributes'].value_counts()
#data['month6_irrelevant'].value_counts()

#weighting advanced consumption as 3x value of basic consumption
data['month6_consumes'] = data['month6_consumes_basic'] + data['month6_consumes_adv']*3
#data['month6_consumes'].value_counts()

data['consumer'] = data['month6_consumes'] > 4
data['contributor'] = data['month6_contributes'] > 4
data['user_type'] = data['consumer'] + 2*data['contributor']
data['user_type'].value_counts().plot(kind='bar')

1-((data['user_type'].value_counts()[3]+data['user_type'].value_counts()[1])/data['user_type'].value_counts().sum())
1-((data['user_type'].value_counts()[3]+data['user_type'].value_counts()[2])/data['user_type'].value_counts().sum())

#user_activity = user_activity.join(data['user_type'])
#user_activity[user_activity['user_type'] >0].hist(column='month_1_count', by='activity_name')

# lets model using random forest
X = data_input.copy(deep=True)
X.drop(['initlogin_ts','title','userenabled'], axis=1, inplace=True)
X.drop(['district','facility','occupation'], axis=1, inplace=True)
X.drop(['value'], axis=1, inplace=True)
X.drop(user_activity_p6.columns, axis=1, inplace=True)
irrelevant_cols = ['month1_Delete','month1_Dissociate','month1_Expire','month1_Hide','month1_Log out','month1_Move','month1_Offline','month1_Online','month1_Reject','month1_Remove watch','month1_Undelete']  
X.drop(irrelevant_cols, axis=1, inplace=True)
X.drop(facility_dummies.columns, axis=1, inplace=True)
y = data['consumer']

rfreg = RandomForestRegressor()

estimator_range = range(10, 310, 10)

# list to store the average RMSE for each value of n_estimators
RMSE_scores = []

# use 5-fold cross-validation with each value of n_estimators (WARNING: SLOW!)
for estimator in estimator_range:
    rfreg = RandomForestRegressor(n_estimators=estimator, random_state=1)
    MSE_scores = cross_val_score(rfreg, X, y, cv=5, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))


# plot n_estimators (x-axis) versus RMSE (y-axis)
plt.plot(estimator_range, RMSE_scores)
plt.xlabel('n_estimators')
plt.ylabel('RMSE (lower is better)')

feature_range = range(1, len(X.columns)+1, 5)

# list to store the average RMSE for each value of max_features
RMSE_scores = []

# use 10-fold cross-validation with each value of max_features (WARNING: SLOW!)
for feature in feature_range:
    rfreg = RandomForestRegressor(n_estimators=100, max_features=feature, random_state=1)
    MSE_scores = cross_val_score(rfreg, X, y, cv=10, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))

# plot max features (x-axis) versus RMSE (y-axis)
plt.plot(feature_range, RMSE_scores)
plt.xlabel('max features')
plt.ylabel('RMSE (lower is better)')

rfreg = RandomForestRegressor(n_estimators=100, max_features=30, oob_score=True, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
rfreg.fit(X_train, y_train)
y_pred_prob = rfreg.predict(X_test)
metrics.roc_auc_score(y_test, y_pred_prob)
y_pred = np.rint(y_pred_prob)

metrics.accuracy_score(y_test, y_pred)

metrics.confusion_matrix(y_test, y_pred)


rfreg.fit(X, y)
rfreg.oob_score_

pd.DataFrame({'feature':X.columns, 'importance':rfreg.feature_importances_}).sort('importance')

X.shape
rfreg.transform(X, threshold=0.1).shape
rfreg.transform(X, threshold='mean').shape
rfreg.transform(X, threshold='median').shape
X_important = rfreg.transform(X, threshold='mean')


# check the RMSE for a Random Forest that only includes important features
rfreg = RandomForestRegressor(n_estimators=100, max_features=30, oob_score=True, random_state=1)
rfreg.fit(X_important, y)
rfreg.oob_score
scores = cross_val_score(rfreg, X_important, y, cv=10, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))


# now repeat for contributors (should define a function)
# lets model using random forest

y2 = data['contributor']

rfreg = RandomForestRegressor()

estimator_range = range(10, 310, 10)

# list to store the average RMSE for each value of n_estimators
RMSE_scores = []

# use 5-fold cross-validation with each value of n_estimators (WARNING: SLOW!)
for estimator in estimator_range:
    rfreg = RandomForestRegressor(n_estimators=estimator, random_state=1)
    MSE_scores = cross_val_score(rfreg, X, y2, cv=5, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))


# plot n_estimators (x-axis) versus RMSE (y-axis)
plt.plot(estimator_range, RMSE_scores)
plt.xlabel('n_estimators')
plt.ylabel('RMSE (lower is better)')

feature_range = range(1, len(X.columns)+1, 5)

# list to store the average RMSE for each value of max_features
RMSE_scores = []

# use 10-fold cross-validation with each value of max_features (WARNING: SLOW!)
for feature in feature_range:
    rfreg = RandomForestRegressor(n_estimators=140, max_features=feature, random_state=1)
    MSE_scores = cross_val_score(rfreg, X, y2, cv=10, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))

# plot max features (x-axis) versus RMSE (y-axis)
plt.plot(feature_range, RMSE_scores)
plt.xlabel('max features')
plt.ylabel('RMSE (lower is better)')

rfreg = RandomForestRegressor(n_estimators=140, max_features=25, oob_score=True, random_state=1)
X_train, X_test, y2_train, y2_test = train_test_split(X, y2, random_state=1)
rfreg.fit(X_train, y2_train)
y2_pred_prob = rfreg.predict(X_test)
metrics.roc_auc_score(y2_test, y2_pred_prob)
y2_pred = np.rint(y2_pred_prob)

metrics.accuracy_score(y2_test, y2_pred)

metrics.confusion_matrix(y2_test, y2_pred)


rfreg.fit(X, y2)
rfreg.oob_score_

pd.DataFrame({'feature':X.columns, 'importance':rfreg.feature_importances_}).sort('importance')
X.shape
rfreg.transform(X, threshold=0.1).shape
rfreg.transform(X, threshold='mean').shape
rfreg.transform(X, threshold='median').shape
X_important = rfreg.transform(X, threshold='mean')
pd.DataFrame({'feature':X.columns, 'importance':rfreg.feature_importances_}).sort('importance')['feature'].tail(12)


# check the RMSE for a Random Forest that only includes important features
rfreg = RandomForestRegressor(n_estimators=140, oob_score=True, random_state=1)
rfreg.fit(X_important, y2)
rfreg.oob_score_
scores = cross_val_score(rfreg, X_important, y2, cv=10, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))


#Decision Tree
from sklearn.tree import DecisionTreeRegressor

# list of values to try
max_depth_range = range(1, 8)

# list to store the average RMSE for each value of max_depth
RMSE_scores = []

dt_cols = ['phone_present','month1_Approve','month1_Search content','month1_Associate','month1_Spotlight search','month1_Download','month1_Follow','month1_Create','month1_Log in','month1_Search users','month1_Modify','month1_View']
X_dt = data_input[dt_cols]


# use LOOCV with each value of max_depth
for depth in max_depth_range:
    treereg = DecisionTreeRegressor(max_depth=depth, random_state=1)
    MSE_scores = cross_val_score(treereg, X_important, y, cv=14, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))


# plot max_depth (x-axis) versus RMSE (y-axis)
plt.plot(max_depth_range, RMSE_scores)
plt.xlabel('max_depth')
plt.ylabel('RMSE (lower is better)')


# max_depth=3 was best, so fit a tree using that parameter
treereg = DecisionTreeRegressor(max_depth=5, random_state=1)
treereg.fit(X_dt, y)


# "Gini importance" of each feature: the (normalized) total reduction of error brought by that feature
pd.DataFrame({'feature':dt_cols, 'importance':treereg.feature_importances_})

from sklearn.tree import export_graphviz
export_graphviz(treereg, out_file='tree.dot', feature_names=dt_cols)



# Lets play with text data for fun
data['value'].isnull().sum()
X = data[data['value'].notnull()]['value']
y = data[data['value'].notnull()]['consumer']

#null accuracy
y.value_counts()[0]/y.count()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
vect.get_feature_names()

nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

X_test_dtm = vect.transform(X_test)
y_pred_class = nb.predict(X_test_dtm)

metrics.accuracy_score(y_test, y_pred_class)

metrics.confusion_matrix(y_test, y_pred_class)
#horribly insensitive

y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
metrics.roc_auc_score(y_test, y_pred_prob)

# print message text for the false positives
X_test[y_test < y_pred_class]
# print message text for the false negatives
X_test[y_test > y_pred_class]
