## Class 10 Homework: Yelp Votes
'''
This assignment uses a small subset of the data from Kaggle's [Yelp Business Rating Prediction](https://www.kaggle.com/c/yelp-recsys-2013) competition.

**Description of the data:**

* `yelp.json` is the original format of the file. `yelp.csv` contains the same data, in a more convenient format. Both of the files are in this repo, so there is no need to download the data from the Kaggle website.
* Each observation in this dataset is a review of a particular business by a particular user.
* The "stars" column is the number of stars (1 through 5) assigned by the reviewer to the business. (Higher stars is better.) In other words, it is the rating of the business by the person who wrote the review.
* The "cool" column is the number of "cool" votes this review received from other Yelp users. All reviews start with 0 "cool" votes, and there is no limit to how many "cool" votes a review can receive. In other words, it is a rating of the review itself, not a rating of the business.
* The "useful" and "funny" columns are similar to the "cool" column.

**Homework tasks:**
'''
#1. Read `yelp.csv` into a DataFrame.
import pandas as pd
import seaborn as sns

yelp = pd.read_csv('yelp.csv')

#adding some baseline visualizations / analysis; useful for later eval
yelp.stars.value_counts()
yelp.stars.plot(kind='hist', bins=5)
yelp.stars.plot(kind='box')

#    * **Bonus:** Ignore the `yelp.csv` file, and construct this DataFrame yourself from `yelp.json`. This involves reading the data into Python, decoding the JSON, converting it to a DataFrame, and adding individual columns for each of the vote types.
# TODO

#2. Explore the relationship between each of the vote types (cool/useful/funny) and the number of stars.
feature_cols = ['cool', 'useful', 'funny']
sns.pairplot(yelp, x_vars='stars', y_vars=feature_cols, kind='reg')
sns.pairplot(yelp, x_vars=feature_cols, y_vars='stars', kind='reg')
#todo add alpah to pairplot (possible?)

sns.lmplot(x='stars', y='cool', data=yelp, aspect=1.5, scatter_kws={'alpha':0.2})
sns.lmplot(x='stars', y='useful', data=yelp, aspect=1.5, scatter_kws={'alpha':0.2})
sns.lmplot(x='stars', y='funny', data=yelp, aspect=1.5, scatter_kws={'alpha':0.2})

#3. Define cool/useful/funny as the features, and stars as the response.
X = yelp[feature_cols]
y = yelp.stars

#4. Fit a linear regression model and interpret the coefficients. Do the coefficients make intuitive sense to you? Explore the Yelp website to see if you detect similar trends.
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)
print(linreg.intercept_)
print(linreg.coef_)
# Intercept of ~3.8 indicates that the average review is almost 4 stars; this
# matches my experience w/ Yelp where a review with little to no content seems
# (the majority of reviews) ends up being about 4 starts (my interpretation
# of 4 starts has always been nothing spcial)
# I've never really known how to interpret "cool" but apparently it indicates 
# that the review is cool which generally indicates the restaurant is cool and
# and thus cool.
# To me, Funny has always been an indicator of a bad review (jokes at expense
# of restaurant) and the - coeficient is an indicator of this.
# I was very surprised to see useful have an even larger negative coefficient
# than funny; but a well known fact of reviews is that people put more time
# into writing negative reviews and time is an indicator fo how much info (thus
# utility) will be in the review.

#5. Evaluate the model by splitting it into training and testing sets and computing the RMSE. Does the RMSE make intuitive sense to you?
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# this seems like a very large error; more than a full star; looking at the
# overall box plot of stars I will be very curious to see if this does
# any better than some basic null models


#6. Try removing some of the features and see if the RMSE improves.
# define a function that accepts a list of features and returns testing RMSE
# starting with Kevin's hand function
# todo, think about how to genericize 
def train_test_rmse(feature_cols):
    X = yelp[feature_cols]
    y = yelp.stars
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))

train_test_rmse(['cool', 'useful', 'funny'])
#1.173

train_test_rmse(['cool', 'useful'])
#1.185

train_test_rmse(['cool', 'funny'])
#1.185

train_test_rmse(['useful', 'funny'])
#1.20

train_test_rmse(['cool'])
#1.2

train_test_rmse(['funny'])
#1.2

train_test_rmse(['useful'])
#1.2

#So it looks like all 3 results in best


#7. **Bonus:** Think of some new features you could create from the existing data that might be predictive of the response. Figure out how to create those features in Pandas, add them to your model, and see if the RMSE improves.
yelp['review_length'] = yelp.text.map(len)
train_test_rmse(['cool', 'useful', 'funny', 'review_length'])
#1.166... this seems to be the winner
X = yelp[['cool', 'useful', 'funny', 'review_length']]
y = yelp.stars
linreg = LinearRegression()
linreg.fit(X, y)
print(linreg.intercept_)
print(linreg.coef_)

#8. **Bonus:** Compare your best RMSE on the testing set with the RMSE for the "null model", which is the model that ignores all features and simply predicts the mean response value in the testing set.
yelp.stars.describe()
feature_cols = ['cool', 'useful', 'funny', 'review_length']
X = yelp[feature_cols]
y = yelp.stars
linreg = LinearRegression()
linreg.fit(X, y)

y_null = np.zeros_like(y, dtype=float)
y_null.fill(y.mean())

np.sqrt(metrics.mean_squared_error(y, y_null))
# the null (1.21) is worse than all three 1.166

#9. **Bonus:** Instead of treating this as a regression problem, treat it as a classification problem and see what testing accuracy you can achieve with KNN.
from sklearn.neighbors import KNeighborsClassifier

k_range = range(1, 101)
training_error = []
testing_error = []
feature_cols = ['cool', 'useful', 'funny', 'review_length']
X = yelp[feature_cols]
y = yelp.stars
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

for k in k_range:

    # instantiate the model with the current K value
    knn = KNeighborsClassifier(n_neighbors=k)

    # calculate training error
    knn.fit(X, y)
    y_pred_class = knn.predict(X)
    training_accuracy = np.sqrt(metrics.mean_squared_error(y, y_pred_class))
    training_error.append(training_accuracy)
    
    # calculate testing error
    knn.fit(X_train, y_train)
    y_pred_class = knn.predict(X_test)
    testing_accuracy = metrics.accuracy_score(y_test, y_pred_class)
    testing_error.append(1 - testing_accuracy)

column_dict = {'K': k_range, 'training error':training_error, 'testing error':testing_error}
df = pd.DataFrame(column_dict).set_index('K').sort_index(ascending=False)
df.sort('testing error').head()

#94 looks good for n
knn = KNeighborsClassifier(n_neighbors=94)

# calculate training error
knn.fit(X, y)
y_pred_class = knn.predict(X)
np.sqrt(metrics.mean_squared_error(y, y_pred_class))
#1.408; noticeably worse


#10. **Bonus:** Figure out how to use linear regression for classification, and compare its classification accuracy with KNN's accuracy.
#todo (isn't that what the next class is about?)
feature_cols = ['cool', 'useful', 'funny', 'review_length']
X = yelp[feature_cols]
y = yelp.stars
linreg = LinearRegression()
linreg.fit(X, y)
y_pred = linreg.predict(X)
y_pred_class = np.round(y_pred)
np.sqrt(metrics.mean_squared_error(y, y_pred_class))
#1.20 is not too terrible but not as good as not rounding first