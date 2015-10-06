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


#2. Create a new DataFrame that only contains the 5-star and 1-star reviews.
yelp1or5stars = yelp[(yelp.stars == 1) | (yelp.stars == 5)]
yelp1or5stars.stars.value_counts()

#3. Split the new DataFrame into training and testing sets, using the review text as the only feature and the star rating as the response.
# define X and y
X = yelp1or5stars.text
y = yelp1or5stars.stars

# split into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
X_train.shape
X_test.shape
#4. Use CountVectorizer to create document-term matrices from X_train and X_test.
#    - **Hint:** If you run into a decoding error, instantiate the vectorizer with the argument `decode_error='ignore'`.
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm.shape
X_test_dtm = vect.transform(X_test)
X_test_dtm.shape


#5. Use Naive Bayes to predict the star rating for reviews in the testing set, and calculate the accuracy.
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)


# calculate accuracy of class predictions
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))


#6. Calculate the AUC.
#    - **Hint 1:** Make sure to pass the predicted probabilities to `roc_auc_score`, not the predicted classes.
#    - **Hint 2:** `roc_auc_score` will get confused if y_test contains fives and ones, so you will need to create a new object that contains ones and zeros instead.
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
y_pred_prob
y_test_binary = y_test.map({1:0, 5:1})
print(metrics.roc_auc_score(y_test_binary, y_pred_prob))

#7. Plot the ROC curve.
import matplotlib.pyplot as plt
fpr, tpr, thresholds = metrics.roc_curve(y_test_binary, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

#8. Print the confusion matrix, and calculate the sensitivity and specificity. Comment on the results.
confusion = metrics.confusion_matrix(y_test, y_pred_class)
print(confusion)
TP = confusion[1][1]
TN = confusion[0][0]
FP = confusion[0][1]
FN = confusion[1][0]

# calculate the sensitivity
sensitivity = TP / float(TP + FN)
print('sensitivity: {0}'.format(sensitivity))

# calculate the specificity
specificity =  TN / float(TN + FP)
print('specificity: {0}'.format(specificity))


#9. Browse through the review text for some of the false positives and false negatives. Based on your knowledge of how Naive Bayes works, do you have any theories about why the model is incorrectly classifying these reviews?
# print message text for the false positives
with pd.option_context('display.max_rows', 999, 'display.max_columns', 3, 'display.max_colwidth', 1000):
    print(X_test[y_test < y_pred_class])

with pd.option_context('display.max_rows', 999, 'display.max_columns', 3, 'display.max_colwidth', 1000):
    print(X_test[y_test > y_pred_class])

#this is too much noise; I'm sure if I stared at this long enough a pattern might emerge
#no time for staring at the spoon, let's generate the list of words that indicate a 5 or a 1
yelp_1star = yelp[yelp.stars==1]
yelp_5star = yelp[yelp.stars==5]
vect.fit(yelp.text)
all_tokens = vect.get_feature_names()
yelp_1star_dtm = vect.transform(yelp_1star.text)
yelp_5star_dtm = vect.transform(yelp_5star.text)

# count how many times EACH token appears across ALL 1star/5star reviews
onestar_counts = np.sum(yelp_1star_dtm.toarray(), axis=0)
fivestar_counts = np.sum(yelp_5star_dtm.toarray(), axis=0)

# create a DataFrame of tokens with their separate 1star/5star counts
token_counts = pd.DataFrame({'token':all_tokens, '1star':onestar_counts, '5star':fivestar_counts})
token_counts.set_index('token', inplace=True)

#todo: come back and do this a better way (e.g. use a convention; although thinking about this more; maybe not a bad approx)
token_counts['1star'] = token_counts['1star'] + 1
token_counts['5star'] = token_counts['5star'] + 1

# calculate ratio of 5star/1star for each token 
token_counts['5star_ratio'] = token_counts['5star'] / token_counts['1star']
token_counts.sort('5star_ratio')



#now lets see which words show up in false positives and see if the overlap w/ words
#that are high on 5'y list
X_test_fp = X_test[y_test < y_pred_class]
FP_dtm = vect.fit_transform(X_test_fp)
fp_tokens = vect.get_feature_names()
fp_counts = FP_dtm.sum(axis=0).tolist()[0]
fp_counts_df = pd.DataFrame({'token':fp_tokens, 'count':fp_counts}).sort('count')
fp_counts_df.set_index('token', inplace=True)

fp_counts_df['5star_ratio'] = token_counts['5star_ratio'][fp_counts_df.index]
fp_counts_df.describe()
with pd.option_context('display.max_rows', 999, 'display.max_columns', 3, 'display.max_colwidth', 1000):
    print(fp_counts_df)

#this seems to be a list of the words causing fals positives
#maybe great and fresh should be eliminated from dtm to see if that
#improves specificity w/o hurting sensitivity
fp_terms = fp_counts_df[fp_counts_df['5star_ratio']> 10]

#vice versa; looking for words that are triggering false negatives
X_test_fn = X_test[y_test > y_pred_class]
FN_dtm = vect.fit_transform(X_test_fn)
fn_tokens = vect.get_feature_names()
fn_counts = FN_dtm.sum(axis=0).tolist()[0]
fn_counts_df = pd.DataFrame({'token':fn_tokens, 'count':fn_counts}).sort('count')
fn_counts_df.set_index('token', inplace=True)

fn_counts_df['5star_ratio'] = token_counts['5star_ratio'][fn_counts_df.index]
fn_counts_df.describe()
with pd.option_context('display.max_rows', 999, 'display.max_columns', 3, 'display.max_colwidth', 1000):
    print(fn_counts_df)

with pd.option_context('display.max_rows', 999, 'display.max_columns', 3, 'display.max_colwidth', 1000):
    print(fn_counts_df)
#i'm not seeing words that really scream to me; maybe removing 'not', 'didn', 'car'


#10. Let's pretend that you want to balance sensitivity and specificity. You can achieve this by changing the threshold for predicting a 5-star review. What threshold approximately balances sensitivity and specificity?
#looking for the leading edge of the plateau around tpr= 0.9 where any higher threshold just adds to fpr
fpr_df = pd.DataFrame(fpr)
fpr_df[0].value_counts().head(20)

tpr_df = pd.DataFrame(tpr)
tpr_df[0].value_counts().head(20)

#looks like fpr is 0.1185567 and tpr 0.87681159
#looking for fprs where fpr < 0.123 (next highest on fpr chain)
print(thresholds[fpr < 0.123][-1])


#11. Let's see how well Naive Bayes performs when all reviews are included, rather than just 1-star and 5-star reviews:
#    - Define X and y using the original DataFrame from step 1. (y should contain 5 different classes.)
X = yelp.text
y = yelp.stars
#    - Split the data into training and testing sets.
# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
X_train.shape
X_test.shape
#    - Calculate the testing accuracy of a Naive Bayes model.
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm.shape
X_test_dtm = vect.transform(X_test)
X_test_dtm.shape
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)
print(metrics.accuracy_score(y_test, y_pred_class))

#    - Compare the testing accuracy with the null accuracy.
yelp.stars.value_counts()
import numpy as np
y_null = np.zeros_like(y_test, dtype=float)
y_null.fill(4)
print(metrics.accuracy_score(y_test, y_null))

# well 49%+ accurate is better than 36%+ accurate so machine learning wins!!!

#    - Print the confusion matrix.
confusion = metrics.confusion_matrix(y_test, y_pred_class)
print(confusion)

#    - Comment on the results.
y_test.stars.value_counts() #shortcut to get positives
#Relative good sensitivitity  for 4's (642/907) and 5's (480/821)
#pretty bad sensitivity for the others
#Vice versa for the 1/2/3

