'''
Pandas Homework with IMDb data
'''

'''
BASIC LEVEL
'''

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from ast import literal_eval

# read in 'imdb_1000.csv' and store it in a DataFrame named movies
movies = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/imdb_1000.csv')

# check the number of rows and columns
movies.shape

# check the data type of each column
movies.dtypes

# calculate the average movie duration
movies.duration.mean()

# sort the DataFrame by duration to find the shortest and longest movies
movies.sort('duration', ascending=True, inplace='True')
movies.duration.iloc[0]
movies.duration.iloc[-1]

# create a histogram of duration, choosing an "appropriate" number of bins
# to figure out # bins i'm going to first describe duration to get a sense
# of distribution
movie_duration_stats = movies.duration.describe()

# Some quick googling plus my vague recollections of college stats class
# leads me to go with Freedman-Diaconis since there are > 200 rows
h = (2 * (movie_duration_stats['75%']-movie_duration_stats['25%']) 
    * math.pow(movie_duration_stats['count'],(-1/3)))
W = round((movie_duration_stats['max'] - movie_duration_stats['min']) / h, 0)

# It seems that calling plot does not allow us to pass in a variable to bins
# see below; so I'm using .hist instead
# movies.duration.plot(kind='hist', bins=W)
movies.duration.hist(bins=W)

# now for kicks let's try Scott's rule
h = 3.5 * movie_duration_stats['std'] * math.pow(movie_duration_stats['count'],(-1/3))
W = round((movie_duration_stats['max'] - movie_duration_stats['min']) / h, 0)
movies.duration.hist(bins=W)

# Scott's rule misses the odd double peak between 100 and 125 minutes

# use a box plot to display that same data
movies.duration.plot(kind='box')

'''
INTERMEDIATE LEVEL
'''

# count how many movies have each of the content ratings
movies.content_rating.value_counts()

# use a visualization to display that same data, including a title and x and y labels
movies.content_rating.value_counts().plot(kind='bar', title='Bar Plot of # Movies by Content Rating')
plt.xlabel('Content Rating')
plt.ylabel('# Movies')

# convert the following content ratings to "UNRATED": NOT RATED, APPROVED, PASSED, GP
# convert the following content ratings to "NC-17": X, TV-MA
# I thought about doing this a few different ways:
#  a) using filters and assigning resutls
#  b) writing a function and using one of the applying functions (probably vectorize as that seems fastest)
#  c) I finally decided to go with map and having a complete set of mappings
content_rating_mapping = { 'R': 'R'
                        , 'PG-13': 'PG-13'
                        , 'PG': 'PG'
                        , 'NOT RATED': 'UNRATED'
                        , 'APPROVED': 'UNRATED'
                        , 'UNRATED': 'UNRATED'
                        , 'G': 'G'
                        , 'PASSED': 'UNRATED'
                        , 'NC-17': 'NC-17'
                        , 'X': 'NC-17'
                        , 'GP': 'UNRATED'
                        , 'TV-MA': 'NC-17'}
movies.content_rating = movies.content_rating.map(content_rating_mapping)

# count the number of missing values in each column
movies.isnull().sum()

# if there are missing values: examine them, then fill them in with "reasonable" values
movies.content_rating.isnull()
# UNRATED seems like logical value for a movie that has no rating.
movies.content_rating.fillna(value='UNRATED', inplace=True)

# calculate the average star rating for movies 2 hours or longer,
# and compare that with the average star rating for movies shorter than 2 hours
movies[(movies.duration >= 120)].star_rating.mean()
movies[(movies.duration < 120)].star_rating.mean()

# use a visualization to detect whether there is a relationship between duration and star rating
movies.star_rating.value_counts()
# there aren't that many values so we can do a box plot grouped by star_rating
movies.boxplot(column='duration', by='star_rating')

# calculate the average duration for each genre
movies.groupby(movies.genre).duration.mean()

'''
ADVANCED LEVEL
'''

# visualize the relationship between content rating and duration
movies.boxplot(column='duration', by='content_rating')
# todo: manually resort conent_rating axis to G, PG, PG-13, R NC-17, UNRATED

# determine the top rated movie (by star rating) for each genre
movies_by_genre = movies.groupby('genre')
for genre, group in movies_by_genre:
    title = group.sort('star_rating').title.head(1)
    print(genre + ' - ' + title)
#todo: do this as a dictionary comprehension

# check if there are multiple movies with the same title, and if so, determine if they are actually duplicates
# first sort by title to make sure duplicated works; not entirely sure this is necessary
movies.sort('title', inplace='true')
# get a list of the duplicated titles; then find all the movies with title in that list
movies[movies.title.isin(movies[movies.duplicated('title')].title.values)]
# all the movies are unique.

# calculate the average star rating for each genre, but only include genres with at least 10 movies
# using the movies_by_genre dataframegroupby; filter to only have genres with at least 10 records
# then take resulting dataframe and groupby genre again to apply star_rating.agg on it

import time
ts1 = time.time()
num_iterations = 10000
i = 0

while i < num_iterations:
    movies_by_genre.filter(lambda x: len(x) >= 10).groupby('genre').star_rating.agg('mean')
    i += 1
    
ts2 = time.time()
i = 0
# there must be a more efficent way to do this; perhaps using multiple functions at once
while i < num_iterations:
    movies_by_genre_star_ratings = movies_by_genre.star_rating.agg(['count', 'mean'])
    movies_by_genre_star_ratings[movies_by_genre_star_ratings['count'] >= 10]
    i += 1
# the above "feels" more efficeint to me because there is only one iteration through the list
# but, in this case, many of the groups are filtered out so maybe its faster to only compute
# the mean for the subset of groups; seems unlikely though; computing the mean on the small
# groups is unlikely to "cost" much.

ts3 = time.time()

print('avg star rating/genre perf test; i={0}; lambda method took {1}, multi agg method took {2}'.format(i,ts2-ts1, ts3-ts2))

'''
BONUS
'''
# Figure out something "interesting" using the actors data!
movies.dtypes
type(literal_eval(movies.actors_list[0]))
# bummer, the actors_list is a string, not a list; googling around can
# use ast.literal_eval to convert to list
# now, I can create a column with first actor (the lead as far as I can tell)
movies['actors_lead'] = movies.actors_list.apply(lambda x: literal_eval(x)[0])

# get the top 11 (11 selected because 10/11 have same # of movies)
movies['actors_lead'].value_counts().head(11)

# plot a histogram with logarithmic scale to see how many actors
# are leads in x movies
plt.hist(movies['actors_lead'].value_counts(), log=True)
