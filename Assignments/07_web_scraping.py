'''
OPTIONAL WEB SCRAPING HOMEWORK

First, define a function that accepts an IMDb ID and returns a dictionary of
movie information: title, star_rating, description, content_rating, duration.
The function should gather this information by scraping the IMDb website, not
by calling the OMDb API. (This is really just a wrapper of the web scraping
code we wrote above.)

For example, get_movie_info('tt0111161') should return:

{'content_rating': 'R',
 'description': u'Two imprisoned men bond over a number of years...',
 'duration': 142,
 'star_rating': 9.3,
 'title': u'The Shawshank Redemption'}

Then, open the file imdb_ids.txt using Python, and write a for loop that builds
a list in which each element is a dictionary of movie information.

Finally, convert that list into a DataFrame.
'''

from bs4 import BeautifulSoup
import requests
import pandas as pd

def get_movie_info(imdb_id):
    result = None #default return value, e.g. movie not found
    url = 'http://www.imdb.com/title/{0}'.format(imdb_id)
    r = requests.get(url)
    if r.status_code == 200:
        result = {'content_rating': None,
                    'description': None,
                    'duration': None,
                    'star_rating': None,
                    'title': None,
                        'imdb_id': imdb_id}
        soup = BeautifulSoup(r.text)

        # get the content rating
        content_rating_tag = soup.find(name='meta', attrs={'itemprop':'contentRating'})
        if content_rating_tag is not None:            
            result['content_rating'] = content_rating_tag['content']
    
        # get the description
        description_tag = soup.find(name='p', attrs={'itemprop':'description'})
        if description_tag is not None:
            result['description'] = description_tag.text

        # get the duration in minutes (as an integer)
        duration_tag = soup.find(name='time', attrs={'itemprop':'duration'})
        if duration_tag is not None:
            duration_text = duration_tag['datetime']
            if duration_text is not None:
                result['duration'] = float(duration_text[2:-1])

        # get the star rating
        result['star_rating'] = float(soup.find(name='span', attrs={'itemprop':'ratingValue'}).text)

        # get the title
        result['title'] = soup.find(name='span', attrs={'itemprop':'name'}).text


    return result
    
movie_list = []
with open('/Users/Micah/Documents/GeneralAssembly/Data Science/DAT8/data/imdb_ids.txt', mode='rU') as f:
    for row in f:
        movie_info = get_movie_info(row)
        if movie_info is not None:        
            movie_list.append(movie_info)

movies = pd.DataFrame(movie_list)


