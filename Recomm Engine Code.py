#%%
# Import Packages
import pandas as pd
#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances 
#import re
from sklearn.preprocessing import StandardScaler

#from scipy.sparse import csr_matrix

#%%

#Reading users file:
userscolNames = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv(r'C:\Users\Satyajit Narayanan\Desktop\589\Project\ml-100k\ml-100k\u.user', sep='|', names=userscolNames,encoding='latin-1')

# 943 users
# Age between 7 and 73
# 21 different Occupations: 'technician', 'other', 'writer', 'executive', 'administrator','student', 'lawyer', 'educator', 'scientist', 'entertainment', 'programmer', 'librarian', 'homemaker', 'artist', 'engineer', 'marketing', 'none', 'healthcare', 'retired', 'salesman', 'doctor'
# 795 Zip Codes


#%%

#Reading ratings file:
ratingscolNames = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings = pd.read_csv(r'C:\Users\Satyajit Narayanan\Desktop\589\Project\ml-100k\ml-100k\u.data', sep='\t', names=ratingscolNames,encoding='latin-1')

# 100k rows
# 5 Unique Ratings
# 1682 movie_ids

#%%

'''
# The dataset has already been divided into train and test by GroupLens where the test data has 10 ratings for each user (9,430 rows in total)

# Reading them into environment

ratings_train = pd.read_csv('ua.base', sep='\t', names=ratingscolNames, encoding='latin-1')
ratings_test = pd.read_csv('ua.test', sep='\t', names=ratingscolNames, encoding='latin-1')

ratings_train.shape, ratings_test.shape

'''


#%%
#Reading movies file:
moviecolNames = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv(r'C:\Users\Satyajit Narayanan\Desktop\589\Project\ml-100k\ml-100k\u.item', sep='|', names=moviecolNames,
encoding='latin-1')

# (1682, 24)

#%%

#
Users= ratings.user_id.unique()


# Number of users
numUsers = ratings.user_id.unique().shape[0]

# Number of movies
numMovies = ratings.movie_id.unique().shape[0]

#%%
# Creating User-Movie matrix

usermovieMatrix = np.zeros((numUsers, numMovies))
for line in ratings.itertuples():
    usermovieMatrix[line[1]-1, line[2]-1] = line[3]
    
len(usermovieMatrix)


#%%
   ''' 
movie_genre = movies.copy()
movie_genre.drop(movie_genre.columns[[0, 1,2,3,4,5]], axis=1, inplace=True)
movieCosineSimilarity = pairwise_distances(movie_genre, metric='cosine')

#%%
    
user_char = users.copy()
user_char.drop(user_char.columns[[0, 4]], axis=1, inplace=True)
userCosineSimilarity = pairwise_distances(user_char.apply(lambda x: x.factorize()[0]), metric='cosine')


'''

#%%

# Calculating the similarity
userCosineSimilarity = pairwise_distances(usermovieMatrix, metric='cosine')
movieCosineSimilarity = pairwise_distances(usermovieMatrix.T, metric='cosine')
    
#%%
# Creating function to make predictions based on these similarities

def predict(ratings, similarity, type='user'):
    if type == 'user':
#        mean_user_rating = ratings.mean(axis=1)
##        #We use np.newaxis so that mean_user_rating has same format as ratings
 #       ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
  #      pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
        pred = similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'movie':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

#%%
# Making predictions
    
user_prediction = predict(usermovieMatrix, userCosineSimilarity, type='user')
movie_prediction = predict(usermovieMatrix, movieCosineSimilarity, type='movie')


#%%
'''
i = 9

# Ranking based on user prediction
userRank = pd.DataFrame(pd.DataFrame(user_prediction).iloc[i,])

userRank['Rank'] = userRank.rank(ascending=False)
userRank.sort_values('Rank')

userRank = pd.merge(userRank, movies['movie title'].to_frame(), how='left', left_index=True, right_index=True)


userRankjoin = pd.merge(userRank, 
pd.DataFrame(usermovieMatrix).iloc[i,].to_frame(), how='left', left_index=True, right_index=True)

#userRankjoin.sort_values('Rank').head()

userRankF = userRankjoin.drop(userRankjoin[userRankjoin[f'{i}_y']>0].index)


userRankF.sort_values('Rank').head()


#%%
i = 9

# Ranking based on movie prediction
movieRank = pd.DataFrame(pd.DataFrame(movie_prediction).iloc[i,])

movieRank['Rank'] = movieRank.rank(ascending=False)
movieRank.sort_values('Rank')

movieRank = pd.merge(movieRank, movies['movie title'].to_frame(), how='left', left_index=True, right_index=True)

movieRankjoin = pd.merge(movieRank, 
pd.DataFrame(usermovieMatrix).iloc[i,].to_frame(), how='left', left_index=True, right_index=True)

movieRankF = movieRankjoin.drop(movieRankjoin[movieRankjoin[f'{i}_y']>0].index)

#movieRankjoin.sort_values('Rank').head()


movieRankF.sort_values('Rank').head()
'''

#%%

combined_pred = user_prediction + movie_prediction

i = 9

# Ranking based on combined prediction
combinedRank = pd.DataFrame(pd.DataFrame(combined_pred).iloc[i,])

combinedRank['Rank'] = combinedRank.rank(ascending=False)
combinedRank.sort_values('Rank')

combinedRank = pd.merge(combinedRank, movies['movie title'].to_frame(), how='left', left_index=True, right_index=True)

combinedRankjoin = pd.merge(combinedRank, 
pd.DataFrame(usermovieMatrix).iloc[i,].to_frame(), how='left', left_index=True, right_index=True)

combinedRankF = combinedRankjoin.drop(combinedRankjoin[combinedRankjoin[f'{i}_y']>0].index)

#combinedRankjoin.sort_values('Rank').head()


combinedRankF.sort_values('Rank').head()


#%%



movieRecomm = pd.DataFrame(movieCosineSimilarity[49]).sort_values([0])

movieRecomm = movieRecomm.drop(movieRecomm[movieRecomm[0].index==49].index)

movieRecomm = pd.merge(movieRecomm.head(), movies['movie title'].to_frame(), how='left', left_index=True, right_index=True)


#%%


