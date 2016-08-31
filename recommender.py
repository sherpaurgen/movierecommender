import pandas as pd
import numpy as np
r_cols=['user_id','movie_id','rating']
ratings=pd.read_csv('/home/urgen/Downloads/ml-latest-small/ratings.csv',sep=',',names=r_cols,usecols=[0,1,2])
m_cols=['movie_id','title']
movies=pd.read_csv('/home/urgen/Downloads/ml-latest-small/movies.csv',sep=',',names=m_cols,usecols=[0,1])
ratings=pd.merge(movies,ratings)
ratings.head()
movieRatings = ratings.pivot_table(index=["user_id"],columns="title",values="rating")
movieRatings.head()
toystoryrating=movieRatings['Toy Story (1995)']
toystoryrating.head()
similarMovies=movieRatings.corrwith(toystoryrating)
# http://tungwaiyip.info/2012/Collaborative%20Filtering.html
# finding the similarity score between the movies..score close to 1 means their tastes are very similar
similarMovies=similarMovies.dropna()
df = pd.DataFrame(similarMovies)
#df
#similarMovies.order(ascending=False)
movieStats=ratings.groupby('title').agg({'rating': [np.size,np.mean]})
movieStats.head()
similarMovies.order(ascending=False)
movieStats.head()
#taking movies rated by more than 100 people
popularMovies=movieStats['rating']['size']>=100
#ascending=false for mean
movieStats[popularMovies].sort_values([('rating','mean')], ascending=False)[:15]
df = movieStats[popularMovies].join(pd.DataFrame(similarMovies,columns=['similarity']))
df.head()
df.sort_values(['similarity'],ascending=False)[:15]






