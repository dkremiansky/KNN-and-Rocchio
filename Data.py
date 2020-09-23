import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

"""
class for processing and working with the data
"""
class Data:

    """
    initalize the data
    """
    def __init__(self, path):
        self.df = pd.read_csv(path)

    """
        organizes the data before running the algorithms on it 
    """
    def preprocess(self):
        self.df = self.df.drop(columns=['content_rating', 'movie_imdb_link', 'plot_keywords'])
        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates(subset=['movie_title'], keep='first')
        # a list of categorical features without genres (will be done separately)
        categorical_features = ['color', 'country', 'language','director_name']
        # a list of numerical features without imdb_score and movie_title
        numerical_features = ['actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes',
                              'aspect_ratio', 'budget', 'cast_total_facebook_likes', 'director_facebook_likes',
                              'duration', 'facenumber_in_poster', 'gross', 'movie_facebook_likes', 'num_critic_for_reviews',
                              'num_user_for_reviews', 'num_voted_users', 'title_year']

        # organize categorical features
        self.df = pd.get_dummies(data=self.df, columns=categorical_features)
        actor1 = pd.get_dummies(self.df['actor_1_name'])
        actor2 = pd.get_dummies(self.df['actor_2_name'])
        actor3 = pd.get_dummies(self.df['actor_3_name'])
        self.df = self.df.drop(columns=['actor_1_name', 'actor_2_name', 'actor_3_name','movie_title'])
        self.df = pd.concat([self.df, actor1, actor2, actor3], axis=1)
        self.df = self.df.groupby(self.df.columns, axis=1).sum()
        self.df = self.df.join(self.df.pop('genres').str.get_dummies(sep='|'))

        # organize numerical features
        for feature in numerical_features:
            tmp_vec = list(self.df[feature].values)
            self.df[feature] = self.normalize_vector(tmp_vec)
        rows = self.df.index

        #define lables
        for movie in rows:
            if (self.df.at[movie, 'imdb_score'] >= 7):
                self.df.at[movie, 'imdb_score'] = int(1)
            else:
                self.df.at[movie, 'imdb_score'] = int(0)
        return self.df

    """
    normalize the vactor
    :param  vec: vector that consist numbers of a numerical feature
    :return vec: vector that consist numbers of a numerical feature after normalization
    """
    def normalize_vector(self, vec):
        sigma = np.std(vec,ddof=0)
        average = np.sum(vec)/len(vec)
        for i in range(len(vec)):
            vec[i] = (vec[i] - average)/sigma
        return vec

    """
    splits to k=5 folds
    :return splited data to folds
    """
    def split_to_k_folds(self):
        tmp_folds = KFold(n_splits=5, shuffle=False, random_state=None)
        return tmp_folds.split(self.df)