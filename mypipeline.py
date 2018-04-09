import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm_notebook
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class MultinomialNaiveBayesLogProbs(BaseEstimator, TransformerMixin):
    def __init__(self, min_df=7, text_column='text', count_weighting="standard"):
        if count_weighting == "standard" or count_weighting == "normalized":
            self.vectorizer = CountVectorizer(stop_words='english', min_df=min_df)
        elif count_weighting == "tfidf":
            self.vectorizer = TfidfVectorizer(stop_words='english', min_df=min_df)
        self.multi_nb = MultinomialNB()
        self.text_column = text_column
        self.count_weighting = count_weighting
        
    def fit(self, df, y=None):
        self.vectorizer.fit(df[self.text_column])
        bow = self.vectorizer.transform(df[self.text_column]).toarray()
        if self.count_weighting == "normalized":
            #(feat - min) / (max - min)
            x_min = bow.min(axis=0)
            x_max = bow.max(axis=0)
            bow = (bow - x_min) / (x_max - x_min)
        self.multi_nb.fit(bow, y)
        return self
    
    def transform(self, df):
        bow = self.vectorizer.transform(df[self.text_column]).toarray()
        arr = self.multi_nb.predict_log_proba(bow)
        speaker_names = ['ballard', 'eyring', 'faust', 'hinckley', 'kimball', 'monson',
       'nelson', 'oaks', 'packer', 'perry']
        out = pd.DataFrame(arr, columns=['nb_log_prob_{}'.format(s) for s in speaker_names])
        return pd.concat([df, out], axis=1)
    
class CleanTable(BaseEstimator, TransformerMixin):
    def __init__(self, remove_text=True, remove_onehots=True):
        self.list_of_columns_to_remove = ['talk_id', 'total_words', 'norm_X']
        if remove_text:
            self.list_of_columns_to_remove += ['text', 'lemmatized_text']
        if remove_onehots:
            self.list_of_columns_to_remove += ['ballard', 'eyring', 'faust', 
                                               'hinckley', 'kimball', 'monson', 
                                               'nelson', 'oaks', 'packer', 'perry'  ]
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, df):
        out = df.drop(self.list_of_columns_to_remove, axis=1).copy()
        out.fillna(75, inplace=True)
        assert out.isnull().values.any() == False
        return out

def test_pipeline(df, nlp_pipeline, y_column='speaker'):
    label = df[y_column].copy()
    X = df.drop(y_column, axis=1).copy()
    
    rskf = StratifiedKFold(n_splits=5, random_state=1)
    accs = []
    for train_index, test_index in rskf.split(X, label):
        X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train, y_test = label[train_index], label[test_index]
                
        nlp_pipeline.fit(X_train.reset_index(), y_train)
        accs.append((nlp_pipeline.predict(X_test.reset_index()) == y_test).mean())
    
    print "avg accuracies:", np.mean(accs)
    return accs
