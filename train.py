import os
import re
import json


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib

def incremental_learning(data):
    
    y = data['is_purchase']
    X = data.drop('is_purchase', axis=1)
    
    clf = joblib.load('model.pkl')
    
    try:
        clf.particial_fit(X, y)
        joblib.dump(clf, 'model.pkl')
    except AttributeError:
        print('Model doesn`t support particial learning :(')
        
if __name__ == '__main__':
    data = pd.read_csv('train.csv')
    incremental_learning(data)
