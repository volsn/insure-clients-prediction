import os
import re
import json


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

def incremental_learning(data_new):
    
    data_old = pd.read_csv('data_old.csv')
    data_new = pipeline.transform(data_new)
    
    data = pd.concat([data_old, data_new], axis=0)
    data.to_csv('data_old.csv', index=False)
    
    y = data['is_purchase'].as_type('bool')
    X = data.drop('is_purchase', axis=1)
    
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    
    joblib.dump(clf, 'model.pkl')
    

data = pd.read_csv('train.csv')
incremental_learning(data)