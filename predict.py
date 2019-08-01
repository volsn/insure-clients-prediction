import os
import re
import json

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

import warnings
warnings.simplefilter("ignore")
warnings.warn("deprecated", DeprecationWarning)
pd.options.mode.chained_assignment = None


class CompaniesTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        data__companies = self._find_num_companies(X)
        X = X.drop_duplicates(['key']).reset_index(drop=True)
        X['Num_of_companies'] = data__companies
        X.drop('key', axis=1, inplace=True)
        
        return X
    
    def _find_num_companies(self, data):
        keys = data.key.unique()
        data__companies = []
         
        for key in keys:
            data__companies.append(
                len(data[data['key'] == key].company.unique())
            )
            
        return data__companies
    
POSSIBLE_COLUMNS = np.load('sport_types.npy').tolist()
    
    
class SportTransform(BaseEstimator, TransformerMixin):
    
    possible_columns = POSSIBLE_COLUMNS
    
    def fit(self, X, y=None):
        
        sports_dummies = pd.get_dummies(X['sports'], prefix='Sport')
        self.possible_columns.extend([dummy for dummy in sports_dummies if dummy not in self.possible_columns])
        
        return self
    
    def transform(self, X, y=None):
        
        sports_dummies = pd.get_dummies(X['sports'], prefix='Sport')
        for column in self.possible_columns:
            if column in sports_dummies.columns:
                X[column] = sports_dummies[column]
            else:
                X[column] = np.zeros(X.shape[0], dtype=np.uint8)
        
        #X = pd.concat([X, sports_dummies], axis=1)
        X.drop('sports', axis=1, inplace=True)
        
        return X
    
class DurationTransform(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X['init_from__days'] = X['init_from'].map(lambda d: int(d.split('-')[2]))
        X['init_from__month'] = X['init_from'].map(lambda w: int(w.split('-')[1]))
        X['init_from__years'] = X['init_from'].map(lambda m: int(m.split('-')[0]))

        X['init_till__days'] = X['init_till'].map(lambda d: int(d.split('-')[2]))
        X['init_till__month'] = X['init_till'].map(lambda w: int(w.split('-')[1]))
        X['init_till__years'] = X['init_till'].map(lambda m: int(m.split('-')[0]))
        
        X['Length'] = (X['init_till__days'] - X['init_from__days']) + \
                        (X['init_till__month'] - X['init_from__month']) * 30 + \
                            (X['init_till__years'] - X['init_from__years']) * 365 + 1
        
        X.drop(['init_from__days', 'init_from__month', \
                   'init_from__years', 'init_till__days', \
                   'init_till__month', 'init_till__years', \
                   'init_from', 'init_till'], axis=1, inplace=True)
        
        X['Short_term'] = X['Length'].map(lambda l: 1 if l < 7 else 0)
        X['Middle_term'] = X['Length'].map(lambda l: 1 if 7 < l < 30  else 0)
        X['Long_term'] = X['Length'].map(lambda l: 1 if 30 < l else 0)
        
        X.drop('Length', axis=1, inplace=True)
        
        return X
    
class GroupTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X['Group_size'] = X['adult'] + X['child']
        
        X['Group_small'] = X['Group_size'].map(lambda x: 1 if x == 1 else 0)
        X['Group_middle'] = X['Group_size'].map(lambda x: 1 if 1 < x < 5 else 0)
        X['Group_large'] = X['Group_size'].map(lambda x: 1 if 10 < x else 0)
        
        X.drop(['Group_size', 'adult', 'child'], axis=1, inplace=True)
        
        return X
    
"""
Possible OSs:
    - Android
    - iPhone
    - Macintosh
    - Windows
"""

class UserAgentTransform(BaseEstimator, TransformerMixin):
    
    possible_columns = ['User_agent_Android', 'User_agent_Macintosh', \
                'User_agent_Others', 'User_agent_Windows', 'User_agent_iPhone']
    
    def fit(self, X, y=None):
        
        X['user_agent'] = X['user_agent'].map(lambda agent: \
            'Android' if re.search('Android', str(agent)) else \
            'iPhone' if re.search('iPhone', str(agent)) else \
            'Macintosh' if re.search('Macintosh', str(agent)) else \
            'Windows' if re.search('Windows', str(agent)) else \
            'Others'
        )
        
        user_agent_dummies = pd.get_dummies(X['user_agent'], prefix='User_agent')
        self.possible_columns.extend([dummy for dummy in user_agent_dummies if dummy not in self.possible_columns])
        
        
        return self
    
    def transform(self, X, y=None):
        
        X['user_agent'] = X['user_agent'].map(lambda agent: \
            'Android' if re.search('Android', str(agent)) else \
            'iPhone' if re.search('iPhone', str(agent)) else \
            'Macintosh' if re.search('Macintosh', str(agent)) else \
            'Windows' if re.search('Windows', str(agent)) else \
            'Others'
        )
        
        user_agent_dummies = pd.get_dummies(X['user_agent'], prefix='User_agent')
        
        for column in self.possible_columns:
            if column in user_agent_dummies.columns:
                X[column] = user_agent_dummies[column]
            else:
                X[column] = np.zeros(X.shape[0], dtype=np.uint8)
        
        #X = pd.concat([X, user_agent_dummies], axis=1)
        X.drop('user_agent', axis=1, inplace=True)
        
        return X
    
class DropingTransform(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X.drop(['status', 'company', 'year', \
                'place', 'created_at', 'timezone', 'ip', \
                'referer', 'is_partner'], axis=1, inplace=True)
        
        return X
    
class BinaryTransform(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X['is_foreigner'] = X['is_foreigner'].map(lambda x: 1 if x == True else 0)
        X['is_adblock_enabled'] = X['is_adblock_enabled'].map(lambda x: 1 if x == True else 0)
        X['action_type'] = X['action_type'].map(lambda x: 1 if x == True else 0)
        #X['is_purchase'] = X['is_purchase'].map(lambda x: 1 if x == True else 0)
        
        return X


pipeline = Pipeline([
    ('companies', CompaniesTransformer()),
    ('sports', SportTransform()),
    ('duration', DurationTransform()),
    ('group', GroupTransformer()),
    ('user_agent', UserAgentTransform()),
    ('binary', BinaryTransform()),
    ('droping', DropingTransform()),
])


def predict_prices(keys, data, clf):
    
    results = {}
    for key, (i, row) in zip(keys, data.iterrows()):
        
        (min_, max_) = (0, 0)
    
        for i in range(20):
            row.loc['premium'] -= row['premium'] * 0.05
            if clf.predict(pd.DataFrame([row])) == [True]:
                min_ = row['premium']
            else:
                break

        for i in range(20):
            row.loc['premium'] += row['premium'] * 0.05
            if clf.predict(pd.DataFrame([row])) == [True]:
                max_ = row['premium']
            else:
                break
        
        results[str(key)] = [str(min_), str(max_)]
        
    return results


def predict(data):
    
    keys = data.copy().drop_duplicates(['key']).reset_index(drop=True)['key'].values
    X = pipeline.transform(data.copy())
    #pipeline.transform(X)
    
    if 'is_purchase' in X.columns:
        X.drop('is_purchase', axis=1, inplace=True)
    data.to_csv('output.csv', index=False)
        
    clf = joblib.load('model.pkl')
    predictions = clf.predict(X)
    
    results = {'proba': {} ,'prices': {}, 'is_purchase': {}}
    for key, pred, in zip(keys, predictions):
        results['is_purchase'][str(key)] = 1 if pred == True else 0
        
    X_proba = X[predictions == True].copy()
    keys_proba = keys[predictions == True]
    
    
    for key, (i, row) in zip(keys_proba, X_proba.iterrows()):
        results['proba'][str(key)] = str(clf.predict_proba(pd.DataFrame([row]))[0][1])
    
    
    X_prices = X[predictions == True].copy()
    keys_prices = keys[predictions == True]
    results['prices'] = predict_prices(keys_prices, X_prices, clf)
        
    return results

data = pd.read_csv('input.csv')
results = predict(data)

with open('output.json', 'w') as f:
    json.dump(results, f)
