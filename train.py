import os
import re
import json

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib


scaler_premium = joblib.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'scaler_premium.pkl'))
scaler_sum = joblib.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'scaler_sum.pkl'))

class CompaniesTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        #data__companies = self._find_num_companies(X)
        X = X.drop_duplicates(['key']).reset_index(drop=True)
        #X['Num_of_companies'] = data__companies
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
    
POSSIBLE_COLUMNS = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sport_types.npy')).tolist()
    
    
class SportTransform(BaseEstimator, TransformerMixin):
    
    possible_columns = [
         'DANCE',
         'GIMNASTIKA',
         'CHIRLIDING',
         'TURIZM',
         'PLAVANIE',
         'BOKS',
         'VINDSYORFING',
         'STRELBA',
         'TANTSY',
         'FUTBOL',
         'GANDBOL',
         'ARMRESLING',
         'BEG',
         'BOJ',
         'DZYUDO',
         'BMX',
         'KATANIE',
         'TRIATLON',
         'KARATE',
         'AEROBIKA'
    ]
    
    possible_columns_dummies = ['sport_{}'.format(sport) for sport in possible_columns]
    
    def fit(self, X, y=None):
        
        sports_dummies = pd.get_dummies(X['sports'], prefix='Sport')
        self.possible_columns = np.concatenate([self.possible_columns, [dummy for dummy in sports_dummies if dummy not in self.possible_columns]])
        
        np.save('sport_types.npy', self.possible_columns)
        
        return self
    
    def transform(self, X, y=None):
        
        for i, row in X.iterrows():
            for sport in self.possible_columns:
                if re.search(sport, X.loc[i].sports):
                    X.loc[i, 'sports'] = sport
                    
            if X.loc[i].sports not in self.possible_columns:
                X.loc[i, 'sports'] = 'Others'
                    
                    
        user_agent_dummies = pd.get_dummies(X['sports'], prefix='sport')
        
        for column in self.possible_columns_dummies:
            if column in user_agent_dummies.columns:
                X[column] = user_agent_dummies[column]
            else:
                X[column] = np.zeros(X.shape[0], dtype=np.uint8)
                
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
        
        X['Group_small'] = X['Group_size'].map(lambda x: 1 if int(x) == 1 else 0)
        X['Group_middle'] = X['Group_size'].map(lambda x: 1 if 1 < int(x) < 5 else 0)
        X['Group_large'] = X['Group_size'].map(lambda x: 1 if 10 < int(x) else 0)
        
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
                'place', 'timezone', 'ip', \
                'referer', 'created_at', 'action_type', 'is_partner', 'is_foreigner', 'is_adblock_enabled'], axis=1, inplace=True)
        
        return X
    
class BinaryTransform(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X['is_foreigner'] = X['is_foreigner'].map(lambda x: 1 if x == True else 0)
        X['is_adblock_enabled'] = X['is_adblock_enabled'].map(lambda x: 1 if x == True else 0)
        X['action_type'] = X['action_type'].map(lambda x: 1 if x == True else 0)
        X['is_partner'] = X['is_partner'].map(lambda x: 1 if x == True else 0)
        
        return X
    
class ScalerTransform(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X['premium'] = scaler_premium.transform(X['premium'].values.reshape(-1, 1))
        X['sum'] = scaler_sum.transform(X['sum'].values.reshape(-1, 1))
        return X

class TimestampTransform(BaseEstimator, TransformerMixin):
    
    possible_hours = ['Hour_{}'.format(hour) for hour in list(range(0, 24))]
    possible_days = ['Day_{}'.format(day) for day in list(range(0, 7))]

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X['created_at'] = X['created_at'].map(lambda stamp: datetime.datetime.fromtimestamp(int(stamp)))
        X['weekday'] = X['created_at'].map(lambda date: date.weekday())
        X['hour'] = X['created_at'].map(lambda date: date.hour)
        
        weekday_dummies = pd.get_dummies(X['weekday'], prefix='Day')
        hour_dummies = pd.get_dummies(X['hour'], prefix='Hour')

        for day in self.possible_days:
            if day in weekday_dummies.columns:
                X[day] = weekday_dummies[day]
            else:
                X[day] = np.zeros(X.shape[0], dtype=np.uint8)

        for hour in self.possible_hours:
            if hour in hour_dummies.columns:
                X[hour] = hour_dummies[hour]
            else:
                X[hour] = np.zeros(X.shape[0], dtype=np.uint8)
                
        X.drop('created_at', axis=1, inplace=True)

        return X


pipeline = Pipeline([
    ('companies', CompaniesTransformer()),
    ('sports', SportTransform()),
    ('duration', DurationTransform()),
    ('group', GroupTransformer()),
    ('user_agent', UserAgentTransform()),
    ('binary', BinaryTransform()),
    ('droping', DropingTransform()),
    ('scaler', ScalerTransform()),
])


def incremental_learning(data):
    
    data = pipeline.transform(data)
    y = data['is_purchase']
    X = data.drop('is_purchase', axis=1)
    
    clf = joblib.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model.pkl'))

    try:
        clf.partial_fit(X, y)
        clf = joblib.dump(clf, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model.pkl'))
    except AttributeError:
        print('Model doesn`t support partial learning :(')

        
if __name__ == '__main__':
    data = pd.read_csv('train.csv')
    incremental_learning(data)
