import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy import stats
from sklearn.base import TransformerMixin
from datetime import datetime
from sklearn import linear_model


class MyTidy:
    def tidy_personal(self, personal):
        personal = personal.drop(columns='Unnamed: 0')
        
        personal['address'] = personal['address'].str.replace('\r\n', ', ')
        personal[['street', 'city', 'postal_code']] = personal.address.str.split(', ', expand=True)
        personal = personal.drop(columns='address')
        personal.loc[personal['postal_code'].isnull(), 'postal_code'] = personal.loc[personal['postal_code'].isnull(), 'city']
        personal.loc[personal['postal_code'] == personal['city'], 'city'] = np.nan
        
        personal.loc[personal['date_of_birth'].str.len() > 10, 'date_of_birth'] = personal.loc[personal['date_of_birth'].str.len() > 10, 'date_of_birth'].apply(lambda x: x[:10])
        personal['date_of_birth'] = pd.to_datetime(personal.date_of_birth)
        personal['date_of_birth'] = personal['date_of_birth'].dt.strftime('%d-%m-%y')
        
        for i in range(len(personal)):
            tmp = str(personal.iloc[i, 3])
            if personal.iloc[i, 1] > 19:
                tmp = tmp[0:6] + '19' + tmp[6:]
            else:
                tmp = tmp[0:6] + '20' + tmp[6:]
            personal.iloc[i, 3] = datetime.strptime(tmp, '%d-%m-%Y').date()
        
        tmp = pd.Series(np.zeros(len(personal)))
        tmp[personal[personal['city'].isnull()].index] = 1
        personal['city_was_nan'] = tmp
        
        return personal
    
    
    def tidy_other(self, other):
        other = other.drop(columns='Unnamed: 0')
        
        other['address'] = other['address'].str.replace('\r\n', ', ')
        other[['street', 'city', 'postal_code']] = other.address.str.split(', ', expand=True)
        other = other.drop(columns='address')
        other.loc[other['postal_code'].isnull(), 'postal_code'] = other.loc[other['postal_code'].isnull(), 'city']
        other.loc[other['postal_code'] == other['city'], 'city'] = np.nan
        
        other.personal_info = other.personal_info.str.replace('|', ',')
        other.personal_info = other.personal_info.str.replace(' -- ', ',')
        other.personal_info = other.personal_info.str.replace('\r\r\n', ',')
        other[['job', 'country', 'status', 'employment','race']] = other.personal_info.str.split(',', expand=True)
        other = other.drop(columns='personal_info')
        
        pregnant = {'FALSE': False, 'f': False, 'F': False, 'TRUE': True, 'T': True, 't': True}
        other = other.replace({'pregnant': pregnant})
        
        other.loc[other['job'] == '?', 'job'] = np.nan
        other.loc[other['country'] == '?', 'country'] = np.nan
        other.loc[other['employment'] == '?', 'employment'] = np.nan
        other.loc[other['race'] == '??', 'race'] = np.nan
        other.loc[other['race'] == 'nan', 'race'] = np.nan
        
        other['job'] = other['job'].str.replace('-', '_')
        
        other['education'] = other['education'].str.strip()
        other['relationship'] = other['relationship'].str.strip()
        other['income'] = other['income'].str.strip()
        
        tmp = pd.Series(np.zeros(len(other)))
        tmp[other[other['city'].isnull()].index] = 1
        other['city_was_nan'] = tmp
        
        duplicated = other[other['name'].duplicated()]['name']
        tmp = other[other['name'].isin(duplicated)].rename_axis('index').sort_values(['name', 'index']).index
        for i in range(0, len(tmp), 2):
            for j in range(27):
                if isinstance(other.iloc[tmp[i], j], str):
                    continue
                if np.isnan(other.iloc[tmp[i], j]):
                    other.iloc[tmp[i], j] = other.iloc[tmp[i + 1], j]
                    
        other = other.drop(duplicated.index, axis=0)
        other = other.reset_index(drop=True)
        
        return other
		
		
class MyImputer(TransformerMixin):
    
    def __init__(self, missing_value=np.nan):
        self.missing_value = missing_value
        self.value = []
        
    def _get_mask(self, X, value_to_mask):
        if np.isnan(value_to_mask):
            return X.isnull()
        else:
            return np.equal(X, value_to_mask)
        
    def fit(self, df, y=None):
        column_name = df.columns
        dictionary = {'income': 'job',
                      'relationship': 'status',
                      'status': 'relationship',
                      'education': 'education-num',
                      'education-num': 'education',
                      'hours-per-week': 'employment'}
        
        for i in column_name:
            mask = self._get_mask(df[i], self.missing_value)
            
            if df[i].dtype == 'float64' and i != 'std_glucose' and i != 'class' and i != 'hours-per-week':
                self.value.append(np.median(df.loc[~mask, i]))
            elif i == 'std_glucose':
                model = linear_model.LinearRegression()
                
                X_train = np.array(df.loc[~df['std_glucose'].isnull(), 'mean_glucose'].dropna()).reshape(-1, 1)
                y_train = np.array(df.loc[~df['std_glucose'].isnull(), 'std_glucose'].dropna()).reshape(-1, 1)
                X_test = np.array(df.loc[df['std_glucose'].isnull(), 'mean_glucose'].dropna()).reshape(-1, 1)
                
                model.fit(X=X_train, y=y_train)
                result = model.predict(X_test)
                
                result = result.reshape(len(result)).tolist()
                self.value.append(result)
            elif i == 'race' or i == 'employment' or i == 'job' or i == 'class' or i == 'city' or i == 'country' or i == 'pregnant':
                self.value.append(df.loc[~mask, i].mode()[0])
            elif i != 'name' and i != 'age' and i != 'sex' and i != 'postal_code' and i != 'date_of_birth' and i != 'street':
                tmp = df.loc[df[i].isnull(), [i, dictionary[i]]]
                
                self.value.append([])
                for j in range(len(tmp)):
                    if isinstance(tmp.iloc[j, 1], str):
                        self.value[-1].append(df.loc[df[tmp.columns[1]] == tmp.iloc[j, 1], i].mode()[0])
                    elif not np.isnan(tmp.iloc[j, 1]):
                        self.value[-1].append(df.loc[df[tmp.columns[1]] == tmp.iloc[j, 1], i].mode()[0])
                    else:
                        self.value[-1].append(df[i].mode()[0])
        
        return self
        
    
    def transform(self, df):
        column_name = df.columns
        
        if 'education-num' in column_name:
            df.loc[df['education-num'] > 99, 'education-num'] /= 100
            df.loc[df['education-num'] < 0, 'education-num'] /= -100
        
        k = 0
        for i in range(len(column_name)):
            mask = self._get_mask(df[column_name[i]], self.missing_value)
            
            if df[column_name[i]].dtype == 'float64' and column_name[i] != 'std_glucose' and column_name[i] != 'class' and column_name[i] != 'hours-per-week':
                df[column_name[i]][mask] = self.value[k]
                k += 1
            elif column_name[i] == 'race' or column_name[i] == 'employment' or column_name[i] == 'job' or column_name[i] == 'city' or column_name[i] == 'country' or column_name[i] == 'pregnant':
                df[column_name[i]][mask] = self.value[k]
                k += 1
            elif column_name[i] == 'class':
                df[column_name[i]][mask] = self.value[k]
                df[column_name[i]] = df[column_name[i]].astype(bool)
                k += 1
            elif column_name[i] != 'name' and column_name[i] != 'age' and column_name[i] != 'sex' and column_name[i] != 'postal_code' and column_name[i] != 'date_of_birth' and column_name[i] != 'street':
                tmp = df.loc[df[column_name[i]].isnull()].index
                
                x = 0
                for j in tmp:
                    df.iloc[j, i] = self.value[k][x]
                    x = (x + 1) % len(self.value[k])
                    
                k += 1
                
        if 'pregnant' in column_name:
            df['pregnant'] = df['pregnant'].astype(bool)
                
        return df
		
		
class MyNormalizator:
    def normalize(self, df):
        columns = df.columns
        
        for column in columns:
            if df[column].dtype != 'float64' or column == 'std_glucose' or column == 'education-num' or column == 'hours-per-week' or column == 'city_was_nan' or column == 'class':
                continue
            
            if column == 'skewness_glucose' or column == 'kurtosis_glucose' or column == 'mean_oxygen':
                df[column] = np.log(df[column] + 3)
            elif column == 'std_oxygen' or column == 'skewness_oxygen':
                df[column] = np.sqrt(df[column] + 2)
            else:
                lower = df[column].quantile(0.05)
                upper = df[column].quantile(0.95)
                
                df.loc[df[column] > upper, column] = upper
                df.loc[df[column] < lower, column] = lower
                
        return df
		
		
class MyMerger:
    def merge(self, df1, df2):
        new = pd.merge(df1, df2, on=['name'])
        new = new.drop(columns=['street_x', 'city_x', 'postal_code_x', 'city_was_nan_x'])
        new = new.rename(columns={'street_y': 'street', 'city_y': 'city','postal_code_y': 'postal_code', 'city_was_nan_y': 'city_was_nan'})
        
        return new
