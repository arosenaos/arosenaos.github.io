#!/usr/bin/env python
# coding: utf-8

# https://www.sciencedirect.com/science/article/pii/S266682702100102X 
# 
# - rainfall hard to predict, important for forecasting floods & impacts to infrastructure
# - variables: pressure, temperature, wind speed, humidity 
# 
# 
# questions: 
# - do i need to add relative humidity and wind speed
# - should i compare the feature significance at 500hpa versus surface? 
# - should i be using surface values for the model? 
# - what do my results mean? what values for my features are most correlated with an APE? 
# - should rainfall predictors be different in different regions? 
# 
# do tos: 
# - three-four figures: interpolation process (show temperature interpolated), apes categorization (a plot with ape example where morning and evening precipitation are less than afternoon), result figures which shows feature importance 
# 

# In[318]:


import pandas as pd
import seaborn as sb
from func import cal_buoyancy, sounding_cal
import glob
import os
import re
import datetime
import metpy.calc as mpcalc
from metpy.units import units
from metpy.cbook import get_test_data
from metpy.plots import add_metpy_logo, SkewT
from pyhdf import SD
from scipy.interpolate import interp1d
from scipy.optimize import bisect
import numpy as np
import pickle
import math
import pint
from pydoc import help
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from pyhdf.SD import *
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
from numpy import load, asarray, save
import pytz
from scipy import stats
from sklearn.linear_model import LinearRegression
from collections import Counter
import pymannkendall as mk

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

from sklearn.tree import export_graphviz
from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# In[345]:


with open('resultdf.pkl', 'rb') as file:
    resultdf = pickle.load(file)


# In[346]:


columns_to_keep = ['interp_pres','interp_temp','interp_dp','q_obs','fwi','APE']

apes = resultdf[columns_to_keep]

new_cols = {'interp_pres': 'pres',
            'interp_temp': 'temp',
            'interp_dp': 'dp',
            'q_obs': 'q',
            'fwi': 'fwi',
            'APE': 'ape'
           }

apes.rename(columns=new_cols, inplace=True)

sfc_df = pd.DataFrame()
for col in apes.columns:
    if isinstance(apes[col][0], np.ndarray) or isinstance(apes[col][0], list):
        sfc_df[col] = apes[col].apply(lambda x: x[0])
    else:
        sfc_df[col] = apes[col]


# In[395]:


features = list(sfc_df.columns)
features.remove('ape')
plt.subplots(figsize=(30,20))
 
for i, col in enumerate(features):
    plt.subplot(8,12, i + 1)
    sb.distplot(sfc_df[col])
plt.tight_layout()
plt.show()


# In[397]:


plt.subplots(figsize=(30,20))
 
for i, col in enumerate(features):
    plt.subplot(8,12, i + 1)
    sb.boxplot(data=sfc_df[col], orient='h')
    plt.xlabel(col)
plt.tight_layout()
plt.show()


# In[ ]:


#some code to remove outliers? 


# In[349]:


plt.figure(figsize=(5,5))
sb.heatmap(sfc_df.corr() > 0.8,
           annot=True,
           cbar=False)
plt.show()


# In[365]:


#code to remove highly correlated q and dp?
sfc_df = sfc_df.drop('q',axis=1)
sfc_df.head()


# In[371]:


features = sfc_df.drop('ape', axis=1)
target = sfc_df.ape

X_train, X_test, y_train, y_test = train_test_split(features,
                                      target,
                                      test_size=0.2,
                                      stratify=target,
                                      random_state=2)
 
ros = RandomOverSampler(sampling_strategy='minority',
                        random_state=22)

X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# scaler = StandardScaler()
# X = scaler.fit_transform(X_resampled)
# X_test = scaler.transform(X_test)

param_dist = {'n_estimators': randint(50, 500),
              'max_depth': randint(1, 50)} 

rf = RandomForestClassifier()

rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

rand_search.fit(X_resampled, y_resampled)

best_rf = rand_search.best_estimator_

print('Best hyperparameters:',  rand_search.best_params_)


# In[372]:


y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# In[373]:


feature_importances = pd.Series(best_rf.feature_importances_, index=X_test.columns).sort_values(ascending=False)
feature_importances.plot.bar();


# In[376]:


best_rf.feature_importances_


# In[374]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[401]:


get_ipython().system('pip install nbformat')


# In[402]:


get_ipython().system('jupyter nbconvert --to pdf ml_project.ipynb')


# In[ ]:




