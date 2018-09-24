#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 10:29:02 2018

@author: jeremywatkins
"""

#import some necessary librairies

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn.cross_validation import train_test_split
from bayes_opt import BayesianOptimization
from operator import itemgetter


pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)


train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])


ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})


all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
    
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)    
    
    
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
    
    
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
    
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

#Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()

#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))

# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
all_data = pd.get_dummies(all_data)


########################
### Global Variables ###
########################
train = all_data[:ntrain]
test = all_data[ntrain:]
features=list(train.columns.values)
train['SalePrice']=y_train
target='SalePrice'
num_models=5
RANDOM_SEED=42

########################
    
def run_single(train, test, params, features, target, random_state=0):

    num_boost_round = 1000
    early_stopping_rounds = 20
    test_size = 0.125
    verbosity=False
    #verbosity=True
   
    
    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    #print('Length train:', len(X_train.index))
    #print('Length valid:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbosity)
    
 
    
    test_prediction = gbm.predict(xgb.DMatrix(test[features],missing = -99), ntree_limit=gbm.best_iteration+1)


    return test_prediction

  
def optim_run_single(train, features, target, params, random_state=0):   
    num_boost_round = 1000
    early_stopping_rounds = 20
    test_size = 0.125
    verbosity=False
    #verbosity=True
   
    
    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    #print('Length train:', len(X_train.index))
    #print('Length valid:', len(X_valid.index))
    y2_train = X_train[target]
    y2_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y2_train)
    dvalid = xgb.DMatrix(X_valid[features], y2_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbosity)
    return gbm.best_score
    #return -1.0 * gbm['test-rmse-mean'].iloc[-1]

def multi_model(train, test, params, features, target,num_models, random_state=0):
    all_preds=[]
    for i in range(num_models):
        preds =run_single(train, test,params, features, target, random_state)
        all_preds.append(preds)
        random_state=random_state+1
    avg_pred=np.mean(np.array(all_preds),axis=0)
    return avg_pred

def xgb_eval_single(min_child_weight,colsample_bytree,max_depth,subsample,gamma,alpha):
    
    random_state=42
    eta=.1
    xtrain=train
    xfeatures=features

    
    params = {
        "objective": "reg:linear",
        "booster" : "gbtree",
        "eval_metric": "rmse",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "silent": 1,
        "seed": random_state,
        #"num_class" : 22,
    }
    
    params['min_child_weight'] = int(min_child_weight)
    params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample, 1), 0)
    params['gamma'] = max(gamma, 0)
    params['alpha'] = max(alpha, 0)
    
    

    score =optim_run_single(xtrain, xfeatures, target, params, random_state)

    
    return -1*score
  
def xgb_eval_multi(min_child_weight,colsample_bytree,max_depth,subsample,gamma,alpha):
    
    random_state=42
    eta=.1
    xtrain=train
    xfeatures=features
    
    params = {
        "objective": "reg:linear",
        "booster" : "gbtree",
        "eval_metric": "rmse",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "silent": 1,
        "seed": random_state,
        #"num_class" : 22,
    }
    
    params['min_child_weight'] = int(min_child_weight)
    params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample, 1), 0)
    params['gamma'] = max(gamma, 0)
    params['alpha'] = max(alpha, 0)
    
    
    all_scores=[]
    for i in range(num_models):
        score =optim_run_single(xtrain, xfeatures, target, params, random_state)
        all_scores.append(score)
        random_state=random_state+1

    avg_score=np.mean(all_scores)
    
    return -1*avg_score

#####################################################################
### bayesian optimization with either single model or multi model ###
#####################################################################
""" 
xgb_bo = BayesianOptimization(xgb_eval_multi, {'min_child_weight': (1, 20),
                                                'colsample_bytree': (0.1, 1),
                                                'max_depth': (3, 15),
                                                'subsample': (0.5, 1),
                                                'gamma': (0, 10),
                                                'alpha': (0, 10),
                                                })
        
"""       
xgb_bo = BayesianOptimization(xgb_eval_single, {'min_child_weight': (1, 20),
                                                'colsample_bytree': (0.1, 1),
                                                'max_depth': (3, 15),
                                                'subsample': (0.5, 1),
                                                'gamma': (0, 10),
                                                'alpha': (0, 10),
                                                })


# Use the expected improvement acquisition function to handle negative numbers
# Optimally needs quite a few more initiation points and number of iterations
xgb_bo.maximize(init_points=5, n_iter=10, acq='ei')
    
params = xgb_bo.res['max']['max_params']
params['max_depth'] = int(params['max_depth'])
        
        
preds=multi_model(train, test,params, features, target,num_models,RANDOM_SEED)


preds=np.expm1(preds)
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = preds
sub.to_csv('submission.csv',index=False)








