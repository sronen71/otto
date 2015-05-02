#!/usr/bin/python
# this is the example script to use xgboost to train 
import inspect
import os
import sys
import numpy as np
import data
from sklearn.metrics import log_loss
from sklearn import cross_validation as cv
import sklearn as sk
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder

test_size = 144368


def augment(X):
    rows=X.shape[0]
    cols=X.shape[1]
    Xnew=np.zeros((rows,cols*(cols+1)/2))
    Xnew[:,0:cols]=X
    newcol=cols
    for i in range(cols-1):
        for j in range (i+1,cols):
            Xnew[:,newcol]=X[:,i]+X[:,j]
            newcol += 1
    print newcol,Xnew.shape[1]
    return Xnew

# path to where the data lies

# load in training data,
X,y,encoder,scaler=data.load_train_data('data/train.csv')

#X=augment(X)

#y=np.reshape(y,(len(y),1))
#enc=OneHotEncoder(sparse=False)
#y=enc.fit_transform(y)
#data=scaler.transform(data)
dtrain = xgb.DMatrix( X, label=y)
# 'eta': 0.3
# setup parameters for xgboost

# more regularization:
# increase gamma, min_child_weight

# bigger:
# max depth
"""
# cv5 -> 0.4638
param = { 'max_depth':10, 'min_child_weight':4,'eta': 0.07,
    'subsmaple': 0.9,'gamma':1.0,'colsample_bytree':0.8,
    'silent':1 ,'objective': 'multi:softprob',
    'eval_metric' : 'mlogloss','num_class' :9,'nthread': 4}

#cv5 -> 0.4605
param = { 'max_depth':16, 'min_child_weight':6,'eta': 0.07,
    'subsmaple': 0.9,'gamma':1.0,'colsample_bytree':0.8,
    'silent':1 ,'objective': 'multi:softprob',
    'eval_metric' : 'mlogloss','num_class' :9,'nthread': 4}

#cv5 -> 0.4590
param = { 'max_depth':16, 'min_child_weight': 6 ,'eta': 0.07,
    'subsmaple': 0.9,'gamma':1.0,'colsample_bytree':0.7,
    'silent':1 ,'objective': 'multi:softprob',
    'eval_metric' : 'mlogloss','num_class' :9,'nthread': 4}


#cv5 -> 0.4566 ~ num_round 520 and then over shoot
param = { 'max_depth':16, 'min_child_weight': 6 ,'eta': 0.07,
    'subsmaple': 0.9,'gamma':1.0,'colsample_bytree':0.5,
    'silent':1 ,'objective': 'multi:softprob',
    'eval_metric' : 'mlogloss','num_class' :9,'nthread': 4}
"""

param = { 'max_depth':16, 'min_child_weight': 6 ,'eta': 0.07,
    'subsmaple': 0.9,'gamma':1,'colsample_bytree':0.5,
    'silent':1 ,'objective': 'multi:softprob',
    'eval_metric' : 'mlogloss','num_class' :9,'nthread': 4}




watchlist = [ (dtrain,'train') ]
num_round = 500 # 1000

#bst = xgb.train( param, dtrain, num_round, watchlist )
#preds=bst.predict(dtest)
#bst.save_model('bst.model')

print 'running cross validation'
xgb.cv(param,dtrain,num_round,nfold=5,seed=1)

"""
X_train,X_val,y_train,y_val=cv.train_test_split(X,y,test_size=0.2,random_state=2)


dtrain = xgb.DMatrix( X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)


bst= xgb.train(param,dtrain,num_round,watchlist)

preds1=bst.predict(dval)
print log_loss(y_val,preds1)


"""




