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

# path to where the data lies

# load in training data,
X,y,encoder,scaler=data.load_train_data('data/train.csv')

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
num_round = 100 # 1000

#bst = xgb.train( param, dtrain, num_round, watchlist )
#preds=bst.predict(dtest)
#bst.save_model('bst.model')

#print 'running cross validation'
#xgb.cv(param,dtrain,num_round,nfold=5,seed=1)

X_train,X_val,y_train,y_val=cv.train_test_split(X,y,test_size=0.2,random_state=2)


dtrain = xgb.DMatrix( X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)


bst= xgb.train(param,dtrain,num_round,watchlist)

preds1=bst.predict(dval)
print log_loss(y_val,preds1)

X_test,ids=data.load_test_data('data/test.csv',scaler)

X_test=X_test[:len(X_test)/6]
dtest = xgb.DMatrix( X_test)

preds_test=bst.predict(dtest)
y_preds_test=np.argmax(preds_test,axis=1)


print y_train.shape,y_preds_test.shape
Xc=np.vstack((X_train,X_test))
yc=np.append(y_train,y_preds_test)
p = np.random.permutation(len(yc))
Xc=Xc[p,:]
yc=yc[p]

dtrain_ext=xgb.DMatrix(Xc,yc)


param = { 'max_depth':16, 'min_child_weight': 6 ,'eta': 0.07,
    'subsmaple': 0.9,'gamma':1,'colsample_bytree':0.5,
    'silent':1 ,'objective': 'multi:softprob',
    'eval_metric' : 'mlogloss','num_class' :9,'nthread': 4}


bst_ext =xgb.train(param,dtrain_ext,num_round,watchlist)
preds2=bst_ext.predict(dval)
print log_loss(y_val,preds2)




