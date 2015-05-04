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
import cPickle as pickle
print 'sklearn version',sk.__version__

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
np.random.seed(1)
X,y,encoder,scaler=data.load_train_data('data/train.csv')
X_test,ids=data.load_test_data('data/test.csv',scaler)
#X=augment(X)

#y=np.reshape(y,(len(y),1))
#enc=OneHotEncoder(sparse=False)
#y=enc.fit_transform(y)
#data=scaler.transform(data)
dtrain = xgb.DMatrix( X, label=y)
dtest = xgb.DMatrix(X_test)
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
    'eval_metric' : 'mlogloss','num_class' :9}




num_round = 500 # 1000

#bst = xgb.train( param, dtrain, num_round, watchlist )
#preds=bst.predict(dtest)
#bst.save_model('bst.model')

#print 'running cross validation'
#xgb.cv(param,dtrain,num_round,nfold=5,seed=1)


scores = []
p = None
folds =5
eps = 1e-4
delta =1 
num=0
scores=np.zeros((y.shape[0],9))
scores_test=np.zeros((X_test.shape[0],9))
prev_loss=10

while (delta>eps):
    num += 1
    skf=cv.StratifiedKFold(y,n_folds=folds,shuffle=True,random_state=num)
    for k,(bag,val) in enumerate(skf):
        X_bag,y_bag=X[bag],y[bag]
        X_val,y_val=X[val],y[val]
        dbag = xgb.DMatrix( X_bag, label=y_bag)
        dval = xgb.DMatrix(X_val,label=y_val)

        watchlist = [ (dval,'validate') ]
        bst=xgb.train(param,dbag,num_round,watchlist)

        predicted=np.array(bst.predict(dval))
        predicted_test=np.array(bst.predict(dtest))
        model_loss=log_loss(y[val],predicted)
        print "iter ",num, "cv", k, "model loss",model_loss
        scores[val,:] += predicted
        scores_test += predicted_test
        filename='xgb-models/xgb--iter%s--cv%s.p' % (str(num),str(k))
	bst.dump_model(filename)
    loss=log_loss(y,scores/num)

    delta=prev_loss-loss
    prev_loss=loss    
    print "*iter ",num, "ensemble loss ",loss, "delta", delta


scores_test /= num
data.write_submission(X_test,ids,scores_test,encoder,name=
        'predictions/xgb--bag.csv')
"""
X_train,X_val,y_train,y_val=cv.train_test_split(X,y,test_size=0.2,random_state=2)


dtrain = xgb.DMatrix( X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)


bst= xgb.train(param,dtrain,num_round,watchlist)

preds1=bst.predict(dval)
print log_loss(y_val,preds1)


"""




