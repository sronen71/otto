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
X,y,encoder,scaler,ids_val=data.load_train_data('data/train.csv')
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

#bst = xgb.train( param, dtrain, num_round, watchlist )
#preds=bst.predict(dtest)
#bst.save_model('bst.model')

#print 'running cross validation'
#xgb.cv(param,dtrain,num_round,nfold=5,seed=1)


scores = []
p = None
folds =5
scores=np.zeros((y.shape[0],9))
scores_test=np.zeros((X_test.shape[0],9))

for num in range(1,15):
    skf=cv.StratifiedKFold(y,n_folds=folds,shuffle=True,random_state=num)
    for k,(bag,val) in enumerate(skf):
        X_bag,y_bag=X[bag],y[bag]
        X_val,y_val=X[val],y[val]
        dbag = xgb.DMatrix( X_bag, label=y_bag)
        dval = xgb.DMatrix(X_val,label=y_val)
        filename='xgb-models/xgb--iter%s--cv%s.model' % (num,k)
        bst=xgb.Booster()
        bst.load_model(filename)
        predicted=np.array(bst.predict(dval))
        predicted_test=np.array(bst.predict(dtest))
        model_loss=log_loss(y[val],predicted)
        print "iter ",num, "cv", k, "model loss",model_loss
        scores[val,:] += predicted
        scores_test += predicted_test
        filename='xgb-models/xgb--iter%s--cv%s.p' % (str(num),str(k))
	bst.dump_model(filename)
    loss=log_loss(y,scores/num)
    print "*iter ",num, "ensemble loss ",loss


scores_test /= num
scores /= num
data.write_submission(ids_val,scores,encoder,name= 
        'predictions/xgb--bag-val.csv')
data.write_submission(ids,scores_test,encoder,name=
        'predictions/xgb--bag.csv')




