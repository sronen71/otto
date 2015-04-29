# 0.4483
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from lasagne.layers import DenseLayer, InputLayer,DropoutLayer
from lasagne.layers import FeaturePoolLayer
from lasagne.nonlinearities import softmax,LeakyRectify
from lasagne.updates import nesterov_momentum
from lasagne.updates import adadelta
from lasagne.init import HeNormal


from nolearn.lasagne import NeuralNet
from sklearn.ensemble import BaggingClassifier
from sklearn import cross_validation as cv
from sklearn.metrics import log_loss
import sklearn as sk

import theano

def tfloat32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self,name,epochs,rates):
        self.name = name
        self.epochs=epochs
        self.rates=rates
        self.state=0
    def __call__(self,nn,train_history):
        epoch=train_history[-1]['epoch']
        if self.state>=len(self.epochs):
            return
        if epoch>self.epochs[self.state]:
            new_value=tfloat32(self.rates[self.state]) 
            getattr(nn,self.name).set_value(new_value)
            #print "step, learing rate:", self.state+1,self.rates[self.state]
            self.state += 1


def load_train_data(path):
    df=pd.read_csv(path)
    X=df.values.copy()
    #
    np.random.seed(1)
    #
    np.random.shuffle(X)
    X,labels=X[:,1:-1].astype(np.float32),X[:,-1]
    #X=np.delete(X,[81,27,48,60,3],1)
    encoder=LabelEncoder()
    y=encoder.fit_transform(labels).astype(np.int32)
    scaler=StandardScaler()
    #X=scaler.fit_transform(X)
    return X,y,encoder,scaler

def load_test_data(path,scaler):
    df=pd.read_csv(path)
    X=df.values.copy()
    X,ids=X[:,1:].astype(np.float32),X[:,0].astype(str)
    #X=scaler.transform(X)
    return X,ids
def make_submission(clf,X_test,ids,encoder,name='submission.csv'):
    y_prob=clf.predict_proba(X_test)
    with open(name,'w') as f:
        f.write('id,'+','.join(encoder.classes_)+'\n')
        for id, probs in zip(ids,y_prob):
            probas=','.join([id]+map(str,probs.tolist()))
            f.write(probas+'\n')
    print "Wrote submission to file",name

np.random.seed(1)
X,y,encoder,scaler= load_train_data('data/train.csv')
X_test,ids=load_test_data('data/test.csv',scaler)
num_classes=len(encoder.classes_)
num_features=X.shape[1]

layers0=[('input',InputLayer),
        ('dropin',DropoutLayer),
        ('dense0',DenseLayer),
        ('dropout0',DropoutLayer),
        ('dense1',DenseLayer),
        ('dropout1',DropoutLayer),
        ('dense2',DenseLayer),
        ('dropout2',DropoutLayer),
        ('output',DenseLayer)]
net0=NeuralNet(layers=layers0,
        input_shape=(None,num_features),
        dropin_p=0.1,
        dense0_num_units=512,
        dense0_W=HeNormal(),
        dense0_nonlinearity=LeakyRectify(0.3),

        dropout0_p=0.2,
        dense1_num_units=800,
        dense1_nonlinearity=LeakyRectify(0.0),
        dense1_W=HeNormal(), 

        dropout1_p=0.3,
        dense2_num_units=1024, # 1024
        dense2_nonlinearity=LeakyRectify(0.0),
        dense2_W=HeNormal(),
        dropout2_p=0.4, 
        
        output_num_units=num_classes,
        output_nonlinearity=softmax,

        update=nesterov_momentum,
        update_learning_rate = theano.shared(tfloat32(0.02)),
        update_momentum = theano.shared(tfloat32(0.9)),
        eval_size=0.1,
        verbose=1,
        max_epochs=150,
        on_epoch_finished=[
            AdjustVariable('update_learning_rate',
                epochs=[50,100],rates=[2e-3,2e-4])],
        regularization_rate=1e-5
        )

skf=cv.StratifiedKFold(y,10)
fold=0
scores=[]
for train, val in skf:
    fold += 1
    X_train,X_val,y_train,y_val=X[train],X[val],y[train],y[val]

    net1=sk.base.clone(net0)
    net1.fit(X_train,y_train)
    p=net1.predict_proba(X_val)
    score= log_loss(y_val,p)
    print fold,score
    scores.append(score)
    exit(0)

print "mean score: ",np.mean(scores)

make_submission(net0,X_test,ids,encoder)


