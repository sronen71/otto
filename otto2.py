# 0.4483
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from lasagne.layers import DenseLayer, InputLayer,DropoutLayer
from lasagne.layers import FeaturePoolLayer
from lasagne.nonlinearities import softmax,LeakyRectify
from lasagne.updates import nesterov_momentum,momentum,adadelta
from lasagne.init import HeNormal, Orthogonal


from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from sklearn.ensemble import BaggingClassifier
from sklearn import cross_validation as cv
from sklearn.metrics import log_loss
import sklearn as sk
import theano

import data

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



np.random.seed(1)
X,y,encoder,scaler= data.load_train_data('data/train.csv')
X_test,ids=data.load_test_data('data/test.csv',scaler)
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
        #dense0_W=Orthogonal('relu'),
        dense0_nonlinearity=LeakyRectify(0.3),

        dropout0_p=0.2,
        dense1_num_units=800,
        dense1_nonlinearity=LeakyRectify(0.0),
        dense1_W=HeNormal(), 
        #dense1_W=Orthogonal('relu'),
 

        dropout1_p=0.3,
        dense2_num_units=1024, # 1024
        dense2_nonlinearity=LeakyRectify(0.0),
        dense2_W=HeNormal(),
        #dense2_W=Orthogonal('relu'),
 
        dropout2_p=0.4, 
        
        output_num_units=num_classes,
        output_nonlinearity=softmax,
        #output_W=Orthogonal(),

        #update=nesterov_momentum,
        update=momentum,
        update_learning_rate = theano.shared(tfloat32(0.02)),
        update_momentum = theano.shared(tfloat32(0.9)),
        eval_size=0.0,
        verbose=1,
        max_epochs=150,
        on_epoch_finished=[
            AdjustVariable('update_learning_rate',
                epochs=[50,100],rates=[2e-3,2e-4])],
        regularization_rate=1e-5,
        batch_iterator_train=BatchIterator(batch_size=128)
        )



skf=cv.StratifiedKFold(y,10)


fold=0
scores=[]
p=None
for train, val in skf:
    fold += 1
    X_train,X_val,y_train,y_val=X[train],X[val],y[train],y[val]

    skf_inner=cv.StratifiedShuffleSplit(y_train,5,random_state=0)
    print len(skf_inner)
    for bag,ignore in skf_inner:
        X_bag,y_bag=X_train[bag],y_train[bag]
        net1=sk.base.clone(net0)    
        net1.fit(X_bag,y_bag)
        p1=np.array(net1.predict_proba(X_val))
        if p is None:
            p= p1
        else:
            p += p1
    p=p/len(skf_inner)
    """
    net1=sk.base.clone(net0)    
    net1.fit(X_train,y_train)
    p=np.array(net1.predict_proba(X_val))
    """
 
    score= log_loss(y_val,p)
    print fold,score
    scores.append(score)
    exit(0)

print "mean score: ",np.mean(scores)

make_submission(net0,X_test,ids,encoder)


