# 0.4483
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder

from lasagne.layers import DenseLayer, InputLayer,DropoutLayer
from lasagne.layers import FeaturePoolLayer
from lasagne.nonlinearities import softmax,LeakyRectify
from lasagne.updates import nesterov_momentum,momentum,adadelta
from lasagne.init import HeNormal, Orthogonal
from lasagne.objectives import categorical_crossentropy
import theano.tensor as T

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from sklearn.ensemble import BaggingClassifier
from sklearn import cross_validation as cv
from sklearn.metrics import log_loss
import sklearn as sk
import theano

import data

def obj_log_loss(y, t, eps=1e-15):
    """
    cross entropy loss, summed over classes, mean over batches
    """
    y = T.clip(y, eps, 1 - eps)
    loss = -T.sum(t * T.log(y)) / y.shape[0].astype(theano.config.floatX)
    return loss


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



def build_net(randomize=False,loss=categorical_crossentropy,
        y_tensor_type=None,dropfactor=1.0,sizefactor=1):

    layers0=[('input',InputLayer),
            ('dropin',DropoutLayer),
            ('dense0',DenseLayer),
            ('dropout0',DropoutLayer),
            ('dense1',DenseLayer),
            ('dropout1',DropoutLayer),
            ('dense2',DenseLayer),
            ('dropout2',DropoutLayer),
            ('output',DenseLayer)]
    n=[int(512*sizefactor),int(800*sizefactor),int(1024*sizefactor)]
    leak=[0.3,0.0,0.0]
    drop=[0.1,0.2,0.3,0.4]
    if randomize:
        for i in range(3):
            n[i] += np.random.randint(low=-n[i]//15,high=n[i]//15)
        """
        for i in range(4):
            drop[i] *= np.random.uniform(0.8,1.2)
        leak[0]=np.random.uniform(0.2,0.3)
        leak[1]=np.random.uniform(0,0.1)
        leak[2]=np.random.uniform(0.0,0.05)
        """
        print "net: ", n,leak,drop

    net0=NeuralNet(layers=layers0,
        input_shape=(None,num_features),
        dropin_p=drop[0]*dropfactor,
        dense0_num_units=n[0],
        dense0_W=HeNormal(),
        dense0_nonlinearity=LeakyRectify(leak[0]),

        dropout0_p=drop[1]*dropfactor,
        dense1_num_units=n[1],
        dense1_nonlinearity=LeakyRectify(leak[1]),
        dense1_W=HeNormal(), 
 

        dropout1_p=drop[2]*dropfactor,
        dense2_num_units=n[2], # 1024
        dense2_nonlinearity=LeakyRectify(leak[2]),
        dense2_W=HeNormal(),
 
        dropout2_p=drop[3]*dropfactor, 
        
        output_num_units=num_classes,
        output_nonlinearity=softmax,

        update=nesterov_momentum,
        update_learning_rate = theano.shared(tfloat32(0.02)),
        update_momentum = theano.shared(tfloat32(0.9)),
        eval_size=0.0,
        verbose=1,
        max_epochs=150,
        on_epoch_finished=[
            AdjustVariable('update_learning_rate',
                epochs=[50,100],rates=[2e-3,2e-4])],
        regularization_rate=1e-5,
        batch_iterator_train=BatchIterator(batch_size=128),
        objective_loss_function= loss,
        y_tensor_type=y_tensor_type
        )

    return net0



np.random.seed(1)
X,y,encoder,scaler= data.load_train_data('data/train.csv')
X_test,ids=data.load_test_data('data/test.csv',scaler)
num_classes=len(encoder.classes_)
num_features=X.shape[1]

skf=cv.StratifiedKFold(y,5)
train,val=next(iter(skf))

X_train,y_train=X[train],y[train]
X_val,y_val=X[val],y[val]
net1=build_net()
net1.fit(X_train,y_train)
predicted_val=np.array(net1.predict_proba(X_val))
loss1=log_loss(y[val],predicted_val)
print "loss1",loss1

test_take=int(0.7*len(X_test))
predicted_test=np.array(net1.predict_proba(X_test))[:test_take]


X_ext=np.vstack((X_train,X_test[:test_take])

enc=OneHotEncoder(sparse=False)
enc.fit(np.reshape(y,(len(y),1)))
Y_train=enc.transform(np.reshape(y_train,(len(y_train),1)))
Y_val=enc.transform(np.reshape(y_val,(len(y_val),1)))
Y_ext=np.vstack((Y_train,predicted_test))

s=np.arange(len(X_ext))
np.random.shuffle(s)
X_ext=X_ext[s]
Y_ext=Y_ext[s]

# dropfactor 0.6, sizefactor=2.0 -> 0.437 
net2=build_net(loss=obj_log_loss,y_tensor_type=T.matrix,dropfactor=0.6,sizefactor=2.0)
net2.fit(X_ext,Y_ext.astype(np.float32))

predicted_val=np.array(net2.predict_proba(X_val))
loss2=log_loss(y[val],predicted_val)
print "loss2",loss2




#make_submission(net0,X_test,ids,encoder)


