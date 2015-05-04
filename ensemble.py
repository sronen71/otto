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
import cPickle as pickle


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
        verbose=0,
        max_epochs=150, #150
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

scores=np.zeros((y.shape[0],9))
scores_test=np.zeros((X_test.shape[0],9))
folds =5 
for num in range(1,15):
    skf=cv.StratifiedKFold(y,n_folds=folds,shuffle=True,random_state=num)
    for k,(bag,val) in enumerate(skf):
        X_bag,y_bag=X[bag],y[bag]
        X_val,y_val=X[val],y[val]
        filename='nn-models/nn--semi--iter%s--cv%s.p' % (str(num),str(k))
        net1=pickle.load(open(filename,'rb'))
        predicted= np.array(net1.predict_proba(X_val))
        model_loss=log_loss(y[val],predicted)
        print "iter ",num, "cv",k,"model loss",model_loss        
        scores[val,:] += predicted
        test_take=int(.5*len(X_test))
        predict_test = np.array(net1.predict_proba(X_test[test_take:]))
        scores_test[test_take:,:] += predict_test
    loss=log_loss(y,scores/num)
    print "*iter ",num, "ensemble loss ",loss
    scores_test /= num

data.write_submission(X_test,ids,scores_test,encoder,name='predictions/semi--nn--bag.csv')


