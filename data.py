import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def load_train_data(path):
    df=pd.read_csv(path)
    X=df.values.copy()
    np.random.seed(1)
    np.random.shuffle(X)
    X,labels=X[:,1:-1].astype(np.float32),X[:,-1]
    encoder=LabelEncoder()
    y=encoder.fit_transform(labels).astype(np.int32)
    scaler=StandardScaler()
    scaler.fit(X)
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

