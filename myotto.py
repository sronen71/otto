import sklearn as sk
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing as pre
import csv
import numpy as np
from matplotlib import pyplot as plt

data=[]
ids=[]
labels=[]
with open('train.csv','rb') as csvfile:
    reader=csv.reader(csvfile)
    reader.next()
    for row in reader:
        ids.append(int(row[0]))
        data.append([float(x) for x in row[1:94]])
        labels.append(int(row[94].split('_')[1]))

data=np.array(data)
#sc= pre.StandardScaler()
#sc.fit(data)
#data=sc.transform(data)
#h=np.bincount(labels)
print h
#plt.bar(range(1,10),h[1:])
#plt.show()


"""
for k in range(data.shape[1]):
    x=data[:,k]
    print k,np.amin(x),np.amax(x),np.mean(x),np.std(x)
    #plt.hist(x,bins=20)
    #plt.show()
"""

Xtrain,Xtest,y_train,y_test=train_test_split(data,labels)





