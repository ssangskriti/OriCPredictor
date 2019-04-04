import pandas as pn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

data=pn.read_csv("OriCFSOne.csv",header=None)
data=np.asarray(data)

y=data[:,-1]
print(y)

print(y.shape)
X=np.delete(data, 1362, axis=1)
print(X.shape)


clf = svm.SVC(gamma=0.001,C=200)
clf.fit(X,y)

from sklearn.externals import joblib
joblib.dump(clf, 'model.pkl')
