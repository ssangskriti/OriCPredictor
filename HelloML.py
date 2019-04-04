import pandas as pn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

data=pn.read_csv("OriCFSOne.csv",header=None)
data=np.asarray(data)

y=data[:,-1]
print(y)

print(y.shape)
X=np.delete(data, 1362, axis=1)
print(X.shape)

# now we have X and y

#clf = svm.SVC(gamma=0.001,C=100)
#clf= KNeighborsClassifier()

classifiers =[svm.SVC(gamma=0.001,C=100),
              KNeighborsClassifier(),
              GaussianNB(),
              DecisionTreeClassifier(),
              AdaBoostClassifier(),
              LogisticRegression(),
              RandomForestClassifier()]
names =["svm",
        "KNN",
        "Naive Bayes",
        "decision tree",
        "Adaboost",
        "Logistic Regression",
        "Random Forest"]

#for name,clf in zip(names,classifiers):
#    clf.fit(X,y)
#    predictedY=clf.predict(X)
#    cm=confusion_matrix(predictedY,y)
#    print(name)
#    print(cm)

clf= svm.SVC(gamma=0.001,C=100)
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=10, shuffle=True)

for train_index,test_index in cv.split(X,y:
    print("In crossfold")
    X_train=X[train_index]
    X_test = X[test_index]

    Y_train=y[train_index]
    Y_test=y[test_index]

    clf.fit(X_train,Y_train)

    predicitedY=clf.predict(X_test)
    cm=confusion_matrix(predicitedY,Y_test)
    print(cm)
