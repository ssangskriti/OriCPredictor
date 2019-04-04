

from sklearn import svm
import pandas as pn
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE

def reduceFeatures(datafile):
    dataAll=pn.read_csv(datafile)
    dataMat = pn.DataFrame.as_matrix(dataAll)

    X=dataMat[:,:-1]
    y=dataMat[:,-1]

    from sklearn import preprocessing
    scaler=preprocessing.StandardScaler().fit(X)
    #scaler=preprocessing.MinMaxScaler().fit(X)
    X_scaled=scaler.transform(X)

    estimator = svm.SVC(kernel="linear")
    cv = KFold(n_splits=10, shuffle=True,random_state=0)
    f = open(datafile + "outlog.txt", "w")
    f.write("Datafile: "+datafile+" \n")
    for num_feat in range(25,101):

        f.write(str(num_feat)+"features\n")
        selector = RFE(estimator, num_feat, step=1)
        selector = selector.fit(X_scaled, y)
        X_selected= selector.transform(X_scaled)
        np.savetxt(str(num_feat)+datafile,X_selected)
        np.savetxt(str(num_feat)+datafile + ".rank", selector.ranking_)
        np.savetxt(str(num_feat) +datafile+ ".sup", selector.support_)

        f.write("using Adaboost\n")
        from sklearn.ensemble import AdaBoostClassifier
        clf =  AdaBoostClassifier() #svm.SVC(kernel='sigmoid', gamma=0.0001, C=100, probability=True)
        score = cross_val_score(clf, X_selected, y, cv=cv)
        f.write(str(score))
        f.write("\n")
        f.write(str(score.mean()))
        f.write("\n")

        f.write("using rbf\n")
        clf=svm.SVC(kernel='rbf',gamma=0.0001,C=100,probability=True)
        score= cross_val_score(clf,X_selected,y,cv=cv)
        f.write(str(score))
        f.write("\n")
        f.write(str(score.mean()))
        f.write("\n")

        f.write("using linear\n")
        clf=svm.SVC(kernel='linear',gamma=0.01,C=1000)
        score= cross_val_score(clf,X_selected,y,cv=cv)
        f.write(str(score))
        f.write("\n")
        f.write(str(score.mean()))
        f.write("\n")
        f.flush()


reduceFeatures("OriCFSOne.csv")
#reduceFeatures("dataAll.csv")
