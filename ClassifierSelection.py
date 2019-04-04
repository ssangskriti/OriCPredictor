from sklearn import svm
import pandas as pn
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
#from core import *

def selectClassifier(datafile):
    dataAll=pn.read_csv(datafile,header=None)

    # Step 02 : Divided: features(X) and classes(y) from dataset(D).
    X = dataAll.iloc[:, :-1]
    X = pn.get_dummies(X).values  # Convert categorical features into OneHotEncoder.
    y = dataAll.iloc[:, -1].values

    #from sklearn.preprocessing import StandardScaler, MinMaxScaler

    #X = StandardScaler().fit_transform(X)

    print("data scaled at this point, now start with all the cross validation with different classifiers")

    # 0 : SVM linear
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    X_new = X
    clf = svm.SVC(kernel='linear', gamma=0.01, C=1000, probability=True)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    ###################
    cm = [[0, 0],
          [0, 0]]

    auprList = []
    mccList = []
    auROCList = []
    ###################

    lw = 2

    i = 0

    for (train, test) in cv.split(X_new, y):
        print(i)
        probas_ = clf.fit(X_new[train], y[train]).predict_proba(X_new[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)

        ######
        y_pred = clf.fit(X_new[train], y[train]).predict(X_new[test])
        auprList.append(recall_score(y[test], y_pred, average='binary'))
        mccList.append(matthews_corrcoef(y[test], y_pred))
        cm += confusion_matrix(y[test], y_pred)
        ##################
        # plt.plot(fpr, tpr, lw=lw,label='ROC fold %d (area = %0.2f)' % (i, roc_auc))


        i += 1
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',label='Random')




    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='b', linestyle='-.',
             label='SVM linear kernel auROC= %0.2f' % mean_auc, lw=lw)

    print("SVM linear")
    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[1, 0] + cm[0, 1])
    print("Accuracy:", accuracy)
    sensitivity = (cm[0, 0]) / (cm[0, 0] + cm[0, 1])
    print("sensitivity:", sensitivity)
    specificity = (cm[1, 1]) / (cm[1, 0] + cm[1, 1])
    print("specificity:", specificity)
    print("auPR:", np.mean(auprList))
    print("MCC:", np.mean(mccList))
    print("auROC:", mean_auc)

    # 1 : SVM rbf
    clf = svm.SVC(kernel='rbf', gamma=0.001, C=200, probability=True)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    ###################
    cm = [[0, 0],
          [0, 0]]

    auprList = []
    mccList = []
    auROCList = []
    ###################

    lw = 2

    i = 0
    for (train, test) in cv.split(X_new, y):
        print(i)
        probas_ = clf.fit(X_new[train], y[train]).predict_proba(X_new[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)

        ######
        y_pred = clf.fit(X_new[train], y[train]).predict(X_new[test])
        auprList.append(recall_score(y[test], y_pred, average='binary'))
        mccList.append(matthews_corrcoef(y[test], y_pred))
        cm += confusion_matrix(y[test], y_pred)
        ##################
        # plt.plot(fpr, tpr, lw=lw,label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        i += 1
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',label='Random')




    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='r', linestyle='-',
             label='SVM rbf kernel auROC= %0.2f' % mean_auc, lw=lw)

    print("SVM rbf")
    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[1, 0] + cm[0, 1])
    print("Accuracy:", accuracy)
    sensitivity = (cm[0, 0]) / (cm[0, 0] + cm[0, 1])
    print("sensitivity:", sensitivity)
    specificity = (cm[1, 1]) / (cm[1, 0] + cm[1, 1])
    print("specificity:", specificity)
    print("auPR:", np.mean(auprList))
    print("MCC:", np.mean(mccList))
    print("auROC:", mean_auc)

    # 3 : AdaBoost
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier()#svm.SVC(kernel='sigmoid', gamma=0.01, C=1000, probability=True)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    ###################
    cm = [[0, 0],
          [0, 0]]

    auprList = []
    mccList = []
    auROCList = []
    ###################

    lw = 2

    i = 0
    for (train, test) in cv.split(X_new, y):
        probas_ = clf.fit(X_new[train], y[train]).predict_proba(X_new[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)

        ######
        y_pred = clf.fit(X_new[train], y[train]).predict(X_new[test])
        auprList.append(recall_score(y[test], y_pred, average='binary'))
        mccList.append(matthews_corrcoef(y[test], y_pred))
        cm += confusion_matrix(y[test], y_pred)
        ##################
        # plt.plot(fpr, tpr, lw=lw,label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        i += 1
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',label='Random')




    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='-',
             label='AdaBoost auROC= %0.2f' % mean_auc, lw=lw)

    print("AdaBoost")
    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[1, 0] + cm[0, 1])
    print("Accuracy:", accuracy)
    sensitivity = (cm[0, 0]) / (cm[0, 0] + cm[0, 1])
    print("sensitivity:", sensitivity)
    specificity = (cm[1, 1]) / (cm[1, 0] + cm[1, 1])
    print("specificity:", specificity)
    print("auPR:", np.mean(auprList))
    print("MCC:", np.mean(mccList))
    print("auROC:", mean_auc)

    # 4 : Random Forest
    #X_new = np.loadtxt(csvfile)
    #clf = svm.SVC(kernel='linear', gamma=0.01, C=1000, probability=True)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    ###################
    cm = [[0, 0],
          [0, 0]]

    auprList = []
    mccList = []
    auROCList = []
    ###################

    lw = 2

    i = 0
    for (train, test) in cv.split(X_new, y):
        probas_ = clf.fit(X_new[train], y[train]).predict_proba(X_new[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)

        ######
        y_pred = clf.fit(X_new[train], y[train]).predict(X_new[test])
        auprList.append(recall_score(y[test], y_pred, average='binary'))
        mccList.append(matthews_corrcoef(y[test], y_pred))
        cm += confusion_matrix(y[test], y_pred)
        ##################
        # plt.plot(fpr, tpr, lw=lw,label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        i += 1
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',label='Random')




    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='cyan', linestyle='-',
             label='Random Forest auROC= %0.2f' % mean_auc, lw=lw)

    print("Random Forest")
    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[1, 0] + cm[0, 1])
    print("Accuracy:", accuracy)
    sensitivity = (cm[0, 0]) / (cm[0, 0] + cm[0, 1])
    print("sensitivity:", sensitivity)
    specificity = (cm[1, 1]) / (cm[1, 0] + cm[1, 1])
    print("specificity:", specificity)
    print("auPR:", np.mean(auprList))
    print("MCC:", np.mean(mccList))
    print("auROC:", mean_auc)

    # 5 : Naive Bayes
    #X_new = np.loadtxt("94185data.csv")
    from sklearn.naive_bayes import GaussianNB
    #clf = svm.SVC(kernel='linear', gamma=0.01, C=1000, probability=True)
    clf = GaussianNB()

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    ###################
    cm = [[0, 0],
          [0, 0]]

    auprList = []
    mccList = []
    auROCList = []
    ###################

    lw = 2

    i = 0
    for (train, test) in cv.split(X_new, y):
        probas_ = clf.fit(X_new[train], y[train]).predict_proba(X_new[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)

        ######
        y_pred = clf.fit(X_new[train], y[train]).predict(X_new[test])
        auprList.append(recall_score(y[test], y_pred, average='binary'))
        mccList.append(matthews_corrcoef(y[test], y_pred))
        cm += confusion_matrix(y[test], y_pred)
        ##################
        # plt.plot(fpr, tpr, lw=lw,label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        i += 1
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',label='Random')




    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='gold', linestyle='-',
             label='Naive Bayesian auROC= %0.2f' % mean_auc, lw=lw)

    print("Naive bayesian")
    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[1, 0] + cm[0, 1])
    print("Accuracy:", accuracy)
    sensitivity = (cm[0, 0]) / (cm[0, 0] + cm[0, 1])
    print("sensitivity:", sensitivity)
    specificity = (cm[1, 1]) / (cm[1, 0] + cm[1, 1])
    print("specificity:", specificity)
    print("auPR:", np.mean(auprList))
    print("MCC:", np.mean(mccList))
    print("auROC:", mean_auc)

    # 5 : Logistic Regression
    from sklearn.linear_model import LogisticRegression
    #clf = svm.SVC(kernel='linear', gamma=0.01, C=1000, probability=True)
    clf = LogisticRegression()

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    ###################
    cm = [[0, 0],
          [0, 0]]

    auprList = []
    mccList = []
    auROCList = []
    ###################

    lw = 2

    i = 0
    for (train, test) in cv.split(X_new, y):
        probas_ = clf.fit(X_new[train], y[train]).predict_proba(X_new[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)

        ######
        y_pred = clf.fit(X_new[train], y[train]).predict(X_new[test])
        auprList.append(recall_score(y[test], y_pred, average='binary'))
        mccList.append(matthews_corrcoef(y[test], y_pred))
        cm += confusion_matrix(y[test], y_pred)
        ##################
        # plt.plot(fpr, tpr, lw=lw,label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        i += 1
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',label='Random')




    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='magenta', linestyle='-',
             label='Logistic Regression auROC= %0.2f' % mean_auc, lw=lw)

    print("Logistic Regression")
    accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + cm[1, 0] + cm[0, 1])
    print("Accuracy:", accuracy)
    sensitivity = (cm[0, 0]) / (cm[0, 0] + cm[0, 1])
    print("sensitivity:", sensitivity)
    specificity = (cm[1, 1]) / (cm[1, 0] + cm[1, 1])
    print("specificity:", specificity)
    print("auPR:", np.mean(auprList))
    print("MCC:", np.mean(mccList))
    print("auROC:", mean_auc)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig("aucCLS185.png")


#reduceFeatures("dataSub.csv")
selectClassifier("OriCFSOne.csv")
