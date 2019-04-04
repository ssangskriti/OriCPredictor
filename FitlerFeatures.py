

# Avoiding warning
import warnings

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

def warn(*args, **kwargs): pass


warnings.warn = warn
# ________________________________


# Essential Library
import pandas as pd
import numpy as np
# ________________________________


# scikit-learn for classifiers :
from sklearn.linear_model import LogisticRegression

# scikit-learn for performance measures :
from sklearn.metrics import accuracy_score, \
    confusion_matrix, \
    roc_auc_score, \
    average_precision_score, \
    f1_score, \
    matthews_corrcoef

# ////////////////////////////////////////////////////////////////////////////////////////////


# Step 01 : Load the dataset :
pima = 'OriCFSOne.csv'
D = pd.read_csv(pima)
D = D.drop_duplicates()  # Return : Unique records/instances.
# __________________________________________________________________________________



# Step 02 : Divided: features(X) and classes(y) from dataset(D).
X = D.iloc[:, :-1]
X = pd.get_dummies(X).values  # Convert categorical features into OneHotEncoder.
y = D.iloc[:, -1].values
# __________________________________________________________________________________




# Step 03 : Handle the missing values
from sklearn.preprocessing import Imputer

X[:, 0:X.shape[1]] = Imputer(strategy='mean').fit_transform(X[:, 0:X.shape[1]])
# We can use more stategy = 'median' or stategy = 'most_frequent'
# __________________________________________________________________________________



# Step 04 : Scaling the features
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#X = StandardScaler().fit_transform(X)

# Step 05 : Encoding y
#from sklearn.preprocessing import LabelEncoder

#y = LabelEncoder().fit_transform(y)
# _________________________________________________________________________________



from sklearn.utils import shuffle

X, y = shuffle(X, y)  # Avoiding bias
# __________________________________________________________________________________



# Step 06 : Spliting with 10-FCV :
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=10, shuffle=True)



from sklearn.model_selection import GridSearchCV
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-4], 'C': [1,1000]}]
score='precision'
#clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=10, scoring='%s_macro' % score)
#clf.fit(X, y)

print("Best parameters set found on development set:")
print()
#print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
#means = clf.cv_results_['mean_test_score']
#stds = clf.cv_results_['std_test_score']
#for mean, std, params in zip(means, stds, clf.cv_results_['params']):
 #  print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

# try feature selection

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

for ktop in range(10,1361):
    Accuray = []
    auROC = []
    avePrecision = []
    F1_Score = []
    AUC = []
    MCC = []
    CM = np.array([
        [0, 0],
        [0, 0],
    ], dtype=int)
    i=0
    print("value of k=" + str(ktop))

    X_new = SelectKBest(chi2, ktop).fit_transform(X, y)

    for train_index, test_index in cv.split(X_new, y):
        #print("In cross fold")
        #print(i)
        i=i+1
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]


        model =svm.SVC(kernel='rbf', gamma=0.0009, C=900, probability=True)
        # svm.SVC(kernel='rbf', gamma=0.0009, C=400, probability=True) #LogisticRegression() #GaussianNB()#RandomForestClassifier()#AdaBoostClassifier()#svm.SVC(kernel='linear', gamma=0.01, C=1000, probability=True)
        model.fit(X_train, y_train)

        yHat = model.predict(X_test)  # predicted labels

        y_proba = model.predict_proba(X_test)[:, 1]

        Accuray.append(accuracy_score(y_pred=yHat, y_true=y_test))
        auROC.append(roc_auc_score(y_test, y_proba))
        avePrecision.append(average_precision_score(y_test, y_proba))  # auPR
        F1_Score.append(f1_score(y_true=y_test, y_pred=yHat))
        MCC.append(matthews_corrcoef(y_true=y_test, y_pred=yHat))

        CM += confusion_matrix(y_pred=yHat, y_true=y_test)

    print('Accuracy: {:.4f} ({:0.2f}%)'.format(np.mean(Accuray), np.mean(Accuray) * 100.0))
    print('auROC: {0:.4f}'.format(np.mean(auROC)))
    print('auPR: {0:.4f}'.format(np.mean(avePrecision)))  # average_Precision
    print('F1-score: {0:.4f}'.format(np.mean(F1_Score)))
    print('MCC: {0:.4f}'.format(np.mean(MCC)))

    TN, FP, FN, TP = CM.ravel()
    print('Sensitivity (+): {0:.4f}'.format((TP) / (TP + FN)))
    print('Specificity (-): {0:.4f}'.format((TN) / (TN + FP)))
    print('Confusion Matrix:')
    print(CM)
    print('___________________________')





