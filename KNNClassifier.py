

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
D = pd.read_csv(pima, header=None)
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


print(X.shape[1])
for i in range(X.shape[1]):
    print("X"+str(i),end=",")

# Step 06 : Spliting with 10-FCV :
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=10, shuffle=True)


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

from sklearn.model_selection import GridSearchCV
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.01,0.005,1e-3, 1e-4],
                     'C': [1, 10, 50, 100, 150,200,250,1000]}]
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
numberofneighbors=25

for train_index, test_index in cv.split(X, y):
    print("In cross fold")
    print(i)
    i=i+1
    X_train = X[train_index]
    X_test = X[test_index]

    y_train = y[train_index]
    y_test = y[test_index]


    # here before fitting of the model lets transform the feature space
    from sklearn.neighbors import NearestNeighbors

    AllNeighbors = NearestNeighbors(n_neighbors=numberofneighbors,algorithm='ball_tree').fit(X_train)
    distances, indices = AllNeighbors.kneighbors(X_train)
    indices=np.asarray(indices)
    print(indices.shape)
    # now we have top neighbors for all the instances
    # from this array, lets calculate the propensity matriix
    FreqPos = np.zeros(2 * numberofneighbors).reshape(2, numberofneighbors)
    FreqNeg = np.zeros(2 * numberofneighbors).reshape(2, numberofneighbors)

    counter=0
    for ti in train_index:
        if y[ti]==1:
            # it is a positive sample
            arraypos=0;
            for n in indices[counter]:
                if y[n]==1: # positive neighbor
                    FreqPos[0,arraypos]=FreqPos[0,arraypos]+1
                else:
                    FreqPos[1,arraypos] = FreqPos[1,arraypos] +1
                arraypos=arraypos+1
        else:
            # it is a negative sample
            arraypos = 0;
            for n in indices[counter]:
                if y[n] == 1:  # positive neighbor
                    FreqNeg[0, arraypos] = FreqNeg[0, arraypos] + 1
                else:
                    FreqNeg[1, arraypos] = FreqNeg[1, arraypos] + 1
                arraypos=arraypos+1
        counter=counter+1
    #print(FreqPos)
    #print(FreqNeg)
    #print(AllNeighbors.kneighbors_graph().toarray())

    # now that the frequency matrix is ready we convert the whole X_train dataset
    t_X_train=np.zeros(counter*numberofneighbors).reshape(counter,numberofneighbors)
    nCounter=0
    for ti in train_index:
        arraypos = 0;
        for n in indices[nCounter]:
            if y[n]==1: # neighbor is positive
                t_X_train[nCounter,arraypos]=1#FreqPos[0,arraypos]-FreqNeg[0,arraypos]
            else:
                t_X_train[nCounter, arraypos] = -1#FreqPos[1, arraypos] - FreqNeg[1, arraypos]
            arraypos=arraypos+1
        nCounter=nCounter+1
    #print(t_X_train)
    # now tha train data is ready

    # lets prepare the test data now
    distances, indices = AllNeighbors.kneighbors(X_test)
    indices = np.asarray(indices)
    print(indices)
    print(indices.shape[0])

    t_X_test = np.zeros(indices.shape[0]* numberofneighbors).reshape(indices.shape[0], numberofneighbors)
    nCounter = 0
    for ti in test_index:
        arraypos = 0;
        for n in indices[nCounter]:
            if y[n] == 1:  # neighbor is positive
                t_X_test[nCounter, arraypos] = FreqPos[0, arraypos] - FreqNeg[0, arraypos]
            else:
                t_X_test[nCounter, arraypos] = FreqPos[1, arraypos] - FreqNeg[1, arraypos]
            arraypos = arraypos + 1
        nCounter = nCounter + 1
    print(t_X_test)
    print(t_X_train)
    print(t_X_test.shape)
    print(nCounter)



    model = AdaBoostClassifier()#RandomForestClassifier()#svm.SVC(kernel='linear', gamma=0.01, C=1000, probability=True)#LogisticRegression()#GaussianNB()#RandomForestClassifier()#AdaBoostClassifier()#
    model.fit(t_X_train, y_train)

    yHat = model.predict(t_X_test)  # predicted labels

    y_proba = model.predict_proba(t_X_test)[:, 1]

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





