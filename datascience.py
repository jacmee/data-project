import more_itertools
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.ticker import NullFormatter
import pandas as pd
#import matplotlib.ticker as ticker
from sklearn import preprocessing
import wget
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import os
import json

def main():
    url1 = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv'
    url2 = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv'

    filename1 = wget.download(url1)
    filename2 = wget.download(url2)
    df = pd.read_csv(filename1)

    df['due_date'] = pd.to_datetime(df['due_date'])
    df['effective_date'] = pd.to_datetime(df['effective_date'])
    df['dayofweek'] = df['effective_date'].dt.dayofweek
    df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
    df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
    df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
    df.groupby(['education'])['loan_status'].value_counts(normalize=True)


    Feature = df[['Principal','terms','age','Gender','weekend']]
    Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
    Feature.drop(['Master or Above'], axis = 1,inplace=True)
    Feature.head()

    X = Feature
    X[0:5]

    y = df['loan_status'].values
    y[0:5]

    X= preprocessing.StandardScaler().fit(X).transform(X)
    X[0:5]


    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4 )
    print ('Train set:', X_train.shape,  y_train.shape)
    print ('Test set:', X_test.shape,  y_test.shape)


    Ks=15
    mean_acc=np.zeros((Ks-1))
    std_acc=np.zeros((Ks-1))
    ConfustionMx=[];
    for n in range(1,Ks):

        model = KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
        yhat = model.predict(X_test)


        mean_acc[n-1]=np.mean(yhat==y_test);

        std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
    mean_acc

    model = KNeighborsClassifier(n_neighbors = 7).fit(X_train,y_train)
    yhat = model.predict(X_test)
    yhat[0:5]


    loanTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
    loanTree.fit(X_train,y_train)
    loanTree

    yhat = loanTree.predict(X_test)
    yhat

    SVM = svm.SVC()
    SVM.fit(X_train, y_train)

    yhat = SVM.predict(X_test)
    yhat

    LR = LogisticRegression(C=0.01).fit(X_train,y_train)
    LR

    yhat = LR.predict(X_test)
    yhat

    test_df = pd.read_csv(filename2)
    test_df.head()

    test_df['due_date'] = pd.to_datetime(test_df['due_date'])
    test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
    test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
    test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
    test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
    test_Feature = test_df[['Principal','terms','age','Gender','weekend']]
    test_Feature = pd.concat([test_Feature,pd.get_dummies(test_df['education'])], axis=1)
    test_Feature.drop(['Master or Above'], axis = 1,inplace=True)
    test_X = preprocessing.StandardScaler().fit(test_Feature).transform(test_Feature)
    test_X[0:5]

    test_y = test_df['loan_status'].values
    test_y[0:5]

    knn_yhat = model.predict(test_X)
    #print("KNN Jaccard index: %.2f" % jaccard_similarity_score(test_y, knn_yhat))
    print("KNN F1-score: %.2f" % f1_score(test_y, knn_yhat, average='weighted') )

    Tree_yhat = loanTree.predict(test_X)
    #print("DT Jaccard index: %.2f" % jaccard_similarity_score(test_y, Tree_yhat))
    print("DT F1-score: %.2f" % f1_score(test_y, Tree_yhat, average='weighted') )

    SVM_yhat = SVM.predict(test_X)
    #print("SVM Jaccard index: %.2f" % jaccard_similarity_score(test_y, SVM_yhat))
    print("SVM F1-score: %.2f" % f1_score(test_y, SVM_yhat, average='weighted') )

    LR_yhat = LR.predict(test_X)
    LR_yhat_prob = LR.predict_proba(test_X)
    #print("LR Jaccard index: %.2f" % jaccard_similarity_score(test_y, LR_yhat))
    print("LR F1-score: %.2f" % f1_score(test_y, LR_yhat, average='weighted') )
    print("LR LogLoss: %.2f" % log_loss(test_y, LR_yhat_prob))

    loan_status = df['loan_status'].tolist()
    Gender = df['Gender'].tolist()
    Principal = df['Principal'].tolist()
    Age = df['age'].tolist()

    graphic1 = {
    "loan_status": loan_status,
    "gender": Gender,
    "Principal": Principal
    }

    graphic2 = {
    "loan_status": loan_status,
    "gender": Gender,
    "age": Age
    }

    res = {
    "graphic1": graphic1,
    "graphic2": graphic2
    }

    os.unlink(filename1)
    os.unlink(filename2)
    return json.dumps(res)
