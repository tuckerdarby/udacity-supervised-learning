import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv("student-data.csv")
df = df.replace('no', 0).replace('yes', 1)


def hot_encode(data, name):
    df_new = pd.get_dummies(data[name])
    for c in range(len(df_new.columns)):
        data[df_new.columns[c]] = df_new[df_new.columns[c]]
    data = data.drop(name, 1)
    return data

fix_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']
for i in range(len(fix_columns)):
    df = hot_encode(df, fix_columns[i])

passed_df = df['passed']
student_df = df.drop('passed', 1)

X, x, Y, y = train_test_split(student_df, passed_df, random_state=1)

clf = SVC(C=2.5, kernel='rbf')
#clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
#clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X, Y)
p = clf.predict(x)
print accuracy_score(y, p)
#imps = clf.feature_importances_
for i in range(0): #range(len(imps)):
    print student_df.columns[i], imps[i]

