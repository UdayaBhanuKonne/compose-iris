from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import pandas as pd
import scipy as sy
import os
import matplotlib.pyplot as plt
import pickle


banknotes = pd.read_csv(r'C:\Users\Family\Downloads\data_banknote_authentication.txt', names=['variance', 'skewness', 'curtosis', 'entropy', 'class'], header=0)

# convert to array
X = banknotes[['variance', 'skewness', 'curtosis', 'entropy']].values
y = banknotes[['class']].values[:,0]

# create training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

logisticRegr = LogisticRegression()
# now let us fit our model
logisticRegr.fit(X_train, y_train)
# predictions
predictions = logisticRegr.predict(X_test)
# get the score of the model
score = logisticRegr.score(X_test, y_test)

pickle.dump(logisticRegr, open('bank_note_pred.pkl', 'wb'))

print(score)
