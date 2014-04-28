from __future__ import division
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.load("../feats/train_formatted.npy")
y = np.load("../feats/train_y.npy")
X_test = np.load("../feats/test_formatted.npy")
y_test = np.load("../feats/test_y.npy")

clf = LogisticRegression(C=0.2)
clf.fit(X, y) 
pred = clf.predict(X_test)
accuracy = sum(pred == y_test)/y_test.size
print 'Logistic Regression Accuracy: ' + str(accuracy)

