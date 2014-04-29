from __future__ import division
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectPercentile, f_classif
 
X = np.load("../feats/train_formatted.npy")
y = np.load("../feats/train_y.npy")
X_test = np.load("../feats/test_formatted.npy")
y_test = np.load("../feats/test_y.npy")

selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X,y)
clf = LogisticRegression()
clf.fit(selector.transform(X), y) 
pred = clf.predict(selector.transform(X_test))
accuracy = sum(pred == y_test)/y_test.size
print 'Logistic Regression Accuracy: ' + str(accuracy)

