from __future__ import division
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_svmlight_file

# Use load_svmlight_file
X, y = load_svmlight_file("../feats/train_formatted.lsvm")
X = X.toarray()
X_test, y_test = load_svmlight_file("../feats/test_formatted.lsvm")
X_test = X_test.toarray()

selector = SelectPercentile(f_classif, percentile=45)
selector.fit(X,y)
clf = LogisticRegression()
clf.fit(selector.transform(X), y) 
pred = clf.predict(selector.transform(X_test))
#clf.fit(X, y) 
#pred = clf.predict(X_test)
accuracy = sum(pred == y_test)/y_test.size
print 'Univariate SVM Accuracy: ' + str(accuracy)

