from __future__ import division
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_svmlight_file

# Use load_svmlight_file
X_train, Y_train = load_svmlight_file("../feats/train_formatted.lsvm")
X_train = X_train.toarray()
X_test, Y_test = load_svmlight_file("../feats/test_formatted.lsvm")
X_test = X_test.toarray()
#print X_train

#X_train = np.load("../feats/train_formatted.npy")
#Y_train = np.load("../feats/train_y.npy")
#X_test = np.load("../feats/test_formatted.npy")
#Y_test = np.load("../feats/test_y.npy")

# Random Forest
clf = RandomForestClassifier(n_estimators = 10)
clf = clf.fit(X_train,Y_train)
randForest_pred = clf.predict(X_test)
accuracy = sum(randForest_pred == Y_test)/Y_test.size
print 'Random Forest Accuracy: ' + str(accuracy)

