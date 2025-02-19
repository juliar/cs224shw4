from __future__ import division
import numpy as np
from sklearn.tree import DecisionTreeClassifier

#X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
#y = np.array([1, 1, 2, 2])

X = np.load("../feats/train_pca.npy")
y = np.load("../feats/train_y.npy")
X_test = np.load("../feats/test_pca.npy")
y_test = np.load("../feats/test_y.npy")

clf = DecisionTreeClassifier()
clf.fit(X, y) 
pred = clf.predict(X_test)
accuracy = sum(pred == y_test)/y_test.size
print 'Decision Tree Accuracy: ' + str(accuracy)

