from __future__ import division
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV

X = np.load("../feats/train_formatted.npy")
y = np.load("../feats/train_y.npy")
X_test = np.load("../feats/test_formatted.npy")
y_test = np.load("../feats/test_y.npy")

clf = LogisticRegression()
selector = RFECV(clf)
selector.fit(X, y)
X = selector.transform(X)
X_test = selector.transform(X_test)

scores = selector.ranking_
print 'Index    :   score'
sortedIdx = [i[0] for i in sorted(enumerate(scores), key=lambda x:x[1])]
top = 384
for i in range(top):
    print str(sortedIdx[i]) + ' :   ' + str(scores[sortedIdx[i]])


clf.fit(X, y) 
pred = clf.predict(X_test)
accuracy = sum(pred == y_test)/y_test.size
print 'Logistic Regression Accuracy: ' + str(accuracy)

