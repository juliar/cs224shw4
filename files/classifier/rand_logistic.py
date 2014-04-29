from __future__ import division
import numpy as np
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.linear_model import LogisticRegression

X = np.load("../feats/train_formatted.npy")
y = np.load("../feats/train_y.npy")
X_test = np.load("../feats/test_formatted.npy")
y_test = np.load("../feats/test_y.npy")

clf = RandomizedLogisticRegression()
clf.fit(X, y) 
scores = clf.scores_
print 'Index    :   score'
sortedIdx = [i[0] for i in sorted(enumerate(scores), key=lambda x:x[1], reverse=True)]
top = 30
for i in range(top):
    print str(sortedIdx[i]) + ' :   ' + str(scores[sortedIdx[i]])

lr = LogisticRegression()
lr.fit(clf.transform(X), y)
pred = lr.predict(clf.transform(X_test))
accuracy = sum(pred == y_test)/y_test.size
print 'Logistic Regression Accuracy: ' + str(accuracy)

