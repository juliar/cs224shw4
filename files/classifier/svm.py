from __future__ import division
import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
import numpy as np
import pylab as pl
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectPercentile, f_classif

#X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
#y = np.array([1, 1, 2, 2])

X = np.load("../feats/train_pca.npy")
y = np.load("../feats/train_y.npy")
X_test = np.load("../feats/test_pca.npy")
y_test = np.load("../feats/test_y.npy")

###########################################################################


# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X, y)
scores = -np.log10(selector.pvalues_)
#scores /= scores.max()
print scores
print 'Index    :   score'
sortedIdx = [i[0] for i in sorted(enumerate(scores), key=lambda x:x[1], reverse=True)]
top = 30
for i in range(top):
    print str(sortedIdx[i]) + ' :   ' + str(scores[sortedIdx[i]])
#print sortedIdx[:top]


print 'plots'

pl.figure(1)
pl.clf()
X_indices = np.arange(X.shape[-1])
pl.bar(X_indices[:12], scores[:12], width=.6, label=r'Univariate score ($-Log(p_{value})$)', color='g')
print len(scores)

pl.title("RMS Energy")
pl.xlabel('Feature number')
pl.yticks(())
pl.axis('tight')
pl.legend(loc='upper right')
pl.show()

pl.figure(2)
pl.clf()
X_indices = np.arange(X.shape[-1])
pl.bar(X_indices[:], scores[:], width=.2, label=r'Univariate score ($-Log(p_{value})$)', color='g')
print len(scores)

pl.title("All features")
pl.xlabel('Feature number')
pl.yticks(())
pl.axis('tight')
pl.legend(loc='upper right')
pl.show()

#Feature selection
fit = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
print fit.get_params()
X = fit.transform(X)
X_test = fit.transform(X_test)

clf = LinearSVC()
clf.fit(X, y) 
pred = clf.predict(X_test)
accuracy = sum(pred == y_test)/y_test.size
print 'SVM Accuracy: ' + str(accuracy)

