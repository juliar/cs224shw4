from __future__ import division
import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
import numpy as np
import pylab as pl
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

from sklearn.datasets import load_svmlight_file

# Use load_svmlight_file
X, y = load_svmlight_file("../feats/train_formatted.lsvm")
X = X.toarray()
X_test, y_test = load_svmlight_file("../feats/test_formatted.lsvm")
X_test = X_test.toarray()


###########################################################################
# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features
#selector = SelectPercentile(f_classif, percentile=10)
#selector.fit(X, y)
#scores = -np.log10(selector.pvalues_)
#scores /= scores.max()

# RFE 
#svc = SVC(kernel="linear", C=1)
#rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
#rfe.fit(X, y)
#scores = rfe.ranking_

# Univariate feature selection
#print 'Index    :   score'
#sortedIdx = [i[0] for i in sorted(enumerate(scores), key=lambda x:x[1], reverse=True)]
#top = 30
#for i in range(top):
#    print str(sortedIdx[i]) + ' :   ' + str(scores[sortedIdx[i]])
#print sortedIdx[:top]


#Feature selection
#fit = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
#print fit.get_params()
#X = fit.transform(X)
#X_test = fit.transform(X_test)

clf = LinearSVC()
clf.fit(X, y) 
pred = clf.predict(X_test)
accuracy = sum(pred == y_test)/y_test.size
print 'SVM Accuracy: ' + str(accuracy)

