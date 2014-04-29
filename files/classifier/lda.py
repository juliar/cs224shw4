from __future__ import division
import numpy
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.datasets import load_svmlight_file

# load training data
train_file = open('../feats/train_formatted.lsvm', 'r')
lines = train_file.readlines()
train_file.close()

# convert training data to sklearn format
X_train_list = []
Y_train_list = []
for i, line in enumerate(lines):
    split = line.split(' ')
    split = split[:-1]
    for j, feature in enumerate(split):
        if j == 0:
            split[0] = int(split[0])
        else:
            split2 = feature.split(':')
            split[j] = float(split2[1])   
    Y_train_list.append(split[0])
    X_train_list.append(split[1:])
X_train = numpy.array(X_train_list)
Y_train = numpy.array(Y_train_list)

# load test data
test_file = open('../feats/test_formatted.lsvm', 'r')
lines = test_file.readlines()
test_file.close()

# convert test data to sklearn format
X_test_list = []
Y_test_list = []
for i, line in enumerate(lines):
    split = line.split(' ')
    split = split[:-1]
    for j, feature in enumerate(split):
        if j == 0:
            split[0] = int(split[0])
        else:
            split2 = feature.split(':')
            split[j] = float(split2[1]) 
    #print split
    Y_test_list.append(split[0])
    X_test_list.append(split[1:])
X_test = numpy.array(X_test_list)
Y_test = numpy.array(Y_test_list)
#print sum(numpy.isinf(X_train))

# Use load_svmlight_file
X_train, Y_train = load_svmlight_file("../feats/train_formatted.lsvm")
X_train = X_train.toarray()
X_test, Y_test = load_svmlight_file("../feats/test_formatted.lsvm")
X_test = X_test.toarray()
#print X_train

# LDA
clf = LDA()
clf.fit(X_train,Y_train)
qda_pred = clf.predict(X_test)
accuracy = sum(qda_pred == Y_test)/Y_test.size
print 'LDA Accuracy: ' + str(accuracy)

# QDA
clf = QDA()
clf.fit(X_train,Y_train)
qda_pred = clf.predict(X_test)
accuracy = sum(qda_pred == Y_test)/Y_test.size
print 'QDA Accuracy: ' + str(accuracy)

