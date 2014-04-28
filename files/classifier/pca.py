from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn.decomposition import PCA
import sys

X_train, y_train = load_svmlight_file("../feats/train_formatted.lsvm")
X_train = X_train.toarray()

X_test, y_test = load_svmlight_file("../feats/test_formatted.lsvm")
X_test = X_test.toarray()

pca = PCA(n_components=int(sys.argv[1]))
pca.fit(X_train)
train_pca = pca.transform(X_train)
test_pca = pca.transform(X_test)


np.save("../feats/train_formatted", X_train)
np.save("../feats/train_pca", train_pca)
np.save("../feats/train_y", y_train)
np.save("../feats/test_formatted", X_test)
np.save("../feats/test_pca", test_pca)
np.save("../feats/test_y", y_test)






