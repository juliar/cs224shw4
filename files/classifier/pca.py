from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn.decomposition import PCA

X_train, y_train = load_svmlight_file("../feats/train_formatted.lsvm")
pca = PCA()
X_pca = pca.fit_transform(X_train.toarray())

np.savetxt("../feats/train_formatted.np", X_train.toarray())
np.savetxt("../feats/train_pca.np", X_pca)








