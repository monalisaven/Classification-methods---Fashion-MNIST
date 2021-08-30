# https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python
# https://www.udemy.com/data-science-supervised-machine-learning-in-python
# This is an example of a Naive Bayes classifier on MNIST data.
#https://github.com/lazyprogrammer/machine_learning_examples/blob/master/supervised_class/nb.py
from __future__ import print_function, division
from future.utils import iteritems
from tensorflow import keras
from sklearn import metrics
import numpy as np
from scipy.stats import multivariate_normal as mvn

class NaiveBayes(object):
    def fit(self, X, Y, smoothing=1e-2):
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for l in labels:
            current_x = X[Y == l]
            self.gaussians[l] = {
                'mean': current_x.mean(axis=0),
                'var': current_x.var(axis=0) + smoothing,
            }
            self.priors[l] = float(len(Y[Y == l])) / len(Y)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in iteritems(self.gaussians):
            mean, variance = g['mean'], g['var']
            P[:, c] = mvn.logpdf(X, mean=mean, cov=variance) + np.log(self.priors[c])
        return np.argmax(P, axis=1)


if __name__ == '__main__':
    fashion_mnist = keras.datasets.fashion_mnist
    (trImages, trLabels), (tImages, tLabels) = fashion_mnist.load_data()

    # normalize
    trImages = trImages.astype('float32')
    tImages = tImages.astype('float32')
    trImages = trImages / 255
    tImages = tImages / 255

    trImages = trImages.reshape(trImages.shape[0], trImages.shape[1] * trImages.shape[2])
    tImages = tImages.reshape(tImages.shape[0], tImages.shape[1] * tImages.shape[2])

    model = NaiveBayes()
    model.fit(trImages, trLabels)
    predictions = model.predict(tImages)
    print("accuracy = %.2f%% " % (metrics.accuracy_score(tLabels, predictions) * 100))
    print("f1_score = %.2f%%" % (metrics.f1_score(tLabels, predictions, average="weighted") * 100))