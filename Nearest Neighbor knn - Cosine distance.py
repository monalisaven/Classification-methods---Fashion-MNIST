#https://notebook.community/YorkUIRLab/eosdb/word-movers-distance-in-python
from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

fashion_mnist = keras.datasets.fashion_mnist
(trImages, trLabels), (tImages, tLabels) = fashion_mnist.load_data()

#normalize
trImages = trImages.astype('float32')
tImages = tImages.astype('float32')
trImages=trImages/255
tImages=tImages/255

trImages = trImages.reshape(trImages.shape[0], trImages.shape[1] * trImages.shape[2])
tImages = tImages.reshape(tImages.shape[0], tImages.shape[1] * tImages.shape[2])

kVals = [1, 5, 10]
accuracies = []

# loop over various values of `k` for the k-Nearest Neighbor classifier
for k in kVals:
    # train the k-Nearest Neighbor classifier with the current value of `k`
    model = KNeighborsClassifier(metric='cosine', algorithm='brute')
    model.fit(trImages, trLabels)

    predictions = model.predict(tImages)
    # evaluate the model and update the accuracies list
    score=model.score(tImages, tLabels)
    print("k=%d, accuracy=%.2f%%" % (k, metrics.accuracy_score(tLabels, predictions) * 100))
    print("k=%d, f1_score=%.2f%%" % (k, metrics.f1_score(tLabels, predictions, average= "weighted") * 100))
    accuracies.append(score)

    # find the value of k that has the largest accuracy
    i = int(np.argmax(accuracies))
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                           accuracies[i] * 100))
    # show a final classification report demonstrating the accuracy of the classifier
    # for each of the label
    print("EVALUATION ON TESTING DATA")
    print(classification_report(tLabels, predictions))