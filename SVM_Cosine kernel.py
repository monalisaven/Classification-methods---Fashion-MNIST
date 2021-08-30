#SVM_COSINE KERNEL
from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
fashion_mnist = keras.datasets.fashion_mnist
(trImages, trLabels), (tImages, tLabels) = fashion_mnist.load_data()

#normalize
trImages = trImages.astype('float32')
tImages = tImages.astype('float32')
trImages=trImages/255
tImages=tImages/255

trImages = trImages.reshape(trImages.shape[0], trImages.shape[1] * trImages.shape[2]) #monodiastata
tImages = tImages.reshape(tImages.shape[0], tImages.shape[1] * tImages.shape[2])

accuracies = []

model = SVC(kernel = metrics.pairwise.cosine_similarity, decision_function_shape='ovr') # Cosine Kernel
model.fit(tImages, tLabels) #fortwnoume to montelo BALAME TA TEST IMAGES , BECAUSE OF MEMORY ISSUES

predictions = model.predict(tImages) #ksekinaei thb ekpaideush

# evaluate the model and update the accuracies list
score=model.score(tImages, tLabels)
print("accuracy=%.2f%%" % (metrics.accuracy_score(tLabels, predictions) * 100))
print("f1_score=%.2f%%" % (metrics.f1_score(tLabels, predictions, average= "weighted") * 100))
accuracies.append(score)

# show a final classification report demonstrating the accuracy of the classifier
# for each of the label
print("EVALUATION ON TESTING DATA")
print(classification_report(tLabels, predictions)) # matrix with our all-statistics