# USAGE
# python race_classification_cslbp.py
# import the necessary packages
from skimage import feature
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from PIL import Image
from PIL import ImageOps as imops
from time import time
from sklearn.metrics import confusion_matrix
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import operateur

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="images",
	help="path to directory containing the 'images' dataset")

# grab all image paths in the input dataset directory, initialize our
# list of extracted features and corresponding labels
print("[INFO] extracting image features...")
imagePaths = paths.list_images(args["dataset"])
data = []
labels = []
N,M = 100,100
# the size of the images NxM
size = (N,M)
extract_time = time()
# the size of the image blocks
size_block = 50
 
print("l'image est divisé en régions de taille ",size_block,"x",size_block)
print("")
print("extraction de caractéristiques lancé....")

# loop over our input images
for imagePath in imagePaths:
	# preprocessing of all the images we have as input
	image = Image.open(imagePath)
	image = image.resize(size)
	image_gray = imops.grayscale(image)
	image_gray = imops.equalize(image_gray,mask=None)
	image_gray = np.array(image_gray)
	i = 0
	features = []
	features = np.asarray(features)
	nb_region = 0
	# Calculation of the cslbp images of each block
	while i <= image_gray.shape[0]-size_block :
		j = 0
		while j <= image_gray.shape[1]-size_block:
			part_image_dslbp = operateur.central_symmetric_lbp(image_gray[i:i+size_block-1,j:j+size_block-1])
			features = np.concatenate((features,np.histogram(part_image_dslbp.ravel(),bins=np.arange(0, 17),range=(0, 16))[0]))

			j = j + size_block
			nb_region+=1
		i = i + size_block
		# print(features)
	data.append(features)

	# extract the class label from the file path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

print("caractéristiques extraites en %0.6fs" % (time() - extract_time))
# encode the labels, converting them from strings to integers

le = LabelEncoder()
labels = le.fit_transform(labels)
np_labels = np.asarray(labels)
np_data = np.asarray(data)
n_classe = np.unique(np_labels).shape[0]
n_features = np_data.shape[1]
n_samples = np_data.shape[0]
print("l'image est divisé en ",nb_region," régions")
print("nombre de classes     : ",n_classe)
print("nombre d'echantillons : ",n_samples)
print("nombre de features    : ",n_features)

# perform a training and testing split, using 75% of the data for
# training and 25% for evaluation
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25,random_state=42)

# train the model
train_time=time()
print("")
print("Phase d'entrainement lancé....")
model = SVC(kernel='linear')
model.fit(trainX, trainY)
print("L'apprentissage a été fait en %0.6fs" % (time() - train_time))
print("")
print("Le score d'apprentissage est de  ",(model.score(trainX,trainY)*100),'%')

# make predictions on our data and show a classification report
print("[INFO] evaluating...")
print("")
pred_time = time()
print("")
print("Phase de test lancé....")

predictions = model.predict(testX)
print("Les prédictions ont été fait en %0.6fs" % (time() - pred_time))
print("")

print("La matrice de confusion est : ")
print("")
print(confusion_matrix(testY,predictions,labels=range(n_classe)))
print(classification_report(testY, predictions, target_names=le.classes_))

print("Le score du model est : ",(model.score(testX,testY)*100),'%')
