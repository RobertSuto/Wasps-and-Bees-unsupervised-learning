#!/usr/bin/env python
# coding: utf-8

# # Importing packages

# In[1]:


from PIL import Image
import numpy as np
from random import randrange
import cv2
from scipy.optimize import linear_sum_assignment

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import confusion_matrix, silhouette_score, accuracy_score
from sklearn.svm import SVC

import os
from os import walk


# # Importing files
# 

# In[3]:


bee_image_paths = 'C:/Users/sutob/Desktop/bobi/kaggle_bee_vs_wasp/bee1/'
wasp_image_paths = 'C:/Users/sutob/Desktop/bobi/kaggle_bee_vs_wasp/wasp1/'


# In[4]:


# Saving all image paths and creating labels

paths = []
for path in os.listdir(bee_image_paths):
    paths.append(bee_image_paths + path)

y_bees = np.zeros((len(os.listdir(bee_image_paths),)), dtype=int)

for path in os.listdir(wasp_image_paths):
    paths.append(wasp_image_paths + path)

y_wasps = np.ones((len(os.listdir(wasp_image_paths),)), dtype=int)
y = np.concatenate((y_bees, y_wasps))


# # Feature Extraction

# In[6]:


"""

    Function for extracting features out of images
    Parameters
    --------
    extraction_type: which represents the way in which we want to extract features, 
                    - can take the value of "HSV" or "BOVW"
    paths_file: which represents an array including the path to all of the images.
    
    If the extraction_type is equal to HSV, the function converts each image into the HSV format and extracts 
    mean average of Hue, Saturation and Value for each position.
    
    IF it's equal to BOVW, it starts by extracting descriptors from the images, which are later used in creating
    histograms.
    
"""
def feature_Extraction(extraction_type, paths_file):
    if extraction_type == 'HSV':
        print('converting to HSV')
        x = []
        for path in paths_file:
            image = Image.open(path)
            image = image.convert('HSV')
            image = image.resize((60, 60))
            image = np.mean(np.array(image), axis=2)
            array = image.flatten()
            x.append(array)
        print('finished computing HSV means')
        return x
    if extraction_type == 'BOVW':
    
        print('extracting descriptors')
        bee_wasp_descriptors = []
        for path in paths:
            bee_wasp_image = cv2.imread(path)
            bee_wasp_image = cv2.cvtColor(bee_wasp_image, cv2.COLOR_BGR2GRAY)
            bee_wasp_image = cv2.resize(bee_wasp_image, (120,120))

            _, image_descriptors = cv2.xfeatures2d.SIFT_create().detectAndCompute(bee_wasp_image, None)
            bee_wasp_descriptors.append(image_descriptors)

        bee_wasp_descriptors = np.array(bee_wasp_descriptors,dtype=object)
        print('finished extracting descriptors')
        
        
        bee_wasp_descriptors = np.concatenate(bee_wasp_descriptors,axis=0)
        descriptors_clustering = KMeans(n_clusters = 50)
        descriptors_clustering.fit(bee_wasp_descriptors)
        
        x = []
        print('creating histograms')
        for path in paths:
            bee_wasp_image = cv2.imread(path)
            bee_wasp_image = cv2.cvtColor(bee_wasp_image, cv2.COLOR_BGR2GRAY)
            bee_wasp_image = cv2.resize(bee_wasp_image, (120,120))

            _, predicted_descriptors = cv2.xfeatures2d.SIFT_create().detectAndCompute(bee_wasp_image, None)

            predicted_descriptors = descriptors_clustering.predict(predicted_descriptors)
            bee_wasp_histogram, _ = np.histogram(predicted_descriptors, 1000)

            x.append(bee_wasp_histogram)
        print('finished creating histograms')
        return x
        
        
        
        


# In[7]:


# x = feature_Extraction('BOVW', paths)
x = feature_Extraction('HSV', paths)


# In[8]:


x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, stratify=y
)


# # Random Chance
# 

# In[9]:


import random

random_train = [random.randint(0, 1) for i in range(len(x_train))]
random_val = [random.randint(0, 1) for i in range(len(x_val))]


print("Random Chance train  |  Random Chance validation")
print(accuracy_score(y_train, random_train), '| ' ,accuracy_score(y_val, random_val))


# # Supervised Baseline

# In[8]:


svc = SVC()
svc.fit(x_train, y_train)

print(accuracy_score(y_train, svc.predict(x_train)))
print(accuracy_score(y_val, svc.predict(x_val)))


# # Models

# In[19]:


# clustering_model , model_name = KMeans(n_clusters = 2,
#                           init= 'k-means++',
#                           algorithm =  'full',
#                           tol =  1e-2), 'KMeans'

clustering_model , model_name = MiniBatchKMeans(n_clusters = 2, batch_size = 1500), 'MiniBatchKMeans'

clustering_model.fit(x_train)
train_predictions = clustering_model.predict(x_train)

val_predictions = clustering_model.predict(x_val)


train_acc = np.mean(train_predictions == y_train)
val_acc = np.mean(val_predictions == y_val)


print(model_name)
print('train Accuracy :',train_acc)
print('val Accuracy :' ,val_acc)
print('silhouette score :',silhouette_score(x_train, clustering_model.labels_, metric='euclidean'))


# In[18]:


clustering_model = AgglomerativeClustering(n_clusters = 2, linkage = 'ward', affinity = 'euclidean')


train_predictions = clustering_model.fit_predict(x_train)
train_accuracy = np.mean(train_predictions == y_train)

val_predictions = clustering_model.fit_predict(x_val)
val_acc = np.mean(val_predictions == y_val)


print('AgglomerativeClustering')
print('Train Accuracy :',train_accuracy)
print('Validation Accuracy :' ,val_acc)
print('Silhouette Score :', silhouette_score(x_train, train_predictions, metric='euclidean'))

