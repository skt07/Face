#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.image import imread
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance


# In[2]:


#separating the images into training and testing lists which contains the path of the images
data_path = "lfwcropped"
train, test = [], [] 
folders = sorted(os.listdir(data_path))
for folder in folders:
    files = [os.path.join(data_path, folder, path) for path in os.listdir(os.path.join(data_path, folder))]
    if len(files) == 1:
        test.extend(files)
    else:
        train.extend(files[:int(len(files)/2)])
        test.extend(files[int(len(files)/2):])


# In[3]:


#resizing the train images
width  = 100
height = 100

for i in range(len(train)):
    img = Image.open(train[i])  
    img = img.resize((width, height), Image.ANTIALIAS)
    img.save(train[i])


# In[4]:


training_image_vector= np.ndarray(shape=(len(train), height*width), dtype=np.float64)
for i in range(len(train)):
    img = cv2.imread(train[i],0)
    training_image_vector[i,:] = np.array(img, dtype='float64').flatten()
print("Size of training image vector:",training_image_vector.shape)
print("Training image vector:",training_image_vector)


# In[5]:


average_face_vector = np.zeros(height*width)
for i in training_image_vector:
    average_face_vector = np.add(average_face_vector,i)
average_face_vector = np.divide(average_face_vector,float(len(train))).flatten()
print("Size of average face vector:",average_face_vector.shape)
print("Average face vector:",average_face_vector)
print("Average face:")
plt.imshow(average_face_vector.reshape(height, width), cmap='gray')#plotting the average face generated
plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')#removing the labels from the plot
plt.show()


# In[6]:


normalised_training_vector = np.ndarray(shape=(len(train), height*width))
for i in range(len(train)):
    normalised_training_vector[i] = np.subtract(training_image_vector[i],average_face_vector)
print("Size of normalised face vector:",normalised_training_vector.shape)
print("Normalised face vector:",normalised_training_vector)


# In[14]:


cov_matrix = np.cov(normalised_training_vector.T)
cov_matrix_1=np.cov(normalised_training_vector)
print("Size of covariance matrix:",cov_matrix.shape)
print("Covariance matrix:",cov_matrix)


# In[15]:


eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
eigenvalues_1, eigenvectors_1, = np.linalg.eig(cov_matrix_1)
print("Eigenvectors of covariance matrix:\n",eigenvectors)
print("\nEigenvalues of covariance matrix:\n",eigenvalues)


# In[16]:


idx = eigenvalues.argsort()[::-1]   
eigvalues_sort = eigenvalues[idx]
eigvectors_sort = eigenvectors[:,idx]
idx_1 = eigenvalues_1.argsort()[::-1]   
eigvalues_sort_1 = eigenvalues_1[idx_1]
eigvectors_sort_1 = eigenvectors_1[:,idx_1]


# In[17]:


k = 100
top_eigenvectors_transpose = np.array(eigvectors_sort[:k])
top_eigenvectors_transpose_1 = np.array(eigvectors_sort_1[:k])
print("Size of the top k eigenvectors",top_eigenvectors_transpose.shape)
print("Top k eigenvectors:\n",top_eigenvectors_transpose)


# In[22]:


proj_data = np.dot(training_image_vector.transpose(),top_eigenvectors_transpose_1.transpose())
proj_data = proj_data.transpose()


# In[23]:


for i in range(50):
    img = proj_data[i].reshape(height,width)
    plt.subplot(5,10,1+i)
    plt.imshow(img, cmap='gray')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')

plt.show()


# In[24]:


for i in range(len(test)):
    img = Image.open(test[i]) 
    img = img.resize((width, height), Image.ANTIALIAS)
    img.save(test[i])


# In[26]:


testing_image_vector= np.ndarray(shape=(len(test), height*width), dtype=np.float64)
for i in range(len(test)):
    img = cv2.imread(test[i],0)
    testing_image_vector[i,:] = np.array(img, dtype='float64').flatten()
print("Size of the testing image vector:",np.shape(testing_image_vector))
print("Testing image vector:\n",testing_image_vector)


# In[27]:


#testing image weights
test_weights = np.ndarray(shape=(len(test), 100))
for i in range(len(test)):
    test_weights[i] = np.dot(top_eigenvectors_transpose,(np.subtract(testing_image_vector[i],average_face_vector)))
print("Size of testing weights:",test_weights.shape)
print("Testing weights:",test_weights)


# In[28]:


threshold = 270
tp, tn, fp, fn = 0, 0, 0, 0
for i in range(len(test_weights)-1):
    for j in range(i+1,len(test_weights)):
        dist = distance.euclidean(test_weights[i], test_weights[j])
        #print(dist)
        if(dist < threshold):
            if test[i].split(os.sep)[1] == test[j].split(os.sep)[1]:
                #print("Correctly Verified")
                tp += 1
            else:
                fp+=1
        else:
            if test[i].split(os.sep)[1] != test[j].split(os.sep)[1]:
                #print("Correctly Unverified")
                tn += 1
            else:
                fn+=1
        
print("True Positive : ", tp)
print("True Negative : ", tn)
print("False Positive : ", fp)
print("False Negative : ", fn)

# Calculating accuracy
accuracy = ((tp + tn)/float(tp+tn+fp+fn))*100.0
print("Accuracy : ", accuracy)


# In[15]:


fpr = [0.002,0.004,0.006,0.011,0.018,0.030,0.040,0.061,0.077,0.102,0.138,0.163,0.199,0.248,0.279,0.333,0.350,0.366,0.391,0.431,0.510,0.548,0.551,0.731,0.853]
tpr = [0.014,0.022,0.029,0.043,0.063,0.096,0.112,0.161,0.181,0.221,0.290,0.312,0.360,0.441,0.458,0.539,0.557,0.553,0.599,0.639,0.712,0.743,0.743,0.875,0.946]
print len(fpr)
plt.figure()
plt.xlabel('False Positive Rate(FPR)')
plt.ylabel('True Positive Rate(TPR)')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.autoscale(enable=True, axis='both', tight='None')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC Curve')
plt.plot(fpr, tpr, color='blue', lw=2)
plt.show()





# In[ ]:




