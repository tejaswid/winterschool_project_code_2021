import os
import numpy as np
import tensorflow as tf
import trimesh.sample
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import pickle

import network
import utils


# If using a GPU keep these lines to avoid CUDNN errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# *********************************************
# STEP 0: Download the dataset and extract it.
# *********************************************
# This is a zip file of 451 MB.
# The dataset consists of CAD models of objects of 10 classes in .off format,
# *********************************************
# 1. Download the file "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
# 2. Extract it and note the path
# 3. Set the path to the extracted folder
DATA_DIR = "ModelNet10"     # <- Set this path correctly

# 4. Familiarize yourself with the dataset by checking the folders and visualizing the models.
# Use the functions trimesh.load to load a mesh and mesh.show() to visualize it
cad_mesh = trimesh.load(os.path.join(DATA_DIR, "bed/train/bed_0010.off"))  # <- Set path to a .off file
cad_mesh.show()

# *************************************************************
# STEP 1: Generate point clouds
# *************************************************************
# We need to generate point clouds by sampling the surfaces of these CAD models.
# This can be done using the trimesh.sample.sample_surface() function as follows.
points = trimesh.sample.sample_surface(cad_mesh, 1024)[0]
# visualize the point cloud using matplotlib
utils.visualize_cloud(points)

# 1. Use the above function to repeat the process and sample points for the entire dataset.
# Develop the skeleton code in the utils.py file.
num_points_per_cloud = 1024     # <- you can modify this number as needed
train_pc, test_pc, train_labels, test_labels, class_ids = utils.create_point_cloud_dataset(DATA_DIR,
                                                                                           num_points_per_cloud)

# 2. For the semantic segmentation dataset, modify the above function.
# Tips: You can generate the point clouds of individual objects and then randomly combine multiple objects to one cloud.

# once loaded save the numpy arrays to pickle files to use later
pickle.dump(train_pc, open("trainpc.pkl", "wb"))
pickle.dump(test_pc, open("testpc.pkl", "wb"))
pickle.dump(train_labels, open("trainlabels.pkl", "wb"))
pickle.dump(test_labels, open("testlabels.pkl", "wb"))
pickle.dump(class_ids, open("class_ids.pkl", "wb"))

# load the data from pickle files if already present
# train_pc = pickle.load(open("trainpc.pkl", "rb"))
# train_labels = pickle.load(open("trainlabels.pkl", "rb"))
# test_pc = pickle.load(open("testpc.pkl", "rb"))
# test_labels = pickle.load(open("testlabels.pkl", "rb"))
# class_ids = pickle.load(open("class_ids.pkl", "rb"))

# Create tensorflow data loaders from the numpy arrays
train_dataset = tf.data.Dataset.from_tensor_slices((train_pc, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_pc, test_labels))

# 3. Perform data augmentation by adding noise and shuffling the dataset.
# In this step you need to fill in the utils.add_noise_and_shuffle function
# to add noise to the sampled points and shuffle the points around
batch_size = 16     # <- You can modify this value as needed
train_dataset = train_dataset.shuffle(len(train_pc)).map(utils.add_noise_and_shuffle).batch(batch_size)
test_dataset = test_dataset.shuffle(len(test_pc)).batch(batch_size)

# ********************************************************************************
# STEP 2: Build the network model - Either for classification or for segmentation
# ********************************************************************************
# 1. Fill in the skeleton code given in the network.py file
inputs = keras.Input(shape=(num_points_per_cloud, 3))
outputs = network.pointnet_classifier(inputs, num_classes=10)
# outputs = network.pointnet_segmenter(inputs, train_labels)

# build the network and visualize its architecture
model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
model.summary()

# 2. Set the loss function, optimizer and metrics to print
model.compile(
    loss="....",     # <- choose a suitable loss function
    optimizer=keras.optimizers.Adam(learning_rate=0.001),      # <- you may modify this if you like
    metrics=[""],    # <- choose a suitable metric, https://www.tensorflow.org/api_docs/python/tf/keras/metrics
)

# train the network
num_epochs = 5      # <- change this value as needed
model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)

# visualize results
data = test_dataset.take(1)
point_clouds, labels = list(data)[0]  # this is one batch of data

# predict labels using the model
preds = model.predict(point_clouds)
preds = tf.math.argmax(preds, -1)

# 3. Display some clouds using matplotlib scatter plot along with true and predicted labels

# 4. Display a confusion matrix