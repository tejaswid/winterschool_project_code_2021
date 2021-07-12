import enum
import os
import glob
import trimesh
import trimesh.sample
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def create_point_cloud_dataset(data_dir, num_points_per_cloud=1024):
    """
    Given the path to the ModelNet10 dataset, samples the models and creates point clouds
    :param data_dir: path to the ModelNet10 dataset
    :type data_dir: str
    :param num_points_per_cloud: number of points to sample per cloud. 1024, 2048....
    :type num_points_per_cloud: int
    :return: tuple of numpy array containing training and test point clouds, their corresponding labels and a list of
    class IDs
    :rtype: tuple
    """

    train_pc = []   # array of training point clouds
    test_pc = []    # array of test point clouds

    train_labels = []   # array of corresponding training labels
    test_labels = []    # array of corresponding test labels

    class_ids = {}   # list of class names

    # get all the folders except the readme file
    folders = glob.glob(os.path.join(data_dir, "[!README]*"))

    for class_id, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))

        # TODO: Fill this part, get the name of the folder (class) and save it
        class_ids[class_id]=os.path.basename(folder)

        # get the files in the train folder
        train_files = glob.glob(os.path.join(folder, "train/*"))
        for f in train_files:
            # TODO: Fill this part
            points = trimesh.sample.sample_surface(trimesh.load(f),num_points_per_cloud)[0]
            train_pc.append(points)
            train_labels.append(class_id)
        # get the files in the test folder
        test_files = glob.glob(os.path.join(folder, "test/*"))
        for f in test_files:
            # TODO: FIll this part
            points = trimesh.sample.sample_surface(trimesh.load(f),num_points_per_cloud)[0]
            test_pc.append(points)
            test_labels.append(class_id)

    encoded_train_labels = []
    for idx, label in enumerate(train_labels):
        one_hot = np.zeros(10)
        one_hot[label] = 1.
        encoded_train_labels.append(one_hot)
    encoded_train_labels = np.array(encoded_train_labels)

    encoded_test_labels = []
    for idx, label in enumerate(train_labels):
        one_hot = np.zeros(10)
        one_hot[label] = 1.
        encoded_test_labels.append(one_hot)
    encoded_test_labels = np.array(encoded_test_labels)

    return (np.array(train_pc), np.array(test_pc),
            np.array(encoded_train_labels), np.array(encoded_test_labels), class_ids)


def visualize_cloud(point_cloud):
    """
    Utility function to visualize a point cloud
    :param point_cloud: input point cloud
    :type point_cloud: numpy array
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
    plt.show()


def add_noise_and_shuffle(point_cloud, label):
    """
    Adds noise to a point cloud and shuffles it
    :param point_cloud: input point cloud
    :type point_cloud: tensor
    :param label: corresponding label
    :type label: tensor
    :return: the processed point cloud and the label
    :rtype: tensors
    """
    dev_in_metres = 0.002   # <- change this value to change amount of noise
    # add noise to the points
    point_cloud += tf.random.uniform(point_cloud.shape, -dev_in_metres, dev_in_metres, dtype=tf.float64)
    # shuffle points
    point_cloud = tf.random.shuffle(point_cloud)
    return point_cloud, label

# if __name__=='__main__':
#     a = create_point_cloud_dataset('ModelNet10/')
