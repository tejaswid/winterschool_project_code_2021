import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def tnet(inputs, num_features):
    """
    This is the core t-net of the pointnet paper
    :param inputs: the input tensor
    :type inputs: tensor
    :param num_features: number of features in the tensor (3 for point cloud, N if N features)
    :type num_features: int
    :return: output tensor
    :rtype: tensor
    """

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())

    # TODO: Build the tnet with the following layers
    # Some convolutional layers (1D) - with batch normalization, RELU activation
    # Global max pooling
    # Some dense fully connected layers - with batch normalization, RELU activation
    x =

    # final layer with custom regularizer on the output
    # TODO: this custom regularizer needs to be defined
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=CustomRegularizer(num_features))(x)
    feat_t = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_t])


class CustomRegularizer(keras.regularizers.Regularizer):
    """
    This class implements a regularizer that makes the output to be orthogonal.
    In other words, it adds a loss |I-AA^T|^2 on the output A. Equation 2 of the paper.
    """
    def __init__(self, dim, weight=0.001):
        """
        Initializes the class
        :param dim: dimensions of the input tensor
        :type dim: int
        :param weight: weight to apply on the regularizer
        :type weight: float
        """
        self.dim = dim
        self.weight = weight

    def __call__(self, x):
        # TODO: define the custom regularizer here
        x = tf.reshape(x, (-1, self.dim, self.dim))
        # compute the outer product and reshape it to batch size x num_features x num_features

        # Compute (I-outerproduct)^2 element wise. use tf.square()

        # Apply weight

        # Compute reduce sum using tf.reduce_sum()

        return output


def pointnet_classifier(inputs, num_classes):
    """
    This is the object classifier version of PointNet
    :param inputs: input point clouds tensor
    :type inputs: tensor
    :param num_classes: number of classes
    :type num_classes: int
    :return: the predicted labels
    :rtype: tensor
    """
    # TODO: build the network using the following layers
    # apply tnet to the input data
    x =
    # extract features using some Convolutional Layers - with batch normalization and RELU activation

    # apply tnet on the feature vector

    # extract features using some Convolutional Layers - with batch normalization and RELU activation

    # apply 1D global max pooling

    # Add a few dense layers with dropout between the layers

    # Finally predict classes using a dense layer with a softmax activation
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return outputs


def pointnet_segmenter(inputs, labels):
    """
    This is the semantic segmentation version of Pointnet
    :param inputs: input point cloud
    :type inputs: tensor
    :param labels: labels for each point of the point cloud
    :type labels: tensor
    :return: predicted labels for each point of the point cloud
    :rtype: tensor
    """
    # TODO: build the network using the following layers
    # apply tnet to the input data
    x =
    # extract features using some Convolutional Layers - with batch normalization and RELU activation

    # apply tnet on the feature vector
    f =
    # extract features using some Convolutional Layers - with batch normalization and RELU activation

    # apply 1D global max pooling

    # concatenate these features with the earlier features (f)
    # you can also use skip connections if you like

    # extract features using some Convolutional Layers - with batch normalization and RELU activation

    # return the output
    return outputs