# Winter School Project: Learning Semantics of 3D point Clouds
This is a group project for the UTS Winter School on SLAM for Deformable Objects, 2021.  

## Problem Description
Understanding what objects are present in a scene or what the constituent parts of an
object are is useful for several robotics applications such as navigation, mapping and interaction. In this project
we will work with 3D point clouds and look at two closely related tasks - object classification and semantic
segmentation. In object classification, we will try to infer the class of an object by learning geometric features
that are unique to a class and in semantic segmentation, we will assign to every point of the cloud a label that
represents the class it belongs to.

## Setup
### Prerequisites
- tensorflow
- trimesh
- matplotlib
- pyglet

### Setup using Pip
If you want to install the prerequisites using pip, use the following commands.  
```bash
pip install tensorflow
pip install trimesh
pip install matplotlib
pip install pyglet
```

### Setup using Conda
If you are using an Anaconda environment then install the prerequisites using the following commands.  

For CPU version of tensorflow
```bash
conda install -c anaconda tensorflow
```

For GPU version of tensorflow
```bash
conda install -c anaconda tensorflow-gpu
```

For trimesh, matplotlib and pyglet
```bash
conda install -c conda-forge trimesh matplotlib pyglet
```

## File Description
- The main code for training is in `train.py`.
- Utility functions for data processing and visualization are in `utils.py`.
- The networks are defined in `network.py`.

## Tasks
The following tasks need to be performed.

- **Generating the datasets**: The ModelNet dataset is a collection of CAD models. The first step is to
generate point clouds by sampling the models. Using the skeleton code provided, write code to accomplish
this task.
  
- **Building the Network**: This is the most important part of the project. In the provided skeleton code,
key parts of the network are missing and need to be built. Build your network using Tensorflow 2.0
functions such as `tf.keras.layers.Conv1D`, `tf.keras.layers.Dense`, `tf.keras.layers.BatchNormalization`,
`tf.keras.layers.Dropout`, etc. A key task is to implement the T-net which is the core of the proposed
architecture. Use the main reference as your guide. Help is provided in the code. Feel free to play
around with the parameters of the network. You need not follow the exact number of layers and neurons
mentioned in the reference. 
  
- **Object classification**: The first objective (easy) is to train the network for object classification, i.e given
the point cloud of an object the network has to predict a single label specifying what object the point
cloud represents. Evaluate the performance of your network with and without noise added to the sampled
point clouds. Report performance as a confusion matrix.

- **Semantic segmentation**: The second objective (hard) is to train the network for semantic segmentation,
i.e. given a point cloud of a scene, the network has to label each point based on the object class that it
belongs to. For this task you will need to create a custom dataset using the same ModelNet10 dataset.
Create a dataset where each input cloud consists of 2-3 objects. Try to make one of suitable size, with a
range of orientations, placement and overlap of the objects. Report the performance on this task.

### Instructions
- Start with the `train.py` script.
- Follow the comments and fill in the missing parts of the code.
- Once done train by running `python3 train.py`.

### References
1. Qi, Charles R., Hao Su, Kaichun Mo, and Leonidas J. Guibas. ”Pointnet: Deep learning on point sets
for 3d classification and segmentation.” In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), 2017.
2. Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang and J. Xiao ”3D ShapeNets: A Deep Representation for
Volumetric Shapes.” In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2015.

### Help
- Refer to the paper for help with theoretical understanding.
- Refer to tensorflow documentation for help with code.
- Reach out to Tejaswi Digumarti (tejaswi.digumarti@sydney.edu.au) for any other help.