# Winter School Project: Learning Semantics of 3D point Clouds
This is a group project for the UTS Winter School on SLAM for Deformable Objects, 2021.  
In this project we will be performing the task of learning semantics of 3D point clouds.

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

### File Description
- The main code for training is in `train.py`.
- Utility functions for data processing and visualization are in `utils.py`.
- The networks are defined in `network.py`.

### Instructions
- Start with the `train.py` script.
- Follow the comments and fill in the missing parts of the code.
- Once done train by running `python3 train.py`.

### Help
- Refer to the paper for help with theoretical understanding.
- Refer to tensorflow documentation for help with code.
- Reach out to Tejaswi Digumarti (tejaswi.digumarti@sydney.edu.au) for any other help. 