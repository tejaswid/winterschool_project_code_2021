import os
import glob
import trimesh
import trimesh.sample
import numpy as np
import matplotlib.pyplot as plt

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


DATA_DIR = "ModelNet10"     # <- Set this path correctly

cad_mesh = trimesh.load(os.path.join(DATA_DIR, "bed/train/bed_0010.off"))  # <- Set path to a .off file
cad_mesh.show()

points = trimesh.sample.sample_surface(cad_mesh, 1024*2)[0]
# visualize the point cloud using matplotlib
visualize_cloud(points)