from stl import mesh
import numpy as np


def load_scatterer(path):
    scatterer = mesh.Mesh.from_file(path)
    return scatterer


def plot_scatterer(scatterer):
    from mpl_toolkits import mplot3d
    from matplotlib import pyplot as plt
    import matplotlib
    matplotlib.use("TkAgg")

    figure = plt.figure()
    axes = figure.add_subplot(projection='3d')

    # Load the STL files and add the vectors to the plot
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(scatterer.vectors))

    # Auto scale to the mesh size
    scale = scatterer.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    plt.show()

def translate(scatterer, dx=0,dy=0,dz=0):
    scatterer.translate(np.array([dx,dy,dz]))




if __name__ == "__main__":
    path = "Media/Bunny-lam1.stl"
    scatterer = load_scatterer(path)
    print(scatterer.x)
    translate(scatterer,dz=-0.06)
    plot_scatterer(scatterer)
  
