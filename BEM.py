import vedo
import numpy as np
import torch

import matplotlib.pyplot as plt

from Utilities import device, TOP_BOARD, forward_model, create_points
import Constants



def load_scatterer(path):
    scatterer = vedo.load(path)
    return scatterer

def get_plane(scatterer, origin=(0,0,0), normal=(1,0,0)):
    intersection = scatterer.clone().intersect_with_plane(origin,normal)
    return intersection

def get_lines_from_plane(scatterer, origin=(0,0,0), normal=(1,0,0)):

    mask = [0,0,0]
    for i in range(3):
        mask[i] =not normal[i]
    mask = np.array(mask)

    intersection = get_plane(scatterer, origin, normal)
    verticies = intersection.vertices
    lines = intersection.lines

    connections = []

    for i in range(len(lines)):
        connections.append([verticies[lines[i][0]][mask],verticies[lines[i][1]][mask]])

    return connections

def plot_plane(connections):
    
    for con in connections:
        xs = [con[0][0], con[1][0]]
        ys = [con[0][1], con[1][1]]
        plt.plot(xs,ys,color = "blue")

    plt.xlim((-0.06,0.06))
    plt.ylim((-0.06,0.06))
    plt.show()

def translate(scatterer, dx=0,dy=0,dz=0):
    scatterer.shift(np.array([dx,dy,dz]))

def compute_G(points, scatterer):
    scatterer.compute_cell_size()
    areas = torch.Tensor(scatterer.celldata["Area"]).to(device)
    B = points.shape[0]
    N = points.shape[2]
    M = areas.shape[0]
    areas = areas.expand((B,N,-1))

    #Compute the partial derivative of Green's Function
    centres = torch.tensor(scatterer.cell_centers).to(device)
    centres = centres.expand((B,N,-1,-1))
    
    p = torch.reshape(points,(B,N,3))
    p = torch.unsqueeze(p,2).expand((-1,-1,M,-1))
    
    distance = torch.sqrt(torch.sum((centres - p)**2,dim=3))

    scatterer.compute_normals()
    norms = torch.tensor(scatterer.cell_normals).to(device)
    
    vecs = points.reshape(B,1,3,N)
    vecs = vecs.expand(-1,M,-1,-1)
    vecs = torch.permute(vecs,(0,3,1,2))
    vecs = vecs - centres


    norms = norms.expand(B,N,-1,-1)

    
    angles = torch.sum(norms*vecs,3) / (torch.norm(norms,2,dim=3)*torch.norm(vecs,2,dim=3))

    partial_greens = (torch.e**(1j*Constants.k*distance))/(4*torch.pi*distance)*(1j*Constants.k - 1/(distance))*angles
    
    G = areas * partial_greens
    return G



if __name__ == "__main__":
    path = "Media/Bunny-lam1.stl"
    # path = "Media/Cactus-lam6.stl"
    scatterer = load_scatterer(path)
    translate(scatterer,dz=-0.06)

    origin = (0,0,-0.06)
    normal = (1,0,0)

    points = create_points(4,2)

    G = compute_G(points, scatterer)
    
    '''
    intersection = get_plane(scatterer,origin,normal)
    lines = intersection.lines

    
    connections = get_lines_from_plane(scatterer,origin,normal)
    plot_plane(connections)

    '''

    # print(scatterer.inside_points())
    # vedo.show(scatterer,intersection).close()

    # vedo.show(scatterer)
