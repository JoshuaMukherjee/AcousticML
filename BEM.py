import vedo
import numpy as np
import torch
import pickle

import matplotlib.pyplot as plt

from Utilities import device, TOP_BOARD, forward_model_batched, create_points
import Constants



def load_scatterer(path, compute_areas = True, compute_normals=True, dx=0,dy=0,dz=0):
    scatterer = vedo.load(path)
    if compute_areas: scatterer.compute_cell_size()
    if compute_normals: scatterer.compute_normals()
    translate(scatterer,dx,dy,dz)
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

def compute_green_derivative(y,x,norms,B,N,M):
    distance = torch.sqrt(torch.sum((y - x)**2,dim=3))

    vecs = x-y


    norms = norms.expand(B,N,-1,-1)

    
    norm_norms = torch.norm(norms,2,dim=3)
    vec_norms = torch.norm(vecs,2,dim=3)
    angles = torch.sum(norms*vecs,3) / (norm_norms*vec_norms)

    partial_greens = (torch.e**(1j*Constants.k*distance))/(4*torch.pi*distance)*(1j*Constants.k - 1/(distance))*angles
    partial_greens[partial_greens.isnan()] = 1


    return partial_greens


def compute_G(points, scatterer):
    areas = torch.Tensor(scatterer.celldata["Area"]).to(device)
    B = points.shape[0]
    N = points.shape[2]
    M = areas.shape[0]
    areas = areas.expand((B,N,-1))

    #Compute the partial derivative of Green's Function

    #Firstly compute the distances from mesh points -> control points
    centres = torch.tensor(scatterer.cell_centers).to(device) #Uses centre points as position of mesh
    centres = centres.expand((B,N,-1,-1))
    
    # print(points.shape)
    # p = torch.reshape(points,(B,N,3))
    p = torch.permute(points,(0,2,1))
    p = torch.unsqueeze(p,2).expand((-1,-1,M,-1))

    #Compute cosine of angle between mesh normal and point
    scatterer.compute_normals()
    norms = torch.tensor(scatterer.cell_normals).to(device)
  
    partial_greens = compute_green_derivative(centres,p,norms, B,N,M)
    
    

    G = areas * partial_greens
    return G

def compute_A(scatterer):

    areas = torch.Tensor(scatterer.celldata["Area"]).to(device)

    centres = torch.tensor(scatterer.cell_centers).to(device)
    m = centres
    M = m.shape[0]
    m = m.expand((M,M,3))

    m_prime = m.clone()
    m_prime = m_prime.permute((1,0,2))

    norms = torch.tensor(scatterer.cell_normals).to(device)

    green = compute_green_derivative(m_prime.unsqueeze_(0),m.unsqueeze_(0),norms,1,M,M)

    A = green * areas * -1
    eye = torch.eye(M).to(bool)
    A[:,eye] = 0.5

    return A.to(torch.complex64)

def compute_bs(scatterer, board):
    centres = torch.tensor(scatterer.cell_centers).to(device).T.unsqueeze_(0)
    bs = forward_model_batched(centres,board)
    return bs.to(torch.complex64)

def compute_H(scatterer, board):
    A = compute_A(scatterer)
    bs = compute_bs(scatterer,board)
    H = torch.linalg.solve(A,bs)

    return H

def compute_E(scatterer, points, board=TOP_BOARD, use_cache_H=True, print_lines=False, H=None):
    if print_lines: print("H...")
    
    if H is None:
        if use_cache_H:
            f_name = scatterer.filename 
            bounds = [str(round(i,2)) for i in scatterer.bounds()]
            f_name = f_name.split("/")[1].split(".")[0]
            f_name = "Media/BEMCache/" + f_name + "".join(bounds)+".bin"

            try:
                H = pickle.load(open(f_name,"rb"))
            except FileNotFoundError:
                # print("Computing H...")
                H = compute_H(scatterer,board)
                f = open(f_name,"wb")
                pickle.dump(H,f)
                f.close()
        else:
            H = compute_H(scatterer,board)
        
    
    if print_lines: print("G...")
    G = compute_G(points, scatterer).to(torch.complex64)
    
    if print_lines: print("F...")
    F = forward_model_batched(points,board)
    
    if print_lines: print("E...")

    E = F+G@H 
    return E.to(torch.complex64)


def propagate_BEM(activations,points,scatterer=None,board=TOP_BOARD,H=None,E=None):


    if E is None:
        if type(scatterer) == str:
            scatterer = load_scatterer(scatterer)
        E = compute_E(scatterer,points,board,H=H)
    
    return E@activations

def propagate_BEM_pressure(activations,points,scatterer=None,board=TOP_BOARD,H=None,E=None):
    point_activations = propagate_BEM(activations,points,scatterer,board,H,E)
    pressures =  torch.abs(point_activations)
    # print(pressures)
    return pressures




if __name__ == "__main__":
    path = "Media/bunny-lam4.stl"
    scatterer = load_scatterer(path)
    translate(scatterer,dz=-0.06)

    origin = (0,0,-0.06)
    normal = (1,0,0)

    N=4
    B = 2
    points = create_points(N,B)


    E = compute_E(scatterer,points,TOP_BOARD) #E=F+GH

    from Solvers import wgs
    _, _,x1 = wgs(E[0,:],torch.ones(N,1).to(device)+0j,200)
    _, _,x2 = wgs(E[1,:],torch.ones(N,1).to(device)+0j,200)
    x = torch.stack([x1,x2])
    

    # print(E)
    # print(x.shape)
    print(propagate_BEM_pressure(x,points,E=E))

    


