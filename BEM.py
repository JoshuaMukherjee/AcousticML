import vedo
import numpy as np
import torch
import pickle

import matplotlib.pyplot as plt

from Utilities import device, TOP_BOARD, TRANSDUCERS, forward_model_batched, create_points, forward_model_grad, forward_model_second_derivative_unmixed
import Constants

def scatterer_file_name(scatterer,board):
    M = board.shape[0]
    f_name = scatterer.filename 
    bounds = [str(round(i,2)) for i in scatterer.bounds()]
    rots = str(scatterer.metadata["rotX"][0]) + str(scatterer.metadata["rotY"][0]) + str(scatterer.metadata["rotZ"][0])
    # if "\\" in f_name:
        # f_name = f_name.split("/")[1].split(".")[0]
    f_name = f_name + "".join(bounds) +"--" + "-".join(rots) +"--" + str(M)
    return f_name

def load_scatterer(path, compute_areas = True, compute_normals=True, dx=0,dy=0,dz=0, rotx=0, roty=0, rotz=0):
    scatterer = vedo.load(path)
    if compute_areas: scatterer.compute_cell_size()
    if compute_normals: scatterer.compute_normals()
    scatterer.metadata["rotX"] = 0
    scatterer.metadata["rotY"] = 0
    scatterer.metadata["rotZ"] = 0

    scatterer.filename = scatterer.filename.split("/")[1]

    rotate(scatterer,(1,0,0),rotx)
    rotate(scatterer,(0,1,0),roty)
    rotate(scatterer,(0,0,1),rotz)

    translate(scatterer,dx,dy,dz)
    

    return scatterer

def load_multiple_scatterers(paths,board,  compute_areas = True, compute_normals=True, dxs=[],dys=[],dzs=[], rotxs=[], rotys=[], rotzs=[]):
    dxs += [0] * (len(paths) - len(dxs))
    dys += [0] * (len(paths) - len(dys))
    dzs += [0] * (len(paths) - len(dzs))

    rotxs += [0] * (len(paths) - len(rotxs))
    rotys += [0] * (len(paths) - len(rotys))
    rotzs += [0] * (len(paths) - len(rotzs))

    scatterers = []
    names= []
    for i,path in enumerate(paths):
        scatterer = load_scatterer(path, compute_areas, compute_normals, dxs[i],dys[i],dzs[i],rotxs[i],rotys[i],rotzs[i])
        f_name = scatterer_file_name(scatterer, board)
        scatterers.append(scatterer)
        names.append(f_name)
    combined = vedo.merge(scatterers)
    combined.filename = "--".join(names)
    return combined

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

def rotate(scatterer, axis, rot):
    if axis[0]:
        scatterer.metadata["rotX"] = scatterer.metadata["rotX"] + rot
    if axis[1]:
        scatterer.metadata["rotY"] = scatterer.metadata["rotZ"] + rot
    if axis[2]:
        scatterer.metadata["rotZ"] = scatterer.metadata["rotZ"] + rot
    scatterer.rotate(rot, axis)

def compute_green_derivative(y,x,norms,B,N,M, return_components=False):
    distance = torch.sqrt(torch.sum((x - y)**2,dim=3))

    vecs = x-y
    norms = norms.expand(B,N,-1,-1)

    
    norm_norms = torch.norm(norms,2,dim=3)
    vec_norms = torch.norm(vecs,2,dim=3)
    angles = torch.sum(norms*vecs,3) / (norm_norms*vec_norms)

    A = (torch.e**(1j*Constants.k*distance))/(4*torch.pi*distance)
    B = (1j*Constants.k - 1/(distance))
    partial_greens = A*B*angles
    partial_greens[partial_greens.isnan()] = 1

    if return_components:
        return partial_greens, A,B,angles
    
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
            
            f_name = scatterer_file_name(scatterer, board)
            f_name = "Media/BEMCache/"  +  f_name + ".bin"

            try:
                if print_lines: print("Trying to load H...")
                H = pickle.load(open(f_name,"rb"))
            except FileNotFoundError:
                if print_lines: print("Not found, computing H...")
                H = compute_H(scatterer,board)
                f = open(f_name,"wb")
                pickle.dump(H,f)
                f.close()
        else:
            if print_lines: print("Computing H...")
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
    
    out = E@activations
    return out

def propagate_BEM_pressure(activations,points,scatterer=None,board=TOP_BOARD,H=None,E=None):
    point_activations = propagate_BEM(activations,points,scatterer,board,H,E)
    pressures =  torch.abs(point_activations)
    # print(pressures)
    return pressures

def get_G_partial(points, scatterer, board=TRANSDUCERS, return_components=False):
    B = points.shape[0]
    N = points.shape[2]
    
    centres = torch.tensor(scatterer.cell_centers).to(device)
    M = centres.shape[0]

    p = torch.permute(points,(0,2,1))
    p = torch.unsqueeze(p,2).expand((-1,-1,M,-1))

    #Y is centres, X is points

    vecs = p-centres #Centres -> Points
    norms = torch.tensor(scatterer.cell_normals).to(device)
    norms = norms.expand(B,N,-1,-1)

    norm_norms = torch.norm(norms,2,dim=3)
    vec_norms = torch.norm(vecs,2,dim=3)
    vec_norms_cube = vec_norms**3

    distance = torch.sqrt(torch.sum(vecs**2,dim=3))
    distance_exp = torch.unsqueeze(distance,3)
    distance_exp = distance_exp.expand(-1,-1,-1,3)
    distance_exp_cube = distance_exp ** 3

    Ca = torch.zeros((B,N,M,3))
    Ca[:,:,:,0] = (norms[:,:,:,0] * (vecs[:,:,:,1]**2 +vecs[:,:,:,2]**2) - vecs[:,:,:,0] * (vecs[:,:,:,1] * norms[:,:,:,1] + vecs[:,:,:,2] * norms[:,:,:,2])) / (norm_norms * vec_norms_cube)
    Ca[:,:,:,1] = (norms[:,:,:,1] * (vecs[:,:,:,0]**2 +vecs[:,:,:,2]**2) - vecs[:,:,:,1] * (vecs[:,:,:,0] * norms[:,:,:,0] + vecs[:,:,:,2] * norms[:,:,:,2])) / (norm_norms * vec_norms_cube)
    Ca[:,:,:,2] = (norms[:,:,:,2] * (vecs[:,:,:,1]**2 +vecs[:,:,:,0]**2) - vecs[:,:,:,2] * (vecs[:,:,:,1] * norms[:,:,:,1] + vecs[:,:,:,0] * norms[:,:,:,0])) / (norm_norms * vec_norms_cube)


    Ba = vecs / distance_exp**2

    Aa = (1j*torch.exp(1j * Constants.k * distance_exp) * (Constants.k * distance_exp + 1j) * vecs) / (4*torch.pi * distance_exp_cube)

    G, A,B,C = compute_green_derivative(centres,p,norms,B,N,M,True)
    A = torch.unsqueeze(A,3)
    A = A.expand(-1,-1,-1,3)

    B = torch.unsqueeze(B,3)
    B = B.expand(-1,-1,-1,3)

    C = torch.unsqueeze(C,3)
    C = C.expand(-1,-1,-1,3)

    areas = torch.Tensor(scatterer.celldata["Area"]).to(device)
    areas = torch.unsqueeze(areas,0)
    areas = torch.unsqueeze(areas,0)
    areas = torch.unsqueeze(areas,3)

    
    Ga = B * (C*Aa + A*Ca) + A*C*Ba
    Ga = areas * Ga
    
    if return_components:
        return Ga[:,:,:,0], Ga[:,:,:,1], Ga[:,:,:,2], A, B, C, Aa, Ba, Ca
    else:
        return Ga[:,:,:,0], Ga[:,:,:,1], Ga[:,:,:,2]

def BEM_forward_model_grad(points, scatterer, board=TRANSDUCERS, use_cache_H=True, print_lines=False, H=None, return_components=False):
    B = points.shape[0]
    if H is None:
        if use_cache_H:
            
            f_name = scatterer_file_name(scatterer, board)
            f_name = "Media/BEMCache/"  +  f_name + ".bin"

            try:
                if print_lines: print("Trying to load H...")
                H = pickle.load(open(f_name,"rb"))
            except FileNotFoundError:
                if print_lines: print("Not found, computing H...")
                H = compute_H(scatterer,board)
                f = open(f_name,"wb")
                pickle.dump(H,f)
                f.close()
        else:
            if print_lines: print("Computing H...")
            H = compute_H(scatterer,board)
    
    Fx, Fy, Fz  = forward_model_grad(points, board)
    Gx, Gy, Gz = get_G_partial(points, scatterer, board)
    
    Gx = Gx.to(torch.complex64)
    Gy = Gy.to(torch.complex64)
    Gz = Gz.to(torch.complex64)

    H = H.expand(B, -1, -1)

    Ex = Fx + Gx@H
    Ey = Fy + Gy@H
    Ez = Fz + Gz@H

    if return_components:
        return Ex, Ey, Ez, Fx, Fy, Fz, Gx, Gy, Gz, H, 
    else:
        return Ex, Ey, Ez
    
def BEM_forward_model_second_derivative_unmixed(points, scatterer, board=TRANSDUCERS, use_cache_H=True, print_lines=False, H=None):
    B = points.shape[0]
    N = points.shape[2]

    if H is None:
        if use_cache_H:
            
            f_name = scatterer_file_name(scatterer, board)
            f_name = "Media/BEMCache/"  +  f_name + ".bin"

            try:
                if print_lines: print("Trying to load H...")
                H = pickle.load(open(f_name,"rb"))
            except FileNotFoundError:
                if print_lines: print("Not found, computing H...")
                H = compute_H(scatterer,board)
                f = open(f_name,"wb")
                pickle.dump(H,f)
                f.close()
        else:
            if print_lines: print("Computing H...")
            H = compute_H(scatterer,board)
    
    centres = torch.tensor(scatterer.cell_centers).to(device)
    M = centres.shape[0]
    
    p = torch.permute(points,(0,2,1))
    p = torch.unsqueeze(p,2).expand((-1,-1,M,-1))

    vecs = p-centres #Centres -> Points
    norms = torch.tensor(scatterer.cell_normals).to(device)
    norms = norms.expand(B,N,-1,-1)

    norm_norms = torch.norm(norms,2,dim=3)
    vec_norms = torch.norm(vecs,2,dim=3)
    vec_norms_cube = vec_norms**3
    vec_norms_five = vec_norms**5

    distance = torch.sqrt(torch.sum(vecs**2,dim=3))
    vecs_square = vecs **2
    distance_exp = torch.unsqueeze(distance,3)
    distance_exp = distance_exp.expand(-1,-1,-1,3)
    
    distance_exp_cube = distance_exp**3

    distaa = torch.zeros_like(distance_exp)
    distaa[:,:,:,0] = (vecs_square[:,:,:,1] + vecs_square[:,:,:,2]) 
    distaa[:,:,:,1] = (vecs_square[:,:,:,0] + vecs_square[:,:,:,2]) 
    distaa[:,:,:,2] = (vecs_square[:,:,:,1] + vecs_square[:,:,:,0])
    distaa = distaa / distance_exp_cube

    dista = vecs / distance_exp


    Aaa = -1 * torch.exp(1j*Constants.k * distance_exp) * \
    (distance_exp*(1-1j*Constants.k*distance_exp))*\
    distaa + dista*(Constants.k**2 * distance_exp**2 + 2*1j*Constants.k * distance_exp -2)\
          / (4*torch.pi * distance_exp_cube)
    
    Baa = (distance_exp * distaa - 2*dista**2) / distance_exp_cube

    Caa = torch.zeros_like(distance_exp)

    vec_dot_norm = vecs[:,:,:,0]*norms[:,:,:,0]+vecs[:,:,:,1]*norms[:,:,:,1]+vecs[:,:,:,2]*norms[:,:,:,2]

    Caa[:,:,:,0] = ((( (3 * vecs[:,:,:,0]**2) / (vec_norms_five) - (1)/(vec_norms_cube))*(vec_dot_norm)) / norm_norms) - ((2*vecs[:,:,:,0]*norms[:,:,:,0]) / (norm_norms*vec_norms_cube**3))
    Caa[:,:,:,1] = ((( (3 * vecs[:,:,:,1]**2) / (vec_norms_five) - (1)/(vec_norms_cube))*(vec_dot_norm)) / norm_norms) - ((2*vecs[:,:,:,1]*norms[:,:,:,1]) / (norm_norms*vec_norms_cube**3))
    Caa[:,:,:,2] = ((( (3 * vecs[:,:,:,2]**2) / (vec_norms_five) - (1)/(vec_norms_cube))*(vec_dot_norm)) / norm_norms) - ((2*vecs[:,:,:,2]*norms[:,:,:,2]) / (norm_norms*vec_norms_cube**3))
    
    Gz, Gy, Gz, A, B, C, Aa, Ba, Ca = get_G_partial(points, scatterer, board, return_components=True)

    Gaa = 2*Ca*(B*Aa + A*Ba) + C*(B*Aaa + 2*Aa*Ba + A*Baa)+ A*B*Caa
    Gaa = Gaa.to(torch.complex64)

    Fxx, Fyy, Fzz = forward_model_second_derivative_unmixed(points)

    Exx = Fxx + Gaa[:,:,:,0]@H
    Eyy = Fyy + Gaa[:,:,:,1]@H
    Ezz = Fzz + Gaa[:,:,:,2]@H

    return Exx, Eyy, Ezz


if __name__ == "__main__":
    from Solvers import wgs_batch
    from Gorkov import gorkov_fin_diff, gorkov_analytical
    from Utilities import add_lev_sig


    paths = ["Media/flat-lam1.stl","Media/flat-lam1.stl"]
    # scatterer = load_scatterer(path)
    board = TRANSDUCERS
    scatterer = load_multiple_scatterers(paths,board,dys=[-0.06,0.06,0],dxs=[0,0,0.06],rotxs=[-90,90,0],rotys=[0,0,90])
    origin = (0,0,-0.06)
    normal = (1,0,0)

    N=4
    B = 2
    points = create_points(N,B)

    E = compute_E(scatterer, points, board)
    Ex, Ey, Ez = BEM_forward_model_grad(points,scatterer,board)
    Exx, Eyy, Ezz = BEM_forward_model_second_derivative_unmixed(points,scatterer,board)

    F = forward_model_batched(points, board)
    _,_,x = wgs_batch(E, torch.ones(N,1).to(device)+0j,200)
    _,_,xF = wgs_batch(F, torch.ones(N,1).to(device)+0j,200)
    x = add_lev_sig(x)
    xF = add_lev_sig(xF)
  
    p = E@x
    px = Ex@x
    py = Ey@x
    pz = Ez@x
    pxx = Exx@x
    pyy = Eyy@x
    pzz = Ezz@x

   

    K1 = Constants.V / (4*Constants.p_0*Constants.c_0**2)
    K2 = 3*Constants.V / (4*(2*Constants.f**2 * Constants.p_0))

    U_ag = gorkov_fin_diff(x,points,prop_function=propagate_BEM,prop_fun_args={"scatterer":scatterer,"board":board},K1=K1, K2=K2)
    print(U_ag)

    U = K1 * torch.abs(p)**2 - K2*(torch.abs(px)**2 + torch.abs(py)**2 + torch.abs(pz)**2)
    print(U.squeeze_())

    UF = gorkov_analytical(xF,points,board).squeeze_()

    print(U / UF)

    x = add_lev_sig(x)
    xF = add_lev_sig(xF)
    p_BEM = E@x
    p_f = F@xF

    print(torch.abs(p_BEM).squeeze_() / torch.abs(p_f).squeeze_())
    # print(K1 * torch.abs(p)**2)


    # vedo.show(scatterer)   


    '''
    E = compute_E(scatterer,points,board,print_lines=True) #E=F+GH

    from Solvers import wgs
    _, _,x1 = wgs(E[0,:],torch.ones(N,1).to(device)+0j,200)
    _, _,x2 = wgs(E[1,:],torch.ones(N,1).to(device)+0j,200)
    x = torch.stack([x1,x2])
    

    # print(E)
    # print(x.shape)
    print(propagate_BEM_pressure(x,points,E=E))
    
    '''



