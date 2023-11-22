import torch
from Utilities import propagate_abs, add_lev_sig, device, create_board
import matplotlib.pyplot as plt


def get_point_pos(A,B,C, points, res=(200,200),flip=True):
    AB = torch.tensor([B[0] - A[0], B[1] - A[1], B[2] - A[2]])
    AC = torch.tensor([C[0] - A[0], C[1] - A[1], C[2] - A[2]])

    ab_dir = AB!=0
    ac_dir = AC!=0

    step_x = AB / res[0]
    step_y = AC / res[1]

    if points.shape[2] > 1:
        points = torch.split(points.squeeze_().T,1)
        points = [pt.squeeze_() for pt in points]
    print(points)

    pts_norm = []

    for pt in points:
        Apt =  torch.tensor([pt[0] - A[0], pt[1] - A[1], pt[2] - A[2]])
        px = Apt / step_x
        py = Apt / step_y
        pt_pos = torch.zeros((2))
        if not flip:
            pt_pos[0]= torch.round(px[ab_dir])
            pt_pos[1]=torch.round(py[ac_dir])
        else:
            pt_pos[1]= torch.round(px[ab_dir])
            pt_pos[0]=torch.round(py[ac_dir])
        
        pts_norm.append(pt_pos)

   

    return pts_norm


def Visualise_single_slow(A,B,C,activation,colour_function=propagate_abs, colour_function_args={}, res=(200,200), flip=True):
    '''
    OLD SLOW VERSION
    Visalises field generated from activation to the plane ABC
    colour_function defined what is plotted, default is pressure 
    '''
    if len(activation.shape) < 3:
        activation.unsqueeze_(0)
    

    AB = torch.tensor([B[0] - A[0], B[1] - A[1], B[2] - A[2]])
    AC = torch.tensor([C[0] - A[0], C[1] - A[1], C[2] - A[2]])

    step_x = AB / res[0]
    step_y = AC / res[1]

    result = torch.zeros(res)
    posX = torch.tensor([0])

    for i in range(0,res[0]):
        posX = A + step_x * i
        for j in range(res[1]):
            pos = (posX + step_y * j).to(device)

            pos.unsqueeze_(0)
            pos.unsqueeze_(2)
            
            field_val = colour_function(activation,pos,**colour_function_args)
            result[i,j] = field_val
        print(i,end=" ")
    if flip:
        # result = torch.flip(result,[0,])
        # result = torch.rot90(result)
        result = torch.rot90(torch.fliplr(result))
    
    
    return result

    # plt.imshow(result.cpu().detach().numpy(),cmap="hot")
    # plt.colorbar()
    

    # plt.show()

def Visualise_single(A,B,C,activation,colour_function=propagate_abs, colour_function_args={}, res=(200,200), flip=True):
    '''
    Visalises field generated from activation to the plane ABC
    colour_function defined what is plotted, default is pressure 
    '''
    if len(activation.shape) < 3:
        activation.unsqueeze_(0)
    

    AB = torch.tensor([B[0] - A[0], B[1] - A[1], B[2] - A[2]])
    AC = torch.tensor([C[0] - A[0], C[1] - A[1], C[2] - A[2]])

    step_x = AB / res[0]
    step_y = AC / res[1]

    positions = torch.zeros((1,3,res[0]*res[1])).to(device)

    for i in range(0,res[0]):
        for j in range(res[1]):
            positions[:,:,i*res[0]+j] = A + step_x * i + step_y * j
           
    field_val = colour_function(activation,positions,**colour_function_args)
    result = torch.reshape(field_val, res)

    if flip:
        result = torch.rot90(torch.fliplr(result))
    
    
    return result

def Visualise(A,B,C,activation,points=[],colour_functions=[propagate_abs], colour_function_args=None, 
              res=(200,200), cmaps=[], add_lines_functions=None, add_line_args=None):
    results = []
    lines = []
    if len(points) > 0:
        pts_pos = get_point_pos(A,B,C,points,res)
        # print(pts_pos)
        pts_pos_t = torch.stack(pts_pos).T


    if colour_function_args is None:
        colour_function_args = [{}]*len(colour_functions)
    
    for i,colour_function in enumerate(colour_functions):
        result = Visualise_single(A,B,C,activation,colour_function, colour_function_args[i], res)
        results.append(result)
        
        if add_lines_functions is not None:
            lines.append(add_lines_functions[i](**add_line_args[i]))

    for i in range(len(results)):
        if len(cmaps) > 0:
            cmap = cmaps[i]
        else:
            cmap = 'hot'
        plt.subplot(1,len(colour_functions),i+1)
        plt.imshow(results[i].cpu().detach().numpy(),cmap=cmap,vmin=0,vmax=9000)
        plt.colorbar()

        if add_lines_functions is not None:
            AB = torch.tensor([B[0] - A[0], B[1] - A[1], B[2] - A[2]])
            AC = torch.tensor([C[0] - A[0], C[1] - A[1], C[2] - A[2]])
            # print(AB,AC)
            norm_x = AB
            norm_y = AC
            AB = AB[AB!=0] / res[0]
            AC = AC[AC!=0] / res[1]
            # AC = AC / torch.abs(AC)
            # print(AB,AC)
            for con in lines[i]:
                xs = [con[0][0]/AB + res[0]/2, con[1][0]/AB + res[0]/2] #Convert real coordinates to pixels - number of steps in each direction
                ys = [con[0][1]/AC + res[1]/2, con[1][1]/AC + res[1]/2] #Add res/2 as 0,0,0 in middle of real coordinates not corner of image
                # print(xs,ys)
                plt.plot(xs,ys,color = "blue")

        
        plt.scatter(pts_pos_t[1],pts_pos_t[0],marker="x")
        
    plt.show()

if __name__ == "__main__":
    # A = torch.tensor((-0.06, 0.06, 0))
    # B = torch.tensor((0.06, 0.06, 0))
    # C = torch.tensor((-0.06, -0.06, 0))

    res=(300,300)

    X = 0
    A = torch.tensor((X,-0.07, 0.07))
    B = torch.tensor((X,0.07, 0.07))
    C = torch.tensor((X,-0.07, -0.07))
    
    from Utilities import create_points, forward_model, device, TOP_BOARD
    from Solvers import wgs
    from Gorkov import gorkov_autograd

    from BEM import propagate_BEM_pressure, load_scatterer,compute_E, compute_H, get_lines_from_plane
    
    N = 4
    points=  create_points(N,x=X)
    print(points.shape)

    path = "Media/bunny-lam1.stl"
    scatterer = load_scatterer(path,dz=-0.06)

    origin = (X,0,-0.06)
    normal = (1,0,0)

    H = compute_H(scatterer,TOP_BOARD)
    E = compute_E(scatterer,points,TOP_BOARD,H=H) #E=F+GH

    _, _, x = wgs(E[0,:],torch.ones(N,1).to(device)+0j,200)
    

    Visualise(A,B,C,x,colour_functions=[propagate_BEM_pressure],points=points,res=res,
              colour_function_args=[{"H":H,"scatterer":scatterer}],
              add_lines_functions=[get_lines_from_plane],add_line_args=[{"scatterer":scatterer,"origin":origin,"normal":normal}])
    
    # Visualise(A,B,C,x,colour_functions=[propagate_abs],points=points,res=res,colour_function_args=[{"board":TOP_BOARD}])