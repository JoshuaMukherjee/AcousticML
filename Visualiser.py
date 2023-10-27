import torch
from Utilities import propagate_abs, add_lev_sig, device
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

def get_point_pos(A,B,C, points, res=(200,200)):
    AB = torch.tensor([B[0] - A[0], B[1] - A[1], B[2] - A[2]])
    AC = torch.tensor([C[0] - A[0], C[1] - A[1], C[2] - A[2]])

    ab_dir = AB!=0
    ac_dir = AC!=0

    step_x = AB / res[0]
    step_y = AC / res[1]

    
    points = torch.split(points.squeeze_().T,1)
    print(points)
    points = [pt.squeeze_() for pt in points]
    print(points)

    pts_norm = []

    for pt in points:
        Apt =  torch.tensor([pt[0] - A[0], pt[1] - A[1], pt[2] - A[2]])
        px = Apt / step_x
        py = Apt / step_y
        pt_pos = torch.zeros((2))
        pt_pos[0]= torch.round(px[ab_dir])
        pt_pos[1]=torch.round(py[ac_dir])
        pts_norm.append(pt_pos)

    return pts_norm


def Visualise_single(A,B,C,activation,colour_function=propagate_abs, colour_function_args={}, res=(200,200)):
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
        print(i)
    
    return result

    # plt.imshow(result.cpu().detach().numpy(),cmap="hot")
    # plt.colorbar()
    

    # plt.show()

def Visualise(A,B,C,activation,points=[],colour_functions=[propagate_abs], colour_function_args=None, res=(200,200), cmaps=[]):
    results = []
    if len(points) > 0:
        pts_pos = get_point_pos(A,B,C,points,res)
        print(pts_pos)
        pts_pos_t = torch.stack(pts_pos).T


    if colour_function_args is None:
        colour_function_args = [{}]*len(colour_functions)
    for i,colour_function in enumerate(colour_functions):
        result = Visualise_single(A,B,C,activation,colour_function, colour_function_args[i], res)
        results.append(result)

    for i in range(len(results)):
        if len(cmaps) > 0:
            cmap = cmaps[i]
        else:
            cmap = 'hot'
        plt.subplot(1,len(colour_functions),i+1)
        plt.imshow(results[i].cpu().detach().numpy(),cmap=cmap)
        plt.colorbar()
        plt.scatter(pts_pos_t[1],pts_pos_t[0],marker="x")
        
    plt.show()

if __name__ == "__main__":
    # A = torch.tensor((-0.06, 0.06, 0))
    # B = torch.tensor((0.06, 0.06, 0))
    # C = torch.tensor((-0.06, -0.06, 0))

    res=(200,200)

    A = torch.tensor((-0.08, 0, 0.08))
    B = torch.tensor((0.08, 0, 0.08))
    C = torch.tensor((-0.08, 0, -0.08))
    
    from Utilities import create_points, forward_model, device
    from Solvers import wgs
    from Gorkov import gorkov_autograd

    N = 4
    points=  create_points(N,y=0)
    print(points)
    F = forward_model(points[0,:]).to(device)
    _, _, x = wgs(F,torch.ones(N,1).to(device)+0j,200)

    Visualise(A,B,C,x,colour_functions=[propagate_abs,gorkov_autograd],points=points,res=res)