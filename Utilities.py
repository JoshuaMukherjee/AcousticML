import torch, math, sys
import Constants



device = 'cuda' if torch.cuda.is_available() else 'cpu'


device = device if '-cpu' not in sys.argv else 'cpu'

# import line_profiler
# profile = line_profiler.LineProfiler()


def create_board(N, z):  
    #Written by Giorgos Christopoulos, 2022
    pitch=0.0105
    grid_vec=pitch*(torch.arange(-N/2+1, N/2, 1)).to(device)
    x, y = torch.meshgrid(grid_vec,grid_vec,indexing="ij")
    x = x.to(device)
    y= y.to(device)
    trans_x=torch.reshape(x,(torch.numel(x),1))
    trans_y=torch.reshape(y,(torch.numel(y),1))
    trans_z=z*torch.ones((torch.numel(x),1)).to(device)
    trans_pos=torch.cat((trans_x, trans_y, trans_z), axis=1)
    return trans_pos
  
def transducers():
    #Written by Giorgos Christopoulos, 2022
  return torch.cat((create_board(17,.234/2),create_board(17,-.234/2)),axis=0).to(device)

TRANSDUCERS = transducers()
TOP_BOARD = create_board(17,.234/2)


def forward_model(points, transducers = TRANSDUCERS):
    #Written by Giorgos Christopoulos, 2022
    m=points.size()[1]
    n=transducers.size()[0]
    k=2*math.pi/0.00865
    radius=0.005
    
    transducers_x=torch.reshape(transducers[:,0],(n,1))
    transducers_y=torch.reshape(transducers[:,1],(n,1))
    transducers_z=torch.reshape(transducers[:,2],(n,1))


    points_x=torch.reshape(points[0,:],(m,1))
    points_y=torch.reshape(points[1,:],(m,1))
    points_z=torch.reshape(points[2,:],(m,1))
    
    dx = (transducers_x.T-points_x) **2
    dy = (transducers_y.T-points_y) **2
    dz = (transducers_z.T-points_z) **2

    distance=torch.sqrt(dx+dy+dz)
    planar_distance=torch.sqrt(dx+dy)

    bessel_arg=k*radius*torch.divide(planar_distance,distance) #planar_dist / dist = sin(theta)

    directivity=1/2-torch.pow(bessel_arg,2)/16+torch.pow(bessel_arg,4)/384
    
    p = 1j*k*distance
    phase = torch.e**(p)
    
    trans_matrix=2*8.02*torch.multiply(torch.divide(phase,distance),directivity)
    return trans_matrix

def forward_model_batched(points, transducers = TRANSDUCERS):
    B = points.shape[0]
    N = points.shape[2]
    M = transducers.shape[0]
    
    # p = torch.permute(points,(0,2,1))
    transducers = torch.unsqueeze(transducers,2)
    transducers = transducers.expand((B,-1,-1,N))
    points = torch.unsqueeze(points,1)
    points = points.expand((-1,M,-1,-1))

    distance_axis = (transducers - points) **2
    distance = torch.sqrt(torch.sum(distance_axis,dim=2))
    planar_distance= torch.sqrt(torch.sum(distance_axis[:,:,0:2,:],dim=2))
    
    bessel_arg=Constants.k*Constants.radius*torch.divide(planar_distance,distance)
    directivity=1/2-torch.pow(bessel_arg,2)/16+torch.pow(bessel_arg,4)/384
    
    p = 1j*Constants.k*distance
    phase = torch.e**(p)

    trans_matrix=2*8.02*torch.multiply(torch.divide(phase,distance),directivity)

    return trans_matrix.permute((0,2,1))
    

def propagate(activations, points,board=TRANSDUCERS):
    out = []
    for i in range(activations.shape[0]):
        A = forward_model(points[i],board).to(device)
       
        out.append(A@activations[i])
    out = torch.stack(out,0)
    return out.squeeze()

def propagate_abs(activations, points,board=TRANSDUCERS):
    out = propagate(activations, points,board)
    return torch.abs(out)

def permute_points(points,index,axis=0):
    if axis == 0:
        return points[index,:,:,:]
    if axis == 1:
        return points[:,index,:,:]
    if axis == 2:
        return points[:,:,index,:]
    if axis == 3:
        return points[:,:,:,index]


def swap_output_to_activations(out_mat,points):
    acts = None
    for i,out in enumerate(out_mat):
        out = out.T.contiguous()
        pressures =  torch.view_as_complex(out)
        A = forward_model(points[i]).to(device)
        if acts == None:
            acts =  A.T @ pressures
        else:
            acts = torch.stack((acts,A.T @ pressures),0)
    return acts


def convert_to_complex(matrix):
    # B x 1024 x N (real) -> B x N x 512 x 2 -> B x 512 x N (complex)
    matrix = torch.permute(matrix,(0,2,1))
    matrix = matrix.view((matrix.shape[0],matrix.shape[1],-1,2))
    matrix = torch.view_as_complex(matrix.contiguous())
    return torch.permute(matrix,(0,2,1))


def convert_pats(board):
    raise Exception("DO NOT USE")
    board[512//2+1:,0] = torch.flipud(board[512//2+1:,0]);
    board[:,1] = torch.flipud(board[:,1]);
    board[:,2] = torch.flipud(board[:,2]);
    return board

def get_convert_indexes():
     #Invert with _,INVIDX = torch.sort(IDX)
    
    board = transducers()
    board[512//2:,0] = torch.flipud(board[512//2:,0]);
    board[:,1] = torch.flipud(board[:,1]);
    board[:,2] = torch.flipud(board[:,2]);
    indexes = []

    for t,row in enumerate(board):
        for b,row_b in enumerate(transducers()):
            if torch.all(row == row_b):
                indexes.append(b)
    indexes = torch.as_tensor(indexes)


    return indexes


def do_NCNN(net, points):
    from Solvers import naive_solver
    '''
    Go from points -> activations using Naive-input Networks

    Adds empty batch if only one point passed (512 -> 1x512)
    Leave batchs if batched passed in
    '''

    naive_acts = []
    for ps in points:
        _, naive_act = naive_solver(ps)
        naive_acts.append(naive_act)
    
    naive_act = torch.stack(naive_acts)

    if len(naive_act.shape) < 2:
        naive_act = torch.unsqueeze(naive_act,0)
        print("Added Batch of 1")
    


    act_in = torch.reshape(naive_act,(naive_act.shape[0],2,16,16))
    act_phases = torch.angle(act_in)
    activation_out_img = net(act_phases) 
    # activation_out = torch.reshape(activation_out_img,(naive_act.shape[0],512)) + 1j
    activation_out = torch.e** (1j*(torch.reshape(activation_out_img,(naive_act.shape[0],512))))

    
    return activation_out.unsqueeze_(2)

def create_points(N,B=1,x=None,y=None,z=None, min_pos=-0.06, max_pos = 0.06):
    points = torch.FloatTensor(B, 3, N).uniform_(min_pos,max_pos).to(device)
    if x is not None:
        points[:,0,:] = x
    
    if y is not None:
        points[:,1,:] = y
    
    if z is not None:
        points[:,2,:] = z

    return points
    
# @profile
def add_lev_sig(activation):
    act = activation.clone().to(device)

    s = act.shape
    B = s[0]

    act = torch.reshape(act,(B,2, 256))

    act[:,0,:] = torch.e**(1j*(torch.pi + torch.angle(act[:,0,:].clone())))
    act = torch.reshape(act,s)

    return act

def generate_gorkov_targets(N,B=1, max_val=0, min_val=-1e-4):
    targets = torch.FloatTensor(B, N,1).uniform_(min_val,max_val).to(device)
    return targets

def generate_pressure_targets(N,B=1, max_val=10000, min_val=7000):
    targets = torch.FloatTensor(B, N,1).uniform_(min_val,max_val).to(device)
    return targets


if __name__ == "__main__":
    from Solvers import wgs
    
    points = create_points(4,2)

    A1 = forward_model(points[0,:])
    _, _, x1 = wgs(A1,torch.ones(4,1).to(device)+0j,200)

    A2 = forward_model(points[1,:])
    _, _, x2 = wgs(A2,torch.ones(4,1).to(device)+0j,200)

    A = forward_model_batched(points)
    x = torch.stack([x1,x2])
    print(A.shape)
    print(x.shape)
    print(torch.abs(A@x))


    '''
    A = forward_model(points[0,:])
    _, _, x = wgs(A,torch.ones(4,1).to(device)+0j,200)
    x = torch.unsqueeze(x,0)
    
    pr = propagate(x,points)
    print(torch.abs(pr))
    
    x2 = add_lev_sig(x)
    pr2 = propagate(x2,points)
    print(torch.abs(pr2))


    from torch.utils.data import DataLoader 
    from Dataset import NaiveDataset
    from Loss_Functions import mse_loss

    net = torch.load("./Models/model_NCNN1_latest.pth")
    # points=  torch.FloatTensor(3,4).uniform_(-.06,.06).to(device)
    dataset = torch.load("./Datasets/NaiveDatasetTrain-4-4.pth")
    data = iter(DataLoader(dataset,1,shuffle=True))

    for p,a,pr,n in data:

        activation_out = do_NCNN(net, p)

        field = propagate(activation_out,p)

        f = torch.unsqueeze(field,0)
        print(mse_loss(torch.abs(f), torch.abs(pr)))
        print(torch.abs(field))

    '''

    '''
    trans_pos(:,2) = flipud(trans_pos(:,2));   
    trans_pos(n/2+1:end,1) = flipud(trans_pos(n/2+1:end,1));
    '''

    # board = transducers()
    # print(board.shape)
    # board[512//2:,0] = torch.flipud(board[512//2:,0]);
    # board[:,1] = torch.flipud(board[:,1]);
    # board[:,2] = torch.flipud(board[:,2]);
    # indexes = []

    # for t,row in enumerate(board):
    #     for b,row_b in enumerate(transducers()):
    #         if torch.all(row == row_b):
    #             indexes.append(b)


    # indexes = torch.as_tensor(indexes)
    # trans = transducers()
    # flipped = trans[indexes]


    # for i,row in enumerate(flipped):
    #     print(row)

    
    
    # print(torch.reshape(board,(16,16,3)))