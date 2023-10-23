import torch, math


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_board(N, z):  
    #Written by Giorgos Christopoulos, 2022
    pitch=0.0105
    grid_vec=pitch*(torch.arange(-N/2+1, N/2, 1))
    x, y = torch.meshgrid(grid_vec,grid_vec,indexing="ij")
    trans_x=torch.reshape(x,(torch.numel(x),1))
    trans_y=torch.reshape(y,(torch.numel(y),1))
    trans_z=z*torch.ones((torch.numel(x),1))
    trans_pos=torch.cat((trans_x, trans_y, trans_z), axis=1)
    return trans_pos
  
def transducers():
    #Written by Giorgos Christopoulos, 2022
  return torch.cat((create_board(17,.234/2),create_board(17,-.234/2)),axis=0).to(device)

def forward_model(points, transducers = transducers()):
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
    

    distance=torch.sqrt((transducers_x.T-points_x)**2+(transducers_y.T-points_y)**2+(transducers_z.T-points_z)**2)
    planar_distance=torch.sqrt((transducers_x.T-points_x)**2+(transducers_y.T-points_y)**2)
    bessel_arg=k*radius*torch.divide(planar_distance,distance)
    directivity=1/2-bessel_arg**2/16+bessel_arg**4/384
    phase=torch.exp(1j*k*distance)
    trans_matrix=2*8.02*torch.multiply(torch.divide(phase,distance),directivity)
    return trans_matrix.to(device)


def propagate(activations, points):
    out = []
    for i in range(activations.shape[0]):
        A = forward_model(points[i]).to(device)
       
        out.append(A@activations[i])
    out = torch.stack(out,0)
    return out.squeeze()


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


    return activation_out



if __name__ == "__main__":

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