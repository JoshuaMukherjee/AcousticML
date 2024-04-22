import torch
from acoustools.Solvers import naive
from acoustools.Utilities import device,DTYPE

def do_NCNN(net, points):
    
    '''
    Go from points -> activations using Naive-input Networks

    Adds empty batch if only one point passed (512 -> 1x512)
    Leave batchs if batched passed in
    '''

    naive_acts = []
    for ps in points:
        naive_act = naive(ps)
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

def do_GCNN(net, points, board,norm=False):
    B = points.shape[0]
    N = points.shape[2]
    M = board.size()[0]

    p = torch.unsqueeze(points,1)
    p = p.expand((B,M,-1,-1))
    
    board = torch.unsqueeze(board,0)
    board = torch.unsqueeze(board,3)
    board = board.expand((B,-1,-1,N))
    
    distance_axis = (board - p) **2
    distance = torch.sqrt(torch.sum(distance_axis,dim=2))
    distance = torch.reshape(distance,(B,2*N,16,16))
    

    green = torch.exp(1j*726.3798*distance) / distance    

    green_ri = torch.cat((green.real,green.imag),1).to(device)
    
    if norm:
        green_ri = torch.nn.functional.normalize(green_ri)
    out = net(green_ri)
    out = torch.reshape(out,(-1,512,1)).to(DTYPE)
    out = torch.e**(1j * out)

    return out

def do_GMLP(net, points, board):
    B = points.shape[0]
    N = points.shape[2]
    M = board.size()[0]

    p = torch.unsqueeze(points,1)
    p = p.expand((B,M,-1,-1))
    
    board = torch.unsqueeze(board,0)
    board = torch.unsqueeze(board,3)
    board = board.expand((B,-1,-1,N))
    
    distance_axis = (board - p) **2
    distance = torch.sqrt(torch.sum(distance_axis,dim=2))
    distance = torch.reshape(distance,(B,2*N,16,16))
    

    green = torch.exp(1j*726.3798*distance) / distance    

    green_ri = torch.cat((green.real,green.imag),1).to(device)
    green_ri = green_ri.reshape(B,-1)
    
    out = net(green_ri)
    out = torch.reshape(out,(-1,512,1)).to(DTYPE)
    out = torch.e**(1j * out)

    return out