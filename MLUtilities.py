import torch
from acoustools.Solvers import naive_solver

def do_NCNN(net, points):
    
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