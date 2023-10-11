import torch
import torch.nn as nn
from Utilities import propagate


def train_PointNet(net, params):

    optimiser = params["optimiser"]
    loss_function = params["loss_function"]
    loss_params = params["loss_params"]
    datasets = params["datasets"]
    test= params["test"]
    supervised= params["supervised"]
    scheduler = params["scheduler"]
    random_stop=params["random_stop"]
    clip= params["clip"]
    clip_args=params["clip_args"]
    norm_loss = params["norm_loss"]
    extra_points_fun=params["extra_point_fun"]
    extra_points_args=params["extra_point_args"]
    maximise_first_N=params["maximise_first_N"]
    
    #TRAINING
    running = 0
    if not test:
        net.train()
    else:
        net.eval()

    
    for dataset in datasets:
        for points, activations, pressures in iter(dataset):
            if not test:
                optimiser.zero_grad()            
            
            #RANDOM STOP NOT IMPLEMENTED, ADD HERE IF REQUIRED
            
            optimiser.zero_grad()            
        
            end = 0
            outputs = []
            phase_reg_val = 0
            pressure_reg_val = 0
            
            if extra_points_fun is not None:
                points = extra_points_fun(points,**extra_points_args)
       
           
            activation_out = net(points)

            #RANDOM STOP CHECK HERE

            # For sine-> Add to points, remember which to minimise & what to maximise

            field = propagate(activation_out,points)
            pressure_out = torch.abs(field)
            outputs.append(pressure_out)
                    
            output = torch.stack(outputs,dim=1).squeeze_()
            target = torch.abs(pressures)

    
           
            # if supervised:
            #     loss = loss_function(pressure_out,torch.abs(pressures[:,end,:]),**loss_params)
            # else:
            #     loss = loss_function(pressure_out,**loss_params)
            if maximise_first_N == -1:
                if supervised:
                    loss = loss_function(output,target,**loss_params) + phase_reg_val + pressure_reg_val
                else:
                    loss = loss_function(output,**loss_params) + phase_reg_val + pressure_reg_val
            else:
                if supervised:
                    loss = loss_function(output,target,maximise_first_N,**loss_params) + phase_reg_val + pressure_reg_val
                else:
                    loss = loss_function(output,maximise_first_N,**loss_params) + phase_reg_val + pressure_reg_val
    

            if norm_loss:
                batch = points.shape[0]
                time =  points.shape[1]
                loss /= (batch*time)

            running += loss.item()
            grad = None
            if not test:
                loss.backward()
                
                if clip:
                    grads = [torch.sum(p.grad) for n, p in net.named_parameters()]
                    grad = []
                    grad.append(grads)
                   
                    nn.utils.clip_grad_norm_(net.parameters(), **clip_args)
                    grads = [torch.sum(p.grad) for n, p in net.named_parameters()]
                    grad.append((sum(grads)/len(grads)).item())
                    # print({n:p.grad for n, p in net.named_parameters()}["encoder.layers.5.weight"])
                
                optimiser.step()
    if not test:
        if scheduler is not None:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(running)
            else:
                scheduler.step()
    
    out= {
        "grad":grad
    }
    
    return running, out



default_functions = {
    "PointNet":train_PointNet,
    "ResPointNet":train_PointNet
}