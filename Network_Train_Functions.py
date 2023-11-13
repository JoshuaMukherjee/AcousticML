import torch
import torch.nn as nn
from Utilities import propagate, generate_gorkov_targets, generate_pressure_targets


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

def train_FCNN(net, params):

    optimiser = params["optimiser"]
    loss_function = params["loss_function"]
    loss_params = params["loss_params"]
    datasets = params["datasets"]
    test= params["test"]
    supervised= params["supervised"]
    scheduler = params["scheduler"]

    running = 0
    if not test:
        net.train()
    else:
        net.eval()

    for dataset in datasets:
        for F, points, activations, pressures in iter(dataset):
            if not test:
                optimiser.zero_grad()            
            
            
            activation_out = net(F) 
            field = torch.abs(F@activation_out).squeeze_()

            target = torch.abs(pressures)


            if supervised:
                loss = loss_function(field,target,**loss_params)
            else:
                loss = loss_function(field,**loss_params) 
            
            running += loss.item()
            if not test:
                loss.backward()
                optimiser.step()

    if not test: #schedule LR on epochs
        if scheduler is not None:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(running)
            else:
                scheduler.step()
    
    return running, {}

def train_FCNN_hologram(net, params):

    optimiser = params["optimiser"]
    loss_function = params["loss_function"]
    loss_params = params["loss_params"]
    datasets = params["datasets"]
    test= params["test"]
    supervised= params["supervised"]
    scheduler = params["scheduler"]

    running = 0
    if not test:
        net.train()
    else:
        net.eval()

    for dataset in datasets:
        for F, points, activations, pressures in iter(dataset):
            if not test:
                optimiser.zero_grad()            
            
            
            activation_out = torch.angle(net(F)).squeeze_()

            target = torch.angle(activations)


            if supervised:
                loss = loss_function(activation_out,target,**loss_params)
            else:
                loss = loss_function(activation_out,**loss_params) 
            
            running += loss.item()
            if not test: #Learn on each batch
                loss.backward()
                optimiser.step()

    if not test: #schedule LR on epochs
        if scheduler is not None:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(running)
            else:
                scheduler.step()
    
    return running, {}
            
def train_FCNN_Slow(net, params):

    optimiser = params["optimiser"]
    loss_function = params["loss_function"]
    loss_params = params["loss_params"]
    datasets = params["datasets"]
    test= params["test"]
    supervised= params["supervised"]
    scheduler = params["scheduler"]

    running = 0
    if not test:
        net.train()
    else:
        net.eval()

    for dataset in datasets:
        for F, points, activations, pressures in iter(dataset):
            if not test:
                optimiser.zero_grad()            
            
            
            activation_out = net(F) 

            field = torch.abs(propagate(activation_out,points)).squeeze_()

            target = torch.abs(pressures)


            if supervised:
                loss = loss_function(field,target,**loss_params)
            else:
                loss = loss_function(field,**loss_params) 
            
            running += loss.item()
            if not test:
                loss.backward()
                optimiser.step()

    if not test: #schedule LR on epochs
        if scheduler is not None:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(running)
            else:
                scheduler.step()
    
    return running, {}         
        
def train_FCNN_hologram_pressure(net, params):

    optimiser = params["optimiser"]
    loss_function = params["loss_function"]
    loss_params = params["loss_params"]
    datasets = params["datasets"]
    test= params["test"]
    supervised= params["supervised"]
    scheduler = params["scheduler"]

    running = 0
    if not test:
        net.train()
    else:
        net.eval()

    for dataset in datasets:
        for F, points, activations, pressures in iter(dataset):
            if not test:
                optimiser.zero_grad()            
            
            act = net(F)
            activation_out = torch.angle(act).squeeze_()
            field = torch.abs(propagate(act,points)).squeeze_()
            target_pressure = torch.abs(pressures)

            target = torch.angle(activations)


            if supervised:
                loss = loss_function(activation_out,target,field,target_pressure,**loss_params)
            else:
                loss = loss_function(activation_out,target,field,**loss_params) 
            
            running += loss.item()
            if not test: #Learn on each batch
                loss.backward()
                optimiser.step()

    if not test: #schedule LR on epochs
        if scheduler is not None:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(running)
            else:
                scheduler.step()
    
    return running, {}

def train_naive(net, params):

    optimiser = params["optimiser"]
    loss_function = params["loss_function"]
    loss_params = params["loss_params"]
    datasets = params["datasets"]
    test= params["test"]
    supervised= params["supervised"]
    scheduler = params["scheduler"]

    running = 0
    if not test:
        net.train()
    else:
        net.eval()

    for dataset in datasets:
        for points, activations, pressures, naive_act in iter(dataset):
            if not test:
                optimiser.zero_grad()            
            

            act_in = torch.reshape(naive_act,(naive_act.shape[0],2,16,16))
            act_phases = torch.angle(act_in)
            activation_out_img = net(act_phases) 
            activation_out = torch.e** (1j*(torch.reshape(activation_out_img,(naive_act.shape[0],512))))
      
            field = torch.abs(propagate(activation_out, points))
            target = torch.abs(pressures)


            if supervised:
                loss = loss_function(field,target,**loss_params)
            else:
                loss = loss_function(field,**loss_params) 
            
            running += loss.item()
            if not test:
                loss.backward()
                optimiser.step()

    if not test: #schedule LR on epochs
        if scheduler is not None:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(running)
            else:
                scheduler.step()
    
    return running, {}

def train_naive_holograam(net, params):

    optimiser = params["optimiser"]
    loss_function = params["loss_function"]
    loss_params = params["loss_params"]
    datasets = params["datasets"]
    test= params["test"]
    supervised= params["supervised"]
    scheduler = params["scheduler"]

    running = 0
    if not test:
        net.train()
    else:
        net.eval()

    for dataset in datasets:
        for points, activations, pressures, naive_act in iter(dataset):
            if not test:
                optimiser.zero_grad()            
            

            act_in = torch.reshape(naive_act,(naive_act.shape[0],2,16,16))
            act_phases = torch.angle(act_in)
            activation_out_img = net(act_phases) 
            activation_out = torch.e** (1j*(torch.reshape(activation_out_img,(naive_act.shape[0],512))))

            act_out_phase = torch.angle(activation_out)
            act_target_phase = torch.angle(activations)


            if supervised:
                loss = loss_function(act_out_phase,act_target_phase,**loss_params)
            else:
                loss = loss_function(act_out_phase,**loss_params) 
            
            running += loss.item()
            if not test:
                loss.backward()
                optimiser.step()

    if not test: #schedule LR on epochs
        if scheduler is not None:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(running)
            else:
                scheduler.step()
    
    return running, {}

def train_naive_hologram_points(net, params):

    optimiser = params["optimiser"]
    loss_function = params["loss_function"]
    loss_params = params["loss_params"]
    datasets = params["datasets"]
    test= params["test"]
    supervised= params["supervised"]
    scheduler = params["scheduler"]

    running = 0
    if not test:
        net.train()
    else:
        net.eval()
    
    if not test:
        optimiser.zero_grad()       

    for dataset in datasets:
        for points, activations, pressures, naive_act in iter(dataset):
            
                    
            

            act_in = torch.reshape(naive_act,(naive_act.shape[0],2,16,16))
            act_phases = torch.angle(act_in)
            activation_out_img = net(act_phases) 
            
            activation_out = torch.e** (1j*(torch.reshape(activation_out_img,(naive_act.shape[0],512,1))))


            if supervised:
                loss = loss_function(activation_out,activations,points,**loss_params)
            else:
                loss = loss_function(activation_out,points,**loss_params) 
            
            running += loss.item()
            if not test:
                optimiser.zero_grad()    
                loss.backward()
                optimiser.step()
                   

    if not test: #schedule LR on epochs
        if scheduler is not None:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(running)
            else:
                scheduler.step()
    
    return running, {}

def train_gorkov_target_mCNN(net,params):
    optimiser = params["optimiser"]
    loss_function = params["loss_function"]
    loss_params = params["loss_params"]
    datasets = params["datasets"]
    test= params["test"]
    supervised= params["supervised"]
    scheduler = params["scheduler"]
    solver = params["solver"]

    running = 0
    if not test:
        net.train()
    else:
        net.eval()
    
    if not test:
        optimiser.zero_grad()       

    for dataset in datasets:
        for points, activations, pressures, naive_act in iter(dataset):
            B = points.shape[0]
            
            targets = generate_gorkov_targets(points.shape[2],B)

            if solver == "naive":
                act_start = naive_act
            elif solver == "wgs":
                act_start = activations

            act_in = torch.reshape(act_start,(B,2,16,16))
            act_phases = torch.angle(act_in)
            
            activation_out_img = net(act_phases, targets) 
            
            activation_out = torch.reshape(activation_out_img,(B,512,1))

            if supervised:
                loss = loss_function(activation_out,points, targets,**loss_params)
            else:
                loss = loss_function(activation_out,points, targets,**loss_params) 
            
            running += loss.item()
            if not test:
                optimiser.zero_grad()    
                loss.backward()
                optimiser.step()
                   

    if not test: #schedule LR on epochs
        if scheduler is not None:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(running)
            else:
                scheduler.step()
    
    return running, {}

def train_gorkov_no_target_mCNN(net,params):
    optimiser = params["optimiser"]
    loss_function = params["loss_function"]
    loss_params = params["loss_params"]
    datasets = params["datasets"]
    test= params["test"]
    supervised= params["supervised"]
    scheduler = params["scheduler"]
    solver = params["solver"]

    running = 0
    if not test:
        net.train()
    else:
        net.eval()
    
    if not test:
        optimiser.zero_grad()       

    for dataset in datasets:
        for points, activations, pressures, naive_act in iter(dataset):
            B = points.shape[0]
            
            targets = generate_gorkov_targets(points.shape[2],B)

            if solver == "naive":
                act_start = naive_act
            elif solver == "wgs":
                act_start = activations

            act_in = torch.reshape(act_start,(B,2,16,16))
            act_phases = torch.angle(act_in)
            
            activation_out_img = net(act_phases, targets) 
            
            activation_out = torch.reshape(activation_out_img,(B,512,1))

            if supervised:
                loss = loss_function(activation_out,points,**loss_params)
            else:
                loss = loss_function(activation_out,points,**loss_params) 
            
            running += loss.item()
            if not test:
                optimiser.zero_grad()    
                loss.backward()
                optimiser.step()
                   

    if not test: #schedule LR on epochs
        if scheduler is not None:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(running)
            else:
                scheduler.step()
    
    return running, {}

def train_pressure_target_mCNN(net,params):
    optimiser = params["optimiser"]
    loss_function = params["loss_function"]
    loss_params = params["loss_params"]
    datasets = params["datasets"]
    test= params["test"]
    supervised= params["supervised"]
    scheduler = params["scheduler"]
    solver = params["solver"]

    running = 0
    if not test:
        net.train()
    else:
        net.eval()
    
    if not test:
        optimiser.zero_grad()       

    for dataset in datasets:
        for points, activations, pressures, naive_act, targets in iter(dataset):
            B = points.shape[0]
            

            if solver == "naive":
                act_start = naive_act
            elif solver == "wgs":
                act_start = activations

            act_in = torch.reshape(act_start,(B,2,16,16))
            act_phases = torch.angle(act_in)
            
            activation_out_img = net(act_phases, targets) 
            
            activation_out = torch.reshape(activation_out_img,(B,512,1))
            pressure_out = torch.abs(propagate(activation_out, points))
            if supervised:
                loss = loss_function(pressure_out, targets,**loss_params)
            else:
                loss = loss_function(pressure_out, targets,**loss_params) 
            
            running += loss.item()
            if not test:
                optimiser.zero_grad()    
                loss.backward()
                optimiser.step()
                   

    if not test: #schedule LR on epochs
        if scheduler is not None:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(running)
            else:
                scheduler.step()
    
    return running, {}


default_functions = {
    "PointNet":train_PointNet,
    "ResPointNet":train_PointNet,
    "F_CNN":train_FCNN,
    "CNN":train_naive,
    "UNET":train_naive,
    "MultiInputCNN":train_gorkov_target_mCNN

}