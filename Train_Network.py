import torch
import torch.nn.functional as F
import torch.nn as nn
'''
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019).
PyTorch: An Imperative Style, High-Performance Deep Learning Library. 
In Advances in Neural Information Processing Systems 32 (pp. 8024–8035). 
Curran Associates, Inc. 
Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf
'''

import Dataset
import time
import pickle
import random
from Symmetric_Functions import SymSum
from Utilities import *

# torch.autograd.set_detect_anomaly(True)

def do_network(net, optimiser,loss_function,loss_params, datasets,test=False, 
                supervised=True, scheduler = None, random_stop=False, clip=False,
                clip_args={}, norm_loss = False):
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
           
           
            activation_out = net(points)

            #RANDOM STOP CHECK HERE


            field = propagate(activation_out,points)
            pressure_out = torch.abs(field)
            outputs.append(pressure_out)
                    
            output = torch.stack(outputs,dim=1).squeeze_()
            target = torch.abs(pressures)

    
           
            # if supervised:
            #     loss = loss_function(pressure_out,torch.abs(pressures[:,end,:]),**loss_params)
            # else:
            #     loss = loss_function(pressure_out,**loss_params)
            if supervised:
                loss = loss_function(output,target,**loss_params) + phase_reg_val + pressure_reg_val
            else:
                loss = loss_function(output,**loss_params) + phase_reg_val + pressure_reg_val
    

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
    return running, grad



def train(net, start_epochs, epochs, train, test, optimiser, 
            loss_function, loss_params, supervised, scheduler, name, 
            batch, random_stop, clip=False, clip_args={}, log_grad =False, norm_loss = False ):
    print(name, "Training....")
    print(device)
    start_time = time.asctime()
    losses = []
    losses_test = []
    best_test = torch.inf

    try:   
        for epoch in range(epochs):
            #Train
            running , grad= do_network(net, optimiser, loss_function, loss_params, train, 
                                        scheduler=scheduler, supervised=supervised, 
                                        random_stop=random_stop, clip=clip, clip_args=clip_args,
                                        norm_loss = norm_loss )
            #Test
            running_test, _ = do_network(net, optimiser, loss_function, loss_params, 
                                         test, test=True, supervised=supervised, 
                                         norm_loss = norm_loss)
            
            losses.append(running) #Store each epoch's losses 
            losses_test.append(running_test)

            print(name,epoch+start_epochs,"Training",running,"Testing",running_test,"Time",time.asctime(),"Start",start_time, end=" ")
            if log_grad:
                print("grad",grad, end = " ")
            
            # if supervised: #what is this for?
            #     running_test = torch.abs(running_test)
            if running_test < best_test: #Only save if the best 
                net.epoch_saved = epoch
                torch.save(net, 'Models/model_' + str(name) + '.pth')
                best_test = running_test
                print("SAVED")
            else:
                print()
            torch.save(net, 'Models/model_' + str(name) + '_latest.pth') #save the newest model too
            loss_to_dump = (losses, losses_test)
            pickle.dump(loss_to_dump, open("Losses/loss_"+ str(name) +'.pth',"wb"))

    except KeyboardInterrupt:
        pass



if __name__ == "__main__":
    from torch.utils.data import DataLoader 
    from Networks import MLP, PointNet
    from Symmetric_Functions import SymSum
    dataset = Dataset.PointDataset(10)


    layers = [[64,64],[64,128,1024],[512,256,128,128,512]]
    norm = torch.nn.BatchNorm1d
    network = PointNet(layers,batch_norm=norm,output_funct=SymSum, input_size=7)
    
    train(network,0,1,[DataLoader(dataset,2,shuffle=True)],[DataLoader(dataset,2,shuffle=True)],None,None)
