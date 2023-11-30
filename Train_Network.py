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
from acoustools.Utilities import *

# torch.autograd.set_detect_anomaly(True)


def train(net, start_epochs, epochs, 
          train, test, 
          optimiser, 
          loss_function, loss_params, supervised, 
          scheduler, 
          name, 
          batch, random_stop, 
          clip=False, clip_args={}, 
          log_grad =False, norm_loss = False,
          extra_points_fun=None, extra_points_args={}, maximise_first_N =-1, 
          training_function=None,
          solver=None):
    
    print(name, "Training....")
    print(device)
    start_time = time.asctime()
    losses = []
    losses_test = []
    best_test = torch.inf

    try:   
        for epoch in range(epochs):
            #Train
            training_params = {
                "optimiser":optimiser,
                "loss_function":loss_function,
                "loss_params":loss_params,
                "datasets":train,
                "test": False,
                "batch":batch,
                "supervised": supervised,
                "scheduler": scheduler,
                "random_stop": random_stop,
                "clip": clip,
                "clip_args":clip_args,
                "norm_loss":norm_loss,
                "extra_point_fun":extra_points_fun,
                "extra_point_args":extra_points_args,
                "maximise_first_N":maximise_first_N,
                "solver":solver
            }
            running , train_out = training_function(net,training_params)
            #Test
            test_params = {
                "optimiser":optimiser,
                "loss_function":loss_function,
                "loss_params":loss_params,
                "datasets":test,
                "test": True,
                "batch":batch,
                "supervised": supervised,
                "scheduler": scheduler,
                "random_stop": random_stop,
                "clip": clip,
                "clip_args":clip_args,
                "norm_loss":norm_loss,
                "extra_point_fun":extra_points_fun,
                "extra_point_args":extra_points_args,
                "maximise_first_N":maximise_first_N,
                "solver":solver
            }
            running_test, test_out = training_function(net, test_params)
            
            losses.append(running) #Store each epoch's losses 
            losses_test.append(running_test)

            print(name,epoch+start_epochs,"Training",running,"Testing",running_test,"Time",time.asctime(),"Start",start_time, end=" ")
            if log_grad and "grad" in train_out:
                print("grad",train_out["grad"], end = " ")
            
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
