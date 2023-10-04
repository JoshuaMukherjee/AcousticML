
import json
import traceback
import torch
from torch.utils.data import DataLoader


from Train_Network import train
import Networks, Loss_Functions
from Utilities import device
from Dataset import PointDataset


files = [
   "PN44","PN45"
]


def parse(params,name):

    start_epochs = params["start-epochs"]
    epochs = params["epochs"]

    if start_epochs == 0:
        net = getattr(Networks, params["net"])(**params["net-args"]).to(device)
    else:
        net = torch.load(name+".pth",map_location=torch.device(device))

    try:
        train_s = [torch.load("./Datasets/"+pth,map_location=torch.device(device)) for pth in params["train"] ]
        test_s =  [torch.load("./Datasets/"+pth,map_location=torch.device(device)) for pth in params["test"]  ]
    except Exception as e:
        print("Datasets not found, please generate using ```python3 Dataset.py```")

    batch = params["batch"]
    train_sets = [DataLoader(d,batch,shuffle=True) for d in train_s]
    test_sets = [DataLoader(d,batch,shuffle=True) for d in test_s]

    optimiser = getattr(torch.optim, params["optimiser"])(net.parameters(),**params["optimiser-args"])
    loss_function = getattr(Loss_Functions, params["loss-function"])
    
    if "loss-params" in params:
        loss_params = params["loss-params"]
    else:
        loss_params = {}
    
    supervised = params["supervised"]
   
    if "scheduler" in params:
        scheduler = getattr(torch.optim.lr_scheduler, params["scheduler"])(optimiser,**params["scheduler-args"])
    else:
        scheduler = None
    
    if "random-stop" in params:
        rand_stop = params["random-stop"]
    else:
        rand_stop = False
    
    if "clip" in params:
        clip = params["clip"]
        if "clip-args" in params:
            clip_params = params["clip-args"]
        else:
            clip_params = {}
    else:
        clip = False
        clip_params = {}
    
    if "log-grad" in params:
        log_grad = params["log-grad"]
    else:
        log_grad = False
    
    if "norm-loss" in params:
        norm_loss = True
    else:
        norm_loss = False

    
    train(net,start_epochs,epochs,train_sets,test_sets,optimiser,
        loss_function,loss_params, supervised, scheduler, 
        name, batch, rand_stop, clip, clip_params, log_grad,
        norm_loss)

if __name__ == '__main__':
    for file in files:
        try:
            print(file, "Parsing....")
            params = json.load(open("Params/"+file+".json","r"))
            parse(params,file)
        except Exception as e:
            print(traceback.format_exc())


